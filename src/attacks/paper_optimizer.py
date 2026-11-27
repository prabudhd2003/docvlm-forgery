"""Checkpointed PGD and bounded query search; no success is inferred from loss."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from attacks.donut import project_linf
from attacks.jpeg_eot import frequency_aware_update
from attacks.qwen import save_reload
from models.donut import pil_to_rgb_tensor
from paper.common import digest, sha256_bytes, sha256_file


def same_pixels(a, b):
    """Same 8-bit image. k/255 can differ by one float ulp between NumPy and CUDA
    division, so compare on the 0..255 grid; off-grid values never match."""
    a, b = a.detach().float() * 255, b.detach().float() * 255
    if a.shape != b.shape or (a - a.round()).abs().max() > 1e-3 or (b - b.round()).abs().max() > 1e-3:
        return False
    return torch.equal(a.round(), b.round())


def atomic_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def input_fingerprint(pages):
    return digest([{"size": list(p.size), "rgb": sha256_bytes(p.convert("RGB").tobytes())} for p in pages])


def rank(verification, loss, unit):
    if unit.get("locator_goal") == "refusal":
        return (int(verification["success"]), -loss)
    if unit.get("locator_goal") == "subtotal":
        return (int(verification["success"]), verification.get("subtotal_iou") or 0.,
                verification["target_count"], -loss)
    return (int(verification["success"]), verification["target_count"], -loss)


def optimize(unit, pages, objective, settings, folder, contract, device="cuda"):
    """Injectable objective supports CPU crash/restart and query-budget tests.

E5a is one 500-step trajectory, with cumulative best candidates at 100/250/500.
Query counts are recorded at each snapshot; only the final snapshot includes
the remaining square-search queries. This avoids pretending prefix snapshots
have each received a separate 100-query search.
"""
    folder = Path(folder).resolve()
    clean = [pil_to_rgb_tensor(p.convert("RGB"), torch.device(device)) for p in pages]
    direct = unit["family"] in ("direct", "locator")
    epsilon, steps = unit["epsilon"], unit["steps"]
    checkpoints = unit.get("milestones", [steps])
    refresh = settings["refresh_interval"] if direct else settings["evaluation_interval"]
    # initial, ceil(steps / refresh) gradient observations, snapshots, final
    needed = 1 + (steps + refresh - 1) // refresh + len(checkpoints)
    if direct and settings["queries"] < needed:
        raise ValueError(f"query budget {settings['queries']} cannot cover {needed} required observations")
    state_path = folder / "state.pt"
    identity = digest({"contract": contract, "unit": unit, "inputs": input_fingerprint(pages)})
    rng = np.random.default_rng(int(unit["unit_id"][:16], 16))
    started = time.perf_counter()
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise RuntimeError("checkpoint code/config/unit/clean image contract changed")
        rng.bit_generator.state = state["rng"]
        for snapshot in state["snapshots"]:
            for image in snapshot["images"]:
                if sha256_file(image["path"]) != image["sha256"]:
                    raise RuntimeError("checkpoint snapshot PNG changed")
    else:
        current = [x.clone() for x in clean]
        if direct:
            current = [project_linf(x + torch.as_tensor(rng.integers(-epsilon, epsilon + 1, x.shape),
                       device=x.device, dtype=x.dtype) / 255, x, epsilon, quantize=True) for x in clean]
        state = {"identity": identity, "step": 0, "queries": 0, "trace": [],
                 "current": [x.cpu() for x in current], "best": None, "best_rank": None,
                 "observation": None, "snapshots": [], "weights": None, "loss": 1e30,
                 "seconds": 0., "diagnostics": [], "locator_reference": getattr(objective, "locator_reference", None)}
    prior = state["seconds"]

    def persist():
        state["rng"] = rng.bit_generator.state
        state["seconds"] = prior + time.perf_counter() - started
        diagnostics = getattr(getattr(objective, "cache", None), "diagnostics", [])
        state["diagnostics"] = prior_diagnostics + list(diagnostics)
        atomic_state(state_path, state)

    prior_diagnostics = list(state["diagnostics"])

    def readback(values, directory):
        output, images = save_reload(values, clean, directory, epsilon)
        for value, original, page in zip(values, clean, output):
            expected = project_linf(value.detach(), original, epsilon, quantize=True)
            if not same_pixels(pil_to_rgb_tensor(page, original.device), expected):
                raise RuntimeError("saved/reloaded PNG is not the candidate")
        return output, images

    def query(values, phase, loss, directory=None, update=True):
        if direct and state["queries"] >= settings["queries"]:
            raise RuntimeError("real-pipeline query budget exhausted")
        reloaded, images = readback(values, directory or folder / "query")
        verified = objective.verify(reloaded)
        state["queries"] += 1
        candidate_rank = rank(verified, loss, unit)
        if update and (state["best_rank"] is None or candidate_rank > tuple(state["best_rank"])):
            state["best_rank"] = candidate_rank
            state["best"] = [x.detach().cpu().clone() for x in values]
            state["best_step"] = state["step"]
        observation = verified.get("observation")
        if observation and (observation.get("band") or observation.get("band_count", 0)):
            state["observation"] = observation
        state["trace"].append({"step": state["step"], "query": state["queries"], "phase": phase,
            "loss": loss, "success": verified["success"], "prediction": verified.get("prediction"),
            "target_count": verified["target_count"], "model_calls": (observation or {}).get("calls")})
        return images, verified

    def snapshot(budget):
        values = [x.to(device) for x in state["best"]]
        images, verified = query(values, "saved_snapshot", state["loss"],
                                  folder / f"step_{budget:04d}", update=False)
        row = {k: v for k, v in unit.items() if k not in ("unit_id", "milestones", "task")}
        row.update(steps=budget, images=images, queries=state["queries"], verification=verified,
                   selected_step=state["best_step"], saved_reload_equal=True,
                   attack_record=str(folder / "result.json"))
        if unit.get("milestones"):
            row["attack_id"] += f"/steps{budget}"
        state["snapshots"].append(row)

    xs = [x.to(device).detach().requires_grad_(True) for x in state["current"]]
    if not state["queries"]:
        if not direct:
            state["loss"], state["weights"] = objective.score(xs)
        query(xs, "initial", state["loss"])
        persist()
    alpha = settings["step_size_255"][str(epsilon)] / 255
    for step in range(state["step"] + 1, steps + 1):
        loss, gradients = objective.gradient(xs, step, state["observation"], state["weights"])
        norm = float(sum(g.float().square().sum() for g in gradients).sqrt())
        if not np.isfinite(loss) or not np.isfinite(norm) or norm <= 0:
            raise RuntimeError("zero/nonfinite attack gradient; cannot count as robustness")
        with torch.no_grad():
            frequency = settings.get("hardview", {}).get("attack", {}).get("frequency_aware", {})
            if unit["family"] == "hardview" and frequency.get("enabled"):
                xs = [frequency_aware_update(x, c, g, alpha=alpha, epsilon_255=epsilon,
                       cutoff=frequency["cutoff"], gradient_mix=frequency["gradient_mix"],
                       delta_mix=frequency["delta_mix"]).detach().requires_grad_(True)
                      for x, c, g in zip(xs, clean, gradients)]
            else:
                xs = [project_linf(x - alpha * g.sign(), c, epsilon, quantize=True).detach().requires_grad_(True)
                      for x, c, g in zip(xs, clean, gradients)]
        del gradients
        state.update(step=step, loss=loss)
        if step % refresh == 0 or step in checkpoints or step == steps:
            if not direct:
                state["loss"], state["weights"] = objective.score(xs)
            query(xs, "gradient", state["loss"])
            state["trace"][-1]["gradient_norm"] = norm
        if step in checkpoints and step != steps:
            snapshot(step)
        if step % settings["checkpoint_interval"] == 0 or step % refresh == 0 or step in checkpoints:
            state["current"] = [x.detach().cpu() for x in xs]
            persist()
    # Full pipeline search addresses the locator's discrete crop selection.
    # Reserve one query for the final *saved* PNG, also on a resumed search.
    if direct:
        while state["queries"] < settings["queries"] - 1 and not state["best_rank"][0]:
            candidate = [x.to(device).clone() for x in state["best"]]
            j = int(rng.integers(len(candidate)))
            _, h, w = candidate[j].shape
            side = max(1, round(min(h, w) * (0.15 if state["queries"] < settings["queries"] // 2 else 0.05)))
            y, x = int(rng.integers(h - side + 1)), int(rng.integers(w - side + 1))
            signs = torch.as_tensor(rng.choice([-1, 1], (3, 1, 1)), device=device)
            candidate[j][:, y:y+side, x:x+side] = (clean[j][:, y:y+side, x:x+side] + signs * epsilon / 255).clamp(0, 1)
            candidate = [project_linf(v, c, epsilon, quantize=True) for v, c in zip(candidate, clean)]
            query(candidate, "query_search", state["loss"])
            persist()
    if not any(s["steps"] == steps for s in state["snapshots"]):
        snapshot(steps)
    persist()
    return {"unit": unit, "contract": contract, "input_hash": input_fingerprint(pages),
            "snapshots": state["snapshots"], "trace": state["trace"], "queries": state["queries"],
            "seconds": state["seconds"], "parity": state["diagnostics"],
            "locator_reference": state["locator_reference"],
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.,
            "memory_strategy": "fixed processor caps; non-reentrant checkpoints; pagewise vision and backward; sequential models"}
