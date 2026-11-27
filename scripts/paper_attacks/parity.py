#!/usr/bin/env python3
"""E0 GPU gate: 3 development receipts and optionally the largest two-page resume.

python scripts/paper_attacks/parity.py M2 --resume
Writes measured processor/logit parity, gradient norms, timings and peak VRAM.
No test images are used and no attack manifests are produced.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    from attacks.paper_runner import attack_settings
    from attacks.paper_objectives import ReaderPool
    from attacks.surrogates import make_objective, page_gradients
    from attacks.donut import project_linf
    from attacks.qwen import save_reload
    from attacks.paper_optimizer import same_pixels
    from models.donut import pil_to_rgb_tensor
    from paper import documents, sets
    from paper.common import RESULTS_DIR, digest, environment, load_config, write_json
    from attacks.paper_protocol import code_hash
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=["M1", "M2", "M3"])
    p.add_argument("--resume", action="store_true", help="also measure the largest two-page Res-dev document")
    p.add_argument("--settings")
    a = p.parse_args()
    cfg, settings = load_config(), attack_settings(a.settings)
    if cfg["readers"][a.model]["revision"] == "PIN_IN_E0":
        raise SystemExit("pin the model revision before recording E0")
    ids = sets.get("R-dev")[:3]
    if len(ids) != 3:
        raise RuntimeError("E0 requires three development receipts")
    if a.resume:
        largest = None
        for rid in sets.get("Res-dev"):
            if documents.meta(rid)["num_pages"] != 2:
                continue
            pages = documents.load_pages([rid])[rid]
            candidate = (sum(p.width * p.height for p in pages), rid)
            if largest is None or candidate > largest:
                largest = candidate
        if largest is None:
            raise RuntimeError("Res-dev has no two-page resume; cannot certify the required memory case")
        ids.append(largest[1])
    pool = ReaderPool(cfg, settings)
    records = []
    try:
        reader = pool.get(a.model)
        for rid in ids:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            pages = documents.load_pages([rid])[rid]
            task = documents.task_of(rid)
            target = documents.meta(rid)["attack_target"]
            xs = [pil_to_rgb_tensor(page, torch.device("cuda")).requires_grad_(True) for page in pages]
            objective = make_objective(reader, pages, cfg["prompts"][task], target)
            stats = objective.parity(xs, settings["logit_tolerance"])
            loss, gradients = page_gradients(objective, xs)
            norm = float(sum(g.float().square().sum() for g in gradients).sqrt())
            if not norm > 0:
                raise RuntimeError("zero gradient in E0")
            updated = [project_linf(x - 2 / 255 * g.sign(), x.detach(), 8, quantize=True)
                       for x, g in zip(xs, gradients)]
            with tempfile.TemporaryDirectory(prefix="paper_parity_") as tmp:
                reloaded, images = save_reload(updated, xs, Path(tmp), 8)
                for x, page in zip(updated, reloaded):
                    if not same_pixels(x, pil_to_rgb_tensor(page, x.device)):
                        raise RuntimeError("PNG reload changed E0 candidate")
                raw = reader.read(reloaded, cfg["prompts"][task])
            torch.cuda.synchronize()
            result = {"record_id": rid, "task": task, "page_sizes": [list(p.size) for p in pages],
                      **stats, "loss": loss, "gradient_norm": norm, "raw_one_step": raw,
                      "max_linf_255": max(i["linf_255"] for i in images),
                      "seconds": time.perf_counter() - started,
                      "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
            records.append(result)
            print(result, flush=True)
            del objective, xs, gradients, updated
        out = RESULTS_DIR / "attack_runs" / "parity" / f"{a.model}_{'resume' if a.resume else 'receipt'}.json"
        write_json(out, {"model": a.model, "reader": reader.settings, "code": code_hash(),
                        "config": digest(cfg), "environment": environment(), "records": records})
        print(out)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
