"""CPU-only, frozen schedules and contracts for paper attack producers."""
from __future__ import annotations

from pathlib import Path

from paper import documents, sets
from paper.common import ROOT, RESULTS_DIR, digest, load_config, read_json, sha256_file

ATTACK_ROOT = RESULTS_DIR / "attack_runs"
BUDGET_FILE = ATTACK_ROOT / "budget.json"
EXPERIMENTS = ("E2a", "E2b", "E2c", "E3b", "E5a", "E5b", "E5c", "E6", "E7")


def settings():
    """Keep the course's step sizes, view pool, sampling and frequency bias."""
    uniform = read_json(ROOT / "configs/jpeg_eot_attack/public_full.json")
    hard = read_json(ROOT / "configs/jpeg_eot_attack/public_frequency_hardview.json")
    return {
        "steps": 100, "checkpoint_interval": 50, "evaluation_interval": 10,
        "step_size_255": {"2": 0.5, "4": 1.0, "8": 2.0, "16": 4.0},
        "queries": 100, "refresh_interval": 10, "logit_tolerance": 0.01,
        "uniform_eot": {k: uniform[k] for k in ("attack", "semantic_prompts", "defense")},
        "hardview": {k: hard[k] for k in ("attack", "semantic_prompts", "defense")},
        "ocr_surrogate": {"model_id": "microsoft/trocr-base-printed", "revision": "PIN_IN_E0"},
        "locator_steps": 100,
        "locator_subtotal_prompt": 'Locate the subtotal and its label on this receipt. Exclude the final total, tax, cash and change. Return only JSON {"box":[left,top,right,bottom]} using coordinates from 0 to 1000 relative to the full image. If absent return {"box":null}.',
        "resume_targets": {"short": ["master", "bachelor"],
                           "long": ["certificate_or_diploma", "postgraduate"]},
    }


def code_hash():
    # Include transitive course helpers and entry points, including untracked
    # files; a git commit alone is insufficient in a working research tree.
    paths = sorted(p for root in (ROOT / "src", ROOT / "scripts/paper_attacks")
                   for p in root.rglob("*") if p.suffix in (".py", ".sbatch"))
    return digest({str(p.relative_to(ROOT)): sha256_file(p) for p in paths})


def set_hashes():
    return {p.name: sha256_file(p) for p in (sets.FROZEN, sets.DERIVED_ATTACK, sets.DERIVED_DIRECT) if p.exists()}


def receipt_targets(meta):
    """Fixed before optimization; change one non-leading digit, and a 3-digit value."""
    from normalization import normalize_receipt_total

    clean = meta["clean_target"]
    positions = [i for i, c in enumerate(clean) if c.isdigit()]
    i = positions[-1]
    digit = clean[:i] + str((int(clean[i]) + 1) % 10) + clean[i + 1:]
    unrelated = next(normalize_receipt_total(str(v), meta["dataset_name"])
                     for v in (731, 419, 863) if normalize_receipt_total(str(v), meta["dataset_name"])
                     not in (clean, meta["attack_target"], digit))
    return [("digit_flip", digit), ("x10", meta["attack_target"]), ("unrelated", unrelated)]


def build_units(experiment, cfg, attack_cfg, get_set=None, get_meta=None, budget=None):
    """One optimizer trajectory per unit; E5a emits three immutable snapshots."""
    if experiment not in EXPERIMENTS:
        raise ValueError(experiment)
    get_set, get_meta = get_set or sets.get, get_meta or documents.meta
    out = []

    def add(rid, source="M1", family="standard", eps=8, restart=0, method=None,
            steps=None, kind=None, target=None, **extra):
        m = get_meta(rid)
        resume = m["task"] != "receipt_total"
        row = {"experiment": experiment, "record_id": rid, "source": source, "family": family,
               "epsilon": eps, "restart": restart, "attacked_method": method,
               "steps": steps or attack_cfg["steps"], "target": target or m["attack_target"],
               "target_kind": kind or ("next_level" if resume else "x10"),
               "task": "resume" if resume else "receipt", **extra}
        row["unit_id"] = digest(row)[:24]
        row["attack_id"] = f"{experiment}/{source}/{family}/eps{eps}/{rid}/r{restart}/{row['unit_id']}"
        out.append(row)

    if experiment in ("E2a", "E2c"):
        for model in ("M1", "M2", "M3"):
            for rid in get_set("R-attack" if experiment == "E2a" else "Res-attack"):
                for eps in ((2, 4, 8, 16) if experiment == "E2a" else (8, 16)):
                    add(rid, source=model, eps=eps)
    elif experiment == "E2b":
        for model in ("M1", "M2"):
            for rid in get_set("R-attack")[:50]:
                for eps in (8, 16):
                    for family in ("uniform_eot", "hardview"):
                        add(rid, source=model, family=family, eps=eps)
    elif experiment == "E3b":
        for rid in get_set("R-attack")[:50]:
            for eps in (8, 16):
                add(rid, source="M1+M2+M3", family="ensemble", eps=eps)
    elif experiment == "E5a":
        for rid in get_set("R-conv")[:20]:
            for restart in range(3):
                add(rid, family="direct", method="D6", restart=restart, steps=500,
                    milestones=[100, 250, 500])
    elif experiment in ("E5b", "E5c"):
        if budget not in (100, 250, 500):
            raise ValueError("freeze B* from completed E5a before planning direct attacks")
        if experiment == "E5c":
            for rid in get_set("Res-direct")[:30]:
                for restart in range(2):
                    add(rid, family="direct", method="D6", eps=16, restart=restart, steps=budget)
        else:
            for method, epsilons, n in (("D6", (4, 8), 50), ("D7", (8,), 50),
                                         ("D8", (8,), 50), ("D4", (8,), 25)):
                for rid in get_set("R-direct")[:n]:
                    for eps in epsilons:
                        for restart in range(2):
                            add(rid, source="M1+M3" if method == "D7" else "M1", family="direct",
                                method=method, eps=eps, restart=restart, steps=budget,
                                surrogate="S" if method == "D8" else None)
    elif experiment == "E6":
        for rid in get_set("R-direct")[:30]:
            for restart in range(2):
                for mode in ("refusal", "subtotal"):
                    add(rid, family="locator", method="locator", restart=restart,
                        steps=attack_cfg["locator_steps"], locator_goal=mode)
    elif experiment == "E7":
        for rid in get_set("R-attack")[:50]:
            for kind, target in receipt_targets(get_meta(rid)):
                add(rid, family="target_ablation", kind=kind, target=target)
        for rid in get_set("Res-attack")[:50]:
            for kind, choices in attack_cfg["resume_targets"].items():
                target = next(v for v in choices if v != get_meta(rid)["clean_target"])
                add(rid, family="target_ablation", kind=kind, target=target)
    if not out:
        raise ValueError("the experiment has no eligible documents")
    if len({u["unit_id"] for u in out}) != len(out):
        raise ValueError("duplicate attack units")
    return out


def required_readers(unit):
    names = unit["source"].split("+")
    if unit.get("attacked_method") == "D8":
        names += ["OCR", "S"]
    return names


def manifest_name(row):
    return f"E2a_{row['source']}" if row["experiment"] == "E2a" else row["experiment"]


def choose_budget(records):
    """Count documents broken across restarts, never count runs as documents."""
    successes = {n: set() for n in (100, 250, 500)}
    for rec in records:
        snapshots = {r["steps"]: r for r in rec["snapshots"]}
        if set(snapshots) != set(successes):
            raise ValueError("incomplete E5a checkpoints")
        for n, snap in snapshots.items():
            if snap["verification"]["success"]:
                successes[n].add(rec["unit"]["record_id"])
    if not records:
        raise ValueError("no budget records")
    best = min(n for n in successes if len(successes[n]) >= len(successes[500]) - 1)
    return {"steps": best, "documents_broken": {str(k): len(v) for k, v in successes.items()},
            "rule": "smallest budget within one broken document of 500 steps"}
