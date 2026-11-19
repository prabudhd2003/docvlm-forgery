"""Attack manifests: the only way attacked images enter the paper_v1 evaluation.

Attack code (written separately, not in this package) must write one JSON object per
line to ``results/paper_v1/manifests/<NAME>.jsonl``:

    {
      "attack_id":   "E2a/M1/standard/eps8/collected_receipts/external_test/c00012/r0",   unique
      "experiment":  "E2a" | "E2b" | "E2c" | "E3b" | "E5a" | "E5b" | "E5c" | "E6" | "E7",
      "record_id":   frozen record ID (must be in data/paper_v1 sets),
      "source":      "M1" | "M2" | "M3" | "M1+M3" | "M1+M2+M3",  VLM(s) whose gradients were used
      "family":      "standard" | "uniform_eot" | "hardview" | "ensemble" | "direct" | "locator" | "target_ablation",
      "attacked_method": null | "D4" | "D6" | "D7" | "D8" | "locator",   for direct attacks (E5*, E6)
      "epsilon":     integer, L-infinity budget on the 0-255 scale,
      "restart":     integer (0 if none),
      "steps":       integer, budget used (E5a: one row per checkpoint 100/250/500),
      "target":      attacker's target string, normalized like the answers,
      "target_kind": "x10" (default) | "digit_flip" | "unrelated" | "next_level" | ...   (E7)
      "locator_goal": "refusal" | "subtotal",               E6 only, kept separate in Table 4
      "surrogate":   "S" | null,                            TrOCR stand-in for D8
      "images":      [{"path": "/scratch1/.../page_000.png", "sha256": "..."}],   one per page, PNG
      "attack_record": optional path to the attack's own JSON (loss trace, runtime, peak memory)
    }

    python -m paper.manifest validate results/paper_v1/manifests/E2a_M1.jsonl
    python -m paper.manifest import-course     # course saved attacks on R-dev -> E4dev tuning only

``validate`` checks: unique IDs, frozen record, PNG exists and hash matches, page count and
size equal the clean document, L-infinity <= epsilon against the clean pixels, and the target
equals the frozen 10x / next-level target unless target_kind says otherwise.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from paper.common import RESULTS_DIR, ROOT, read_json, read_jsonl, sha256_file, write_jsonl

MANIFEST_DIR = RESULTS_DIR / "manifests"
EXPERIMENTS = {"E2a", "E2b", "E2c", "E3b", "E5a", "E5b", "E5c", "E6", "E7", "course"}
REQUIRED = ("attack_id", "experiment", "record_id", "source", "family", "epsilon", "target", "images")


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def validate(path: Path | str, check_pixels: bool = True) -> list[dict[str, Any]]:
    import numpy as np

    from paper import documents, sets

    rows = read_jsonl(path)
    frozen = sets.frozen()
    known = set(frozen["documents"])
    seen, errors = set(), []
    clean_cache: dict[str, list] = {}
    for i, r in enumerate(rows):
        where = f"line {i + 1} ({r.get('attack_id')})"
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            errors.append(f"{where}: missing {missing}"); continue
        if r["attack_id"] in seen:
            errors.append(f"{where}: duplicate attack_id"); continue
        seen.add(r["attack_id"])
        if r["experiment"] not in EXPERIMENTS:
            errors.append(f"{where}: unknown experiment {r['experiment']}")
        if r["record_id"] not in known:
            errors.append(f"{where}: record not in frozen sets"); continue
        meta = frozen["documents"][r["record_id"]]
        if r.get("target_kind", "x10") in ("x10", "next_level") and r["target"] != meta["attack_target"]:
            errors.append(f"{where}: target {r['target']!r} != frozen {meta['attack_target']!r}")
        if not isinstance(r["epsilon"], int) or not 0 < r["epsilon"] <= 32:
            errors.append(f"{where}: epsilon must be an int in 1..32")
        for img in r["images"]:
            p = _abs(img["path"])
            if p.suffix.lower() != ".png" or not p.is_file():
                errors.append(f"{where}: missing PNG {p}"); break
            if sha256_file(p) != img["sha256"]:
                errors.append(f"{where}: sha256 mismatch {p}"); break
        else:
            if check_pixels:
                if r["record_id"] not in clean_cache:
                    clean_cache.clear()
                    clean_cache[r["record_id"]] = documents.load_pages([r["record_id"]])[r["record_id"]]
                clean = clean_cache[r["record_id"]]
                if len(clean) != len(r["images"]):
                    errors.append(f"{where}: {len(r['images'])} pages, clean has {len(clean)}"); continue
                for c, img in zip(clean, r["images"]):
                    a = documents.load_image_file(_abs(img["path"]))
                    if a.size != c.size:
                        errors.append(f"{where}: size {a.size} != clean {c.size}"); break
                    linf = int(np.abs(np.asarray(a, dtype=np.int16) - np.asarray(c, dtype=np.int16)).max())
                    if linf > r["epsilon"]:
                        errors.append(f"{where}: L-inf {linf} > epsilon {r['epsilon']}"); break
    if errors:
        raise SystemExit(f"{len(errors)} problem(s) in {path}:\n  " + "\n  ".join(errors[:50]))
    print(f"OK: {len(rows)} attacks in {path}")
    return rows


def load(names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        p = Path(name)
        p = p if p.suffix == ".jsonl" else MANIFEST_DIR / f"{name}.jsonl"
        rows.extend(read_jsonl(p))
    ids = [r["attack_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate attack_id across manifests")
    return rows


def import_course() -> Path:
    """The course's saved attacks on the 120 R-dev receipts (images still on CARC)."""
    plan = read_json(ROOT / "data" / "receipt_study" / "plan.json")
    rows = []
    for split in ("development", "evaluation"):
        for d in plan[split]:
            for a in d["attacks"]:
                rows.append({"attack_id": f"course/{a['family']}/eps{a['epsilon']}/{d['record_id']}", "experiment": "course",
                             "record_id": d["record_id"], "source": "M1", "family": a["family"], "attacked_method": None,
                             "epsilon": int(a["epsilon"]), "restart": 0, "steps": None, "target": d["frozen_attack_target"],
                             "target_kind": "x10", "images": a["images"]})
    out = MANIFEST_DIR / "course_rdev.jsonl"
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} course attacks to {out}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["validate", "import-course"])
    p.add_argument("path", nargs="?")
    p.add_argument("--no-pixels", action="store_true")
    a = p.parse_args()
    if a.command == "validate":
        validate(a.path, check_pixels=not a.no_pixels)
    else:
        import_course()


if __name__ == "__main__":
    main()
