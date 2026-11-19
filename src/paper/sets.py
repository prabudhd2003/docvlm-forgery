"""Freeze the paper_v1 document sets, then derive the eligible subsets.

    python -m paper.sets freeze            # once, before any model reads a test document
    python -m paper.sets derive-attack     # after E1: R-attack, Res-attack
    python -m paper.sets derive-direct     # after E4 clean pass: R-direct, R-conv, Res-direct
    python -m paper.sets show

Every list is in the fixed hash order digest([566, 'paper_v1', record_id]).
Eligibility uses clean outputs only.
"""
from __future__ import annotations

import argparse
from typing import Any

from paper.common import ROOT, SETS_DIR, digest, hash_order, load_config, read_json, sha256_file, write_json
from paper import documents

FROZEN = SETS_DIR / "sets.json"
DERIVED_ATTACK = SETS_DIR / "derived_attack.json"
DERIVED_DIRECT = SETS_DIR / "derived_direct.json"
ROLES = ROOT / "data" / "receipt_study" / "data_roles.json"


def build(cfg: dict[str, Any]) -> dict[str, Any]:
    salt, seed = cfg["order_salt"], cfg["seed"]
    order = lambda ids: hash_order(ids, salt, seed)  # noqa: E731
    roles = read_json(ROLES)
    r_dev = roles["public_development"] + roles["collected_development"] + roles["collected_reserved_evaluation"]
    if len(r_dev) != 120 or len(set(r_dev)) != 120:
        raise SystemExit(f"expected 120 distinct receipt development documents, found {len(set(r_dev))}")
    collected = documents.collected_index()
    if len(collected) != 114:
        raise SystemExit(f"expected 114 collected receipts, found {len(collected)}")
    unused = [rid for rid in collected if rid not in set(r_dev)]
    want = cfg["sets"]["receipt_test"]
    if len(unused) != want["collected_unused"]:
        raise SystemExit(f"expected {want['collected_unused']} unused collected receipts, found {len(unused)}")
    pub = documents.public_index()
    pick = lambda ds, split, task: [r for r, m in pub.items() if m["dataset_name"] == ds and m["experiment_split"] == split and m["task"] == task]  # noqa: E731
    cord = order(pick("cord_v2", "test", "receipt_total"))[: want["cord_v2_test"]]
    sroie = order(pick("sroie", "test", "receipt_total"))[: want["sroie_test"]]
    r_test = order(unused + cord + sroie)
    if set(r_test) & set(r_dev):
        raise SystemExit("receipt test overlaps development")
    res_dev = order(pick("resume_parsing_vision", "test", "resume_highest_degree"))
    res_test = order(pick("resume_parsing_vision", "train", "resume_highest_degree"))[: cfg["sets"]["resume_test"]["resume_train"]]
    if len(res_dev) != 75:
        raise SystemExit(f"expected 75 resume development documents, found {len(res_dev)}")
    sets = {"R-dev": order(r_dev), "R-test": r_test, "Res-dev": res_dev, "Res-test": res_test}
    entries = {rid: documents.meta(rid) for ids in sets.values() for rid in ids}
    sources = {p: sha256_file(ROOT / p) for p in [
        "data/receipt_study/data_roles.json", "data/annotations/collected_receipt_labels.csv",
        "data/processed/master_manifest.csv"]}
    body = {"version": cfg["version"], "order": [seed, salt], "sets": sets, "documents": entries, "sources": sources}
    body["hash"] = digest({k: body[k] for k in ("sets", "documents", "sources")})
    return body


def freeze(cfg):
    body = build(cfg)
    if FROZEN.exists():
        old = read_json(FROZEN)
        if old["hash"] != body["hash"]:
            raise SystemExit("data/paper_v1/sets.json already exists with different contents; refusing to change frozen sets")
        print("Already frozen, unchanged:", body["hash"][:12])
    else:
        write_json(FROZEN, body)
        print("Frozen:", body["hash"][:12])
    for k, v in body["sets"].items():
        print(f"  {k}: {len(v)}")


def frozen() -> dict[str, Any]:
    if not FROZEN.exists():
        raise SystemExit("run `python -m paper.sets freeze` first")
    body = read_json(FROZEN)
    if digest({k: body[k] for k in ("sets", "documents", "sources")}) != body["hash"]:
        raise SystemExit("data/paper_v1/sets.json was edited after freezing")
    return body


def get(name: str) -> list[str]:
    for path in (FROZEN, DERIVED_ATTACK, DERIVED_DIRECT):
        if path.exists():
            body = frozen() if path == FROZEN else read_json(path)
            if name in body["sets"]:
                return list(body["sets"][name])
    raise KeyError(f"set {name} not frozen yet")


def _first(ids: list[str], ok: set[str], n: int) -> list[str]:
    return [r for r in ids if r in ok][:n]


def _write_derived(path, sets, inputs):
    body = {"sets": sets, "inputs": inputs, "parent": frozen()["hash"]}
    body["hash"] = digest(body)
    if path.exists() and read_json(path)["hash"] != body["hash"]:
        raise SystemExit(f"{path.name} already exists with different contents; refusing to change it")
    write_json(path, body)
    for k, v in sets.items():
        print(f"  {k}: {len(v)}")


def derive_attack(cfg):
    from paper.report import clean_correct

    s = cfg["sets"]
    ok = {m: clean_correct("E1", reader=m) for m in ("M1", "M2", "M3")}
    both = ok["M1"] & ok["M2"] & ok["M3"]
    sets = {"R-attack": _first(get("R-test"), both, s["receipt_attack"]),
            "Res-attack": _first(get("Res-test"), both, s["resume_attack"])}
    if len(sets["R-attack"]) < s["receipt_attack"]:
        print(f"NOTE: only {len(sets['R-attack'])} receipts are clean-correct under M1, M2 and M3; using all of them")
    _write_derived(DERIVED_ATTACK, sets, {m: sorted(v) for m, v in ok.items()})


def derive_direct(cfg):
    from paper.report import clean_correct

    s = cfg["sets"]
    ebv_test = clean_correct("E4", reader="M1", method="D6")
    ebv_dev = clean_correct("E4dev", reader="M1", method="D6")
    sets = {"R-direct": _first(get("R-attack"), ebv_test, s["receipt_direct"]),
            "R-conv": _first(get("R-dev"), ebv_dev, s["receipt_conv"]),
            "Res-direct": _first(get("Res-attack"), ebv_test, s["resume_direct"])}
    _write_derived(DERIVED_DIRECT, sets, {"E4": sorted(ebv_test), "E4dev": sorted(ebv_dev)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["freeze", "derive-attack", "derive-direct", "show"])
    p.add_argument("--config")
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.command == "freeze":
        freeze(cfg)
    elif a.command == "derive-attack":
        derive_attack(cfg)
    elif a.command == "derive-direct":
        derive_direct(cfg)
    else:
        for path in (FROZEN, DERIVED_ATTACK, DERIVED_DIRECT):
            if path.exists():
                for k, v in read_json(path)["sets"].items():
                    print(f"{k}: {len(v)}")


if __name__ == "__main__":
    main()
