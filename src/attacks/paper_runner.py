"""Plan/submit/run/status/merge for attack producers, separate from evaluation."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import traceback

from attacks import paper_protocol as protocol
from paper import documents, manifest, sets
from paper.common import (ROOT, digest, environment, git_commit, load_config,
                          read_json, read_jsonl, sha256_file, write_json, write_jsonl)
from paper.run import POLICY_FILE, policy_hash


def attack_settings(path=None):
    default = ROOT / "configs/paper_attacks.json"
    return read_json(path or default) if path or default.exists() else protocol.settings()


def directory(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("run name must contain only letters, digits, underscores or hyphens")
    return protocol.ATTACK_ROOT / name


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield


def plan(experiment, tag=None, settings_path=None, scratch=None):
    cfg, settings = load_config(), attack_settings(settings_path)
    out = directory(f"{experiment}_{tag}" if tag else experiment)
    frozen = sets.frozen()
    policy = None
    if experiment in ("E5a", "E5b", "E5c", "E6"):
        if not POLICY_FILE.exists() or read_json(POLICY_FILE)["policy_hash"] != policy_hash(cfg):
            raise ValueError("freeze the reviewed E4dev policy before planning direct/locator attacks")
        policy = sha256_file(POLICY_FILE)
    budget = read_json(protocol.BUDGET_FILE) if experiment in ("E5b", "E5c") else None
    units = protocol.build_units(experiment, cfg, settings, budget=budget["steps"] if budget else None)
    # Capture the exact selected documents, not merely the parent frozen set.
    inputs = {u["record_id"]: frozen["documents"][u["record_id"]] for u in units}
    contract = {"experiment": experiment, "code": protocol.code_hash(), "config": digest(cfg),
                "settings": settings, "sets": protocol.set_hashes(), "units": digest(units),
                "documents": digest(inputs), "policy_file": policy, "budget": budget,
                "scratch": str(Path(scratch or cfg["scratch_root"]).expanduser().resolve()),
                "settings_path": str(Path(settings_path or ROOT / "configs/paper_attacks.json").resolve())}
    with lock(out / ".plan.lock"):
        if (out / "contract.json").exists():
            old = read_json(out / "contract.json")
            if {k: v for k, v in old.items() if k != "git"} != contract:
                raise ValueError("existing attack plan differs; use a new --tag")
            print(f"{out.name}: unchanged, {len(units)} units")
            return
        contract["git"] = git_commit()
        write_jsonl(out / "units.jsonl", units)
        write_json(out / "contract.json", contract)
    print(f"{out.name}: planned {len(units)} optimizer runs; manifests written only by merge")


def checked_plan(name):
    out = directory(name)
    contract = read_json(out / "contract.json")
    units = read_jsonl(out / "units.jsonl")
    cfg = load_config()
    current_sets = protocol.set_hashes()
    if (contract["code"] != protocol.code_hash() or contract["config"] != digest(cfg)
            or contract["sets"] != {k: current_sets.get(k) for k in contract["sets"]}
            or contract["units"] != digest(units)):
        raise ValueError("code, config, frozen sets or units changed since planning")
    current = attack_settings(contract["settings_path"] if Path(contract["settings_path"]).exists() else None)
    if current != contract["settings"]:
        raise ValueError("attack settings changed since planning")
    if contract["policy_file"] and (not POLICY_FILE.exists() or sha256_file(POLICY_FILE) != contract["policy_file"]):
        raise ValueError("frozen EBV policy changed since planning")
    if contract["budget"] and read_json(protocol.BUDGET_FILE) != contract["budget"]:
        raise ValueError("frozen attack budget changed since planning")
    return out, contract, units, cfg, digest(contract)


def completed(out, unit, cid, check_images=False):
    path = out / "records" / f"{unit['unit_id']}.json"
    if not path.exists():
        return None
    rec = read_json(path)
    if rec.get("contract") != cid or rec.get("unit") != unit:
        raise ValueError("finished result has a different contract/unit")
    if check_images:
        for snapshot in rec["snapshots"]:
            for image in snapshot["images"]:
                if not Path(image["path"]).is_file() or sha256_file(image["path"]) != image["sha256"]:
                    raise ValueError("finished PNG missing or changed; restore artifact before resuming")
    return rec


def run(name, indices):
    import torch
    from attacks.paper_objectives import DirectObjective, ReaderPool, SavedObjective
    from attacks.paper_optimizer import input_fingerprint, optimize

    out, contract, units, cfg, cid = checked_plan(name)
    settings = contract["settings"]
    pool = ReaderPool(cfg, settings)
    failures = []
    try:
        for index in indices:
            if not 0 <= index < len(units):
                raise ValueError(f"unit index {index} outside 0..{len(units)-1}")
            u = units[index]
            folder = Path(contract["scratch"]) / "paper_attacks" / name / cid[:16] / u["unit_id"]
            error_path = out / "errors" / f"{u['unit_id']}.json"
            with lock(folder / ".lock"):
                objective = None
                try:
                    pages = documents.load_pages([u["record_id"]])[u["record_id"]]
                    done = completed(out, u, cid, check_images=True)
                    if done:
                        if done["input_hash"] != input_fingerprint(pages):
                            raise ValueError("clean images changed since the completed run")
                        (folder / "state.pt").unlink(missing_ok=True)
                        print(f"SKIP {index} {u['attack_id']}", flush=True)
                        continue
                    for name_ in protocol.required_readers(u):
                        spec = settings["ocr_surrogate"] if name_ == "S" else cfg["readers"][name_]
                        if spec["revision"] == "PIN_IN_E0":
                            raise ValueError(f"pin {name_}'s revision before running attacks")
                    if not torch.cuda.is_available():
                        raise RuntimeError("paper model attacks require a CUDA GPU; CPU tests use injected toy objectives")
                    torch.cuda.reset_peak_memory_stats()
                    if u["family"] in ("direct", "locator"):
                        reference = None
                        if (folder / "state.pt").exists():
                            reference = torch.load(folder / "state.pt", map_location="cpu", weights_only=False).get("locator_reference")
                        objective = DirectObjective(u, cfg, settings, pool, pages, reference)
                    else:
                        objective = SavedObjective(u, cfg, settings, pool)
                    # The frozen labels and runtime clean loader must agree.
                    if documents.meta(u["record_id"]) != sets.frozen()["documents"][u["record_id"]]:
                        raise ValueError("clean metadata drift since set freezing")
                    rec = optimize(u, pages, objective, settings, folder, cid)
                    rec.update(environment=environment(), git=git_commit(),
                               readers={k: r.settings for k, r in pool.readers.items()})
                    if u["experiment"] == "E7":
                        # Descriptive covariate from clean localization only.
                        from attacks.paper_objectives import DirectObjective
                        clean_objective = DirectObjective({**u, "attacked_method": "D6"}, cfg, settings, pool, pages)
                        obs = clean_objective.observe(pages)
                        if u["task"] == "receipt":
                            b = obs.get("band")
                            area = (b[2]-b[0])*(b[3]-b[1]) if b else 0
                        else:
                            area = sum((b[2]-b[0])*(b[3]-b[1]) for r in obs["regions"] for b, e in r["pairs"])
                        rec["clean_band_area_fraction"] = area / sum(p.width*p.height for p in pages)
                        token_records = [p for p in rec["parity"] if p["source"] == "M1" and p["target"] == u["target"]]
                        if token_records:
                            rec["target_token_ids"] = token_records[0]["target_token_ids"]
                            rec["target_token_count"] = token_records[0]["target_tokens"]
                        for snapshot in rec["snapshots"]:
                            snapshot["clean_band_area_fraction"] = rec["clean_band_area_fraction"]
                            snapshot["target_token_count"] = rec.get("target_token_count")
                    write_json(folder / "result.json", rec)
                    write_json(out / "records" / f"{u['unit_id']}.json", rec)
                    (folder / "state.pt").unlink(missing_ok=True)
                    error_path.unlink(missing_ok=True)
                    print(f"DONE {index} {u['attack_id']} success={rec['snapshots'][-1]['verification']['success']}", flush=True)
                    del objective, rec
                except Exception as exc:
                    failures.append(index)
                    write_json(error_path, {"unit": u, "contract": cid, "error": repr(exc),
                                            "traceback": traceback.format_exc(), "environment": environment()})
                    print(f"FAILED {index}: {exc}", flush=True)
                    objective = None
                    pool.close()
                    pool = ReaderPool(cfg, settings)
    finally:
        pool.close()
    if failures:
        raise SystemExit(f"failed units {failures}; recorded under {out / 'errors'}")


def smoke_indices(units):
    # First cover distinct gradient sources, then fill to at least three units.
    selected, seen = [], set()
    for i, u in enumerate(units):
        if u["source"] not in seen:
            selected.append(i)
            seen.add(u["source"])
    selected.extend(i for i in range(len(units)) if i not in selected and len(selected) < 3)
    return selected


def status(name):
    out, contract, units, cfg, cid = checked_plan(name)
    done, errors, missing = 0, 0, []
    for i, u in enumerate(units):
        if completed(out, u, cid):
            done += 1
        else:
            missing.append(i)
            errors += int((out / "errors" / f"{u['unit_id']}.json").exists())
    print(json.dumps({"name": name, "done": done, "total": len(units), "errors": errors, "missing_units": missing}))
    return missing


def merge(name):
    from attacks.paper_optimizer import input_fingerprint

    out, contract, units, cfg, cid = checked_plan(name)
    records = [completed(out, u, cid, check_images=True) for u in units]
    if any(r is None for r in records):
        raise ValueError("cannot publish a partial manifest: finish or retry every planned unit")
    groups = {}
    for rec in records:
        rid = rec["unit"]["record_id"]
        if input_fingerprint(documents.load_pages([rid])[rid]) != rec["input_hash"]:
            raise ValueError(f"clean input changed since attack: {rid}")
        for snapshot in rec["snapshots"]:
            groups.setdefault(protocol.manifest_name(snapshot), []).append(snapshot)
    for key, rows in groups.items():
        # Validate before replacing the public manifest; E3/E4 never see half a run.
        candidate = out / f"{key}.candidate.jsonl"
        write_jsonl(candidate, rows)
        manifest.validate(candidate)
        target = manifest.MANIFEST_DIR / f"{key}.jsonl"
        with lock(manifest.MANIFEST_DIR / f".{key}.lock"):
            if target.exists() and read_jsonl(target) != rows:
                raise ValueError(f"{key} already belongs to a different run; move/archive it explicitly before publishing")
            write_jsonl(target, rows)
        candidate.unlink()
    write_jsonl(out / "records.jsonl", records)
    print(f"published {sum(len(v) for v in groups.values())} validated attacks: {', '.join(groups)}")


def freeze_budget(name):
    out, contract, units, cfg, cid = checked_plan(name)
    if contract["experiment"] != "E5a":
        raise ValueError("B* must come from E5a")
    records = [completed(out, u, cid, check_images=True) for u in units]
    if any(r is None for r in records):
        raise ValueError("finish every E5a unit before choosing B*")
    value = {**protocol.choose_budget(records), "run": name, "contract": cid,
             "records_hash": digest(records)}
    if protocol.BUDGET_FILE.exists() and read_json(protocol.BUDGET_FILE) != value:
        raise ValueError("a different B* is already frozen")
    write_json(protocol.BUDGET_FILE, value)
    print(json.dumps(value, indent=2))


def submit(name, smoke=False, dry=False):
    import shlex

    out, contract, units, cfg, cid = checked_plan(name)
    gate = smoke_indices(units)
    if smoke:
        indices = gate
    else:
        for i in gate:
            rec = completed(out, units[i], cid, check_images=True)
            if rec is None or not rec["parity"] or not all(s["saved_reload_equal"] for s in rec["snapshots"]):
                raise ValueError("three-unit smoke gate incomplete; submit --smoke, then inspect its records")
        indices = [i for i, u in enumerate(units) if not completed(out, u, cid, check_images=True)]
    if not indices:
        print("all units complete")
        return
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    s = cfg["slurm"]
    command = ["sbatch", f"--job-name=attack_{name}", f"--partition={s['partition']}", f"--gres={s['gres']}",
               f"--cpus-per-task={s['cpus']}", f"--mem={s['mem']}", f"--time={s['time']}", "--requeue",
               f"--array={','.join(map(str, indices))}%{s['concurrency']}", f"--output={logs}/%A_%a.out",
               f"--export=ALL,PP_ATTACK_RUN={name},PP_REPO_ROOT={ROOT}",
               str(ROOT / "scripts/paper_attacks/array.sbatch")]
    print(shlex.join(command))
    if not dry:
        subprocess.run(command, check=True)


def main(allowed=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "submit", "run", "smoke", "status", "merge", "freeze-budget", "init-settings"])
    p.add_argument("experiment", nargs="?")
    p.add_argument("--tag")
    p.add_argument("--settings")
    p.add_argument("--scratch")
    p.add_argument("--unit", type=int, nargs="+")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.command == "init-settings":
        path = Path(a.settings or ROOT / "configs/paper_attacks.json")
        if path.exists():
            raise SystemExit(f"settings already exist: {path}")
        write_json(path, protocol.settings())
        print(path)
        return
    if not a.experiment:
        p.error("experiment/run name is required")
    if allowed and a.experiment.split("_")[0] not in allowed:
        p.error(f"this entry point handles {', '.join(allowed)}")
    if a.command == "plan":
        plan(a.experiment, a.tag, a.settings, a.scratch)
    elif a.command == "run":
        if not a.unit:
            p.error("--unit is required")
        run(a.experiment, a.unit)
    elif a.command == "smoke":
        run(a.experiment, smoke_indices(checked_plan(a.experiment)[2]))
    elif a.command == "submit":
        submit(a.experiment, a.smoke, a.dry_run)
    elif a.command == "status":
        status(a.experiment)
    elif a.command == "merge":
        merge(a.experiment)
    elif a.command == "freeze-budget":
        freeze_budget(a.experiment)


if __name__ == "__main__":
    main()
