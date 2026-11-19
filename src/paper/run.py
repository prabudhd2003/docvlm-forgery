"""Plan, submit, run, check and merge paper_v1 evaluation experiments.

    python -m paper.run plan E1
    python -m paper.run plan E4 --manifest E2a_M1 E2a_M2 E2a_M3 E2b E2c
    python -m paper.run plan E9 --manifest E2a_M1 --successful-on M1       # keeps M1 eps 8/16 attacks that worked on M1 (needs merged E3)
    python -m paper.run submit E1 [--dry-run] [--tasks 3,7]
    python -m paper.run status E1
    python -m paper.run merge E1
    python -m paper.run run E1 --task 0                                      # what each array task runs
    python -m paper.run freeze-policy                                        # after reviewing E4dev; before E4/E4r/E5/E8lat/E9

Each experiment lives in results/paper_v1/<EXP>/: units.jsonl, contract.json, records/<unit_id>.json,
errors/<unit_id>.json, records.jsonl (after merge). A unit whose record exists with the same
contract is skipped, so re-submitting only reruns what is missing or failed.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

from paper.common import RESULTS_DIR, ROOT, digest, git_commit, load_config, read_json, read_jsonl, sha256_file, write_json, write_jsonl

CODE_FILES = ["src/paper/common.py", "src/paper/documents.py", "src/paper/readers.py", "src/paper/ebv.py",
              "src/paper/experiments.py", "src/paper/manifest.py", "src/normalization.py", "src/defenses.py",
              "src/models/qwen.py", "src/models/donut.py", "src/data.py"]
POLICY_FILE = RESULTS_DIR / "E4dev" / "policy_frozen.json"


def exp_dir(name: str) -> Path:
    return RESULTS_DIR / name


def policy_hash(cfg) -> str:
    return digest({k: cfg[k] for k in ("ebv_policy", "baselines", "prompts")})


def config_hash(cfg) -> str:
    return digest({k: cfg[k] for k in ("readers", "generation", "prompts", "ebv_policy", "baselines", "upload_transforms")})


def code_hash() -> str:
    return digest({f: sha256_file(ROOT / f) for f in CODE_FILES})


def plan(name: str, manifests: list[str], tag: str | None, successful_on: str | None):
    from paper import experiments, manifest, sets

    cfg = load_config()
    exp = name
    out = exp_dir(f"{name}_{tag}" if tag else name)
    if exp in experiments.NEEDS_FROZEN_POLICY:
        if not POLICY_FILE.exists() or read_json(POLICY_FILE)["policy_hash"] != policy_hash(cfg):
            raise SystemExit("EBV policy is not frozen (or configs/paper_v1.json changed since). Review E4dev, then `python -m paper.run freeze-policy`.")
    attacks = manifest.load(manifests) if manifests else []
    if successful_on:
        from paper.report import load_records

        won = {r["unit"]["attack_id"] for r in load_records("E3") if r["unit"]["reader"] == successful_on and r["methods"]["D0"]["targeted"]}
        attacks = [a for a in attacks if a["source"] == successful_on and a["epsilon"] in (8, 16) and a["attack_id"] in won]
        print(f"{len(attacks)} attacks succeeded on {successful_on} at eps 8/16")
    us = experiments.units(exp, cfg, attacks)
    us.sort(key=lambda u: (experiments.readers_needed(u), u["unit_id"]))
    if len({u["unit_id"] for u in us}) != len(us):
        raise SystemExit("duplicate unit IDs")
    contract = {"experiment": exp, "directory": out.name, "units": digest([u["unit_id"] for u in us]),
                "config": config_hash(cfg), "code": code_hash(), "sets": sets.frozen()["hash"],
                "manifests": {m: sha256_file(manifest.MANIFEST_DIR / f"{m}.jsonl") if not m.endswith(".jsonl") else sha256_file(m) for m in manifests},
                "policy": policy_hash(cfg) if exp in experiments.NEEDS_FROZEN_POLICY else None}
    cpath = out / "contract.json"
    if cpath.exists():
        old = read_json(cpath)
        if {k: v for k, v in old.items() if k != "git"} != contract:
            changed = [k for k in contract if old.get(k) != contract[k]]
            raise SystemExit(f"{out.name} was planned with different {changed}. Finish it as is, or plan a new one with --tag.")
        print(f"{out.name}: already planned, unchanged ({len(us)} units)")
        return
    contract["git"] = git_commit()
    if contract["git"].get("dirty"):
        print("WARNING: uncommitted changes in src/configs/scripts; commit before submitting.")
    write_jsonl(out / "units.jsonl", us)
    write_json(cpath, contract)
    print(f"{out.name}: planned {len(us)} units -> {out}")


def _contract_id(out: Path) -> str:
    return digest(read_json(out / "contract.json"))[:16]


def run_task(name: str, task: int, per_task: int):
    try:
        import torch
    except ImportError:  # CPU tests only
        torch = None

    from paper import documents, experiments, readers
    from paper.common import environment

    cfg = load_config()
    out = exp_dir(name)
    contract = read_json(out / "contract.json")
    if contract["code"] != code_hash() or contract["config"] != config_hash(cfg):
        raise SystemExit("code or config changed since this experiment was planned; check out the planned commit")
    cid = _contract_id(out)
    us = read_jsonl(out / "units.jsonl")[task * per_task:(task + 1) * per_task]
    todo = []
    for u in us:
        p = out / "records" / f"{u['unit_id']}.json"
        if p.exists() and read_json(p).get("contract") == cid:
            continue
        todo.append(u)
    print(f"{name} task {task}: {len(us)} units, {len(todo)} to run", flush=True)
    if not todo:
        return
    env = environment()
    extras = {"environment": env}
    if name.split("_")[0] == "E8":
        import lpips

        extras["lpips"] = lpips.LPIPS(net="alex").eval().to("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    clean_ids = sorted({u["record_id"] for u in todo})
    clean_pages = documents.load_pages(clean_ids)
    R: dict = {}
    failures = 0
    try:
        for i, u in enumerate(todo):
            for need in experiments.readers_needed(u):
                if need not in R:
                    R[need] = readers.load_reader(need, cfg)
            try:
                rec = experiments.execute(u, R, cfg, clean_pages, extras)
                rec["contract"] = cid
                write_json(out / "records" / f"{u['unit_id']}.json", rec)
                err = out / "errors" / f"{u['unit_id']}.json"
                if err.exists():
                    err.unlink()
                print(f"[{i + 1}/{len(todo)}] {u['unit_id']} {u['record_id']} {u.get('attack_id') or 'clean'} "
                      f"{ {k: v['prediction'] for k, v in rec.get('methods', {}).items()} } {rec['seconds']:.1f}s", flush=True)
            except Exception as exc:  # record and continue; status shows it
                failures += 1
                write_json(out / "errors" / f"{u['unit_id']}.json", {"unit": u, "error": repr(exc), "traceback": traceback.format_exc()})
                print(f"ERROR {u['unit_id']}: {exc!r}", flush=True)
                if type(exc).__name__ == "OutOfMemoryError" and torch is not None:
                    torch.cuda.empty_cache()
    finally:
        for r in R.values():
            r.unload()
    if failures:
        raise SystemExit(f"{failures} unit(s) failed; see {out / 'errors'}")


def status(name: str, per_task: int):
    out = exp_dir(name)
    us = read_jsonl(out / "units.jsonl")
    cid = _contract_id(out)
    done, errors, missing_tasks = 0, [], set()
    for i, u in enumerate(us):
        p = out / "records" / f"{u['unit_id']}.json"
        if p.exists() and read_json(p).get("contract") == cid:
            done += 1
            continue
        e = out / "errors" / f"{u['unit_id']}.json"
        if e.exists():
            errors.append(read_json(e)["error"])
        missing_tasks.add(i // per_task)
    print(f"{name}: {done}/{len(us)} done, {len(errors)} failed, {len(us) - done - len(errors)} not run")
    for e in errors[:5]:
        print("  error:", e[:200])
    if missing_tasks:
        print("  tasks to rerun:", ",".join(map(str, sorted(missing_tasks))))
    return done == len(us)


def merge(name: str, partial: bool):
    out = exp_dir(name)
    us = read_jsonl(out / "units.jsonl")
    cid = _contract_id(out)
    rows, missing = [], 0
    for u in us:
        p = out / "records" / f"{u['unit_id']}.json"
        if p.exists() and (r := read_json(p)).get("contract") == cid:
            rows.append(r)
        else:
            missing += 1
    if missing and not partial:
        raise SystemExit(f"{missing} units missing; finish them or pass --partial")
    write_jsonl(out / "records.jsonl", rows)
    print(f"merged {len(rows)} records into {out / 'records.jsonl'}" + (f" ({missing} missing)" if missing else ""))


def freeze_policy():
    cfg = load_config()
    if not (exp_dir("E4dev") / "records.jsonl").exists():
        raise SystemExit("merge E4dev first")
    body = {"policy_hash": policy_hash(cfg), "ebv_policy": cfg["ebv_policy"], "baselines": cfg["baselines"],
            "e4dev_records_sha256": sha256_file(exp_dir("E4dev") / "records.jsonl"), "git": git_commit()}
    if POLICY_FILE.exists() and read_json(POLICY_FILE)["policy_hash"] != body["policy_hash"]:
        raise SystemExit("a different policy is already frozen; test results would no longer be untouched")
    write_json(POLICY_FILE, body)
    print("EBV policy frozen:", body["policy_hash"][:12])


def submit(name: str, per_task: int, dry: bool, tasks: str | None):
    cfg = load_config()["slurm"]
    us = read_jsonl(exp_dir(name) / "units.jsonl")
    n = math.ceil(len(us) / per_task)
    array = tasks or f"0-{n - 1}"
    logs = exp_dir(name) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cpu_only = name.split("_")[0] == "E8"
    gpu = [] if cpu_only else [f"--gres={cfg['gres']}"]
    partition = cfg.get("cpu_partition", "main") if cpu_only else cfg["partition"]
    cmd = ["sbatch", f"--job-name=pp_{name}", f"--partition={partition}", *gpu, f"--cpus-per-task={cfg['cpus']}",
           f"--mem={cfg['mem']}", f"--time={cfg['time']}", f"--array={array}%{cfg['concurrency']}", "--requeue",
           f"--output={logs}/%A_%a.out", f"--export=ALL,PP_EXP={name},PP_PER_TASK={per_task},PP_REPO_ROOT={ROOT}",
           str(ROOT / "scripts" / "paper" / "array.sbatch")]
    print(" ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["plan", "submit", "run", "status", "merge", "freeze-policy"])
    p.add_argument("experiment", nargs="?")
    p.add_argument("--manifest", nargs="*", default=[])
    p.add_argument("--tag")
    p.add_argument("--successful-on")
    p.add_argument("--task", type=int)
    p.add_argument("--per-task", type=int, default=int(os.environ.get("PP_PER_TASK", 0)) or None)
    p.add_argument("--tasks")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--partial", action="store_true")
    a = p.parse_args()
    k = a.per_task or load_config()["slurm"]["units_per_task"]
    if a.command == "plan":
        plan(a.experiment, a.manifest, a.tag, a.successful_on)
    elif a.command == "submit":
        submit(a.experiment, k, a.dry_run, a.tasks)
    elif a.command == "run":
        run_task(a.experiment, a.task, k)
    elif a.command == "status":
        sys.exit(0 if status(a.experiment, k) else 1)
    elif a.command == "merge":
        merge(a.experiment, a.partial)
    else:
        freeze_policy()


if __name__ == "__main__":
    main()
