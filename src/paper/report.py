"""Tables and figures from merged paper_v1 records.

    python -m paper.report all          # writes results/paper_v1/tables/*.csv, tables.md, figures/*.png

Each table is built only from the experiments that are merged; missing ones are skipped.
Rates come with 95% Wilson intervals over documents (a document counts as broken if any
of its runs/restarts in the group got the attacker's target accepted).
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path
from typing import Any

from paper.common import RESULTS_DIR, read_json, read_jsonl, wilson

TABLES = RESULTS_DIR / "tables"
FIGURES = RESULTS_DIR / "figures"
METHOD_NAMES = {"D0": "No defense", "D1": "JPEG-85", "D2": "Resize vote", "D3": "Multi-view vote",
                "D4": "Full page = band", "D5": "One band", "D6": "EBV-Base", "D7": "EBV-Cross", "D8": "EBV-OCR"}


def load_records(exp: str) -> list[dict[str, Any]]:
    d = RESULTS_DIR / exp
    if (d / "records.jsonl").exists():
        return read_jsonl(d / "records.jsonl")
    if (d / "records").is_dir():
        return [read_json(p) for p in sorted((d / "records").glob("*.json"))]
    return []


def source_group(record_id: str) -> str:
    if record_id.startswith("collected_receipts"):
        return "collected"
    return record_id.split("/")[0].replace("resume_parsing_vision", "resume").replace("cord_v2", "cord")


def clean_correct(exp: str, reader: str | None = None, method: str = "D0") -> set[str]:
    rows = load_records(exp)
    if not rows:
        raise SystemExit(f"no records for {exp}; run and merge it first")
    out = set()
    for r in rows:
        u = r["unit"]
        if u["kind"] != "clean":
            continue
        if reader and u.get("reader") not in (None, reader):
            continue
        if r["methods"][method]["correct"]:
            out.add(u["record_id"])
    return out


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci_low": round(lo, 4), "ci_high": round(hi, 4)}


def _write(name: str, rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    TABLES.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (TABLES / f"{name}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {name}.csv ({len(rows)} rows)")
    return rows


def table1_clean():
    rows = load_records("E1")
    g = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        key = (r["unit"]["reader"], source_group(r["unit"]["record_id"]))
        g[key][0] += r["methods"]["D0"]["correct"]
        g[key][1] += 1
    return _write("table1_clean_accuracy", [{"reader": m, "source": s, **_rate(*v)} for (m, s), v in sorted(g.items())])


def table2_attack_success():
    """Attack success on the model attacked (diagonal of E3) and transfer to every reader."""
    rows = load_records("E3")
    g = collections.defaultdict(dict)
    for r in rows:
        u = r["unit"]
        key = (u["source"], u["reader"], u["family"], u["epsilon"], "receipt" if source_group(u["record_id"]) != "resume" else "resume")
        g[key][u["record_id"]] = g[key].get(u["record_id"], False) or r["methods"]["D0"]["targeted"]
    out = []
    for (src, rd, fam, eps, task), docs in sorted(g.items(), key=lambda kv: tuple(map(str, kv[0]))):
        out.append({"source": src, "reader": rd, "same_model": src == rd, "family": fam, "epsilon": eps, "task": task,
                    **_rate(sum(docs.values()), len(docs))})
    return _write("table2_attack_success_and_transfer", out)


def table3_defenses():
    rows = load_records("E4")
    g = collections.defaultdict(list)
    for r in rows:
        u = r["unit"]
        task = "resume" if source_group(u["record_id"]) == "resume" else "receipt"
        g[(task, u["family"], u["epsilon"])].append(r)
    out = []
    for (task, fam, eps), rs in sorted(g.items(), key=lambda kv: tuple(map(str, kv[0]))):
        for m in sorted({m for r in rs for m in r["methods"]}):
            ok = [r for r in rs if m in r["methods"]]
            c = sum(r["methods"][m]["correct"] for r in ok)
            a = sum(r["methods"][m]["accepted"] for r in ok)
            t = sum(r["methods"][m]["targeted"] for r in ok)
            docs = collections.defaultdict(bool)
            for r in ok:
                docs[r["unit"]["record_id"]] |= r["methods"][m]["targeted"]
            out.append({"task": task, "family": fam, "epsilon": eps, "method": m, "name": METHOD_NAMES.get(m, m),
                        "images": len(ok), "correct": c, "wrong": a - c, "refused": len(ok) - a, "targeted": t,
                        "docs": len(docs), "docs_targeted": sum(docs.values()),
                        "correct_rate": round(c / len(ok), 4) if ok else None,
                        **{f"targeted_docs_{k}": v for k, v in _rate(sum(docs.values()), len(docs)).items() if k != "n"}})
    return _write("table3_defenses", out)


def table4_direct():
    rows = load_records("E5")
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        u = r["unit"]
        key = (u["attack_experiment"], u.get("attacked_method"), u["epsilon"], u.get("steps"), u.get("locator_goal"))
        m = u.get("attacked_method")
        decision = "D6" if m in (None, "locator") else m
        g[key][u["record_id"]].append(r["methods"][decision])
    out = []
    for (exp, m, eps, steps, goal), docs in sorted(g.items(), key=lambda kv: tuple(map(str, kv[0]))):
        broken = sum(any(o["targeted"] for o in v) for v in docs.values())
        refused_runs = sum(not o["accepted"] for v in docs.values() for o in v)
        runs = sum(len(v) for v in docs.values())
        out.append({"attack": exp, "attacked_method": m, "locator_goal": goal,
                    "epsilon": eps, "steps": steps, "docs_broken": broken,
                    **{f"docs_{k}": v for k, v in _rate(broken, len(docs)).items()},
                    "runs": runs, "runs_targeted": sum(o["targeted"] for v in docs.values() for o in v), "runs_refused": refused_runs})
    return _write("table4_direct_attacks", out)


def table5_cost():
    out = []
    stealth = collections.defaultdict(list)
    for r in load_records("E8"):
        u = r["unit"]
        for p in r["pages"]:
            stealth[(u["source"], u["family"], u["epsilon"])].append(p)
    for (src, fam, eps), ps in sorted(stealth.items(), key=lambda kv: tuple(map(str, kv[0]))):
        med = lambda k: sorted(x[k] for x in ps if k in x)[len(ps) // 2] if any(k in x for x in ps) else None  # noqa: E731
        out.append({"part": "stealth", "source": src, "family": fam, "epsilon": eps, "pages": len(ps),
                    "ssim_median": med("ssim"), "psnr_median": med("psnr"), "lpips_median": med("lpips")})
    lat = collections.defaultdict(list)
    for r in load_records("E8lat"):
        lat[r["unit"]["method"]].append(r["latency"])
    for m, xs in sorted(lat.items()):
        out.append({"part": "latency", "method": m, "name": METHOD_NAMES[m], "docs": len(xs),
                    "seconds_mean": round(sum(x["seconds"] for x in xs) / len(xs), 2),
                    "peak_gpu_gb_max": round(max(x["peak_gpu_gb"] for x in xs), 1)})
    up = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in load_records("E9"):
        u = r["unit"]
        for m, o in r["methods"].items():
            up[(u["transform"], u["epsilon"], m)][u["record_id"]].append(o["targeted"])
    for (t, eps, m), docs in sorted(up.items(), key=lambda kv: tuple(map(str, kv[0]))):
        out.append({"part": "upload", "transform": t, "epsilon": eps, "method": m,
                    **{f"still_targeted_{k}": v for k, v in _rate(sum(any(v) for v in docs.values()), len(docs)).items()}})
    return _write("table5_stealth_cost_upload", out)


def fig_transfer(rows2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in rows2 if r["family"] in ("standard", "ensemble") and r["task"] == "receipt" and r["epsilon"] in (8, 16)]
    if not rows:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    readers = sorted({r["reader"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, eps in zip(axes, (8, 16)):
        srcs = sorted({r["source"] for r in rows if r["epsilon"] == eps})
        mat = np.full((len(srcs), len(readers)), np.nan)
        for r in rows:
            if r["epsilon"] == eps:
                mat[srcs.index(r["source"]), readers.index(r["reader"])] = 100 * r["rate"]
        im = ax.imshow(mat, vmin=0, vmax=100, cmap="Reds")
        ax.set_xticks(range(len(readers)), readers)
        ax.set_yticks(range(len(srcs)), srcs)
        ax.set_xlabel("reader")
        ax.set_ylabel("attack built on")
        ax.set_title(f"Receipts, targeted success %, eps={eps}")
        for i in range(len(srcs)):
            for j in range(len(readers)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(FIGURES / "transfer_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_cwr(rows3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows3 if r["task"] == "receipt" and r["family"] != "clean"]
    if not rows:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        a = agg[r["method"]]
        a[0] += r["correct"]; a[1] += r["wrong"] - r["targeted"]; a[2] += r["targeted"]; a[3] += r["refused"]
    ms = sorted(agg)
    tot = [sum(agg[m]) for m in ms]
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = [0.0] * len(ms)
    for idx, (label, color) in enumerate([("correct", "#4c9a5e"), ("other wrong", "#d9a441"), ("attacker's total", "#c0392b"), ("refused", "#9aa5b1")]):
        vals = [100 * agg[m][idx] / t for m, t in zip(ms, tot)]
        ax.bar([METHOD_NAMES[m] for m in ms], vals, bottom=bottom, color=color, label=label)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("% of attacked receipts")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(FIGURES / "defense_outcomes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_budget():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in load_records("E5") if r["unit"]["attack_experiment"] == "E5a"]
    if not rows:
        return
    by = collections.defaultdict(lambda: collections.defaultdict(bool))
    for r in rows:
        by[r["unit"]["steps"]][r["unit"]["record_id"]] |= r["methods"]["D6"]["targeted"]
    steps = sorted(by)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(steps, [100 * sum(by[s].values()) / len(by[s]) for s in steps], marker="o")
    ax.set_xlabel("gradient steps")
    ax.set_ylabel("% of receipts broken (any restart)")
    ax.set_title("Direct attack on EBV-Base, eps=8")
    fig.savefig(FIGURES / "budget_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def markdown(tables: dict[str, list[dict]]):
    lines = ["# paper_v1 tables (generated by `python -m paper.report all`)", ""]
    for name, rows in tables.items():
        if not rows:
            continue
        keys = list(dict.fromkeys(k for r in rows for k in r))
        lines += [f"## {name}", "", "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        lines += ["| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + " |" for r in rows]
        lines.append("")
    (RESULTS_DIR / "tables.md").write_text("\n".join(lines))
    print("wrote tables.md")


def main():
    argparse.ArgumentParser().parse_args()
    t = {"table1": table1_clean(), "table2": table2_attack_success(), "table3": table3_defenses(),
         "table4": table4_direct(), "table5": table5_cost()}
    fig_transfer(t["table2"])
    fig_cwr(t["table3"])
    fig_budget()
    markdown(t)


if __name__ == "__main__":
    main()
