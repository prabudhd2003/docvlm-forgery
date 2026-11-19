# Poisoned Paperwork: targeted pixel attacks on document VLMs, and evidence-band verification

This folder holds every piece of code we ran (cluster and Colab, in one tree), the frozen
data lists, and the per-case result records. The paper plan, with exact models, set sizes
and budgets, is in [`PLAN.md`](PLAN.md).

## The question

Can someone who edits only the pixels of an uploaded receipt or resume (L∞ ≤ ε/255,
ε ∈ {2, 4, 8, 16}) make a vision-language model return a value of their choosing (10× the
receipt total, or the next degree up)? And can that be stopped without retraining?
An attack counts only if the **saved and reloaded PNG** returns the **exact** target.
Every defense is scored three ways: **C** correct and accepted, **W** wrong but accepted,
**R** rejected (sent to a person).

**EBV** (Evidence-Band Verification; called CBVR in the course write-ups and `csev` in the course code) asks the model where
the answer is printed, cuts a tight band and a taller band around it, reads each band
separately, and returns the answer only if both agree.

## Current results (all on Qwen2.5-VL-3B-Instruct unless noted)

**Clean accuracy:** CORD-val 96/100, SROIE-val 123/126, resumes 74/75, our 114 phone-photo
receipts 110/114. Donut: CORD 75/100, SROIE 46/126, ours 90/114.

**Standard targeted PGD, success on clean-correct documents**

| Dataset | ε=2 | ε=4 | ε=8 | ε=16 |
|---|---:|---:|---:|---:|
| CORD (val) | 0.0% | 62.5% | 87.5% | 89.6% |
| SROIE (val) | 0.0% | 37.4% | 67.5% | 91.1% |
| Our receipts (110) | 0.0% | 70.9% | 88.2% | 99.1% |
| Resumes | 0.0% | 1.4% | 5.4% | 23.0% |

**Image cleaning fails under EOT.** Against a plain attack, JPEG-85, resize-vote, multi-view and
re-worded questions cut success at ε=16 from 73.4% to 40–55%. On CORD at ε=16, uniform JPEG-aware
EOT beats all four at once on 87.5% of documents.

**EBV, reserved evaluation** (45 collected receipts frozen before evaluation; 396 saved attacks =
standard / uniform EOT / hard-view EOT × ε 4/8/16)

| Method | Clean C/W/R (45) | Attacked C/W/R (396) | Attacker's total accepted |
|---|---|---|---:|
| No defense | 44 / 1 / 0 | 64 / 332 / 0 | 313 |
| JPEG | | 115 / 281 / 0 | 260 |
| Full page and band must agree | | 60 / 0 / 336 | 0 |
| One located band | | 356 / 28 / 12 | 0 |
| **EBV** | **39 / 1 / 5** | **354 / 7 / 35** | **0** |

**Direct attacks on EBV** (100 gradient steps on both bands + 100 pipeline queries, 2 tries,
19 eligible receipts in a frozen hash order): broken on **4/19 at ε=4** (95% CI 9–43%) and
**8/19 at ε=8** (23–64%). Every success was the exact 10× target accepted by both bands;
every other run returned the true total. 80 runs, 11.7 A100-hours.

**Resumes** (60 reserved): clean 59/1/0 → recovery 56/2/2; 236 saved attacks 214/22/0 → 224/1/11.
Direct attack on recovery: **0/40 runs** (10 resumes, ε 8/16) produced the fake degree.

**Donut as reader** (Qwen locates, Donut reads): Qwen-built attacks do not transfer
(0 targeted under every method), so EBV only costs answers there (344 → 276 correct).

Not done yet: InternVL or any second white-box model, direct attacks on the comparison
defenses (5 exploratory runs only), direct attacks on the Donut pipeline.

## Layout

| Path | What |
|---|---|
| `src/` | course code: models, attacks, defenses, CBVR/EBV (`src/csev/`), staged pipeline (`src/stages/`). **`src/paper/`**: the paper's evaluation code (frozen sets, readers, EBV, runner, reports) |
| `scripts/` | cluster launchers (`submit_stage.sh`, `slurm/`), stage config builders, merge/status tools, Colab runners (`colab_*.py`), input bundling (`colab_inputs/`), diagnostics (localization, band, Donut geometry, method figure), and superseded versions we also ran (`earlier_versions/`) |
| `configs/` | frozen JSON config for every cluster run |
| `notebooks/` | 01–03 data; 04–05 receipt CBVR study and direct attack; 06–08 resumes; 09–10 Donut; `earlier_versions/` |
| `analysis/` | midterm reporting package and the cluster-era analysis notebooks (run them from `analysis/`) |
| `data/` | our 114 receipt photos + labels, `processed/` manifests and splits (`master_manifest.*`, degree mapping, exclusions), receipt-study role lists and hashes |
| `results/` | summaries and contracts per study, plus `raw/` per-case records (JSON/CSV/logs). `results/cluster/` = the CARC `outputs/`, `reports/` and `logs/` trees as merged tables, each record with runtime, peak GPU memory, SSIM/LPIPS and model revision. `results/paper_v1/` = paper experiments |
| `figures/` | poster and figures |
| `inputs/` | **not in Git.** Frozen input bundles with the saved attacks: `csev_full_inputs.zip` (4.4 GB), `resume_complete_inputs_v1.zip` (1.2 GB), `resume_development_inputs_v1.zip`; and `carc_records_full_2026-11-19.tar.gz` (all CARC records including per-worker shards) |
| `tests/` | CPU tests (`python -m pytest tests -q`; `analysis/` has its own) |

Kept elsewhere: CORD / SROIE / resume images (re-download with `notebooks/01`), the attack PNGs in the CARC
`outputs/` tree (`/scratch1`), and the
Colab working folders on Drive (optimizer state, attacked PNGs from the direct attacks).

## Running

```bash
bash scripts/setup_carc.sh && source scripts/activate_paperwork.sh   # conda env "paperwork", Python 3.11
export PYTHONPATH="$PWD/src"
python -m pytest tests -q
```

Cluster stages: `python scripts/prepare_stage_config.py <stage>` → `bash scripts/submit_stage.sh <stage> all <runtime_config>`
→ `python scripts/stage_status.py <stage> <run_id>`. Re-running a submit resumes it.
Colab studies: upload the zip from `inputs/` to Drive, set `DRIVE_FOLDER`, run the notebook.
Every runner writes a `contract.json` (hashes of code, inputs, settings) and refuses to resume into
a folder whose contract differs; change anything → new study ID.

Pinned readers: `Qwen/Qwen2.5-VL-3B-Instruct`, `naver-clova-ix/donut-base-finetuned-docvqa` (revisions in `configs/`).
Environment: torch 2.6.0+cu124, transformers 5.x (exact versions in `env/requirements.lock.txt`).

## Paper experiments (`src/paper/`)

Frozen sets, six readers + OCR, EBV and the comparison defenses, a Slurm array runner, and
the tables. Attacked images come in through attack manifests written by separate attack code
(format in `src/paper/manifest.py`). The attack producers now live in `src/attacks/paper_*.py`,
`src/attacks/surrogates.py`, and `scripts/paper_attacks/`. They cover E2a/b/c, E3b, E5a/b/c,
E6 and E7; pretrained GPU parity, memory and attack-success measurements remain to be run.
See the [attack commands and implementation notes](scripts/paper_attacks/README.md), then
[`PLAN.md`](PLAN.md#code) for the evaluation order.

```bash
export PYTHONPATH="$PWD/src"
python -m pytest tests/test_paper.py -q      # CPU
python -m paper.sets freeze                  # once
python -m paper.run plan E1 && python -m paper.run submit E1
python -m paper.run status E1 && python -m paper.run merge E1
python -m paper.report all                   # results/paper_v1/tables.md, tables/, figures/
```
