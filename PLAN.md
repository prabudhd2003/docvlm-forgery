# Paper plan (frozen 2026-11-19)

**Target:** ICDAR 2027, full paper due **20 Feb 2027** (check the abstract deadline on icdar2027.org).
Backup: a CVPR 2027 archival workshop; then SaTML 2028 (its deadline is usually late September) with a longer version.
Work starts after the course ends (final report due 7 Dec 2026). All jobs run on CARC L40S.

## Claims the paper will make

1. Instruction-tuned document VLMs from three families can be made to return an attacker-chosen
   **number** from a phone-photo receipt at small budgets; replacing a **word** (resume degree) is much harder.
2. Image cleaning (JPEG, resize vote, multi-view, re-worded questions) fails once the attack optimizes through it.
3. Evidence-Band Verification (EBV; CBVR in the course write-ups) recovers the true answer without retraining; it is scored as correct / wrong / refused.
4. An attacker who targets EBV partly breaks it. We measure exactly how much, and test whether
   reading the two bands with **different readers** (another VLM family, or OCR) closes the gap.
   Claim 4b is a question: the paper is written to stand whether it closes the gap or not.

## Models (pin the exact revision hash in E0; refuse to run on a different one)

| ID | Hugging Face model | Role | Why |
|---|---|---|---|
| M1 | `Qwen/Qwen2.5-VL-3B-Instruct` | white-box attack + reader | all existing results |
| M2 | `Qwen/Qwen2.5-VL-7B-Instruct` | white-box attack + reader | scale, same architecture |
| M3 | `OpenGVLab/InternVL3_5-8B-HF` | white-box attack + reader | different vision encoder (InternViT, tiling) |
| T1 | `google/gemma-3-4b-it` | transfer only | SigLIP encoder, no shared weights or tokenizer |
| T2 | `Qwen/Qwen3-VL-8B-Instruct` | transfer only | newer Qwen generation |
| T3 | `naver-clova-ix/donut-base-finetuned-docvqa` | transfer only | bridge to Pintore et al. (ICDAR 2026) |
| OCR | PaddleOCR PP-OCRv4 English recognizer, run on bands only | second reader in EBV-OCR | non-ViT, non-differentiable |
| S | `microsoft/trocr-base-printed` | attacker's differentiable stand-in for OCR | for the adaptive attack on EBV-OCR |

Optional, only if E3 is done by 4 Jan: one commercial API on the 100 ensemble images, capped at $10.

## Frozen document sets

Selection order is always the hash order `digest([566, 'paper_v1', record_id])`, the same scheme as the
existing studies. The lists and their SHA-256 are written to `data/paper_v1/` in E0, before any
model reads a test document. Eligibility is decided from clean outputs only.

**Receipts**

| Set | Size | Contents | Use |
|---|---:|---|---|
| R-dev | 120 | 60 CORD/SROIE validation + 15 collected dev + the 45 old "reserved" collected receipts | all tuning; anything already looked at lives here |
| **R-test** | **154** | the **54** collected receipts not used in the course CBVR study + first **50 CORD test** + first **50 SROIE test** (never attacked or inspected) | every reported number |
| R-attack | 100 | first 100 R-test receipts that M1, M2 and M3 all read correctly when clean (if fewer than 100, all of them) | saved attacks, transfer, defenses |
| R-direct | 50 | first 50 R-attack receipts that EBV-Base (M1) answers correctly when clean | direct attacks on defenses |
| R-conv | 20 | first 20 R-dev receipts that EBV-Base answers correctly when clean | choosing the attack budget only |

**Resumes**

| Set | Size | Contents | Use |
|---|---:|---|---|
| Res-dev | 75 | the test split already used (15 dev + 60 reserved) | tuning |
| **Res-test** | **100** | first 100 of the 850 train-split resumes (models are frozen; we never used this split) | reported numbers |
| Res-attack | 50 | first 50 Res-test resumes all three white-box models read correctly | saved attacks |
| Res-direct | 30 | first 30 Res-attack resumes that recovery answers correctly when clean | direct attack |

Targets as before: 10× the normalized total; the next degree level up.

## Defenses compared (same images for all)

D0 none · D1 JPEG-85 · D2 resize vote · D3 multi-view vote · D4 full-page and band must agree · D5 one located band ·
**D6 EBV-Base** (one model reads both bands; run with M1, M2 and M3 as reader) ·
**D7 EBV-Cross** (tight band read by M1, taller band by M3) ·
**D8 EBV-OCR** (tight band read by M1, taller band by OCR).
Dropped: random crop jitter (EOT is known to beat it), subtotal + tax arithmetic (missing on many receipts, and an attacker can forge all three), masked-band (kept only as an existing result).

## Experiments

Settings not listed keep the values in the existing configs (step size, max image size, prompts).
"Run" = one (document, model, ε, attack, restart) unit. Hours are **L40S-hour estimates** from measured M1 costs
in `results/cluster/` (median per attack: receipt PGD 0.8 min / 19.3 GB peak; receipt uniform EOT 2.4 min and
hard-view EOT 3.2 min / 20.4 GB; resume PGD 4.1 min / **42.3 GB**) and the EBV direct attack (8.8 A100-min per run),
with 7-8B models at about 2.5× the time. E0 measures M2 and M3 and replaces this column.

**Memory:** M1 on resumes already uses 42 of the L40S's 48 GB, so M2 and M3 will not fit on resumes as the
code stands. E0 must add, and test on the largest two-page resume: gradient checkpointing, one page per backward
pass (sum the page gradients), and a fixed max image size per page. Receipts (about 20 GB on M1) should fit M2
and M3 with checkpointing; the EBV-Cross attack (M1 + M3 in memory together) is the other case to measure first.

| # | Experiment | Exact design | Runs | Est. h | Fills |
|---|---|---|---:|---:|---|
| E0 | Setup and parity | pin revisions; differentiable preprocessing for M2 (same code path as M1) and M3 (tile grid fixed from the clean image); parity with the real processor (max abs logit gap ≤ 0.01, current M1 value 0.0045); 3-doc smoke per model; timing and peak memory on L40S; write `data/paper_v1/` lists | — | 10 | Methods |
| E1 | Clean accuracy | 6 readers × (R-test 154 + Res-test 100), full page | 1,524 | 5 | Table 1 |
| E2a | Standard PGD | M1, M2, M3 × R-attack 100 × ε {2,4,8,16}, 100 steps | 1,200 | 35 | Table 2 |
| E2b | EOT attacks | uniform EOT + hard-view EOT × ε {8,16} × M1, M2 × first 50 of R-attack | 400 | 35 | Table 2, Table 3 |
| E2c | Resume PGD | M1, M2, M3 × Res-attack 50 × ε {8,16} | 300 | 40 | Table 2 |
| E3a | Transfer matrix | every ε 8/16 image from E2a (600) read by all 6 readers | 3,600 inferences | 10 | Fig. 2 |
| E3b | Ensemble attack | joint loss over M1+M2+M3, first 50 of R-attack × ε {8,16} → read by T1, T2, T3 | 100 | 35 | Fig. 2 |
| E4 | Defense comparison | tune band padding / fallback on R-dev only (goal: clean refusals < 5%), freeze the policy hash, then one pass of D0–D8 on R-test clean + all E2 receipt images | ~1,750 images | 25 | Table 3, Fig. 3 |
| E5a | Attack budget | EBV-Base (M1) on R-conv 20, ε 8, 3 restarts, 500 steps with checkpoints at 100/250/500, 100 queries | 60 | 40 | Fig. 4 (appendix) |
| E5b | Direct attacks | at the chosen budget B*: D6 (M1) × ε {4,8} × 2 restarts (200 runs); D7 joint loss on M1+M3, ε 8 × 2 (100); D8 with TrOCR stand-in + 100 real queries, ε 8 × 2 (100); D4 on 25 receipts, ε 8 × 2 (50), all on R-direct | 450 | 190 | Table 4 |
| E5c | Resume direct attack | recovery on Res-direct 30, ε 16 × 2 restarts at B* | 60 | 25 | Table 4 |
| E6 | Locator attack | R-direct first 30, ε 8 × 2 restarts; push the predicted box off the total (refusal) and onto the subtotal (forgery) | 60 | 10 | Table 4 |
| E7 | Why numbers break | M1 standard ε 8 on R-attack 50 with 3 targets (one digit changed, 10×, unrelated 3-digit value); resumes 50 with short vs long degree target; log target tokens and band area | 250 | 13 | Fig. 5 |
| E8 | Stealth and cost | SSIM, PSNR, LPIPS for all E2 images; latency and peak VRAM per document for D0, D6, D7, D8 on 50 clean receipts | — | 5 | Table 5 |
| E9 | Upload pipeline | successful M1 ε 8/16 attacks through JPEG q75, q85 and downscale to 1600 px max side, re-read by D0 and D6 | ~600 inferences | 5 | Table 5 |
| | **Total** | | | **≈ 480** | |

**B\*** = the smallest step count in E5a whose receipt success rate is within one receipt of the 500-step result.
If E0 timings put the total above 600 L40S-hours, cut in this order: EOT on M2 → resume attacks on M3 → D4 direct
attacks → ensemble down to 25 receipts → D8 direct attack.

Dropped from the Gemini plan: 5×5 white-box matrix (3 white-box + 3 transfer-only instead), physical printing,
invoice domain, 500 steps × 3 restarts everywhere (replaced by E5a), EBV-Jitter, EBV-Arith.

## Paper tables

1. Clean accuracy, 6 readers × 4 sources. 2. Targeted success by model × ε × attack, with 95% CIs over documents.
3. Defenses, C/W/R and attacker-total counts on clean and attacked R-test. 4. Direct attacks on D4, D6, D7, D8 and
the locator, documents broken / runs. 5. Stealth, latency, VRAM, upload survival. Figures: method, transfer matrix,
C/W/R bars, budget curve, target ablation.

## Schedule

| Dates | Work |
|---|---|
| 8–14 Dec | E0, freeze `data/paper_v1/`, E1 |
| 15–28 Dec | E2 (largest queue; holidays should leave L40S free); start E3a as attacks finish |
| 29 Dec – 4 Jan | E3, E4 (tune on dev → freeze → test), E5a |
| 5–18 Jan | E5b, E5c, E6 |
| 19–25 Jan | E7, E8, E9. **All tables frozen 25 Jan** |
| 26 Jan – 13 Feb | writing; send the draft to Prof. Zhao by 6 Feb |
| 20 Feb | submit |

## Rules for CARC jobs

- One entry point per experiment, e.g. `python -m paper.run E2a --unit $SLURM_ARRAY_TASK_ID` (new code in `src/paper/`,
  reusing `src/attacks` and `src/csev`). The unit list (`units.jsonl`) is written once and hashed into the contract.
- Each unit writes `runs/<unit_hash>/result.json` plus its PNG, and **skips itself if a matching result exists**,
  so a crash or timeout loses one unit, never a batch. 500-step units checkpoint every 50 steps.
- Slurm array with a concurrency cap (`%8`), a per-unit time limit and `--requeue`.
- Smoke gate before every full array: 3 units must pass (reloaded PNG equals the candidate, L∞ ≤ ε, target check).
- Every result records GPU name, torch / transformers versions, model revision and git commit.
- Heavy outputs go to `/scratch1`; only records are copied into `results/paper_v1/`. No images or tensors in Git.
- Keep the resume checkpoint (`state.pt`) only while a unit is running; delete it when the unit's `result.json` is written.
  The old runs left 1,872 of them (299 GB).
- Commits carry only human authors (no AI co-author lines).

## Integrity

Test lists are frozen before any model reads them. Anything tuned is tuned on dev. Every run is reported,
including failures. Rates come with 95% Wilson intervals over documents, not over runs, and both counts are given.
Agree authorship and who continues among the five of us, and ask Prof. Zhao to advise, before E0.

## Code

`src/paper/` holds everything that **reads, defends and evaluates**. Everything that **creates attacked
images** is separate, under `src/attacks/` and `scripts/paper_attacks/` (list below). The two meet in one file format: the
**attack manifest** (`src/paper/manifest.py` documents every field). Attack code writes
`results/paper_v1/manifests/<NAME>.jsonl`; `python -m paper.manifest validate <file>` checks hashes,
page sizes, L∞ ≤ ε against the clean pixels, and that the target is the frozen one.

| File | What it does |
|---|---|
| `configs/paper_v1.json` | readers + revisions, prompts, EBV policy, comparison-defense settings, set sizes, Slurm resources |
| `src/paper/sets.py` | freezes R-dev 120 / R-test 154 / Res-dev 75 / Res-test 100; later derives R-attack, Res-attack (after E1) and R-direct, R-conv, Res-direct (after E4) |
| `src/paper/documents.py` | labels, targets and clean pages for every record (collected-receipt IDs match the course: checked on all 60) |
| `src/paper/readers.py` | M1, M2 (course Qwen wrapper), M3, T1, T2 (any chat VLM), T3 Donut, PaddleOCR; `pin` and `smoke` commands; OCR total extraction |
| `src/paper/ebv.py` | EBV for receipts and resumes (geometry identical to the course code, tested), D0, D4, D5, D6, D7, D8 from one observation |
| `src/paper/experiments.py` | units and execution for E1, E4dev, E4, E4r, E3, E5, E7, E8, E8lat, E9 |
| `src/paper/run.py` | plan / submit / run / status / merge / freeze-policy; contracts; skip-if-done; per-unit error files |
| `src/paper/report.py` | Tables 1–5 with Wilson intervals over documents, transfer matrix, defense outcome bars, budget curve |
| `scripts/paper/array.sbatch` | one Slurm array task |
| `tests/test_paper.py` | 18 CPU tests (`python -m pytest tests/test_paper.py -q`) |

### Attack implementation (GPU validation pending)

The producers below are implemented. See [`scripts/paper_attacks/README.md`](scripts/paper_attacks/README.md)
for commands, memory measures, query accounting and the E0 parity gate. CPU checks do not establish
pretrained-model parity, L40S memory use or empirical attack success. The E6 schedule was clarified
to **120 runs: 60 refusal + 60 subtotal-forgery**, rather than the 60 combined runs in the original
experiment estimate above; its time estimate must be updated after E0.

| Gap | Produces manifest | Notes |
|---|---|---|
| Differentiable preprocessing + target loss for M2 and M3, with the parity check | — | `src/attacks/surrogates.py`; `scripts/paper_attacks/parity.py` |
| E2a/E2b/E2c saved attacks on R-attack / Res-attack | `E2a_M1`, `E2a_M2`, `E2a_M3`, `E2b`, `E2c` | `scripts/paper_attacks/saved.py`; one row per attacked document/budget/family/source |
| E3b ensemble attack | `E3b` | `scripts/paper_attacks/ensemble.py`; source `M1+M2+M3` |
| E5a budget curve, E5b/E5c direct attacks, E6 locator attack | `E5a`, `E5b`, `E5c`, `E6` | `scripts/paper_attacks/direct.py` and `locator.py`; final reloaded PNG per run/checkpoint |
| E7 target ablation | `E7` | `scripts/paper_attacks/targets.py`; target kinds, token counts and clean band area |

### Order of commands (each GPU step: `plan` → `submit` → `status` → `merge`)

```bash
export PYTHONPATH="$PWD/src"
python -m paper.readers pin                    # put the printed hashes in configs/paper_v1.json, commit, pull on CARC
python -m paper.readers smoke M2               # one GPU each for M2, M3, T1, T2, OCR: loads, reads 3 receipts, prints peak memory
python -m paper.sets freeze                    # commit data/paper_v1/sets.json
python -m paper.run plan E1;    python -m paper.run submit E1
python -m paper.sets derive-attack             # R-attack, Res-attack; commit
python -m paper.manifest import-course
python -m paper.run plan E4dev --manifest course_rdev; python -m paper.run submit E4dev
#   review E4dev; tune ebv_policy in the config on dev only; re-plan with --tag if changed
python -m paper.run freeze-policy              # commit results/paper_v1/E4dev/policy_frozen.json
#   [attack side] E2 manifests
python -m paper.run plan E3 --manifest E2a_M1 E2a_M2 E2a_M3 E2b E2c
python -m paper.run plan E4 --manifest E2a_M1 E2a_M2 E2a_M3 E2b E2c
python -m paper.run plan E4r --manifest E2a_M1 E2a_M2 E2a_M3
python -m paper.sets derive-direct             # needs merged E4 and E4dev
#   [attack side] E3b, E5a/b/c, E6, E7 manifests  (re-plan E3 with --tag ens --manifest E3b)
python -m paper.run plan E5 --manifest E5a E5b E5c E6
python -m paper.run plan E7 --manifest E7
python -m paper.run plan E8 --manifest E2a_M1 E2a_M2 E2a_M3 E2b E2c
python -m paper.run plan E8lat
python -m paper.run plan E9 --manifest E2a_M1 --successful-on M1
python -m paper.report all
```

Before the first GPU job on CARC, link the processed public datasets (only their manifests are in Git):
`for d in cord_v2 sroie resume_parsing_vision; do ln -s $OLD/data/processed/$d data/processed/$d; done`
with `OLD=/project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork`, and install the extra packages into
`paperwork`: `pip install paddleocr paddlepaddle lpips`. If `readers smoke` fails for M3/T1/T2 with the
current transformers, make a second env rather than upgrading `paperwork`.
