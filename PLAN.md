# Paper plan (frozen 2026-11-19)

**Target:** ICDAR 2027, full paper due **20 Feb 2027** (check the abstract deadline on icdar2027.org).
Backup: a CVPR 2027 archival workshop; then SaTML 2028 (its deadline is usually late September) with a longer version.
Work starts after the course ends (final report due 7 Dec 2026). All jobs run on CARC L40S.

## Claims the paper will make

1. Instruction-tuned document VLMs from three families can be made to return an attacker-chosen
   **number** from a phone-photo receipt at small budgets; replacing a **word** (resume degree) is much harder.
2. Image cleaning (JPEG, resize vote, multi-view, re-worded questions) fails once the attack optimizes through it.
3. Evidence-band verification (CBVR) recovers the true answer without retraining; it is scored as correct / wrong / refused.
4. An attacker who targets CBVR partly breaks it. We measure exactly how much, and test whether
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
| OCR | PaddleOCR PP-OCRv4 English recognizer, run on bands only | second reader in CBVR-OCR | non-ViT, non-differentiable |
| S | `microsoft/trocr-base-printed` | attacker's differentiable stand-in for OCR | for the adaptive attack on CBVR-OCR |

Optional, only if E3 is done by 4 Jan: one commercial API on the 100 ensemble images, capped at $10.

## Frozen document sets

Selection order is always the hash order `digest([566, 'paper_v1', record_id])`, the same scheme as the
existing studies. The lists and their SHA-256 are written to `data/paper_v1/` in E0, before any
model reads a test document. Eligibility is decided from clean outputs only.

**Receipts**

| Set | Size | Contents | Use |
|---|---:|---|---|
| R-dev | 120 | 60 CORD/SROIE validation + 15 collected dev + the 45 old "reserved" collected receipts | all tuning; anything already looked at lives here |
| **R-test** | **154** | the **54** collected receipts not used in the CBVR study + first **50 CORD test** + first **50 SROIE test** (never attacked or inspected) | every reported number |
| R-attack | 100 | first 100 R-test receipts that M1, M2 and M3 all read correctly when clean (if fewer than 100, all of them) | saved attacks, transfer, defenses |
| R-direct | 50 | first 50 R-attack receipts that CBVR-Base (M1) answers correctly when clean | direct attacks on defenses |
| R-conv | 20 | first 20 R-dev receipts that CBVR-Base answers correctly when clean | choosing the attack budget only |

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
**D6 CBVR-Base** (one model reads both bands; run with M1, M2 and M3 as reader) ·
**D7 CBVR-Cross** (tight band read by M1, taller band by M3) ·
**D8 CBVR-OCR** (tight band read by M1, taller band by OCR).
Dropped: random crop jitter (EOT is known to beat it), subtotal + tax arithmetic (missing on many receipts, and an attacker can forge all three), masked-band (kept only as an existing result).

## Experiments

Settings not listed keep the values in the existing configs (step size, max image size, prompts).
"Run" = one (document, model, ε, attack, restart) unit. Hours are **L40S-hour estimates** scaled from the
one measured rate (8.8 A100-min per 100-step direct-attack run on M1), with 7-8B models at about 2.5×.
E0 measures the real per-unit times and replaces this column.

| # | Experiment | Exact design | Runs | Est. h | Fills |
|---|---|---|---:|---:|---|
| E0 | Setup and parity | pin revisions; differentiable preprocessing for M2 (same code path as M1) and M3 (tile grid fixed from the clean image); parity with the real processor (max abs logit gap ≤ 0.01, current M1 value 0.0045); 3-doc smoke per model; timing and peak memory on L40S; write `data/paper_v1/` lists | — | 10 | Methods |
| E1 | Clean accuracy | 6 readers × (R-test 154 + Res-test 100), full page | 1,524 | 5 | Table 1 |
| E2a | Standard PGD | M1, M2, M3 × R-attack 100 × ε {2,4,8,16}, 100 steps | 1,200 | 125 | Table 2 |
| E2b | EOT attacks | uniform EOT + hard-view EOT × ε {8,16} × M1, M2 × first 50 of R-attack | 400 | 70 | Table 2, Table 3 |
| E2c | Resume PGD | M1, M2, M3 × Res-attack 50 × ε {8,16} | 300 | 30 | Table 2 |
| E3a | Transfer matrix | every ε 8/16 image from E2a (600) read by all 6 readers | 3,600 inferences | 10 | Fig. 2 |
| E3b | Ensemble attack | joint loss over M1+M2+M3, first 50 of R-attack × ε {8,16} → read by T1, T2, T3 | 100 | 35 | Fig. 2 |
| E4 | Defense comparison | tune band padding / fallback on R-dev only (goal: clean refusals < 5%), freeze the policy hash, then one pass of D0–D8 on R-test clean + all E2 receipt images | ~1,750 images | 25 | Table 3, Fig. 3 |
| E5a | Attack budget | CBVR-Base (M1) on R-conv 20, ε 8, 3 restarts, 500 steps with checkpoints at 100/250/500, 100 queries | 60 | 40 | Fig. 4 (appendix) |
| E5b | Direct attacks | at the chosen budget B*: D6 (M1) × ε {4,8} × 2 restarts (200 runs); D7 joint loss on M1+M3, ε 8 × 2 (100); D8 with TrOCR stand-in + 100 real queries, ε 8 × 2 (100); D4 on 25 receipts, ε 8 × 2 (50), all on R-direct | 450 | 190 | Table 4 |
| E5c | Resume direct attack | recovery on Res-direct 30, ε 16 × 2 restarts at B* | 60 | 25 | Table 4 |
| E6 | Locator attack | R-direct first 30, ε 8 × 2 restarts; push the predicted box off the total (refusal) and onto the subtotal (forgery) | 60 | 10 | Table 4 |
| E7 | Why numbers break | M1 standard ε 8 on R-attack 50 with 3 targets (one digit changed, 10×, unrelated 3-digit value); resumes 50 with short vs long degree target; log target tokens and band area | 250 | 13 | Fig. 5 |
| E8 | Stealth and cost | SSIM, PSNR, LPIPS for all E2 images; latency and peak VRAM per document for D0, D6, D7, D8 on 50 clean receipts | — | 5 | Table 5 |
| E9 | Upload pipeline | successful M1 ε 8/16 attacks through JPEG q75, q85 and downscale to 1600 px max side, re-read by D0 and D6 | ~600 inferences | 5 | Table 5 |
| | **Total** | | | **≈ 600** | |

**B\*** = the smallest step count in E5a whose receipt success rate is within one receipt of the 500-step result.
If E0 timings put the total above 650 L40S-hours, cut in this order: EOT on M2 → resume attacks on M3 → D4 direct
attacks → ensemble down to 25 receipts → D8 direct attack.

Dropped from the Gemini plan: 5×5 white-box matrix (3 white-box + 3 transfer-only instead), physical printing,
invoice domain, 500 steps × 3 restarts everywhere (replaced by E5a), CBVR-Jitter, CBVR-Arith.

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
- Commits carry only human authors (no AI co-author lines).

## Integrity

Test lists are frozen before any model reads them. Anything tuned is tuned on dev. Every run is reported,
including failures. Rates come with 95% Wilson intervals over documents, not over runs, and both counts are given.
Agree authorship and who continues among the five of us, and ask Prof. Zhao to advise, before E0.
