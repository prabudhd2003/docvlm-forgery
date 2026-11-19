# Donut final validation results

Generated November 1, 2026 from existing completed attack records. No new model inference or GPU attack is performed by this analysis.

Open `05_attack_analysis.html` for the complete rendered report. The executed source is `../../../notebooks/05_attack_analysis.ipynb` relative to this directory. Method details are in `../../../docs/05_donut_attack.md`.

## Findings

The same 121 clean-correct validation receipts (75 CORD, 46 SROIE) were attacked at each budget. Targeted successes were 7, 17, 50, 99, and 115 at epsilon 2, 4, 8, 16, and 32 respectively. These are conditional validation rates, not test-set or full-dataset rates. Image similarity decreases as the budget increases.

All 605 completed attacks used margin loss with Adam and completed 100 steps. Exact fresh-process GPU restart reproducibility remains unverified; every completed record has `checkpoint_resumed=false`, so none of these results depended on checkpoint restart.

## Artifacts

- `01_success_rates.*`: overall and per-dataset targeted success.
- `02_image_quality.*`: SSIM and LPIPS distributions.
- `03_runtime.*`: median recorded attack runtime.
- `example_*.png` / `.pdf`: six original/attacked/difference comparisons, with one success and one failure for CORD epsilon 2, SROIE epsilon 8, and CORD epsilon 32.
- `results_overall.csv`, `results_by_dataset.csv`: numerical results.
- `pixel_bounds.csv`, `runtime.csv`, `provenance.csv`: validation and provenance tables.
- `example_selection.csv`: deterministic example identities and metrics.
- `analysis_inputs.json`: SHA-256 hashes of the audit input files.

The difference panels are amplified eightfold and are not model inputs. The examples are selected by median SSIM within each outcome group and are illustrative, not a human perceptibility study.

## Re-execution on CARC

Use a CPU compute allocation and the project's `paperwork` Python environment. Run the notebook from the project root or set `PP_DONUT_REPO_ROOT` to it. A Jupyter kernel must point to that environment. Required inputs are `outputs/inspection/donut-audit-20261101/{audit.json,combined_attacks.csv}`, the recorded run configs and saved images in the sibling `kbali-donut-worktree` and `kbali-donut-eps2-4-worktree`, and the processed validation datasets in `kbali-donut-worktree/data/processed/`.

The notebook reads those saved inputs and overwrites only its report artifacts. It does not run attacks or inference. The exported HTML embeds the plots and can be read without CARC access. Recorded quality metrics and predictions are analyzed rather than recomputed. Existing audit checks cover file hashes, coverage, provenance and pixel bounds.
