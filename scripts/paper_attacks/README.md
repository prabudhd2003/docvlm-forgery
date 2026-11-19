# Paper attack producers

These commands create the saved PNGs consumed by `paper.manifest` and
`paper.run`. They use the existing frozen IDs, clean document loader, reader
wrappers, normalization and EBV geometry. They do not select new test sets.

## Implemented

| Entry point | Experiments | Output manifests |
|---|---|---|
| `saved.py` | E2a standard PGD, E2b uniform/hard-view EOT, E2c resume PGD | `E2a_M1`, `E2a_M2`, `E2a_M3`, `E2b`, `E2c` |
| `ensemble.py` | E3b mean target loss over M1+M2+M3 | `E3b` |
| `direct.py` | E5a budget checkpoints; E5b D4/D6/D7/D8; E5c resume recovery | `E5a`, `E5b`, `E5c` |
| `locator.py` | E6 refusal and subtotal forgery, **60 separate runs each** | `E6` |
| `targets.py` | E7 receipt digit-flip/x10/unrelated and short/long resume labels | `E7` |
| `parity.py` | E0 processor/logit/gradient/memory measurements | no manifest |

`src/attacks/surrogates.py` contains the M1/M2 Qwen and M3 InternVL gradient
paths, and the TrOCR stand-in for D8. InternVL uses the installed Hugging Face
tile-selection helper, with clean-image grids frozen for each objective.
Official processor pixels are used forward; differentiable resize/normalize
operations supply BPDA gradients. Processor differences and teacher-forced
logit differences are recorded. A logit difference above 0.01 stops the run.

Memory measures: frozen weights, non-reentrant gradient checkpointing,
answer-position-only logits, one page's vision graph and backward pass at a
time while retaining the complete multi-page prompt, and one VLM resident on
the GPU at a time. Ensemble and cross-reader gradients are accumulated before
the pixel update. Other model weights are held in host RAM. Qwen's fixed
`generation.max_pixels` and InternVL's processor `max_patches` bound the encoder
input per page; these are not changed adaptively after an OOM. Original PNG
dimensions are preserved. E0 must measure whether the pinned models fit L40S.

## Before GPU runs

From the repository root:

```bash
export PYTHONPATH="$PWD/src"
# The regular paper prerequisites still apply:
# pin M2/M3/etc.; freeze sets; finish E1; derive R-attack / Res-attack.
python -m paper.readers pin

# For D8, also put this exact hash into configs/paper_attacks.json:
python -c 'from huggingface_hub import model_info; print(model_info("microsoft/trocr-base-printed").sha)'

python scripts/paper_attacks/parity.py M1 --resume
python scripts/paper_attacks/parity.py M2 --resume
python scripts/paper_attacks/parity.py M3 --resume
```

Parity uses three R-dev receipts and, with `--resume`, the largest two-page
Res-dev document by original pixel area. It records timings, gradient norms,
peak VRAM, and a saved/reloaded one-step PNG check under
`results/paper_v1/attack_runs/parity/`. It does not claim an attack succeeded.

The implementation has CPU coverage, including real HF processors and tiny
random Qwen/InternVL models. **The pretrained GPU paths, their achieved attack
success, and L40S memory usage have not been measured locally.** The installed
transformers version must support these model families, `logits_to_keep`, and
non-reentrant checkpointing. Unsupported layouts fail explicitly.

## Saved attacks

```bash
python scripts/paper_attacks/saved.py plan E2a
python scripts/paper_attacks/saved.py submit E2a --smoke
# Inspect the completed smoke records, then:
python scripts/paper_attacks/saved.py submit E2a
python scripts/paper_attacks/saved.py status E2a
python scripts/paper_attacks/saved.py merge E2a

# Repeat plan -> submit --smoke -> submit -> status -> merge for E2b and E2c.
```

At least three completed smoke units, covering distinct source models when
present, are required before full submission. Success at hitting the target
is not required to pass the smoke gate: valid gradients, parity, bounded PNGs,
and completed saved-image verification are. Each later unit also checks parity.
No Slurm jobs are submitted by `plan`; `submit --dry-run` prints its command.

Run a single unit directly with
`python scripts/paper_attacks/saved.py run E2a --unit 0`.
`--unit 0 1 2` runs several units, records individual failures, and continues.
Use `--scratch /path/on/scratch` when planning to override `scratch_root`.

Uniform and hard-view EOT retain the course's visual/semantic view pool,
three sampled views per step, exact resize-vote views, hardness weights, and
hard-view frequency filtering. Saved verification checks the real JPEG,
resize-vote, multi-view and semantic-consistency defenses as well as D0.

## Remaining experiments

Use the same five-command cycle with the corresponding entry point:

```bash
python scripts/paper_attacks/ensemble.py plan E3b

# After E4dev policy freezing and deriving R-conv/R-direct/Res-direct:
python scripts/paper_attacks/direct.py plan E5a
# Finish E5a, including merge, before choosing B*:
python scripts/paper_attacks/direct.py freeze-budget E5a
python scripts/paper_attacks/direct.py plan E5b
python scripts/paper_attacks/direct.py plan E5c

python scripts/paper_attacks/locator.py plan E6
python scripts/paper_attacks/targets.py plan E7
```

The generic `python -m attacks.paper_runner` entry point accepts every experiment.
Use `--tag new_name` at planning time for a changed configuration, then use the
full run name (for example `E2a_new_name`) in later commands. Publishing refuses
to overwrite an existing, different canonical manifest; archive it explicitly
before publishing a replacement. Settings live in `configs/paper_attacks.json`;
`--settings path.json` selects another complete settings file at planning time.

## Direct-attack accounting and interpretation

- D6 optimizes both bands; D4 optimizes full page and tight band; D7 uses M1 on
  the tight band and M3 on the taller band. D8 substitutes TrOCR only for the
  gradient term. All D8 success checks use the actual PaddleOCR pipeline.
- Each real pipeline query relocalizes the current pixels. Gradient steps
  hold those crop coordinates fixed until the next query. Query search uses
  bounded square perturbations and reserves the final query for the saved PNG.
  The recorded budget is **pipeline invocations**, not individual VLM calls;
  the trace also records underlying model-call counts.
- E5a uses 60 continuous 500-step trajectories, with immutable cumulative-best
  PNGs at 100/250/500. Each trajectory has at most 100 pipeline queries total.
  Query counts at the intermediate snapshots are recorded; the final snapshot
  includes the remaining query search. The three points therefore have different
  accumulated query counts, not three independent 100-query budgets.
- B* uses broken **documents across restarts**, selecting the smallest budget
  within one document of the 500-step result. E5b/E5c refuse to plan without it.
- E6 has 120 trajectories: 30 receipts × 2 restarts × 2 objectives. Refusal
  targets invalid off-page locator coordinates. Subtotal forgery freezes a
  clean M1 subtotal box, targets its locator tokens, and also optimizes both
  subtotal bands toward the frozen x10 answer. Subtotal localization failures
  are explicit failed units requiring inspection, not silently excluded cases.
  A reported producer-side subtotal success requires both >=0.5 box IoU and
  the exact frozen target accepted by D6. The evaluator separately records
  targeted acceptance and refusal, with locator objectives separated in Table 4.
- E7 receipt targets are deterministic. Resume targets are valid incorrect
  labels: `master` (fallback `bachelor`) versus `certificate_or_diploma`
  (fallback `postgraduate`). These compare different label lengths **and
  semantics**, not synonymous answers. Token IDs/counts (including EOS) and
  clean evidence-band area are recorded. Exact-label normalization is retained.

## Files and recovery

Plans, contracts, compact per-unit records and failures live under
`results/paper_v1/attack_runs/<run>/`. PNGs, temporary query images and active
`state.pt` checkpoints live under
`<scratch_root>/paper_attacks/<run>/<contract>/<unit>/`.

Completed units are skipped only when their contract, unit, clean-input hash
and saved-image hashes match. A crash resumes the last recorded optimizer
state and deterministic random stream. Checkpoints are written every 50
gradient steps or sooner at observations. `state.pt` is deleted after the
finished result is written. Errors are retained and retries do not silently
change image size, budgets or preprocessing.

`merge` requires every planned unit, validates the real PNGs against the clean
pages, and then publishes `results/paper_v1/manifests/<name>.jsonl`. No concurrent
workers append to a shared manifest. Those manifests can be passed directly
to the existing evaluation commands in `PLAN.md`.
