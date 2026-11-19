#!/usr/bin/env bash
set -euo pipefail
: "${PP_ATTACK_RUN:?}" "${PP_REPO_ROOT:?}" "${SLURM_ARRAY_TASK_ID:?}"
cd "$PP_REPO_ROOT"
source scripts/activate_paperwork.sh
export PYTHONPATH="$PP_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
python -u -m attacks.paper_runner run "$PP_ATTACK_RUN" --unit "$SLURM_ARRAY_TASK_ID"
