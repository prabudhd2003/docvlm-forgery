#!/usr/bin/env bash
# One array task of a paper_v1 experiment. Submit with: python -m paper.run submit <EXP>
# Resources come from configs/paper_v1.json ("slurm"), passed on the sbatch command line.
set -euo pipefail
: "${PP_EXP:?}" "${PP_PER_TASK:?}" "${PP_REPO_ROOT:?}" "${SLURM_ARRAY_TASK_ID:?}"

cd "$PP_REPO_ROOT"
source scripts/activate_paperwork.sh
export PYTHONPATH="$PP_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
umask 0002

echo "job ${SLURM_JOB_ID:-?} task $SLURM_ARRAY_TASK_ID exp $PP_EXP host $(hostname) commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u -m paper.run run "$PP_EXP" --task "$SLURM_ARRAY_TASK_ID" --per-task "$PP_PER_TASK"
