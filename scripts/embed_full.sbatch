#!/usr/bin/env bash
#SBATCH --job-name=embed-bcfull
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH --partition=h24gpu
#SBATCH --qos=normal
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Dense doc-embedding cache builder, parameterized by EMBED_MODEL (e.g.
# Qwen/Qwen3-Embedding-0.6B, Qwen/Qwen3-Embedding-4B) and DATASET (default
# browsecomp_plus_structured_full; pass browsecomp_plus for the original text). Writes the same
# indexes/dense/<model-key>/<dataset-key>/ cache layout agent_search.evaluation.build_indexes always uses.
#
# The cache key carries the encoding settings, so export them to build a second cache for the
# same checkpoint: DENSE_SEQ_LENGTH (document tokens, ITER uses 512), DENSE_POOLING,
# DENSE_DTYPE. A run picks the matching cache from the same knobs in its experiment file.
#
#   sbatch --export=ALL,EMBED_MODEL=ielabgroup/ITER-Qwen3-Embedding-4B,DATASET=browsecomp_plus,\
#          DENSE_SEQ_LENGTH=4096,DENSE_POOLING=last_token,DENSE_DTYPE=bfloat16 scripts/embed_full.sbatch
set -euo pipefail
if [ -n "${SLURM_JOB_ID:-}" ]; then
  cd "${SLURM_SUBMIT_DIR:-.}"
  [ -d agent_search ] || cd ..
else
  cd "$(dirname "$0")/.."
fi
if command -v module >/dev/null 2>&1; then
  module load miniforge3 2>/dev/null || true
  module load cuda 2>/dev/null || true
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
EMBED_MODEL="${EMBED_MODEL:?set EMBED_MODEL, e.g. Qwen/Qwen3-Embedding-0.6B}"
DATASET="${DATASET:-browsecomp_plus_structured_full}"
echo ">> building dense cache: $EMBED_MODEL over $DATASET (seq_length=${DENSE_SEQ_LENGTH:-default})"
# the project env when submitted from the main checkout; otherwise the python on PATH (a worktree
# has no envs/; the cluster env exported into the job carries torch and transformers)
PY=./envs/bin/python; [ -x "$PY" ] || PY="$(command -v python)"
PYTHONPATH=. "$PY" -m agent_search.evaluation.build_indexes \
  --dataset "$DATASET" --retriever dense \
  --index-root indexes --model "$EMBED_MODEL"
echo ">> done"
