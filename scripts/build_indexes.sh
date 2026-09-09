#!/usr/bin/env bash
#SBATCH --job-name=agent-search-index
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --qos=express
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --output=slurm_logs/indexes/%x-%A_%a.out
#SBATCH --error=slurm_logs/indexes/%x-%A_%a.err
# Pre-build persistent per-corpus indexes (bm25_pyserini / dense) so the eval doesn't
# build them lazily under --workers (threads racing on one index dir).
#
# Dual-mode (like run.sh): `bash scripts/build_indexes.sh` builds in this shell;
# `sbatch --array=0-7 scripts/build_indexes.sh` shards the work by REPO across array
# tasks (a repo's commits stay in one shard, so concurrent tasks never check out the
# same clone). NSHARDS defaults to the array size.
#   DATASET=swebench_lite RETRIEVER=dense sbatch --array=0-7 scripts/build_indexes.sh
set -euo pipefail
if [ -n "${SLURM_JOB_ID:-}" ]; then
  cd "${SLURM_SUBMIT_DIR:-.}"
  [ -d agent_search ] || cd ..
else
  cd "$(dirname "$0")/.."
fi
if command -v module >/dev/null 2>&1; then
  module load miniforge3 2>/dev/null || module load miniconda3 2>/dev/null || true
  module load cuda 2>/dev/null || true
fi
ENV_PATH="${ENV_PATH:-$PWD/envs}"
if [ -d "$ENV_PATH" ] && [ -z "${CONDA_PREFIX:-}" ]; then
  source activate "$ENV_PATH" 2>/dev/null || conda activate "$ENV_PATH" 2>/dev/null || true
fi
mkdir -p slurm_logs/indexes
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"      # offline node: fail loud, don't re-download
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"  # avoid leaked-semaphore warning at exit
# resource_tracker's leaked-semaphore warning is emitted by its OWN daemon process
# (a Python warnings.filter cannot reach it) and is benign — the semaphore is reclaimed
# at exit. Source is usually vLLM/torch multiprocessing shutdown, not our code/results.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:resource_tracker:UserWarning}"

if [ -n "${PYTHON:-}" ]; then
  :
elif [ -x "${ENV_PATH:-$PWD/envs}/bin/python" ]; then
  PYTHON="${ENV_PATH:-$PWD/envs}/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi

# Env knobs:
#   DATASET=swebench_verified|swebench_lite|loc_bench|fixture
#   RETRIEVER=bm25_pyserini|dense · DENSE_MODEL=<optional override>
#   REPO_CACHE=data/repos · INDEX_ROOT=indexes · REBUILD=0|1
#   LIMIT=<int> · CORPUS_LIMIT=<int> (shared document-corpus cap) · NSHARDS=<int>
DATASET="${DATASET:-swebench_verified}"
RETRIEVER="${RETRIEVER:-bm25_pyserini}"
DENSE_MODEL="${DENSE_MODEL:-}"   # empty: build_indexes picks per domain (code vs docs)
REPO_CACHE="${REPO_CACHE:-data/repos}"
INDEX_ROOT="${INDEX_ROOT:-indexes}"
REBUILD="${REBUILD:-0}"
LIMIT="${LIMIT:-}"
CORPUS_LIMIT="${CORPUS_LIMIT:-}"

# SLURM array -> shard by repo; bash -> single shard (everything).
SHARD_ARGS=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  NSHARDS="${NSHARDS:-$(( ${SLURM_ARRAY_TASK_COUNT:-1} ))}"
  SHARD_ARGS=(--shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS")
  echo ">> index shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS}"
elif [ -n "${NSHARDS:-}" ]; then
  SHARD_ARGS=(--shard "${SHARD:-0}" --nshards "$NSHARDS")
fi

REBUILD_ARG=()
[ "$REBUILD" = "1" ] && REBUILD_ARG=(--rebuild)
EXTRA=()
[ -n "$LIMIT" ] && EXTRA+=(--limit "$LIMIT")
[ -n "$CORPUS_LIMIT" ] && EXTRA+=(--corpus-limit "$CORPUS_LIMIT")

"$PYTHON" -m agent_search.evaluation.build_indexes \
  --dataset "$DATASET" \
  --retriever "$RETRIEVER" \
  ${DENSE_MODEL:+--model "$DENSE_MODEL"} \
  --repo-cache "$REPO_CACHE" \
  --index-root "$INDEX_ROOT" \
  ${SHARD_ARGS[@]+"${SHARD_ARGS[@]}"} \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  ${REBUILD_ARG[@]+"${REBUILD_ARG[@]}"}
