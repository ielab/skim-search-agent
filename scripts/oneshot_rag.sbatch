#!/usr/bin/env bash
#SBATCH --job-name=oneshot-rag
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
# One-shot RAG baseline (no agent loop): serve Tongyi vLLM on :8101 (SAME serve invocation
# scripts/run.sh uses), wait for it, then run scripts/oneshot_rag.py for each retriever in
# $RETRIEVERS sequentially (default "bm25 dense", unchanged; set RETRIEVERS="bm25 dense hybrid"
# to add the NEW, additive-only RRF-fused bm25+dense baseline). `bash scripts/oneshot_rag.sbatch`
# runs interactively; `sbatch scripts/oneshot_rag.sbatch` submits it as a SLURM job.
set -euo pipefail

# ── dual-mode: `bash scripts/X.sbatch` runs interactively; `sbatch scripts/X.sbatch` submits
# a job (the #SBATCH lines above are comments to bash). Override any resource on the CLI:
# `sbatch --time=48:00:00 scripts/oneshot_rag.sbatch`.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  cd "${SLURM_SUBMIT_DIR:-.}"                 # $0 is a spooled copy under sbatch
  [ -d agent_search ] || cd ..                   # submitted from scripts/? hop up
else
  cd "$(dirname "$0")/.."
fi
# cluster env (idempotent, both interactive and sbatch; no-op on machines without `module`)
if command -v module >/dev/null 2>&1; then
  module load miniforge3 2>/dev/null || module load miniconda3 2>/dev/null || true
  module load cuda 2>/dev/null || true
fi
ENV_PATH="${ENV_PATH:-$PWD/envs}"
if [ -d "$ENV_PATH" ] && [ -z "${CONDA_PREFIX:-}" ]; then
  source activate "$ENV_PATH" 2>/dev/null || conda activate "$ENV_PATH" 2>/dev/null || true
fi
mkdir -p slurm_logs

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:resource_tracker:UserWarning}"
# MoE backend: FORCE Triton (see scripts/run.sh — LOAD-BEARING, do not remove).
export VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-0}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"

if [ -n "${PYTHON:-}" ]; then
  :
elif [ -x "${ENV_PATH:-$PWD/envs}/bin/python" ]; then
  PYTHON="${ENV_PATH:-$PWD/envs}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi

# ===================== EDIT THESE (the knobs that matter) =====================
DATASET="${DATASET:?set DATASET, e.g. hotpotqa_structured | browsecomp_plus_structured}"
MODEL="${MODEL:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"
LIMIT="${LIMIT:-200}"
PORT="${PORT:-8101}"
TP="${TP:-1}"
WORKERS="${WORKERS:-4}"
OUT_ROOT="${OUT_ROOT:-runs/_oneshot}"
INDEX_ROOT="${INDEX_ROOT:-indexes}"
# =============================================================================

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm is not on PATH. Activate the cluster env before running this." >&2
  exit 127
fi

# STEP 0: the dense retriever needs its persisted doc-embedding cache staged BEFORE the model
# server claims the GPU (same reasoning as scripts/run.sh's prebuild_corpus_indexes) — build
# it here if missing (idempotent: an existing cache is a no-op load-check inside oneshot_rag.py
# itself, this just materializes it up front so the GPU-heavy embed never races vLLM startup).
echo ">> [step 0] ensuring the dense doc-embedding cache exists for $DATASET ..."
"$PYTHON" -m evaluation.build_indexes --dataset "$DATASET" --retriever dense \
  --index-root "$INDEX_ROOT" --model BAAI/bge-base-en-v1.5 ${LIMIT:+--limit "$LIMIT"} || {
  echo "ERROR: [step 0] dense index build failed — see error above." >&2
  exit 1
}

SERVE_ARGS=(--tensor-parallel-size "$TP" --port "$PORT" --gpu-memory-utilization "${GPU_UTIL:-0.9}")
SERVE_ARGS+=(--max-model-len "${MAX_MODEL_LEN:-131072}")
if [ "${EAGER:-0}" = "1" ]; then
  SERVE_ARGS+=(--enforce-eager)
else
  SERVE_ARGS+=(--compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH:-PIECEWISE}\"}")
fi
[ -n "${VLLM_ARGS:-}" ] && SERVE_ARGS+=($VLLM_ARGS)

# torch.compile / inductor cache on NODE-LOCAL disk (same rationale as scripts/run.sh — avoids
# stale-file-handle races on Lustre/NFS when co-scheduled jobs compile the same graph at once).
_cache_base="${VLLM_COMPILE_CACHE_BASE:-${TMPDIR:-/tmp}/$USER/vllm-compile}"
export VLLM_CACHE_ROOT="$_cache_base/vllm"
export TORCHINDUCTOR_CACHE_DIR="$_cache_base/inductor"
export TRITON_CACHE_DIR="$_cache_base/triton"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo ">> serving $MODEL on :$PORT (tp=$TP); compile cache -> $_cache_base"
vllm serve "$MODEL" "${SERVE_ARGS[@]}" &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
echo ">> waiting for the server to be ready ..."
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM exited during startup (OOM? bad model path? see log above)." >&2
    echo "       Fallbacks: EAGER=1, VLLM_ARGS='--gpu-memory-utilization 0.8'." >&2
    exit 1
  fi
  sleep 5
done
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

# RETRIEVERS (NEW, additive-only env override): space-separated list of --retriever values to
# run sequentially. Defaults to the OLD hardcoded pair ('bm25 dense'), so an existing invocation
# that never sets RETRIEVERS is byte-identical. Set RETRIEVERS="bm25 dense hybrid" to also run
# the RRF-fused bm25+dense baseline (scripts/oneshot_rag.py --retriever hybrid).
RETRIEVERS="${RETRIEVERS:-bm25 dense}"

for RETRIEVER in $RETRIEVERS; do
  echo ">> [oneshot_rag] dataset=$DATASET retriever=$RETRIEVER model=$MODEL limit=$LIMIT"
  "$PYTHON" scripts/oneshot_rag.py \
    --dataset "$DATASET" --retriever "$RETRIEVER" --model "$MODEL" \
    --api-base "http://127.0.0.1:$PORT/v1" --limit "$LIMIT" --workers "$WORKERS" \
    --index-root "$INDEX_ROOT" --out-dir "$OUT_ROOT/$DATASET/$RETRIEVER"
done

echo ">> done. rows in $OUT_ROOT/$DATASET/{$RETRIEVERS// /,}/rows.jsonl"
