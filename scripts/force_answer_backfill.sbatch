#!/usr/bin/env bash
#SBATCH --job-name=force-backfill
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
# Offline forced-answer backfill (scripts/force_answer_backfill.py): serve Tongyi vLLM on :8102
# (SAME serve invocation scripts/run.sh / scripts/oneshot_rag.sbatch use), wait for it, then run
# the backfill over every browsecomp factorial cell (CELLS env, newline- or space-separated glob
# patterns; default below covers every cell docs/factorial_snapshot_20260710.md diagnosed).
# `bash scripts/force_answer_backfill.sbatch` runs interactively; `sbatch scripts/force_answer_backfill.sbatch`
# submits it as a SLURM job. Never mutates any rows.jsonl — writes only
# <condition_dir>/recovered_answers.jsonl (additive, idempotent, safe to re-run/resume).
set -euo pipefail

# ── dual-mode: `bash scripts/X.sbatch` runs interactively; `sbatch scripts/X.sbatch` submits
# a job (the #SBATCH lines above are comments to bash). Override any resource on the CLI:
# `sbatch --time=48:00:00 scripts/force_answer_backfill.sbatch`.
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
MODEL="${MODEL:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"
PORT="${PORT:-8102}"
TP="${TP:-1}"
WORKERS="${WORKERS:-8}"
DRY_RUN="${DRY_RUN:-0}"                    # set DRY_RUN=1 to only report counts (no GPU needed)
# CELLS: newline- or space-separated glob pattern(s), one per condition-dir group. Default: every
# browsecomp cell docs/factorial_snapshot_20260710.md diagnosed (the _headline_validation /
# _dense_validation / _visit_uncapped / _fullvisit / _fullvisit_dense factorial dirs, plus the
# main-sweep browsecomp_plus_structured / browsecomp_plus_flat condition dirs).
if [ -n "${CELLS:-}" ]; then
  read -r -a CELL_ARGS <<< "$CELLS"
else
  CELL_ARGS=(
    "runs/_headline_validation/agent/browsecomp_plus_*/*/agent_*"
    "runs/_dense_validation/agent/browsecomp_plus_*/*/agent_*"
    "runs/_visit_uncapped/agent/browsecomp_plus_*/*/agent_*"
    "runs/_fullvisit/agent/browsecomp_plus_*/*/agent_*"
    "runs/_fullvisit_dense/agent/browsecomp_plus_*/*/agent_*"
    "runs/agent/browsecomp_plus_structured/*/agent_*"
    "runs/agent/browsecomp_plus_flat/*/agent_*"
  )
fi
# =============================================================================

DRY_RUN_ARGS=()
if [ "$DRY_RUN" = "1" ]; then
  DRY_RUN_ARGS=(--dry-run)
  echo ">> DRY_RUN=1 — reporting counts only, no vLLM server started."
  "$PYTHON" scripts/force_answer_backfill.py "${CELL_ARGS[@]}" "${DRY_RUN_ARGS[@]}"
  exit $?
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm is not on PATH. Activate the cluster env before running this." >&2
  exit 127
fi

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
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export MODEL

echo ">> [force_answer_backfill] model=$MODEL workers=$WORKERS cells=${#CELL_ARGS[@]}"
"$PYTHON" scripts/force_answer_backfill.py "${CELL_ARGS[@]}" --workers "$WORKERS"

echo ">> done. recovered_answers.jsonl written under each matched condition dir."
