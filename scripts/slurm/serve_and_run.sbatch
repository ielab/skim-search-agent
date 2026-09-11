#!/usr/bin/env bash
#SBATCH --job-name=ssa-run
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Serve an open-weight model with vLLM INSIDE the job, run one experiment file against it, and
# stop the server before exiting. --export=ALL,EXPERIMENT=configs/paper/x.yaml,MODEL=<hf id>
# [,TP=2,PORT=8000,MAX_MODEL_LEN=131072,VLLM_PYTHON=/path/to/env/bin/python,OVERRIDES="model.name=... output.runs_dir=..."]
# VLLM_PYTHON is the interpreter that has vLLM installed (default: the active env's python).
# With MODEL unset the experiment file's own model.name is used (an API model needs no server).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/_common.sh"
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT=<experiment file> via --export}"
PORT="${PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}"   # jobs can share a node: one port per job; TP="${TP:-2}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
SERVER_PID=""
if [ -n "${MODEL:-}" ]; then
  export VLLM_USE_FLASHINFER_MOE_FP16=0
  "${VLLM_PYTHON:-python}" -m vllm.entrypoints.openai.api_server --model "$MODEL" --tensor-parallel-size "$TP" \
      --max-model-len "$MAX_MODEL_LEN" --host 127.0.0.1 --port "$PORT" --trust-remote-code \
      > "slurm_logs/vllm-${SLURM_JOB_ID:-local}.log" 2>&1 &
  SERVER_PID=$!
  trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true' EXIT
  for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "vLLM died; see slurm_logs/vllm-*.log" >&2; exit 1; }
    sleep 10
  done
  MODEL_OVERRIDES="model.name=$MODEL model.backend=api model.api_base=http://127.0.0.1:$PORT/v1"
fi
# shellcheck disable=SC2086
skimsearchagent run "$EXPERIMENT" ${MODEL_OVERRIDES:-} ${OVERRIDES:-}
echo ">> run finished: $EXPERIMENT"
