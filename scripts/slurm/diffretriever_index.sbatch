#!/usr/bin/env bash
#SBATCH --job-name=index-diffretriever
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Build a DiffRetriever index (agent_search/retrievers/learned/diffretriever.py). The model runs
# in its own environment behind scripts/serve_diffretriever.py on this job's GPU; the library
# sends it page text and stores the passage vectors under indexes/diffretriever/.
#
#   sbatch --account=YOUR_ACCOUNT --export=ALL,DATASET=browsecomp_plus,DIFFRETRIEVER_DOC_LENGTH=512 \
#          scripts/slurm/diffretriever_index.sbatch
#
# DIFFRETRIEVER_PYTHON is the model's environment (transformers 4.54 and peft); DIFFRETRIEVER_MODEL
# the checkpoint; DIFFRETRIEVER_DOC_LENGTH the text tokens of a page after the prompt, the knob a
# run reads to find the index.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1
DATASET="${DATASET:?set DATASET, e.g. browsecomp_plus}"
DR_PYTHON="${DIFFRETRIEVER_PYTHON:?set DIFFRETRIEVER_PYTHON to the model environment python}"
export DIFFRETRIEVER_MODEL="${DIFFRETRIEVER_MODEL:-ielabgroup/diffretriever-dream-7b-multi-q4-p16}"
PORT_FILE="slurm_logs/diffretriever-index-${SLURM_JOB_ID:-local}.port"; rm -f "$PORT_FILE"
"$DR_PYTHON" scripts/serve_diffretriever.py --model "$DIFFRETRIEVER_MODEL" --device cuda:0 \
    --port-file "$PORT_FILE" > "slurm_logs/diffretriever-index-${SLURM_JOB_ID:-local}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  [ -s "$PORT_FILE" ] && break
  kill -0 $SERVER 2>/dev/null || { echo "server died; see slurm_logs/diffretriever-index-${SLURM_JOB_ID:-local}.log" >&2; exit 1; }
  sleep 10
done
export DIFFRETRIEVER_URL="http://127.0.0.1:$(cat "$PORT_FILE")"
echo ">> $(curl -s "$DIFFRETRIEVER_URL/health")"
PY=./envs/bin/python; [ -x "$PY" ] || PY="$(command -v python)"
PYTHONPATH=. "$PY" -m agent_search.evaluation.build_indexes --dataset "$DATASET" --retriever diffretriever \
  --index-root indexes --model "$DIFFRETRIEVER_MODEL"
echo ">> done"
