#!/usr/bin/env bash
# Supplementary smoke: the 2 shared-retrieval read-strategy arms not in the main driver
# (research_bm25_dci = grep over retrieved, research_bm25_fetch = section-fetch over retrieved).
# Waits for the main driver (smoke_rebuild.sh) to finish so the two runs don't contend on the
# single login-node vLLM, then validates these methods so all 5 doc arms are proven before the
# full sweep. Same endpoint / env / logging conventions as smoke_rebuild.sh.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"
API_BASE="${API_BASE:-http://127.0.0.1:8765/v1}"
RUNS_DIR="${RUNS_DIR:-runs/_rebuild_smoke}"
WORKERS="${WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-50}"
LIMIT_DOC="${LIMIT_DOC:-12}"
PY="${PY:-envs/bin/python}"
LOGDIR="$RUNS_DIR/logs"; mkdir -p "$LOGDIR"
MAIN_PID="${MAIN_PID:-}"

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export AGENT_DRIVER="${AGENT_DRIVER:-loop}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

DOC_DATASETS=(hotpotqa 2wiki musique browsecomp_plus)
EXTRA_METHODS=(agent_research_bm25_dci agent_research_bm25_fetch)
ts() { date +%H:%M:%S; }

# --- wait for the main driver to finish (pid if given, else the COMPLETE marker) ----------
echo "[$(ts)] extra-smoke waiting for main driver (pid=${MAIN_PID:-n/a}) ..."
while true; do
  if [ -n "$MAIN_PID" ]; then
    kill -0 "$MAIN_PID" 2>/dev/null || { echo "[$(ts)] main pid gone"; break; }
  else
    grep -q "SMOKE COMPLETE" "$LOGDIR/DRIVER.log" 2>/dev/null && break
  fi
  sleep 30
done
echo "[$(ts)] main driver finished — starting extra methods"

for ds in "${DOC_DATASETS[@]}"; do
  for rt in "${EXTRA_METHODS[@]}"; do
    log="$LOGDIR/${ds}__${rt}.log"
    echo "[$(ts)] >> $ds / $rt (limit=$LIMIT_DOC) -> $log"
    "$PY" -m evaluation.run_eval \
      --dataset "$ds" --retriever "$rt" \
      --policy llm --backend api --api-base "$API_BASE" --model "$MODEL" \
      --max-steps "$MAX_STEPS" --workers "$WORKERS" --limit "$LIMIT_DOC" \
      --runs-dir "$RUNS_DIR" --k 1 3 5 10 --seeds 42 >"$log" 2>&1
    echo "[$(ts)]    exit=$? $ds / $rt"
  done
done
echo "[$(ts)] EXTRA SMOKE COMPLETE"
