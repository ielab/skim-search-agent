#!/usr/bin/env bash
# Post-rebuild SMOKE: validate every fix end-to-end against the already-running vLLM
# on :8765 (Tongyi-DeepResearch-30B-A3B @ 128k). Runs per dataset x per method for BOTH
# arms, then the CODE arm's whole-fix -> RESOLVE grading (apptainer, warmed SIFs).
#
# Goal (what this proves): none-rate down (parser fix), observations FULL (no 2k cap),
# total_tokens_once sane (model-processed, not raw tool bytes), canonical EM/F1 + support_f1
# populate, code patches synthesize + grade for real FAIL_TO_PASS/PASS_TO_PASS.
#
# Bypasses run.sh's serve step ON PURPOSE (vLLM is already up filling the GPU; double-serve
# would OOM). Logs IN-REPO (runs/_rebuild_smoke/logs), never /tmp.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"
API_BASE="${API_BASE:-http://127.0.0.1:8765/v1}"
RUNS_DIR="${RUNS_DIR:-runs/_rebuild_smoke}"
WORKERS="${WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-50}"
LIMIT_DOC="${LIMIT_DOC:-12}"
LIMIT_CODE="${LIMIT_CODE:-15}"
PY="${PY:-envs/bin/python}"
LOGDIR="$RUNS_DIR/logs"
mkdir -p "$LOGDIR"

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
# Tongyi's native <tool_call> TEXT protocol: served vLLM (--backend api) otherwise defaults
# to the SDK driver (tool_choice=auto) and 400s because vLLM wasn't started with
# --enable-auto-tool-choice. loop = the validated Tongyi path (retriever.py:443, memory).
export AGENT_DRIVER="${AGENT_DRIVER:-loop}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"   # loop needs a non-empty key; real key => browsecomp judge works

DOC_DATASETS=(hotpotqa 2wiki musique browsecomp_plus)
DOC_METHODS=(agent_research agent_research_bm25 agent_research_dci)
CODE_DATASET="swebench_verified"
CODE_METHODS=(agent_codefix_patch agent_codefix_grep_patch)

ts() { date +%H:%M:%S; }

run_one() {   # dataset retriever limit
  local ds="$1" rt="$2" lim="$3"
  local log="$LOGDIR/${ds}__${rt}.log"
  echo "[$(ts)] >> $ds / $rt (limit=$lim) -> $log"
  "$PY" -m evaluation.run_eval \
    --dataset "$ds" --retriever "$rt" \
    --policy llm --backend api --api-base "$API_BASE" --model "$MODEL" \
    --max-steps "$MAX_STEPS" --workers "$WORKERS" --limit "$lim" \
    --runs-dir "$RUNS_DIR" --k 1 3 5 10 --seeds 42 \
    >"$log" 2>&1
  local rc=$?
  echo "[$(ts)]    exit=$rc  $ds / $rt"
  return $rc
}

echo "=============================================================="
echo " SMOKE start $(date)  model=$MODEL  runs=$RUNS_DIR"
echo " doc: ${DOC_DATASETS[*]} x ${DOC_METHODS[*]} (limit $LIMIT_DOC)"
echo " code: $CODE_DATASET x ${CODE_METHODS[*]} (limit $LIMIT_CODE) + RESOLVE grading"
echo "=============================================================="

# ---- DOC ARM ---------------------------------------------------------------
for ds in "${DOC_DATASETS[@]}"; do
  for rt in "${DOC_METHODS[@]}"; do
    run_one "$ds" "$rt" "$LIMIT_DOC"
  done
done

# ---- CODE ARM (whole-fix) --------------------------------------------------
for rt in "${CODE_METHODS[@]}"; do
  run_one "$CODE_DATASET" "$rt" "$LIMIT_CODE"
done

echo "=============================================================="
echo " RUNS DONE $(date) — starting CODE resolve grading"
echo "=============================================================="

# ---- CODE RESOLVE GRADING (apptainer, warmed SIFs, offline-safe) -----------
module load apptainer/1.5.0 2>/dev/null || module load apptainer 2>/dev/null || true
export APPTAINER_CACHEDIR="$PWD/.cache/apptainer" SINGULARITY_CACHEDIR="$PWD/.cache/apptainer"
export APPTAINER_TMPDIR="/tmp/${USER}-apptainer-build" SINGULARITY_TMPDIR="/tmp/${USER}-apptainer-build"
mkdir -p "$APPTAINER_TMPDIR"
SWE_PY="${SWE_PY:-.venv_swebench/bin/python}"

model_tag="${MODEL##*/}"
for rt in "${CODE_METHODS[@]}"; do
  run_dir="$RUNS_DIR/$CODE_DATASET/$model_tag/$rt"
  [ -d "$run_dir" ] || run_dir=$(dirname "$(find "$RUNS_DIR/$CODE_DATASET" -name rows.jsonl -path "*${rt}*" 2>/dev/null | head -1)")
  glog="$LOGDIR/RESOLVE__${rt}.log"
  echo "[$(ts)] >> resolve grade $rt  run_dir=$run_dir -> $glog"
  if [ ! -f "$run_dir/rows.jsonl" ]; then echo "   !! no rows.jsonl at $run_dir — skip"; continue; fi
  {
    echo "== make preds =="
    "$PY" scripts/make_swebench_preds.py "$run_dir" --apply-check --dataset "$CODE_DATASET" || true
    echo "== apptainer resolve grade =="
    "$SWE_PY" -m evaluation.swebench_apptainer "$run_dir" \
      --subset verified --workers 8 --timeout 1800 --no-pull || true
  } >"$glog" 2>&1
  echo "[$(ts)]    done grading $rt"
done

echo "=============================================================="
echo " SMOKE COMPLETE $(date)"
echo "=============================================================="
