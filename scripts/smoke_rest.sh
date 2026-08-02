#!/usr/bin/env bash
# Priority-reordered remainder of the rebuild smoke, after browsecomp's per-condition corpus
# build was found to block the sequential queue. Order by user priority + GPU economics:
#   A) CODE arm now (GPU is idle during browsecomp's CPU corpus build — free capacity)
#   B) extra doc read-strategy arms on the SMALL datasets (hotpotqa/2wiki/musique) — bm25_dci,
#      bm25_fetch — so all 5 doc methods are validated there (browsecomp excluded: 3GB corpus
#      makes dci/grep arms pathological; its decisive test is BQL-vs-whole-doc token cost).
#   C) browsecomp_bm25 once the in-flight browsecomp/research (pid $BC_PID) frees the GPU.
#   D) CODE resolve grading (apptainer, warmed SIFs).
#   E) BrowseComp LLM-judge + consolidated SUMMARY.txt.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"
API_BASE="${API_BASE:-http://127.0.0.1:8765/v1}"
RUNS_DIR="${RUNS_DIR:-runs/_rebuild_smoke}"
MAX_STEPS="${MAX_STEPS:-50}"
PY="${PY:-envs/bin/python}"
LOGDIR="$RUNS_DIR/logs"; mkdir -p "$LOGDIR"
BC_PID="${BC_PID:-}"   # in-flight browsecomp/research python pid to wait on for GPU-heavy phases

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export AGENT_DRIVER="${AGENT_DRIVER:-loop}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
ts() { date +%H:%M:%S; }

run_one() {   # dataset retriever limit workers
  local ds="$1" rt="$2" lim="$3" wk="$4"
  local log="$LOGDIR/${ds}__${rt}.log"
  echo "[$(ts)] >> $ds / $rt (limit=$lim workers=$wk) -> $log"
  "$PY" -m evaluation.run_eval --dataset "$ds" --retriever "$rt" \
    --policy llm --backend api --api-base "$API_BASE" --model "$MODEL" \
    --max-steps "$MAX_STEPS" --workers "$wk" --limit "$lim" \
    --runs-dir "$RUNS_DIR" --k 1 3 5 10 --seeds 42 >"$log" 2>&1
  echo "[$(ts)]    exit=$? $ds / $rt"
}

wait_pid() {  # wait for a pid to exit (GPU-heavy phase gate)
  local p="$1" what="$2"
  [ -z "$p" ] && return 0
  echo "[$(ts)] waiting for $what (pid=$p) to free the GPU ..."
  while kill -0 "$p" 2>/dev/null; do sleep 20; done
  echo "[$(ts)] $what finished"
}

echo "==== SMOKE REST start $(date) ===="

# ---- A) CODE arm (whole-fix). workers=4 to leave KV headroom while browsecomp may co-run ----
for rt in agent_codefix_patch agent_codefix_grep_patch; do
  run_one swebench_verified "$rt" 15 4
done

# ---- B) extra doc arms on SMALL datasets (browsecomp excluded on purpose) --------------------
for ds in hotpotqa 2wiki musique; do
  for rt in agent_research_bm25_dci agent_research_bm25_fetch; do
    run_one "$ds" "$rt" 12 8
  done
done

# ---- C) browsecomp_bm25 (decisive large-doc token comparison vs BQL) — after research frees GPU
wait_pid "$BC_PID" "browsecomp/research"
run_one browsecomp_plus agent_research_bm25 12 8

# ---- D) CODE resolve grading ----------------------------------------------------------------
module load apptainer/1.5.0 2>/dev/null || module load apptainer 2>/dev/null || true
export APPTAINER_CACHEDIR="$PWD/.cache/apptainer" SINGULARITY_CACHEDIR="$PWD/.cache/apptainer"
export APPTAINER_TMPDIR="/tmp/${USER}-apptainer-build" SINGULARITY_TMPDIR="/tmp/${USER}-apptainer-build"
mkdir -p "$APPTAINER_TMPDIR"
SWE_PY="${SWE_PY:-.venv_swebench/bin/python}"
model_tag="${MODEL##*/}"
for rt in agent_codefix_patch agent_codefix_grep_patch; do
  run_dir="$RUNS_DIR/agent/swebench_verified/$model_tag/$rt"
  glog="$LOGDIR/RESOLVE__${rt}.log"
  echo "[$(ts)] >> resolve grade $rt  run_dir=$run_dir -> $glog"
  [ -f "$run_dir/rows.jsonl" ] || { echo "   !! no rows.jsonl — skip"; continue; }
  {
    "$PY" scripts/make_swebench_preds.py "$run_dir" --apply-check --dataset swebench_verified || true
    "$SWE_PY" -m evaluation.swebench_apptainer "$run_dir" --subset verified --workers 8 --timeout 1800 --no-pull || true
  } >"$glog" 2>&1
  echo "[$(ts)]    done grading $rt"
done

# ---- E) BrowseComp judge + summary ----------------------------------------------------------
echo "EXTRA SMOKE COMPLETE (rest-driver)" >> "$LOGDIR/DRIVER_extra.log"   # release finalize's wait gate
EXTRA_PID="" bash scripts/smoke_finalize.sh >>"$LOGDIR/DRIVER_finalize.log" 2>&1 || true
echo "==== SMOKE REST COMPLETE $(date) ===="
