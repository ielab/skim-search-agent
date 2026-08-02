#!/usr/bin/env bash
# Resubmission after the method/harness fixes (soft-AND fallback, live bm25_fetch/bm25_dci,
# budget 100 for browsecomp/musique). Encodes the CORRECT sequence for the suite's resume
# semantics (see memory: reference-suite-submission-logic):
#   resume/check-complete is keyed ONLY on runs/agent/<ds>/<model>/<cond> — so every condition
#   being re-run with changed code or budget MUST have its dir moved aside FIRST, or the suite
#   will silently skip it (complete) or MIX old/new rows (partial).
#
# What is re-run and why:
#   browsecomp/musique structured (5 arms) + flats @ MAX_STEPS=100  — budget change (74%/55% were
#     capped at 50) + method change (research) + live-retrieval fixes (bm25_dci/bm25_fetch)
#   hotpotqa/2wiki structured: research (method change) + bm25_dci/bm25_fetch (live fix) @ 50
# What stays (valid, untouched): wiki research_bm25 / research_dci / flats @50.
# The moved @50 dirs become the budget/no-fallback ABLATION set (runs_old/ablation_50steps_*).
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d_%H%M)
ABL="runs_old/ablation_50steps_$STAMP"
MODEL_DIR="Tongyi-DeepResearch-30B-A3B"

echo "== 1) cancel still-running tasks whose condition is being re-run =="
# 2wiki research (old method, becomes partial ablation). Cancel BEFORE moving the dir
# (the job holds an open fd on rows.jsonl and would keep appending into the moved file).
for spec in $(squeue -u "$USER" -r -h -o "%i|%j|%K" 2>/dev/null | grep -E "2wiki_structured" | awk -F'|' '$3==0{print $1}'); do
  scancel "$spec" && echo "  cancelled $spec (2wiki research, old method)"
done
sleep 3

echo "== 2) archive the superseded @50 dirs (ablation set) =="
mkdir -p "$ABL"
move() {  # move <runs/agent-relative dir> if it exists
  local d="runs/agent/$1"
  [ -d "$d" ] || { echo "  (absent: $1)"; return 0; }
  local tgt="$ABL/$(echo "$1" | sed 's#/#__#g')"
  mv "$d" "$tgt" && echo "  archived $1"
}
move "browsecomp_plus_structured/$MODEL_DIR/agent_research"
move "browsecomp_plus_structured/$MODEL_DIR/agent_research_bm25"
move "browsecomp_plus_flat/$MODEL_DIR/agent_research_bm25"
move "musique_structured/$MODEL_DIR/agent_research"
move "musique_structured/$MODEL_DIR/agent_research_bm25"
move "musique_flat/$MODEL_DIR/agent_research_bm25"
move "hotpotqa_structured/$MODEL_DIR/agent_research"
move "2wiki_structured/$MODEL_DIR/agent_research"

echo "== 3) resubmit (QOS=auto: express till its cpu-min cap, overflow to normal) =="
export MODEL="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B" AGENT_DRIVER=loop SUBMIT=1 QOS=auto
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
FIVE="agent_research agent_research_bm25 agent_research_dci agent_research_bm25_dci agent_research_bm25_fetch"

# budget-100 datasets (browsecomp/musique): all arms + flats
for DS in browsecomp_plus_structured musique_structured; do
  DATASET=$DS AGENTS="$FIVE" MAX_STEPS=100 SKIP_CHECK=1 bash scripts/eval_agent_suite.sh 2>&1 | grep -E ">> qos|>> [0-9]|complete"
done
for DS in browsecomp_plus_flat musique_flat; do
  DATASET=$DS AGENTS="agent_research_bm25" MAX_STEPS=100 SKIP_CHECK=1 bash scripts/eval_agent_suite.sh 2>&1 | grep -E ">> qos|>> [0-9]|complete"
done
# budget-50 wiki datasets: only the changed arms (research = new method; hybrids = live fix)
for DS in hotpotqa_structured 2wiki_structured; do
  DATASET=$DS AGENTS="agent_research agent_research_bm25_dci agent_research_bm25_fetch" SKIP_CHECK=1 bash scripts/eval_agent_suite.sh 2>&1 | grep -E ">> qos|>> [0-9]|complete"
done

echo "== 4) final queue =="
squeue -u "$USER" -r -h -o "%j|%q|%r" 2>/dev/null | sed 's/as-suite-//;s/-Tongyi[^|]*//' | sort | uniq -c
echo "done. ablation set: $ABL"
