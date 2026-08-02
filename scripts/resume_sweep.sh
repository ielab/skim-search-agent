#!/usr/bin/env bash
# Resume the paused sweep — safe to run any time: the suite's --check-complete gate queues ONLY
# conditions whose run dir is incomplete, and children resume per-instance from rows.jsonl.
# (Code + budgets are unchanged since the pause, so same-dir resume is the CORRECT case here —
# see memory: reference-suite-submission-logic. Do NOT change code before resuming without
# archiving the affected dirs first.)
# Budgets: browsecomp/musique @ MAX_STEPS=100, wiki @ 50 (defaults). QOS=auto piles express
# until its cpu-min cap, then overflows to normal.
set -uo pipefail
cd "$(dirname "$0")/.."
export MODEL="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B" AGENT_DRIVER=loop SUBMIT=1 QOS="${QOS:-auto}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
FIVE="agent_research agent_research_bm25 agent_research_dci agent_research_bm25_dci agent_research_bm25_fetch"

for DS in browsecomp_plus_structured musique_structured; do
  DATASET=$DS AGENTS="$FIVE" MAX_STEPS=100 bash scripts/eval_agent_suite.sh 2>&1 | grep -E ">> qos|>> [0-9]|already complete"
done
for DS in browsecomp_plus_flat musique_flat; do
  DATASET=$DS AGENTS="agent_research_bm25" MAX_STEPS=100 bash scripts/eval_agent_suite.sh 2>&1 | grep -E ">> qos|>> [0-9]|already complete"
done
for DS in hotpotqa_structured 2wiki_structured; do
  DATASET=$DS AGENTS="$FIVE" bash scripts/eval_agent_suite.sh 2>&1 | grep -E ">> qos|>> [0-9]|already complete"
done
for DS in hotpotqa_flat 2wiki_flat; do
  DATASET=$DS AGENTS="agent_research_bm25" bash scripts/eval_agent_suite.sh 2>&1 | grep -E ">> qos|>> [0-9]|already complete"
done
echo "== queue =="
squeue -u "$USER" -r -h -o "%j|%q|%t" 2>/dev/null | sed 's/as-suite-//;s/-Tongyi[^|]*//' | sort | uniq -c
