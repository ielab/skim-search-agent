#!/usr/bin/env bash
# HEADLINE VALIDATION (pre-registered, task #8): paired comparison on browsecomp_plus_structured
# that decides the paper's headline. Arms:
#   bm25 + research(v1)  : REUSED from the running full @100 sweep (first-200 rows, same instances)
#   research_v2          : NEW — typed date ranges + coverage feedback (LIMIT=200)
#   research_indri       : NEW — graded Indri QL backend (LIMIT=200)
# Plus the wiki parity gate: research_v2 + research_indri on hotpotqa_structured LIMIT=200 @50
# (v1/bm25 wiki rows already exist). All QOS=normal (user directive). Separate RUNS_DIR so these
# validation runs never touch the main sweep's dirs (resume-semantics rule).
set -uo pipefail
cd "$(dirname "$0")/.."
export MODEL="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B" AGENT_DRIVER=loop SUBMIT=1 QOS=normal SKIP_CHECK=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export RUNS_DIR="runs/_headline_validation"

DATASET=browsecomp_plus_structured AGENTS="agent_research_v2 agent_research_indri" \
  MAX_STEPS=100 LIMIT=200 bash scripts/eval_agent_suite.sh 2>&1 | grep -E ">> qos|>> [0-9]"
DATASET=hotpotqa_structured AGENTS="agent_research_v2 agent_research_indri" \
  LIMIT=200 bash scripts/eval_agent_suite.sh 2>&1 | grep -E ">> qos|>> [0-9]"

echo "== queue =="
squeue -u "$USER" -r -h -o "%j|%q|%t" 2>/dev/null | sed 's/as-suite-//;s/-Tongyi[^|]*//' | sort | uniq -c
echo "NOTE: first browsecomp indri job pays the one-time Indri index build, then persists to indexes/indri/."
