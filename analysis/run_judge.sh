#!/usr/bin/env bash
# Auto-judge, BROWSECOMP ONLY (its canonical metric is LLM-as-judge per BrowseComp-Plus's own
# protocol; HotpotQA/MuSiQue are EM/F1 benchmarks, so judging them is neither needed nor
# methodologically appropriate). Login node (needs internet). Resumable: the hash-keyed
# judge_cache.jsonl means re-running only pays for NEW rows.
#
# ONE PROCESS PER CELL, sequentially. judge_cells.py loads a whole rows.jsonl into memory, and
# these files run to 16GB; a single process loading one cost 44.5GB RSS on the shared login node
# (2026-07-23) and had to be killed. One process per cell caps peak RSS at the largest single
# file (1.3GB here) and releases it at exit, and a crash on one cell keeps every other cell's
# verdicts, since judge_cache.jsonl is appended per cell.
#
# EXCLUDED: agent_research_dense_autoread (4.9GB) and agent_research_bm25_autoread (16GB), the
# full-text-dump baselines. bm25_autoread is already fully cached; dense_autoread would cost more
# RSS than the rest of this sweep combined to fill one table cell that is currently withheld.
# --cell-exact (not --cell) is REQUIRED below: 'agent_research_bm25' is a substring of
# 'agent_research_bm25_autoread', so a substring filter would load the 16GB file for nothing.
set -uo pipefail
cd ${REPO_ROOT:-.}

export OPENAI_API_KEY='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'
mkdir -p analysis/judge_logs
LOG=analysis/judge_logs/judge_$(date +%m%d_%H%M).log
XB=runs/_xbackbone/agent/browsecomp_plus_structured/Qwen-AgentWorld-35B-A3B

run() {  # run <description> <judge_cells.py args...>
  local what="$1"; shift
  echo "=== $(date +%H:%M:%S)  $what" >>"$LOG"
  PYTHONPATH=. envs/bin/python scripts/judge_cells.py --workers 12 "$@" >>"$LOG" 2>&1
  echo "    exit=$?" >>"$LOG"
}

# Cross-backbone pair FIRST: the replication claim currently rests on EM alone, so these two
# cells are the highest-value judge coverage in the set. They are unreachable via the REGISTRY
# (their model dir is not compare_cells.MODEL_DIR), hence --extra-cell.
run "xbackbone baseline (Qwen-AgentWorld bm25)"        --extra-cell "$XB/agent_research_bm25"
run "xbackbone Sieve (Qwen-AgentWorld bql_dense_snip)" --extra-cell "$XB/agent_research_bql_dense_snip"

# Tongyi cells, largest pending count first.
for c in agent_research_bql_dense_fetch agent_research_dense_fetch_plain \
         agent_research_bql_dense_visit agent_research_dci \
         agent_research_indri_visit agent_research_snip agent_research_bql_visit; do
  run "$c" --cell-exact "$c"
done

echo "=== $(date +%H:%M:%S)  ALL DONE" >>"$LOG"
