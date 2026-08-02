#!/usr/bin/env bash
# Audit every final-table cell: print any that is INCOMPLETE *and* whose rows.jsonl has not been
# written in >STALE_MIN minutes (default 25) = a job that finished/died short and has no live writer.
# Output: one "DEAD|<dataset>|<runs_dir>|<condition>|<n>|<target>|<env>" line per dead cell (nothing
# if all incomplete cells are live or complete). Read-only; never submits or cancels anything.
# Env override: STALE_MIN=25. Used by the hourly watcher + manual `bash scripts/audit_dead_cells.sh`.
cd "$(dirname "$0")/.."
M="Tongyi-DeepResearch-30B-A3B"; NOW=$(date +%s); STALE=${STALE_MIN:-25}
chk() { # <rows.jsonl> <target> <dataset> <runs_dir> <cond> <env>
  local f="$1" tgt="$2"; local n; n=$(wc -l < "$f" 2>/dev/null || echo 0)
  [ "$n" -ge "$tgt" ] && return
  # No rows.jsonl yet = freshly launched / still loading model (not a mid-fill death). A cell
  # that died short ALWAYS has an existing file, so skipping the missing-file case avoids
  # false-alarming on just-submitted cells; a genuinely-failed launch is caught by the queue count.
  [ -f "$f" ] || return
  local m; m=$(stat -c %Y "$f" 2>/dev/null || echo 0); local age=$(( (NOW-m)/60 ))
  [ "$age" -gt "$STALE" ] && echo "DEAD|$3|$4|$5|$n|$tgt|$6"
}
bc() { chk "runs/$1/agent/browsecomp_plus_structured/$M/$2/rows.jsonl" "$3" browsecomp_plus_structured "$1" "$2" "$4"; }
ag() { chk "runs/agent/browsecomp_plus_structured/$M/$1/rows.jsonl" 830 browsecomp_plus_structured agent "$1" "$2"; }
wk() { chk "runs/$2/agent/${1}_structured/$M/$3/rows.jsonl" "$4" "${1}_structured" "$2" "$3" "$5"; }
# --- browsecomp final-table cells (WORKERS=2) ---
bc _visit_uncapped        agent_research_bm25            830 "STRUCTURED_BACKEND=python"
bc _visit_uncapped        agent_research_bm25_autoread   830 "STRUCTURED_BACKEND=lucene"
bc _visit_uncapped        agent_research_dense_autoread  830 "STRUCTURED_BACKEND=lucene"
bc _visit_uncapped_k10    agent_research_bm25            830 "STRUCTURED_BACKEND=lucene|BM25_VISIT_TOPK=10"
bc _fullvisit             agent_research_dense           830 "STRUCTURED_BACKEND=lucene"
bc _fullvisit             agent_research_indri_visit     830 "STRUCTURED_BACKEND=lucene"
bc _fullvisit             agent_research_bql_visit       830 "STRUCTURED_BACKEND=lucene"
bc _fullvisit             agent_research_hybrid          830 "STRUCTURED_BACKEND=lucene"
bc _fullvisit             agent_research_bql_dense_visit 830 "STRUCTURED_BACKEND=lucene"
bc _fullvisit_dense       agent_research_indri_visit     830 "STRUCTURED_BACKEND=lucene|INDRI_DENSE=1"
bc _qwen_fullvisit_dense  agent_research_indri_visit     830 "STRUCTURED_BACKEND=lucene|INDRI_DENSE=1|DENSE_MODEL=Qwen/Qwen3-Embedding-0.6B"
bc _headline_validation   agent_research                 830 "STRUCTURED_BACKEND=lucene"
bc _headline_validation   agent_research_snip            830 "STRUCTURED_BACKEND=lucene"
bc _headline_validation   agent_research_indri           830 "STRUCTURED_BACKEND=lucene"
bc _headline_validation   agent_research_indri_snip      830 "STRUCTURED_BACKEND=lucene"
bc _headline_validation   agent_research_dense_fetch     830 "STRUCTURED_BACKEND=lucene"
bc _headline_validation   agent_research_bm25_fetch_snip 830 "STRUCTURED_BACKEND=lucene"
bc _headline_validation   agent_research_hybrid_fetch_snip 830 "STRUCTURED_BACKEND=lucene"
bc _headline_validation   agent_research_bql_dense_snip  830 "STRUCTURED_BACKEND=lucene"
bc _dense_validation      agent_research_indri_snip      830 "STRUCTURED_BACKEND=lucene|INDRI_DENSE=1"
ag agent_research_dci     "STRUCTURED_BACKEND=lucene"
ag agent_research_bm25_dci "STRUCTURED_BACKEND=lucene"
chk "runs/_visit_uncapped/agent/browsecomp_plus_flat/$M/agent_research_bm25/rows.jsonl" 830 browsecomp_plus_flat _visit_uncapped agent_research_bm25 "STRUCTURED_BACKEND=python"
# --- wiki (hotpotqa WORKERS=6, musique WORKERS=3; full corpus targets) ---
wk hotpotqa _visit_uncapped_k10  agent_research_bm25         7343 "STRUCTURED_BACKEND=lucene|BM25_VISIT_TOPK=10|W=6"
wk hotpotqa _fullvisit_dense     agent_research_indri_visit  7343 "STRUCTURED_BACKEND=lucene|INDRI_DENSE=1|W=6"
wk hotpotqa _headline_validation agent_research_bql_dense_snip 7343 "STRUCTURED_BACKEND=lucene|W=6"
wk musique  _visit_uncapped      agent_research_bm25         2409 "STRUCTURED_BACKEND=lucene|W=3"
wk musique  _visit_uncapped_k10  agent_research_bm25         2409 "STRUCTURED_BACKEND=lucene|BM25_VISIT_TOPK=10|W=3"
wk musique  _fullvisit_dense     agent_research_indri_visit  2409 "STRUCTURED_BACKEND=lucene|INDRI_DENSE=1|W=3"
wk musique  _headline_validation agent_research_bql_dense_snip 2409 "STRUCTURED_BACKEND=lucene|W=3"
# --- controlled decomposition cell added 2026-07-14 (bql+dense fetch = winner minus snippet) ---
bc _headline_validation   agent_research_bql_dense_fetch 830 "STRUCTURED_BACKEND=lucene"
wk hotpotqa _headline_validation agent_research_bql_dense_fetch 7343 "STRUCTURED_BACKEND=lucene|W=6"
wk musique  _headline_validation agent_research_bql_dense_fetch 2409 "STRUCTURED_BACKEND=lucene|W=3"
# --- plain dense fetch (no-excerpt control) added 2026-07-14 ---
bc _headline_validation   agent_research_dense_fetch_plain 830 "STRUCTURED_BACKEND=lucene"
wk hotpotqa _headline_validation agent_research_dense_fetch_plain 7343 "STRUCTURED_BACKEND=lucene|W=6"
wk musique  _headline_validation agent_research_dense_fetch_plain 2409 "STRUCTURED_BACKEND=lucene|W=3"
# --- whole-doc dense visit on wiki (baseline symmetry vs browsecomp) added 2026-07-15 ---
wk hotpotqa _fullvisit  agent_research_dense 7343 "STRUCTURED_BACKEND=lucene|W=6"
wk musique  _fullvisit  agent_research_dense 2409 "STRUCTURED_BACKEND=lucene|W=3"
# --- wiki dci (default RUNS_DIR=runs; brute-force reference; was missing from audit → stalled 5d unnoticed) 2026-07-15 ---
chk "runs/agent/hotpotqa_structured/$M/agent_research_dci/rows.jsonl" 7343 hotpotqa_structured agent agent_research_dci "STRUCTURED_BACKEND=lucene|W=2"
chk "runs/agent/musique_structured/$M/agent_research_dci/rows.jsonl"  2409 musique_structured  agent agent_research_dci "STRUCTURED_BACKEND=lucene|W=2"
# --- wiki baselines (auto-read x2, hybrid visit, bm25+snip fetch) launched sharded 2026-07-16 ---
wk hotpotqa _visit_uncapped      agent_research_bm25_autoread   7343 "STRUCTURED_BACKEND=lucene|W=6"
wk hotpotqa _visit_uncapped      agent_research_dense_autoread  7343 "STRUCTURED_BACKEND=lucene|W=6"
wk hotpotqa _fullvisit           agent_research_hybrid          7343 "STRUCTURED_BACKEND=lucene|W=6"
wk hotpotqa _headline_validation agent_research_bm25_fetch_snip 7343 "STRUCTURED_BACKEND=lucene|W=6"
wk musique  _visit_uncapped      agent_research_bm25_autoread   2409 "STRUCTURED_BACKEND=lucene|W=3"
wk musique  _visit_uncapped      agent_research_dense_autoread  2409 "STRUCTURED_BACKEND=lucene|W=3"
wk musique  _fullvisit           agent_research_hybrid          2409 "STRUCTURED_BACKEND=lucene|W=3"
wk musique  _headline_validation agent_research_bm25_fetch_snip 2409 "STRUCTURED_BACKEND=lucene|W=3"
