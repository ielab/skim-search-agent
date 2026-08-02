#!/usr/bin/env bash
# Resume every cell interrupted by the filesystem outage on 2026-07-10.
#
# SAME-DIR RESUME IS CORRECT HERE: code + budgets (MAX_STEPS/LIMIT) are unchanged since each
# cell's LAST surviving submission, so resubmitting the exact original RUNS_DIR + env is safe —
# children resume per-instance from rows.jsonl (see memory: reference-suite-submission-logic).
# SKIP_CHECK=1 is used for the small validation cells (1-5) so submission doesn't pay a
# full-dataset --check-complete scan; each child still self-gates on completeness before
# claiming a GPU (run.sh's own `run_eval --check-complete` before serving vLLM), so an
# already-finished condition just fast-exits. The main sweep (cell 8) is delegated to the
# existing scripts/resume_sweep.sh UNCHANGED, which deliberately does NOT set SKIP_CHECK (it
# wants the real completeness scan) — so SKIP_CHECK/QOS are scoped per-block below and never
# exported globally, to avoid leaking into that call.
#
# CRITICAL — env vars that describe a NON-DEFAULT run (INDRI_DENSE, MAX_VISIT_TOKENS, QOS,
# SKIP_CHECK) are set inside a `( subshell )` per cell, never left `export`ed at top level,
# so cell N's knobs can never leak into cell N+1's submission.
#
# Row counts at interruption (verified against rows.jsonl / config.json on disk; see the
# report for exact evidence per cell):
#   1. runs/_headline_validation  browsecomp_plus_structured:
#        agent_research_snip        94/200
#        agent_research_indri_snip  99/200
#        agent_research_dense_fetch  2/200
#        agent_research_v2           0/200
#        agent_research_indri      198/200
#      (hotpotqa_structured v2/indri are ALREADY 200/200 complete for this RUNS_DIR — NOT
#      resubmitted here, unlike scripts/validate_headline.sh which also touches hotpotqa.)
#   2. runs/_dense_validation  browsecomp_plus_structured:
#        agent_research_indri_snip 124/200   (INDRI_DENSE=1)
#   3. runs/_visit_uncapped  browsecomp_plus_structured:
#        agent_research_bm25       106/200   (MAX_VISIT_TOKENS=12000)
#   4. runs/_fullvisit  browsecomp_plus_structured  (MAX_VISIT_TOKENS=12000, lexical — NO
#      INDRI_DENSE):
#        agent_research_dense        19/200
#        agent_research_indri_visit   4/200
#        agent_research_bm25q         0/200  (dir doesn't exist yet — see cell 6 note)
#   5. runs/_fullvisit_dense  browsecomp_plus_structured  (MAX_VISIT_TOKENS=12000 +
#      INDRI_DENSE=1):
#        agent_research_indri_visit   3/200
#   6. "bm25q cell" — NOT a separate RUNS_DIR. Evidence (slurm_logs/agent_runs/
#      browsecomp_plus_structured/as-suite-*-28633688_{0,1,2}.out + sacct) shows job array
#      28633688 (submitted 2026-07-10T12:43:22) ran task0=agent_research_indri_visit (port
#      8101), task1=agent_research_dense (port 8102), task2=agent_research_bm25q (port 8103)
#      as ONE array — i.e. ONE eval_agent_suite.sh call, so all three tasks share the SAME
#      RUNS_DIR/env. Tasks 0/1's config.json both say "runs_dir": "runs/_fullvisit"; task 2
#      (bm25q) was still in vLLM CUDA-graph capture (no "eval:" line ever printed in its .err)
#      when the outage's SIGNAL Terminated hit at 16:53:45, so it never got to write rows.jsonl
#      or config.json of its own — hence "run dir may not exist yet". Folded into cell 4 below.
#   7. oneshot_rag (job name oneshot-rag), DATASET=browsecomp_plus_structured: runs/_oneshot is
#      currently EMPTY (the prior capped attempt was archived to
#      runs_old/oneshot_capped_20260710_1323/; the uncapped context-fit retry, job 28638714, was
#      killed by the outage before writing anything) — this is a fresh, non-resumable run
#      (oneshot_rag.py has no per-instance resume; it always (re)writes all 200 rows per
#      retriever). ONESHOT_DOC_CAP is NOT set: scripts/oneshot_rag.py's default (0) already means
#      "context-fit" (auto-splits a 120000-token stuffing budget across the k retrieved docs —
#      see the `stuff_docs()` docstring) — this is now the in-code default, matching the task.
#   8. Main sweep (runs/agent, jobs 28613495-28613512 etc.) — delegated verbatim to
#      scripts/resume_sweep.sh; not duplicated here.
#
# QOS NOTE: `sacct -u $USER --format=JobID,JobName,QOS,State` shows EVERY validation-family job
# (cells 1-6, including the retired/superseded attempts) submitted with QOS=normal — this matches
# scripts/validate_headline.sh's explicit `QOS=normal  # (user directive)` and is used verbatim
# below for cells 1-5, NOT the generic QOS=auto the main sweep uses.
set -uo pipefail
cd "$(dirname "$0")/.."

# Shared, safe-to-leak defaults (every cell wants these; resume_sweep.sh re-exports its own
# copies anyway, so exporting them here is harmless idempotent overlap, not a leak).
export MODEL="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
export AGENT_DRIVER=loop
export SUBMIT=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

echo "###### cell 1/8: runs/_headline_validation (browsecomp_plus_structured) ######"
(
  export RUNS_DIR="runs/_headline_validation"
  export QOS=normal
  export SKIP_CHECK=1
  DATASET=browsecomp_plus_structured \
    AGENTS="agent_research_snip agent_research_indri_snip agent_research_dense_fetch agent_research_v2 agent_research_indri" \
    MAX_STEPS=100 LIMIT=200 \
    bash scripts/eval_agent_suite.sh 2>&1 | grep -E ">> qos|>> [0-9]|already complete"
)

echo "###### cell 2/8: runs/_dense_validation (browsecomp_plus_structured, INDRI_DENSE=1) ######"
(
  export RUNS_DIR="runs/_dense_validation"
  export QOS=normal
  export SKIP_CHECK=1
  export INDRI_DENSE=1
  DATASET=browsecomp_plus_structured \
    AGENTS="agent_research_indri_snip" \
    MAX_STEPS=100 LIMIT=200 \
    bash scripts/eval_agent_suite.sh 2>&1 | grep -E ">> qos|>> [0-9]|already complete"
)

echo "###### cell 3/8: runs/_visit_uncapped (browsecomp_plus_structured, MAX_VISIT_TOKENS=12000) ######"
(
  export RUNS_DIR="runs/_visit_uncapped"
  export QOS=normal
  export SKIP_CHECK=1
  export MAX_VISIT_TOKENS=12000
  DATASET=browsecomp_plus_structured \
    AGENTS="agent_research_bm25" \
    MAX_STEPS=100 LIMIT=200 \
    bash scripts/eval_agent_suite.sh 2>&1 | grep -E ">> qos|>> [0-9]|already complete"
)

echo "###### cell 4/8: runs/_fullvisit (browsecomp_plus_structured, MAX_VISIT_TOKENS=12000, lexical) ######"
echo "     (includes agent_research_bm25q — see cell 6 note in the header comment above)"
(
  export RUNS_DIR="runs/_fullvisit"
  export QOS=normal
  export SKIP_CHECK=1
  export MAX_VISIT_TOKENS=12000
  DATASET=browsecomp_plus_structured \
    AGENTS="agent_research_dense agent_research_indri_visit agent_research_bm25q" \
    MAX_STEPS=100 LIMIT=200 \
    bash scripts/eval_agent_suite.sh 2>&1 | grep -E ">> qos|>> [0-9]|already complete"
)

echo "###### cell 5/8: runs/_fullvisit_dense (browsecomp_plus_structured, MAX_VISIT_TOKENS=12000 + INDRI_DENSE=1) ######"
(
  export RUNS_DIR="runs/_fullvisit_dense"
  export QOS=normal
  export SKIP_CHECK=1
  export MAX_VISIT_TOKENS=12000
  export INDRI_DENSE=1
  DATASET=browsecomp_plus_structured \
    AGENTS="agent_research_indri_visit" \
    MAX_STEPS=100 LIMIT=200 \
    bash scripts/eval_agent_suite.sh 2>&1 | grep -E ">> qos|>> [0-9]|already complete"
)

echo "###### cell 7/8: oneshot_rag baseline (job name oneshot-rag), browsecomp_plus_structured ######"
(
  # oneshot_rag.sbatch hardcodes #SBATCH --qos=normal itself; DATASET is a required var the
  # script enforces with `${DATASET:?...}`. ONESHOT_DOC_CAP is deliberately left UNSET — 0 is
  # both the CLI default and now means "context-fit" in-code (see header note).
  DATASET=browsecomp_plus_structured sbatch scripts/oneshot_rag.sbatch
)

echo "###### cell 8/8: main sweep — delegating to scripts/resume_sweep.sh (unmodified) ######"
bash scripts/resume_sweep.sh

echo "== queue summary (as-suite + oneshot-rag jobs) =="
squeue -u "$USER" -r -h -o "%j|%q|%t" 2>/dev/null \
  | grep -E "^as-suite-|^oneshot-rag" \
  | sed 's/as-suite-//;s/-Tongyi[^|]*//' \
  | sort | uniq -c
