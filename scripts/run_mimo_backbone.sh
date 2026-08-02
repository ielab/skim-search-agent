#!/usr/bin/env bash
# Third-backbone (mimo-v2.5) finisher — LOGIN NODE ONLY (API model; compute nodes have no internet).
# Resumes runs/_xiaomi in place (rows.jsonl append; already-scored instances are skipped).
# Usage:  MIMO_API_KEY=... nohup bash scripts/run_mimo_backbone.sh >> logs/mimo_backbone.log 2>&1 &
#    or:  put the key (single line) in ~/.mimo_key (chmod 600) and run without the env var.
set -euo pipefail
cd "$(dirname "$0")/.."
KEY="${MIMO_API_KEY:-$( [ -f "$HOME/.mimo_key" ] && cat "$HOME/.mimo_key" || true )}"
[ -n "$KEY" ] || { echo "FATAL: no MIMO_API_KEY and no ~/.mimo_key"; exit 1; }
export OPENAI_API_KEY="$KEY"
export VLLM_API_BASE="https://api.xiaomimimo.com/v1"
export AGENT_DRIVER=sdk
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
LIMIT="${LIMIT:-830}"   # FULL collection (user order 2026-07-27)
WORKERS="${WORKERS:-2}"
for COND in agent_research_bm25 agent_research_bql_dense_snip; do
  echo "###### $(date '+%F %T') $COND (limit=$LIMIT) ######"
  ./envs/bin/python -m evaluation.run_eval \
    --dataset browsecomp_plus_structured --retriever "$COND" \
    --model mimo-v2.5 --backend api --api-base "$VLLM_API_BASE" \
    --runs-dir runs/_xiaomi --max-steps 100 --workers "$WORKERS" \
    --limit "$LIMIT" --seed 42 --temperature 0.6 || echo "WARN: $COND exited nonzero (resumable)"
done
echo "###### $(date '+%F %T') both conditions done ######"
