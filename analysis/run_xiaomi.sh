#!/usr/bin/env bash
# Xiaomi MiMo API backbone runs. MUST run on the LOGIN NODE (compute nodes have no internet).
# Logs to an in-repo dir (never /tmp — wiped mid-run). Gentle concurrency for API rate limits.
cd ${REPO_ROOT:-.}
export OPENAI_API_KEY='sk-c4hmif0112itqxpvags3dtpwjqr2t8nc761zuka796tskped'
export OPENAI_BASE_URL='https://api.xiaomimimo.com/v1'
export BM25_BACKEND=pyserini STRUCTURED_BACKEND=lucene MAX_VISIT_TOKENS=12000 MAX_STEPS=100
export AGENT_DRIVER=sdk
COND="$1"; LIMIT="${2:-100}"; W="${3:-2}"; RUNS="${4:-runs/_xiaomi}"
PYTHONPATH=. envs/bin/python -m evaluation.run_eval \
  --dataset browsecomp_plus_structured --retriever "$COND" \
  --policy llm --backend api --api-base "$OPENAI_BASE_URL" \
  --model mimo-v2.5 --max-steps 100 --temperature 0.6 --seed 42 \
  --workers "$W" --limit "$LIMIT" --k 1 3 5 10 --runs-dir "$RUNS"
