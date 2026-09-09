#!/usr/bin/env bash
# Pre-stage everything for OFFLINE eval. RUN ON A NODE WITH INTERNET (login node).
# Saves datasets to data/<name> (loaded offline via load_from_disk) and clones the
# repos into data/repos (read offline via git archive). Also pre-download the MODEL
# into your HF cache here (the GPU node has no internet).
set -euo pipefail
cd "$(dirname "$0")/.."   # always resolve data/ relative to the repo root

DATA="${AGENT_SEARCH_DATA:-data}"
DATASET="${1:-all}"   # all | swebench_verified | swebench_lite | loc_bench (code datasets)
mkdir -p "$DATA"

# Keep dataset download caches with the staged data by default. This avoids
# permission problems on managed clusters and makes the staged directory portable.
export HF_HOME="${HF_HOME:-$DATA/hf_cache}"

if [ -n "${PYTHON:-}" ]; then
  :
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi


echo ">> saving SWE-bench splits to $DATA/ ..."
AGENT_SEARCH_DATA="$DATA" AGENT_SEARCH_DOWNLOAD_DATASET="$DATASET" "$PYTHON" - <<PY
from datasets import load_dataset
import os
DATA = os.environ.get("AGENT_SEARCH_DATA", "data")
choice = os.environ.get("AGENT_SEARCH_DOWNLOAD_DATASET", "all")
names = {
    "swebench_verified": ["princeton-nlp/SWE-bench_Verified"],
    "swebench_lite": ["princeton-nlp/SWE-bench_Lite"],
    "loc_bench": ["czlll/Loc-Bench_V1"],
    "all": ["princeton-nlp/SWE-bench_Verified", "princeton-nlp/SWE-bench_Lite", "czlll/Loc-Bench_V1"],
}
if choice not in names:
    raise SystemExit(f"unknown dataset {choice!r}; use all|swebench_verified|swebench_lite|loc_bench")
for hf in names[choice]:
    out = os.path.join(DATA, hf.split("/")[-1])
    if os.path.isdir(out) and os.listdir(out):
        print("  exists, skip", out)          # already staged -> don't re-download
        continue
    load_dataset(hf, split="test").save_to_disk(out)
    print("  saved", out)
PY

echo ">> cloning repos into $DATA/repos (git archive reads these offline) ..."
# The code-localization arm (SWE-bench repositories) is retained in the package but is not
# part of this release's document-research workflow; no repository prefetcher ships with it.
echo "!! SWE-bench repository staging is not supported in this release (dataset=$DATASET)." >&2
echo "   Stage a DOCUMENT corpus instead: see corpus_build/README.md." >&2
exit 2

echo ">> (optional) pre-download the model into your HF cache, e.g.:"
echo "     hf download Alibaba-NLP/Tongyi-DeepResearch-30B-A3B   # the agent model"
echo "     hf download nomic-ai/CodeRankEmbed"
echo "     hf download BAAI/bge-base-en-v1.5                      # document-domain dense model"
echo ">> document-domain datasets are not auto-downloaded here; pull the published corpora"
echo "   (wshuai190/browsecomp-plus-structured-full, wshuai190/hotpotqa-structured,"
echo "   wshuai190/musique-structured) and stage them per corpus_build/README.md, or place"
echo "   BEIR-style corpus.jsonl, queries.jsonl, and qrels under $DATA/browsecomp_plus/"
echo "   (multi-hop sets: scripts/stage_multihop.py)."
echo ">> done. Copy $DATA/ to shared storage the GPU node can read."
