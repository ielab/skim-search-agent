#!/usr/bin/env bash
# Create a LOCAL (prefix) conda env on the cluster, then pip-install the deps.
# Does not touch the shared conda. Usage: bash scripts/setup_env.sh [PREFIX]
set -euo pipefail

PREFIX="${1:-./envs}"   # match ENV_PATH=$PWD/envs used by run.sh / eval_agent_suite.sh / e2e_cluster.sh

echo ">> conda create --prefix ${PREFIX} (python + JDK + ripgrep)"
conda create --prefix "${PREFIX}" -c conda-forge python=3.10 openjdk=21 ripgrep -y

echo
echo ">> next (run these yourself):"
echo "     conda activate ${PREFIX}"
echo "     pip install -r requirements.txt"
echo "     #  GPU dense retrieval: pip install torch --index-url https://download.pytorch.org/whl/cu121"
echo
echo ">> then VERIFY:"
echo "     python -m pytest -q                       # pure-Python core (no Java)"
echo "     skimsearchagent dataset=doc_fixture strategy=sieve_bm25    # eval pipeline, dep-free"
echo "     skimsearchagent dataset=doc_fixture strategy=bm25_lucene   # the real Pyserini indexing path"
