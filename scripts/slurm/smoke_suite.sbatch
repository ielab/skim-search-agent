#!/usr/bin/env bash
#SBATCH --job-name=ssa-smoke
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# The full test suite plus every strategy on the fixtures, on a compute node (no GPU needed).
# Java 21+ is needed for the Lucene tests: pass JAVA_HOME_OVERRIDE=/path/to/jdk via --export.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/_common.sh"
python -m pytest -q -p no:cacheprovider --junitxml="slurm_logs/junit-${SLURM_JOB_ID:-local}.xml"
OUT="runs/smoke-${SLURM_JOB_ID:-local}"
for st in search_visit search_fetch autoread dci bounded_dci sieve_bm25 indri bm25 codefix codefix_grep; do
  ds=doc_fixture; case "$st" in codefix*) ds=code_fixture;; esac
  skimsearchagent dataset=$ds strategy=$st runs_dir="$OUT" index_root="$OUT/idx"
done
skimsearchagent-build-indexes --dataset doc_fixture --retriever dense --index-root "$OUT/idx"
for st in sieve sieve_dense search_visit_dense search_fetch_dense search_visit_hybrid rag_bm25 rag_dense dedup_dense; do
  skimsearchagent dataset=doc_fixture strategy=$st runs_dir="$OUT" index_root="$OUT/idx"
done
echo ">> smoke suite OK -> $OUT"
