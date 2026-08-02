#!/usr/bin/env bash
#SBATCH --job-name=agent-search-floors
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --output=slurm_logs/submitters/%x-%j.out
#SBATCH --error=slurm_logs/submitters/%x-%j.err
# (sbatch'ing THIS file only submits per-unit child jobs, so it needs no GPU)
# Run non-agent sanity floors. These are not the headline comparison.
# Env knobs:
#   DATASET=swebench_verified|swebench_lite|loc_bench|fixture|browsecomp_plus|hotpotqa|2wiki|musique|*_fixture
#   LEVEL=function|file
#   FLOORS="grep bm25_pyserini dense"  # add bql only when queries are already BQL
#   DENSE_MODEL=nomic-ai/CodeRankEmbed
#   REPO_CACHE=data/repos
#   INDEX_ROOT=indexes
#   LIMIT=optional integer
#   CORPUS_LIMIT=shared document-corpus cap
#   RUNS_DIR=runs
set -euo pipefail
# ── dual-mode: `bash scripts/X.sh` runs interactively; `sbatch scripts/X.sh`
# submits a job (the #SBATCH lines above are comments to bash). Override any
# resource on the CLI: `sbatch --time=48:00:00 --gres=gpu:2 scripts/X.sh`.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  cd "${SLURM_SUBMIT_DIR:-.}"                 # $0 is a spooled copy under sbatch
  [ -d agent_search ] || cd ..                   # submitted from scripts/? hop up
else
  cd "$(dirname "$0")/.."
fi
# cluster env (idempotent, both interactive and sbatch; no-op on machines
# without `module`): module load cuda + miniforge3, then activate envs/
if command -v module >/dev/null 2>&1; then
  module load miniforge3 2>/dev/null || module load miniconda3 2>/dev/null || true
  module load cuda 2>/dev/null || true
fi
ENV_PATH="${ENV_PATH:-$PWD/envs}"
if [ -d "$ENV_PATH" ] && [ -z "${CONDA_PREFIX:-}" ]; then
  source activate "$ENV_PATH" 2>/dev/null || conda activate "$ENV_PATH" 2>/dev/null || true
fi
mkdir -p slurm_logs slurm_logs/submitters
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"  # avoid leaked-semaphore warning at exit
# resource_tracker's leaked-semaphore warning is emitted by its OWN daemon process
# (a Python warnings.filter cannot reach it) and is benign — the semaphore is reclaimed
# at exit. Source is usually vLLM/torch multiprocessing shutdown, not our code/results.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:resource_tracker:UserWarning}"

if [ -n "${PYTHON:-}" ]; then
  :
elif [ -x "${ENV_PATH:-$PWD/envs}/bin/python" ]; then
  PYTHON="${ENV_PATH:-$PWD/envs}/bin/python"   # project conda env (cluster)
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi

DATASET="${DATASET:-swebench_verified}"
LEVEL="${LEVEL:-function}"
FLOORS="${FLOORS:-grep bm25_pyserini dense}"
DENSE_MODEL="${DENSE_MODEL:-}"   # empty: run_eval picks per domain (CodeRankEmbed=code, bge=docs)
REPO_CACHE="${REPO_CACHE:-data/repos}"
INDEX_ROOT="${INDEX_ROOT:-indexes}"
LIMIT="${LIMIT:-}"
CORPUS_LIMIT="${CORPUS_LIMIT:-}"
if [ -n "${RUNS_DIR:-}" ]; then
  :
elif [[ "$DATASET" == fixture || "$DATASET" == *_fixture ]]; then
  RUNS_DIR="runs/debug"
else
  RUNS_DIR="runs"
fi

# ── fan-out: `sbatch scripts/eval_floors.sh` (or SUBMIT=1 bash ...) submits ONE
# job per floor instead of running them consecutively — GPU only where needed.
if [ "${SUBMIT:-0}" = "1" ] || [ "${SLURM_JOB_NAME:-}" = "agent-search-floors" ]; then
  # Split by resource need: grep/bm25_* are pure CPU -> ONE gpu:0 array
  # (JOBID_0, JOBID_1, ...); dense needs a GPU -> one separate gpu:1 job.
  CPU_FLOORS=()
  GPU_FLOORS=()
  for retriever in $FLOORS; do
    case "$retriever" in
      dense) GPU_FLOORS+=("$retriever") ;;
      *)     CPU_FLOORS+=("$retriever") ;;
    esac
  done
  COMMON_EXPORT="DATASET=$DATASET,LEVEL=$LEVEL,LIMIT=$LIMIT,CORPUS_LIMIT=$CORPUS_LIMIT,DENSE_MODEL=$DENSE_MODEL,REPO_CACHE=$REPO_CACHE,INDEX_ROOT=$INDEX_ROOT,RUNS_DIR=${RUNS_DIR:-runs}"
  LOGDIR="slurm_logs/retrieve/$DATASET"; mkdir -p "$LOGDIR"   # logs by type (retrieve) then dataset
  if [ ${#CPU_FLOORS[@]} -gt 0 ]; then
    n=$(( ${#CPU_FLOORS[@]} - 1 ))
    jid=$(sbatch --parsable --array=0-${n} --gres=gpu:0 \
      --partition="${CPU_PARTITION:-h24}" \
      --time=04:00:00 --mem=32g --cpus-per-task=8 \
      --job-name="as-floors-${DATASET}" \
      --output="$LOGDIR/%x-%A_%a.out" --error="$LOGDIR/%x-%A_%a.err" \
      --export=ALL,AGENT_SEARCH_CONDS="${CPU_FLOORS[*]}",$COMMON_EXPORT \
      scripts/run.sh)
    for i in $(seq 0 $n); do echo ">> ${jid}_${i} -> ${CPU_FLOORS[$i]} (gpu:0)"; done
    echo ">> cancel CPU floors: scancel $jid"
  fi
  for retriever in "${GPU_FLOORS[@]}"; do
    jid=$(sbatch --parsable --gres=gpu:1 --partition="${GPU_PARTITION:-h24gpu}" \
      --time=04:00:00 --mem=32g --cpus-per-task=4 \
      --job-name="as-${retriever}-${DATASET}" \
      --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
      --export=ALL,RETRIEVER=$retriever,$COMMON_EXPORT \
      scripts/run.sh)
    echo ">> ${jid} -> ${retriever} (gpu:1)"
  done
  echo ">> watch: squeue -u \$USER ; results land in \${RUNS_DIR:-runs}/ (resumable)"
  exit 0
fi

floor_deps_ok () {  # one loud check per floor instead of one error per instance
  case "$1" in
    bm25_pyserini)
      "$PYTHON" -c "import pyserini" 2>/dev/null || {
        echo "!! skipping bm25_pyserini: '$PYTHON' has no pyserini." >&2
        echo "   fix: activate the project env (conda activate ./envs) or" >&2
        echo "        pip install -r requirements.txt   (needs Java 11+ too)" >&2
        return 1; } ;;
    dense)
      "$PYTHON" -c "import sentence_transformers" 2>/dev/null || {
        echo "!! skipping dense: '$PYTHON' has no sentence-transformers." >&2
        echo "   fix: activate the project env or pip install -r requirements.txt" >&2
        return 1; } ;;
  esac
}

echo ">> using PYTHON=$PYTHON"
for retriever in $FLOORS; do
  echo ">> floor: $retriever"
  floor_deps_ok "$retriever" || continue
  EXTRA=()
  if [ -n "$LIMIT" ]; then
    EXTRA+=(--limit "$LIMIT")
  fi
  if [ -n "$CORPUS_LIMIT" ]; then
    EXTRA+=(--corpus-limit "$CORPUS_LIMIT")
  fi
  "$PYTHON" -m evaluation.run_eval \
    --dataset "$DATASET" \
    --retriever "$retriever" \
    ${DENSE_MODEL:+--dense-model $DENSE_MODEL} \
    --repo-cache "$REPO_CACHE" \
    --index-root "$INDEX_ROOT" \
    --level "$LEVEL" \
    --k 1 3 5 10 \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    --runs-dir "$RUNS_DIR"
done
