# Sourced by every launcher: cd to the repo, activate the env, go offline, make the log dir.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  cd "${SLURM_SUBMIT_DIR:-.}"
else
  cd "$(dirname "${BASH_SOURCE[0]}")/../.."
fi
if command -v module >/dev/null 2>&1; then
  module load miniforge3 2>/dev/null || module load miniconda3 2>/dev/null || true
  module load cuda 2>/dev/null || true
fi
[ -d envs ] && { source activate "$PWD/envs" 2>/dev/null || conda activate "$PWD/envs" 2>/dev/null || true; }
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
[ -n "${JAVA_HOME_OVERRIDE:-}" ] && export JAVA_HOME="$JAVA_HOME_OVERRIDE" PATH="$JAVA_HOME_OVERRIDE/bin:$PATH"
# Pyserini starts a JVM on import and an old Java aborts the process without a message; say so once
if [ -z "${JAVA_HOME:-}" ]; then
  echo ">> JAVA_HOME is not set: anything that imports Pyserini (the test suite, BM25_BACKEND=pyserini) needs JDK 21; pass --export=ALL,JAVA_HOME_OVERRIDE=/path/to/jdk-21" >&2
fi
mkdir -p slurm_logs
echo ">> $(date) host=$(hostname) job=${SLURM_JOB_ID:-interactive} cwd=$PWD python=$(command -v python)"
