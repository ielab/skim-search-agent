#!/usr/bin/env bash
#SBATCH --job-name=dense-bcfull
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH --partition=h24gpu
#SBATCH --qos=normal
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Build the dense doc-embedding cache (BAAI/bge-base-en-v1.5) for the FULL BrowseComp-Plus
# corpus (browsecomp_plus_structured_full, 100,195 docs) — one GPU pass, resumable
# (evaluation/build_indexes.py skips an already-cached corpus). ADDITIVE sibling of the
# pooled dataset's cache: writes indexes/dense/BAAI__bge-base-en-v1.5-sl1024/
# browsecomp_plus_structured_full/, never touching the pooled cache. Header mirrors
# scripts/oneshot_rag.sbatch (1 GPU h24gpu, 24h). `bash` runs interactively; `sbatch` submits.
set -euo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then
  cd "${SLURM_SUBMIT_DIR:-.}"                 # $0 is a spooled copy under sbatch
  [ -d agent_search ] || cd ..                   # submitted from scripts/? hop up
else
  cd "$(dirname "$0")/.."
fi
# cluster env (idempotent, both interactive and sbatch; no-op without `module`)
if command -v module >/dev/null 2>&1; then
  module load miniforge3 2>/dev/null || module load miniconda3 2>/dev/null || true
  module load cuda 2>/dev/null || true
fi
ENV_PATH="${ENV_PATH:-$PWD/envs}"
if [ -d "$ENV_PATH" ] && [ -z "${CONDA_PREFIX:-}" ]; then
  source activate "$ENV_PATH" 2>/dev/null || conda activate "$ENV_PATH" 2>/dev/null || true
fi
mkdir -p slurm_logs logs

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=true

DATASET="${DATASET:-browsecomp_plus_structured_full}"
MODEL="${DENSE_MODEL:-BAAI/bge-base-en-v1.5}"
LOG=logs/fullcorpus_build.log
echo "=== $(date '+%F %T') START dense $DATASET ($MODEL) job=${SLURM_JOB_ID:-interactive} ===" >> "$LOG"
PYTHONPATH=. python -m evaluation.build_indexes \
  --dataset "$DATASET" --retriever dense --index-root indexes --model "$MODEL" \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "=== $(date '+%F %T') END dense $DATASET rc=$rc ===" >> "$LOG"
exit "$rc"
