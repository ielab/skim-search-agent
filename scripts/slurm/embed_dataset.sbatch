#!/usr/bin/env bash
#SBATCH --job-name=embed
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH --partition=h24gpu
#SBATCH --qos=normal
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# The dense document-embedding cache of one dataset with one encoder:
#   DATASET=browsecomp_plus_structured EMBED_MODEL=BAAI/bge-small-en-v1.5 sbatch --account=... --export=ALL scripts/slurm/embed_dataset.sbatch
# The cache lands under indexes/dense/<model key>/<dataset>/ (agent_search.evaluation.build_indexes).
# DENSE_SEQ_LENGTH / DENSE_DTYPE / DENSE_POOLING in the environment shape the cache key as in a run.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -d agent_search ] || cd ..
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
DATASET="${DATASET:?set DATASET}"; EMBED_MODEL="${EMBED_MODEL:?set EMBED_MODEL}"
PY=./envs/bin/python; [ -x "$PY" ] || PY="$(command -v python)"
echo ">> dense cache: $EMBED_MODEL over $DATASET (DENSE_SEQ_LENGTH=${DENSE_SEQ_LENGTH:-1024} DENSE_DTYPE=${DENSE_DTYPE:-float32})"
PYTHONPATH=. "$PY" -m agent_search.evaluation.build_indexes --dataset "$DATASET" --retriever dense --index-root indexes --model "$EMBED_MODEL"
echo ">> done"
