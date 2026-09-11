#!/usr/bin/env bash
#SBATCH --job-name=floors
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Retrieval-only floors: no model, no server. Runs `skimsearchagent run` for every experiment
# file in $EXPERIMENTS (space-separated). One GPU for the dense encoder and the reranker; the
# BM25 and BQL floors would run on a CPU node too.
#
#   sbatch --export=ALL,EXPERIMENTS="configs/floors/bcp_s_bm25.yaml configs/floors/bcp_s_reranked.yaml" \
#          scripts/slurm/floors.sbatch
#
# `bash scripts/slurm/floors.sbatch` runs the same thing interactively (not on a login node).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/_common.sh"
EXPERIMENTS="${EXPERIMENTS:?space-separated experiment files}"
echo "[$(date)] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
FAILED=""
for exp in $EXPERIMENTS; do
  echo ">> $exp"
  skimsearchagent run "$exp" ${OVERRIDES:-} || FAILED="$FAILED $exp"
done
echo "[$(date)] done${FAILED:+; FAILED:$FAILED}"
[ -z "$FAILED" ]
