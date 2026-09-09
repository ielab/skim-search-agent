#!/bin/bash
#SBATCH --job-name=ssa-iter-traj
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160g
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Stage 1 of the ITER smoke pipeline: generate trajectories on a few InfoSeek training questions
# with the dedup search strategy over the 11.2M-chunk wiki corpus (served from disk through the
# prebuilt HNSW index; the query encoder runs on CPU). An API backbone needs no GPU.
#   sbatch --export=ALL[,EXPERIMENT=configs/iter/smoke_infoseek_train_gpt4omini.yaml,OVERRIDES=...] scripts/slurm/iter_smoke_traj.sbatch
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/_common.sh"
EXPERIMENT="${EXPERIMENT:-configs/iter/smoke_infoseek_train_gpt4omini.yaml}"
echo "[$(date)] node=$(hostname) experiment=$EXPERIMENT"
skimsearchagent validate "$EXPERIMENT"
# shellcheck disable=SC2086
skimsearchagent run "$EXPERIMENT" ${OVERRIDES:-}
echo "[$(date)] done"
