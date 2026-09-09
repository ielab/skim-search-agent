#!/usr/bin/env bash
#SBATCH --job-name=ssa-train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Train a dense retriever from trajectory triples (ITER recipe) on a compute node.
# --export=ALL,TRAIN=train.yaml[,TRAIN_ENV=/path/to/training/env]
# The training environment is separate from the eval env (FlagEmbedding 1.3.5 pins an older
# transformers); create it once on a node with internet:
#   python -m venv envs-train && envs-train/bin/pip install -e ".[train]" && \
#   envs-train/bin/skimsearchagent-train-retriever patch
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/_common.sh"
TRAIN="${TRAIN:?set TRAIN=<training yaml> via --export}"
if [ -n "${TRAIN_ENV:-}" ]; then
  source "$TRAIN_ENV/bin/activate" 2>/dev/null || conda activate "$TRAIN_ENV" 2>/dev/null || true
fi
export MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 10000))}"
skimsearchagent-train-retriever check
skimsearchagent-train-retriever run "$TRAIN"
echo ">> training finished: $(python - <<PY
import yaml; print(yaml.safe_load(open("$TRAIN"))["output_dir"])
PY
)"
