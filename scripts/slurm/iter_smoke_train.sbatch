#!/bin/bash
#SBATCH --job-name=ssa-iter-train
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80g
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Stage 2 of the ITER smoke pipeline: triples from the stage-1 run, then a short fine-tune.
#   sbatch --export=ALL,RUNS=runs/iter_smoke/<run dir>[,DATASET=infoseek_train,LABELLER=answer,TRAIN=configs/iter/train_smoke_i2.yaml] scripts/slurm/iter_smoke_train.sbatch
#   sbatch --export=ALL,TRIPLES=train_data/check_all_i2.jsonl,TRAIN=configs/iter/train_check_i2.yaml scripts/slurm/iter_smoke_train.sbatch   # prebuilt triples
# The training env is envs-train (FlagEmbedding 1.3.5 with the ITER patch applied; see docs/TRAINING.md).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/_common.sh"
TRAIN="${TRAIN:-configs/iter/train_smoke_i2.yaml}"
TRIPLES="${TRIPLES:-train_data/iter_smoke_i2.jsonl}"
PY="${TRAIN_PYTHON:-$PWD/envs-train/bin/python}"
echo "[$(date)] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} runs=${RUNS:-<prebuilt triples>}"
mkdir -p train_data
if [ -n "${RUNS:-}" ]; then
  skimsearchagent-build-triples --runs $RUNS --dataset "${DATASET:-infoseek_train}" --out "$TRIPLES" --query-style i2 --labeller "${LABELLER:-answer}"
fi
wc -l "$TRIPLES"
[ -s "$TRIPLES" ] || { echo "no triples were built from $RUNS (no read document contained the gold answer); nothing to train on" >&2; exit 1; }
"$PY" -m agent_search.training.retriever check
"$PY" -m agent_search.training.retriever run "$TRAIN"
echo "[$(date)] done"
