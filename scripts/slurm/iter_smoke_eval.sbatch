#!/bin/bash
#SBATCH --job-name=ssa-iter-eval
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60g
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Stage 3 of the ITER smoke pipeline: retriever-only evaluation of the trained checkpoint against
# its base model on the triples, over a subset corpus (the triples' documents + the first N chunks).
#   sbatch --export=ALL[,MODEL=models/iter-smoke-qwen3-0.6b,BASE=Qwen/Qwen3-Embedding-0.6B,SUBSET=3000] scripts/slurm/iter_smoke_eval.sbatch
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/_common.sh"
MODEL="${MODEL:-models/iter-smoke-qwen3-0.6b}"
BASE="${BASE:-Qwen/Qwen3-Embedding-0.6B}"
SUBSET="${SUBSET:-3000}"
TRIPLES="${TRIPLES:-train_data/iter_smoke_i2.jsonl}"
DATASET="${DATASET:-infoseek_train}"
OUT="${OUT:-runs/iter_smoke}"
# each retriever family uses its own query instruction (a trained checkpoint: its serving note);
# export DENSE_QUERY_INSTRUCTION yourself to force one for both models
echo "[$(date)] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
mkdir -p "$OUT"
skimsearchagent-eval-retriever --triples "$TRIPLES" --dataset "$DATASET" --dense-model "$MODEL" --subset "$SUBSET" --out "$OUT/retriever_eval_trained.json"
skimsearchagent-eval-retriever --triples "$TRIPLES" --dataset "$DATASET" --dense-model "$BASE" --subset "$SUBSET" --out "$OUT/retriever_eval_base.json"
echo "[$(date)] done"
