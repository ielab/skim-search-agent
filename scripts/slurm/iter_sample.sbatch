#!/bin/bash
#SBATCH --job-name=ssa-iter-sample
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120g
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# ITER's paper settings on samples: index the sample corpora with a retriever, serve the backbone
# on the node, run the experiment files. Samples come from skimsearchagent-sample-dataset.
#   sbatch --export=ALL[,RETRIEVER=ielabgroup/ITER-Qwen3-Embedding-0.6B,DATASETS="browsecomp_plus_chunks_sample infoseek_eval_sample",
#          EXPERIMENTS="configs/iter/sample_bcp_chunks_iter06b_tongyi.yaml configs/iter/sample_infoseek_eval_iter06b_tongyi.yaml",
#          MODEL=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B,TP=1,MAX_MODEL_LEN=98304,VLLM_PYTHON=/path/to/env/bin/python,OVERRIDES=...] scripts/slurm/iter_sample.sbatch
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/_common.sh"
RETRIEVER="${RETRIEVER:-ielabgroup/ITER-Qwen3-Embedding-0.6B}"
DATASETS="${DATASETS:-browsecomp_plus_chunks_sample infoseek_eval_sample}"
EXPERIMENTS="${EXPERIMENTS:-configs/iter/sample_bcp_chunks_iter06b_tongyi.yaml configs/iter/sample_infoseek_eval_iter06b_tongyi.yaml}"
MODEL="${MODEL:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"
PORT="${PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}"   # jobs can share a node: one port per job; TP="${TP:-1}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-98304}"
echo "[$(date)] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} retriever=$RETRIEVER"
# 1. the sample indexes (encoded on this GPU; a few minutes for tens of thousands of chunks).
#    Built in the precision the experiment files serve (DTYPE, default bfloat16): the caches differ by precision.
export DENSE_DTYPE="${DTYPE:-bfloat16}"
for ds in $DATASETS; do
  skimsearchagent-build-indexes --dataset "$ds" --retriever dense --model "$RETRIEVER" --index-root "${INDEX_ROOT:-indexes}"
done
# 2. the backbone
export VLLM_USE_FLASHINFER_MOE_FP16=0
"${VLLM_PYTHON:-python}" -m vllm.entrypoints.openai.api_server --model "$MODEL" --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" --host 127.0.0.1 --port "$PORT" --trust-remote-code ${VLLM_EXTRA_ARGS:-} \
    > "slurm_logs/vllm-${SLURM_JOB_ID:-local}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "vLLM died; see slurm_logs/vllm-*.log" >&2; exit 1; }
  sleep 10
done
# 3. the runs (the files name the retriever; the backbone and its endpoint come from this job)
FAILED=""
for exp in $EXPERIMENTS; do
  # shellcheck disable=SC2086
  skimsearchagent run "$exp" model.name="$MODEL" model.backend=api model.api_base="http://127.0.0.1:$PORT/v1" \
      retrieval.dense_model="$RETRIEVER" ${OVERRIDES:-} || FAILED="$FAILED $exp"
done
echo "[$(date)] done${FAILED:+; FAILED:$FAILED}"
[ -z "$FAILED" ]
