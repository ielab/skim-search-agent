#!/usr/bin/env bash
#SBATCH --job-name=shard-cell
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --output=slurm_logs/submitters/%x-%j.out
#SBATCH --error=slurm_logs/submitters/%x-%j.err
# (sbatch'ing THIS file only submits per-shard child jobs, so it needs no GPU)
#
# INSTANCE-SHARDING: split ONE evaluation cell's remaining (not-yet-scored) episodes
# across NUM_SHARDS parallel SLURM jobs, each serving its own vLLM on its own GPU, then
# merge back with scripts/merge_shards.py. Purely ADDITIVE: relies only on the new
# `--only-instances` flag on evaluation/run_eval.py (evaluation/run_eval.py's normal,
# non-sharded call path is untouched) and on distinct per-shard run-dirs so appends
# never collide with the canonical cell or with each other while shards are in flight.
#
# Dual-mode, same pattern as scripts/eval_agent_suite.sh + scripts/run.sh:
#   `bash scripts/shard_cell.sh`  (or a plain `sbatch scripts/shard_cell.sh`, no --array)
#       = SUBMITTER: locates the cell, computes remaining = all_ids - done_ids, writes
#         NUM_SHARDS id-files, and submits itself as a SLURM array (one GPU task/shard).
#   (SLURM sets SLURM_ARRAY_TASK_ID automatically for that array)
#       = WORKER: serves its own vLLM on a distinct port and runs run_eval with
#         --only-instances <its shard file> --runs-dir <its own shard run-dir>.
#
# Env knobs (submitter reads all of these; the array export below carries the
# RESOLVED values through to each worker task):
#   DATASET       required, e.g. browsecomp_plus_structured
#   RUNS_DIR      required: the TIER root the canonical cell lives under, e.g.
#                 runs/_visit_uncapped (NOT the cell dir itself — see CANONICAL_DIR below)
#   CONDITION     required: the retriever/condition name, e.g. agent_research_bm25_autoread
#                 (an agent_* condition — this script's serving setup assumes an LLM policy)
#   NUM_SHARDS    default 10
#   MODEL / DENSE_MODEL / TP / WORKERS / LEVEL / REPO_CACHE / INDEX_ROOT / MAX_STEPS /
#   SEEDS / SEED / TEMPERATURE / LIMIT / CORPUS_LIMIT / GPU_UTIL / EAGER / CUDAGRAPH /
#   VLLM_ARGS / QUANT / DTYPE / KV_CACHE_DTYPE / MAX_MODEL_LEN / LOAD_FORMAT /
#   AGENT_SEARCH_DENSE_DEVICE / PREBUILD  — same meaning + defaults as scripts/run.sh
#   QOS           default 'normal' (eval_agent_suite's QOS=normal path — a fixed policy
#                 here, not that script's split/auto scheduling logic, since N GPU jobs
#                 landing together is exactly the case express's per-user cap punishes)
#   JOB_TIME=24:00:00 · GPU_PARTITION=h24gpu · JOB_CPUS=4 · JOB_MEM=<domain default>
#
# NOTE on LIMIT: --limit governs which instances COUNT as this cell's universe (must
# match whatever LIMIT the canonical cell itself was originally run with, if any) — it
# is used ONLY here, to compute `remaining` correctly. It is deliberately NEVER passed
# to the per-shard run_eval call (see the worker section below): run_eval.py applies
# --only-instances AFTER --limit, so passing --limit there too could truncate the
# dataset before the shard's own ids are ever reached.
set -euo pipefail

# ── dual-mode: `bash scripts/X.sh` runs interactively; `sbatch scripts/X.sh`
# submits a job (the #SBATCH lines above are comments to bash). Override any
# resource on the CLI: `sbatch --time=48:00:00 scripts/X.sh`.
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

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:resource_tracker:UserWarning}"
export VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-0}"   # see run.sh: LOAD-BEARING on H100/SM90

if [ -n "${PYTHON:-}" ]; then
  :
elif [ -x "${ENV_PATH:-$PWD/envs}/bin/python" ]; then
  PYTHON="${ENV_PATH:-$PWD/envs}/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi

: "${DATASET:?set DATASET, e.g. browsecomp_plus_structured}"
: "${RUNS_DIR:?set RUNS_DIR to the TIER root the cell lives under, e.g. runs/_visit_uncapped}"
: "${CONDITION:?set CONDITION, e.g. agent_research_bm25_autoread}"
NUM_SHARDS="${NUM_SHARDS:-10}"
MODEL="${MODEL:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"
# BASE-CONFIG GUARD (2026-07-28): these knobs MUST match every published cell. shard_cell used to
# rely on the submitting shell's env; when absent, code defaults (1200/local/python) silently
# produced incomparable cells. Now pinned here; override only for a deliberate variation.
export MAX_VISIT_TOKENS="${MAX_VISIT_TOKENS:-12000}"
export MAX_SECTION_TOKENS="${MAX_SECTION_TOKENS:-12000}"
export BM25_BACKEND="${BM25_BACKEND:-pyserini}"
export STRUCTURED_BACKEND="${STRUCTURED_BACKEND:-lucene}"
DENSE_MODEL="${DENSE_MODEL:-}"
TP="${TP:-1}"
LEVEL="${LEVEL:-function}"
REPO_CACHE="${REPO_CACHE:-data/repos}"
INDEX_ROOT="${INDEX_ROOT:-indexes}"
LIMIT="${LIMIT:-}"
CORPUS_LIMIT="${CORPUS_LIMIT:-}"
SEEDS="${SEEDS:-42}"
TEMPERATURE="${TEMPERATURE:-0.6}"
# 2026-07-31: default raised 50 -> 100. Every paper cell runs THE BASE config (100 steps); the
# inherited 50 default silently produced ~3.7k truncated-config rows when a submission omitted
# MAX_STEPS (all deleted after repair). Pass 50 explicitly if a
# non-paper pilot ever wants it.
MAX_STEPS="${MAX_STEPS:-100}"
DOMAIN=$("$PYTHON" -c "from evaluation.datasets import dataset_domain; print(dataset_domain('$DATASET'))" 2>/dev/null || echo code)
if [ "$DOMAIN" = "general" ]; then
  WORKERS="${WORKERS:-8}"; JOB_MEM="${JOB_MEM:-256g}"
else
  WORKERS="${WORKERS:-16}"; JOB_MEM="${JOB_MEM:-128g}"
fi
JOB_TIME="${JOB_TIME:-24:00:00}"
QOS="${QOS:-normal}"                     # eval_agent_suite's QOS=normal path (fixed, not split/auto)
GPU_PARTITION="${GPU_PARTITION:-h24gpu}"
JOB_CPUS="${JOB_CPUS:-4}"
# g047: black-hole node 2026-07-31 — vLLM dies at init with cudaErrorDevicesUnavailable, the
# 90-second fail makes the slot look free so SLURM refills it (174 failed tasks in 24h).
# Set EXCLUDE_NODES= (empty) once the node is drained/fixed.
EXCLUDE_NODES="${EXCLUDE_NODES:-g047}"

MODEL_TAG="${MODEL##*/}"
# CANONICAL cell dir — the SAME layout evaluation/config.py:results_dir_for computes
# (kind=agent, since CONDITION is an agent_* condition here): <RUNS_DIR>/agent/<DATASET>/<MODEL_TAG>/<CONDITION>
CANONICAL_DIR="$RUNS_DIR/agent/$DATASET/$MODEL_TAG/$CONDITION"
CANONICAL_ROWS="$CANONICAL_DIR/rows.jsonl"
SHARD_ROOT="$RUNS_DIR/__shards/$CONDITION"

# =============================================================================
# WORKER: SLURM sets SLURM_ARRAY_TASK_ID for each task of the array WE submit below.
# =============================================================================
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  i="$SLURM_ARRAY_TASK_ID"
  SHARD_FILE="$SHARD_ROOT/shard_${i}of${NUM_SHARDS}.txt"
  SHARD_RUNS_DIR="$SHARD_ROOT/shard_${i}of${NUM_SHARDS}"
  if [ ! -s "$SHARD_FILE" ]; then
    echo "ERROR: shard file missing/empty: $SHARD_FILE (was the split step run first?)" >&2
    exit 1
  fi
  PORT="${PORT:-$((8101 + i))}"
  API_BASE="http://127.0.0.1:${PORT}/v1"
  echo ">> shard task ${i}/${NUM_SHARDS}: CONDITION=$CONDITION DATASET=$DATASET PORT=$PORT"
  echo ">>   shard file  : $SHARD_FILE ($(wc -l < "$SHARD_FILE") ids)"
  echo ">>   shard runs  : $SHARD_RUNS_DIR"

  SEED_ARGS=(--temperature "$TEMPERATURE")
  if [ -n "${SEED:-}" ]; then
    SEED_ARGS+=(--seed "$SEED")
  else
    SEED_ARGS+=(--seeds "$(echo "$SEEDS" | tr ' ' ',')")
  fi

  run_eval_shard () {
    # NEVER pass --limit here (see the NOTE at the top of this file) — --only-instances
    # already carries the exact shard partition of `remaining`.
    "$PYTHON" -m evaluation.run_eval \
      --dataset "$DATASET" --retriever "$CONDITION" --level "$LEVEL" \
      --policy llm --backend api --api-base "$API_BASE" \
      --model "$MODEL" ${DENSE_MODEL:+--dense-model "$DENSE_MODEL"} \
      --max-steps "$MAX_STEPS" "${SEED_ARGS[@]}" \
      --workers "$WORKERS" --repo-cache "$REPO_CACHE" --index-root "$INDEX_ROOT" \
      --k 1 3 5 10 --runs-dir "$SHARD_RUNS_DIR" \
      --only-instances "$SHARD_FILE" \
      ${CORPUS_LIMIT:+--corpus-limit "$CORPUS_LIMIT"} "$@"
  }

  # finished settings never rerun: check BEFORE paying the vLLM startup cost (this
  # shard's OWN run-dir, so it only re-reports itself, never the canonical cell).
  if run_eval_shard --check-complete; then
    echo ">> shard ${i} already complete — nothing to run."
    exit 0
  fi
  if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: vllm is not on PATH. Activate the cluster env before running agent_* retrievers." >&2
    exit 127
  fi

  # STEP 0 (mirrors run.sh): materialize this run's PERSISTENT indexes before vLLM.
  # Idempotent/resumable (evaluation/build_indexes.py skips already-built work), and in
  # the intended sharding use case the canonical cell already built whatever persistent
  # index its condition needs (it has SOME scored rows already) — so this is normally a
  # fast no-op. Residual risk: sharding a condition/dataset pair with NO prior scored
  # rows means every shard task races to build the SAME persistent index concurrently.
  # PREBUILD=0 to skip (recommended if you haven't verified build_indexes.py's
  # concurrent-build path); safest of all: run scripts/build_indexes.sh once, un-sharded,
  # before the first shard_cell.sh submission for a new condition.
  if [ "${PREBUILD:-1}" != "0" ]; then
    kinds=$("$PYTHON" -c "from evaluation.build_indexes import prebuildable_for; print(' '.join(prebuildable_for('$CONDITION')))" 2>/dev/null || echo "")
    for kind in $kinds; do
      echo ">> [step 0] building persistent '$kind' index (once; skips if already built) ..."
      ( unset AGENT_SEARCH_DENSE_DEVICE
        export TOKENIZERS_PARALLELISM=true
        "$PYTHON" -m evaluation.build_indexes --dataset "$DATASET" --retriever "$kind" \
          --index-root "$INDEX_ROOT" --repo-cache "$REPO_CACHE" \
          ${DENSE_MODEL:+--model "$DENSE_MODEL"} ) \
      || { echo "ERROR: [step 0] '$kind' index build FAILED — NOT serving vLLM." >&2; exit 1; }
    done
  fi

  SERVE_ARGS=(--tensor-parallel-size "$TP" --port "$PORT"
              --gpu-memory-utilization "${GPU_UTIL:-0.9}")
  [ -n "${QUANT:-}" ]          && SERVE_ARGS+=(--quantization "$QUANT")
  [ -n "${DTYPE:-}" ]          && SERVE_ARGS+=(--dtype "$DTYPE")
  [ -n "${KV_CACHE_DTYPE:-}" ] && SERVE_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  SERVE_ARGS+=(--max-model-len "${MAX_MODEL_LEN:-131072}")   # see run.sh: matches the agent's ctx-budget window
  [ -n "${LOAD_FORMAT:-}" ]    && SERVE_ARGS+=(--load-format "$LOAD_FORMAT")
  case "$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')" in
    *agentworld*) SERVE_ARGS+=(--language-model-only) ;;    # see run.sh: LOAD-BEARING for that family
  esac
  if [ "${EAGER:-0}" = "1" ]; then
    SERVE_ARGS+=(--enforce-eager)
  else
    SERVE_ARGS+=(--compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH:-PIECEWISE}\"}")
  fi
  [ -n "${VLLM_ARGS:-}" ] && SERVE_ARGS+=($VLLM_ARGS)
  [ -n "${AGENT_SEARCH_DENSE_DEVICE:-}" ] && export AGENT_SEARCH_DENSE_DEVICE

  # node-local torch.compile cache, shared across co-scheduled servers (see run.sh for
  # the Lustre "Stale file handle" rationale).
  _cache_base="${VLLM_COMPILE_CACHE_BASE:-${TMPDIR:-/tmp}/$USER/vllm-compile}"
  export VLLM_CACHE_ROOT="$_cache_base/vllm"
  export TORCHINDUCTOR_CACHE_DIR="$_cache_base/inductor"
  export TRITON_CACHE_DIR="$_cache_base/triton"
  mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

  echo ">> serving $MODEL on :$PORT (tp=$TP${EAGER:+, eager}); shard ${i}/${NUM_SHARDS}"
  vllm serve "$MODEL" "${SERVE_ARGS[@]}" &
  VLLM_PID=$!
  trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT
  echo ">> waiting for the server to be ready ..."
  until curl -sf "${API_BASE}/models" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "ERROR: vLLM exited during startup (OOM? bad model path? see log above)." >&2
      exit 1
    fi
    sleep 5
  done
  echo ">> server up."

  # see run.sh: local vLLM has no --enable-auto-tool-choice, so force the native
  # <tool_call> text loop; Tongyi is trained on that protocol.
  export AGENT_DRIVER="${AGENT_DRIVER:-loop}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
  echo ">> running shard ${i}/${NUM_SHARDS} of $CONDITION on $DATASET (AGENT_DRIVER=$AGENT_DRIVER)"
  run_eval_shard

  echo ">> shard ${i}/${NUM_SHARDS} done -> $SHARD_RUNS_DIR/"
  exit 0
fi

# =============================================================================
# SUBMITTER: compute remaining, split into NUM_SHARDS id-files, submit the array.
# =============================================================================
mkdir -p "$SHARD_ROOT"
SPLIT_LOG=$(mktemp)
"$PYTHON" - "$DATASET" "$CANONICAL_ROWS" "$SHARD_ROOT" "$NUM_SHARDS" "${LIMIT:-}" <<'PYEOF' | tee "$SPLIT_LOG"
import sys
dataset, canonical_rows, shard_root, num_shards, limit = sys.argv[1:6]
num_shards = int(num_shards)
limit = int(limit) if limit else None

from evaluation.datasets import load_dataset_by_name
from evaluation.run_eval import _load_rows

instances = load_dataset_by_name(dataset, limit=limit)
all_ids = [inst.instance_id for inst in instances]
_, done = _load_rows(canonical_rows)
remaining = [x for x in all_ids if x not in done]

print(f"TOTAL {len(all_ids)}")
print(f"DONE {len(done)}")
print(f"REMAINING {len(remaining)}")

if remaining:
    shards = [[] for _ in range(num_shards)]
    for idx, inst_id in enumerate(remaining):          # round-robin: balanced to within 1
        shards[idx % num_shards].append(inst_id)
    import os
    for i, ids in enumerate(shards):
        path = os.path.join(shard_root, f"shard_{i}of{num_shards}.txt")
        with open(path, "w") as fh:
            fh.write("\n".join(ids) + ("\n" if ids else ""))
        print(f"SHARD {i} {len(ids)}")
PYEOF

TOTAL=$(awk '/^TOTAL /{print $2}' "$SPLIT_LOG")
DONE=$(awk '/^DONE /{print $2}' "$SPLIT_LOG")
REMAINING=$(awk '/^REMAINING /{print $2}' "$SPLIT_LOG")
rm -f "$SPLIT_LOG"

echo ">> $CANONICAL_ROWS : $DONE/$TOTAL done, $REMAINING remaining"
if [ -z "$REMAINING" ] || [ "$REMAINING" -eq 0 ]; then
  echo ">> already complete -- nothing to shard."
  exit 0
fi

LOGDIR="slurm_logs/agent_runs/$DATASET"; mkdir -p "$LOGDIR"
n=$(( NUM_SHARDS - 1 ))
JOB_NAME="shard-${CONDITION}-${DATASET}-${MODEL_TAG}"
jid=$(sbatch --parsable --array=0-${n} --time="$JOB_TIME" \
  ${EXCLUDE_NODES:+--exclude=$EXCLUDE_NODES} \
  --partition="$GPU_PARTITION" --qos="$QOS" --gres=gpu:1 \
  --cpus-per-task="$JOB_CPUS" --mem="$JOB_MEM" \
  --job-name="$JOB_NAME" \
  --output="$LOGDIR/%x-%A_%a.out" --error="$LOGDIR/%x-%A_%a.err" \
  --export=ALL,MAX_VISIT_TOKENS=$MAX_VISIT_TOKENS,MAX_SECTION_TOKENS=$MAX_SECTION_TOKENS,BM25_BACKEND=$BM25_BACKEND,STRUCTURED_BACKEND=$STRUCTURED_BACKEND,DATASET=$DATASET,RUNS_DIR=$RUNS_DIR,CONDITION=$CONDITION,NUM_SHARDS=$NUM_SHARDS,MODEL=$MODEL,DENSE_MODEL=$DENSE_MODEL,TP=$TP,WORKERS=$WORKERS,LEVEL=$LEVEL,REPO_CACHE=$REPO_CACHE,INDEX_ROOT=$INDEX_ROOT,MAX_STEPS=$MAX_STEPS,SEEDS="$SEEDS",SEED=${SEED:-},TEMPERATURE=$TEMPERATURE,CORPUS_LIMIT=$CORPUS_LIMIT,PREBUILD=${PREBUILD:-1} \
  scripts/shard_cell.sh)

echo ">> submitted array ${jid}_[0-${n}] (qos=$QOS, ${JOB_MEM} mem, ${WORKERS} workers/shard)"
for i in $(seq 0 $n); do
  echo ">>   ${jid}_${i} -> $SHARD_ROOT/shard_${i}of${NUM_SHARDS}.txt (vLLM on :$((8101 + i)))"
done
echo ">> cancel all: scancel $jid ; watch: squeue -u \$USER"
echo ">> when done, merge back: ./envs/bin/python scripts/merge_shards.py --runs-dir $RUNS_DIR --dataset $DATASET --condition $CONDITION --num-shards $NUM_SHARDS"
