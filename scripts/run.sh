#!/usr/bin/env bash
#SBATCH --job-name=agent-search-run
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH --partition=h24gpu
#SBATCH --qos=express
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Run one agent-search condition. `bash scripts/run.sh` runs in this shell;
# `sbatch scripts/run.sh` submits it as a SLURM job. For agent_* retrievers it
# serves the model on this node, waits for it, runs the eval, and shuts the server
# down on exit. For baselines it just runs the eval.
set -euo pipefail
# ── dual-mode: `bash scripts/X.sh` runs interactively; `sbatch scripts/X.sh`
# submits a job (the #SBATCH lines above are comments to bash). Override any
# resource on the CLI: `sbatch --time=48:00:00 --gres=gpu:2 scripts/X.sh`.
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
mkdir -p slurm_logs

# GPU node is offline: never silently re-download. If something isn't staged in
# data/ or the HF cache, fail loudly instead of hitting the network. (Override with
# HF_HUB_OFFLINE=0 on a node with internet.)
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"  # avoid leaked-semaphore warning at exit
# resource_tracker's leaked-semaphore warning is emitted by its OWN daemon process
# (a Python warnings.filter cannot reach it) and is benign — the semaphore is reclaimed
# at exit. Source is usually vLLM/torch multiprocessing shutdown, not our code/results.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:resource_tracker:UserWarning}"
# MoE backend: FORCE Triton. vLLM 0.22 auto-selects FlashInfer CUTLASS MoE,
# whose JIT fused_moe_90.so build FAILS on H100/SM90 (crashes profile_run).
# This env var (deprecated, but works in 0.22.1; removed in 0.23 -> then use
# VLLM_ARGS="--moe-backend triton") forces native Triton, which the official
# Tongyi repo also uses. LOAD-BEARING: do not remove.
export VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-0}"

if [ -n "${PYTHON:-}" ]; then
  :
elif [ -x "${ENV_PATH:-$PWD/envs}/bin/python" ]; then
  PYTHON="${ENV_PATH:-$PWD/envs}/bin/python"   # project conda env (cluster)
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi

# Job-array support: the suite submits its conditions as ONE array job
# (ids JOBID_0, JOBID_1, ...). Task N runs the Nth condition in AGENT_SEARCH_CONDS,
# on its own port so co-scheduled tasks never collide.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${AGENT_SEARCH_CONDS:-}" ]; then
  read -r -a _agent_search_conds <<< "$AGENT_SEARCH_CONDS"
  RETRIEVER="${_agent_search_conds[$SLURM_ARRAY_TASK_ID]}"
  PORT="${PORT:-$((8101 + SLURM_ARRAY_TASK_ID))}"
  export RETRIEVER PORT
  echo ">> array task ${SLURM_ARRAY_TASK_ID}: RETRIEVER=$RETRIEVER PORT=$PORT"
fi

# ===================== EDIT THESE (the knobs that matter) =====================
DATASET="${DATASET:-swebench_verified}"   # swebench_* | loc_bench | browsecomp_plus | hotpotqa | 2wiki | musique | *_fixture
# THE method (search -> fetch): agent_codefix (code) | agent_research (deep-research).
# Baselines: agent_codefix_grep (code: regex grep + read) | agent_research_bm25
# (retrieve-then-visit) | agent_research_dci (docs: bash+read, no retriever).
# floors: grep | bm25_pyserini | dense
RETRIEVER="${RETRIEVER:-agent_codefix}"
MODEL="${MODEL:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"  # agent LLM (the real agentic model)
# if it OOMs on your GPU, fall back: MODEL=Qwen/Qwen2.5-Coder-7B-Instruct bash scripts/run.sh
DENSE_MODEL="${DENSE_MODEL:-}"   # empty: run_eval picks per domain (CodeRankEmbed=code, bge=docs)  # embedder for dense / agent_dense
TP="${TP:-1}"                             # GPUs for vLLM — MUST equal #GPUs on this node
WORKERS="${WORKERS:-3}"                  # concurrent instances (vLLM continuous-batches them)
LEVEL="${LEVEL:-function}"                # function | file
LIMIT="${LIMIT:-}"                        # e.g. 5 for a quick smoke; empty = all instances
CORPUS_LIMIT="${CORPUS_LIMIT:-}"          # shared document-corpus cap
INDEX_ROOT="${INDEX_ROOT:-indexes}"       # persistent dense/bm25 indexes (built once, reused)
# =============================================================================

PORT="${PORT:-8000}"
REPO_CACHE="${REPO_CACHE:-data/repos}"
# agent runs are stochastic -> pin a SINGLE fixed seed (42) so a run is reproducible. SEED
# overrides the value; SEEDS="0 1 2" (space-separated) opts into a seed-to-seed variance band
# (each = its own seed=<N> run dir). TEMPERATURE sets sampling temp. Floors ignore these.
SEEDS="${SEEDS:-42}"
TEMPERATURE="${TEMPERATURE:-0.6}"
if [ -n "${RUNS_DIR:-}" ]; then
  :
elif [[ "$DATASET" == fixture || "$DATASET" == *_fixture ]]; then
  RUNS_DIR="runs/debug"
else
  RUNS_DIR="runs"
fi

# agent seed/temperature args, shared by --check-complete and the real run so the
# completeness check inspects the SAME seed=<N> run dirs. SEED (single) wins over SEEDS.
SEED_ARGS=(--temperature "$TEMPERATURE")
if [ -n "${SEED:-}" ]; then
  SEED_ARGS+=(--seed "$SEED")
else
  SEED_ARGS+=(--seeds "$(echo "$SEEDS" | tr ' ' ',')")
fi

run_eval () {
  EXTRA=("--runs-dir" "$RUNS_DIR")
  if [ -n "$LIMIT" ]; then
    EXTRA+=(--limit "$LIMIT")
  fi
  if [ -n "$CORPUS_LIMIT" ]; then
    EXTRA+=(--corpus-limit "$CORPUS_LIMIT")
  fi
  "$PYTHON" -m agent_search.evaluation.run_eval \
    --dataset "$DATASET" --retriever "$RETRIEVER" --level "$LEVEL" \
    --repo-cache "$REPO_CACHE" --workers "$WORKERS" --index-root "$INDEX_ROOT" \
    --k 1 3 5 10 "${EXTRA[@]}" "$@"
}

# STEP 0: materialize this run's PERSISTENT indexes BEFORE launching vLLM, so the
# GPU-heavy embedding owns the whole GPU (no server contention) and the agent only
# LOADS them. Uniform for code AND documents — only the corpora COUNT differs (1 shared
# doc corpus; N per-query repos, honoring LIMIT). Which indexes are needed is derived
# from the condition's toolset (dense / bm25_pyserini); grep/bm25_local/search_bql are
# in-memory/index-free and need nothing. Idempotent: an existing index is skipped, so
# pre-running scripts/build_indexes.sh (array) makes this a no-op. PREBUILD=0 disables.
prebuild_corpus_indexes () {
  local kinds
  kinds=$("$PYTHON" -c "from agent_search.evaluation.build_indexes import prebuildable_for; print(' '.join(prebuildable_for('$RETRIEVER')))" 2>/dev/null || echo "")
  [ -z "$kinds" ] && return 0                 # nothing persistent to build (index-free toolset)
  [ "${PREBUILD:-1}" = "0" ] && { echo ">> [step 0] prebuild skipped (PREBUILD=0)"; return 0; }
  for kind in $kinds; do
    echo ">> [step 0] building persistent '$kind' index over the run's corpora (once) ..."
    if ! ( unset AGENT_SEARCH_DENSE_DEVICE   # one-time embed belongs on the GPU, not CPU
           export TOKENIZERS_PARALLELISM=true   # build is single-process: parallel tokenize is safe + fast
           "$PYTHON" -m agent_search.evaluation.build_indexes --dataset "$DATASET" --retriever "$kind" \
             --index-root "$INDEX_ROOT" --repo-cache "$REPO_CACHE" \
             ${DENSE_MODEL:+--model "$DENSE_MODEL"} ${LIMIT:+--limit "$LIMIT"} ); then
      echo "ERROR: [step 0] '$kind' index build FAILED — NOT serving vLLM. Fix the error" >&2
      echo "       above and re-run (ensure agent_search/ on this node is the latest sync)." >&2
      exit 1
    fi
  done
  echo ">> [step 0] done — the agent will load the persisted indexes (no eval-time build)."
}

if [ "${PREFLIGHT:-1}" = "0" ]; then
  echo ">> preflight skipped (PREFLIGHT=0)"
elif [[ "$DATASET" == swebench_* ]]; then
  echo ">> (no preflight for code datasets in this release)"
else
  echo ">> preflight skipped for $DATASET (not a SWE-bench repo-cache dataset)"
fi

if [[ "$RETRIEVER" == agent_* || "$RETRIEVER" = "agent" ]]; then
  # finished settings never rerun: check BEFORE paying the vLLM startup cost
  if run_eval --check-complete --policy llm --backend api \
       --model "$MODEL" ${DENSE_MODEL:+--dense-model $DENSE_MODEL} \
       --max-steps "${MAX_STEPS:-50}" "${SEED_ARGS[@]}"; then
    echo ">> already complete — nothing to run (delete the run dir to redo)."
    exit 0
  fi
  if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: vllm is not on PATH. Activate the cluster env or install vllm before running agent_* retrievers." >&2
    exit 127
  fi
  prebuild_corpus_indexes      # STEP 0: all corpus indexing finishes BEFORE vLLM/agent

  # GPU_UTIL=0.9 default (96GB H100: 86.4GB pool = ~57GB weights + ~27GB KV;
  # ~9GB left over). agent_dense co-locates the 137M embedder in that headroom —
  # if it OOMs, either GPU_UTIL=0.85 (more headroom) or AGENT_SEARCH_DENSE_DEVICE=cpu
  # (embeddings cached per corpus; one-time cost). User VLLM_ARGS wins (last).
  SERVE_ARGS=(--tensor-parallel-size "$TP" --port "$PORT"
              --gpu-memory-utilization "${GPU_UTIL:-0.9}")
  # quantized / serving knobs: each unset => flag omitted => vLLM default (a run with
  # none set is byte-identical to before), EXCEPT --max-model-len (see below). Serve a
  # quantized model with one env var:
  #   QUANT=awq|gptq|fp8|bitsandbytes  DTYPE=bfloat16  KV_CACHE_DTYPE=fp8
  #   MAX_MODEL_LEN=65536  LOAD_FORMAT=bitsandbytes   (anything else: VLLM_ARGS).
  [ -n "${QUANT:-}" ]          && SERVE_ARGS+=(--quantization "$QUANT")
  [ -n "${DTYPE:-}" ]          && SERVE_ARGS+=(--dtype "$DTYPE")
  [ -n "${KV_CACHE_DTYPE:-}" ] && SERVE_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  # --max-model-len ALWAYS defaults to 131072 (128k) for agent runs when MAX_MODEL_LEN is
  # unset — the agent's own context-budget window (AgentPolicy.ctx_chars, ~115k tokens) is
  # sized to fill that window, so the SERVED context must actually be 128k too, not whatever
  # a given model's config.json happens to default to (e.g. Tongyi-DeepResearch-30B-A3B's
  # max_position_embeddings is 131072, so this is a no-op there, but it also guards any
  # model whose own default is smaller). Lower it explicitly (e.g. MAX_MODEL_LEN=65536) for
  # more KV-cache batching headroom if your episodes fit; raise it if long episodes get
  # rejected for length.
  SERVE_ARGS+=(--max-model-len "${MAX_MODEL_LEN:-131072}")
  [ -n "${LOAD_FORMAT:-}" ]    && SERVE_ARGS+=(--load-format "$LOAD_FORMAT")
  # Qwen-AgentWorld ships VISUAL component definitions in its config but only LM weights
  # in the checkpoint; without --language-model-only vLLM tries to init the vision tower
  # and CRASHES on load. Auto-add it for that model family. LOAD-BEARING. (It's a language
  # world model whose agentic training transfers to tool-use, so we drive it as an agent.)
  # (AgentWorld also defaults to a 262144/256K window if --max-model-len were ever omitted;
  # the unconditional 131072 default above already caps that, so no separate case is needed.)
  case "$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')" in
    *agentworld*)
      SERVE_ARGS+=(--language-model-only)
      ;;
  esac
  # Compile stays ON, but capture defaults to PIECEWISE: vLLM >=0.22 defaults to
  # FULL_AND_PIECEWISE (51 sizes + combo-kernel benchmarking) = 10-15 min of
  # silence after "Model loading took ..."; the old vLLM you first tested with
  # captured in ~1-3 min. PIECEWISE restores that startup at ~no decode cost.
  # CUDAGRAPH=FULL_AND_PIECEWISE brings the heavy mode back; EAGER=1 skips all.
  if [ "${EAGER:-0}" = "1" ]; then
    SERVE_ARGS+=(--enforce-eager)
  else
    SERVE_ARGS+=(--compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH:-PIECEWISE}\"}")
  fi
  [ -n "${VLLM_ARGS:-}" ] && SERVE_ARGS+=($VLLM_ARGS)     # e.g. VLLM_ARGS="--max-model-len 65536"
  # agent_dense: with the 0.85 cap the embedder fits on the GPU (loads after the
  # server reserves its pool). Still OOM on a smaller card? AGENT_SEARCH_DENSE_DEVICE=cpu
  # (embeddings are cached per corpus, so CPU encoding is a one-time cost).
  [ -n "${AGENT_SEARCH_DENSE_DEVICE:-}" ] && export AGENT_SEARCH_DENSE_DEVICE
  # torch.compile / inductor cache on NODE-LOCAL disk, SHARED across this node's servers.
  # WHY: inductor caches each compiled kernel by writing a temp file then atomically REPLACING
  # the target (keyed by a graph+config hash, so every vLLM compiling THIS model targets the
  # same path). On Lustre/NFS, a process holding the old handle when another replaces it gets
  # "[Errno 116] Stale file handle" and the compile crashes — which is what happens when array
  # tasks cold-compile the same graph at once. Local filesystems keep an open inode valid across
  # replace, so the shared cache is race-safe there. Keeping ONE shared cache (NOT per-task) means
  # the graph compiles ONCE per node and is reused by every co-scheduled server and future run —
  # no recompile-each-time, no duplicated artifacts. Override the base with VLLM_COMPILE_CACHE_BASE=
  # (e.g. a persistent local path to survive across jobs); EAGER=1 skips compilation entirely.
  _cache_base="${VLLM_COMPILE_CACHE_BASE:-${TMPDIR:-/tmp}/$USER/vllm-compile}"
  export VLLM_CACHE_ROOT="$_cache_base/vllm"
  export TORCHINDUCTOR_CACHE_DIR="$_cache_base/inductor"
  export TRITON_CACHE_DIR="$_cache_base/triton"          # triton JIT also races on ~/.triton (Lustre) otherwise
  mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  echo ">> serving $MODEL on :$PORT (tp=$TP${EAGER:+, eager}); compile cache -> $_cache_base (node-local, shared)"
  vllm serve "$MODEL" "${SERVE_ARGS[@]}" &
  VLLM_PID=$!
  trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
  echo ">> waiting for the server to be ready ..."
  until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "ERROR: vLLM exited during startup (OOM? bad model path? see log above)." >&2
      echo "       Fallbacks: EAGER=1, VLLM_ARGS='--gpu-memory-utilization 0.8'," >&2
      echo "       or MODEL=Qwen/Qwen2.5-Coder-7B-Instruct." >&2
      exit 1
    fi
    sleep 5
  done
  # A LOCALLY-served vLLM (this branch) is started WITHOUT --enable-auto-tool-choice, so the
  # SDK driver's tool_choice="auto" 400s ('"auto" tool choice requires --enable-auto-tool-choice').
  # retriever.py defaults --backend api to the SDK driver, so force the native text LOOP here
  # unless the caller overrode it. Tongyi is trained on the <tool_call> text protocol; loop is the
  # validated vLLM path. OPENAI_API_KEY must be non-empty for the OpenAI client (pointed at vLLM).
  export AGENT_DRIVER="${AGENT_DRIVER:-loop}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
  echo ">> running $RETRIEVER (AGENT_DRIVER=$AGENT_DRIVER)"
  run_eval --policy llm --backend api --api-base "http://127.0.0.1:$PORT/v1" \
    --model "$MODEL" ${DENSE_MODEL:+--dense-model $DENSE_MODEL} --max-steps "${MAX_STEPS:-50}" \
    "${SEED_ARGS[@]}"
elif [ "$RETRIEVER" = "dense" ]; then
  echo ">> running the dense baseline ($DENSE_MODEL)"
  run_eval ${DENSE_MODEL:+--dense-model $DENSE_MODEL}
else
  echo ">> running the $RETRIEVER baseline"
  run_eval
fi

echo ">> done. results in $RUNS_DIR/<config>/ (results.json + rows.jsonl)."
echo "   tip: each agent row now carries queries/hits_per_step/stopped for debugging."
