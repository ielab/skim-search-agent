#!/usr/bin/env bash
#SBATCH --job-name=agent-search-suite
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --output=slurm_logs/submitters/%x-%j.out
#SBATCH --error=slurm_logs/submitters/%x-%j.err
# (sbatch'ing THIS file only submits per-unit child jobs, so it needs no GPU)
# Run the method's conditions (default: agent_codefix + agent_research + agent_research_bm25),
# or an explicit AGENTS list for any subset (e.g. a method-vs-baseline pair only). In
# interactive mode it starts one vLLM server and runs pending conditions sequentially; in
# submit mode it fans pending conditions out as a SLURM array, with one server per task.
#
# Env knobs:
#   DATASET=swebench_verified|swebench_lite|loc_bench|fixture|browsecomp_plus|hotpotqa|2wiki|musique|*_fixture
#   AGENTS="agent_codefix agent_research agent_research_bm25"   # default (both arms + doc baseline)
#          (code method vs its grep baseline: "agent_codefix agent_codefix_grep")
#          (doc method vs BOTH baselines: "agent_research agent_research_bm25 agent_research_dci")
#   MODEL=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B   (fallback: Qwen/Qwen2.5-Coder-7B-Instruct)
#   DENSE_MODEL=nomic-ai/CodeRankEmbed
#   TP=1                 # GPUs for vLLM (must equal #GPUs on this node)
#   PORT=8000
#   WORKERS=16 · LEVEL=function|file · REPO_CACHE=data/repos · INDEX_ROOT=indexes
#   LIMIT=<int>          # optional, e.g. 10 for a quick smoke
#   CORPUS_LIMIT=<int>   # shared document-corpus cap
#   RUNS_DIR=runs
set -euo pipefail

# --- QOS budget library (pure functions, no side effects) ------------------
# Used live by the `auto` QOS policy further down. Also sourced in isolation by
# tests/test_qos_auto.sh via `QOS_LIB_ONLY=1 source scripts/eval_agent_suite.sh`,
# which returns (see guard below) before any of the script's real work (cd,
# module load, mkdir, python calls, sbatch) runs. Kept dependency-free (only
# $USER + external squeue/awk) so it's cleanly unit-testable.

# parse_time_to_minutes SLURM_TIME -> whole minutes (seconds truncated). Accepts
# both squeue TimeLimit / --time forms: HH:MM:SS and D-HH:MM:SS (the JOB_TIME
# default below, "24:00:00", is the HH:MM:SS form).
parse_time_to_minutes () {
  local t="$1" days=0 rest="$1" h m s
  if [[ "$t" == *-* ]]; then
    days="${t%%-*}"
    rest="${t#*-}"
  fi
  IFS=: read -r h m s <<< "$rest"
  h="${h:-0}"; m="${m:-0}"
  echo $(( 10#${days:-0} * 1440 + 10#$h * 60 + 10#$m ))
}

# express_usage_cpumin -> echoes $USER's current express-QOS usage in cpu-minutes,
# summed over RUNNING+PENDING jobs across ALL projects (the cap is per-user, not
# per-project). Returns 1 (prints nothing) if squeue is unavailable, so callers
# fall back cleanly on a non-cluster machine. The awk formula (sum of
# cpus-per-task x TimeLimit-minutes over the user's express jobs) is lifted
# straight from scripts/qos_budget.sh — credit there; keep the two in sync if
# the formula ever changes.
express_usage_cpumin () {
  command -v squeue >/dev/null 2>&1 || return 1
  local out
  out=$(squeue -u "$USER" -r -h -O "QOS,cpus-per-task,TimeLimit" 2>/dev/null) || return 1
  awk '
    $1=="express" {
      split($3, t, "[-:]")
      if (length(t)==4)      mins = t[1]*1440 + t[2]*60 + t[3]   # D-HH:MM:SS
      else if (length(t)==3) mins = t[1]*60 + t[2]                # HH:MM:SS
      else                   mins = t[1]                          # MM:SS fallback
      sum += $2 * mins
    } END { print sum+0 }' <<< "$out"
}

# footprint_cpumin N_TASKS JOB_CPUS JOB_TIME -> this submission's own cpu-min
# footprint (what it would ADD to express usage if scheduled there).
footprint_cpumin () {
  local n="$1" cpus="$2" mins
  mins=$(parse_time_to_minutes "$3")
  echo $(( n * cpus * mins ))
}

# decide_qos_budget USAGE FOOTPRINT CAP -> echoes "express" if usage+footprint
# fits under CAP, else "fallback".
decide_qos_budget () {
  if [ $(( $1 + $2 )) -gt "$3" ]; then echo "fallback"; else echo "express"; fi
}

if [ "${QOS_LIB_ONLY:-0}" = "1" ]; then
  return 0
fi

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
mkdir -p slurm_logs slurm_logs/submitters

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"      # GPU node is offline; fail loud, don't re-download
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

DATASET="${DATASET:-swebench_verified}"
# THE method, both arms (search -> fetch): agent_codefix (code, <fix>) + agent_research
# (deep-research, <answer>); default baseline agent_research_bm25 (retrieve-then-visit).
# Other in-loop access baselines (add to AGENTS explicitly, not in the default):
# agent_codefix_grep (code: regex grep + read) and agent_research_dci (docs: bash+read over
# the raw corpus filesystem, no retriever — chen2026dci / RISE's brute-force reference).
AGENTS="${AGENTS:-agent_codefix agent_research agent_research_bm25}"
MODEL="${MODEL:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"
DENSE_MODEL="${DENSE_MODEL:-}"   # empty: run_eval picks per domain (CodeRankEmbed=code, bge=docs)
TP="${TP:-1}"
PORT="${PORT:-8000}"
API_BASE="${API_BASE:-http://127.0.0.1:${PORT}/v1}"
# Auto-size resources by domain: the shared-corpus deep-research datasets (browsecomp /
# hotpotqa / 2wiki / musique) hold a large in-memory corpus, so they default to
# more RAM and fewer concurrent episodes than the small per-query code repos. Any explicit
# WORKERS=/JOB_MEM= still wins. (Falls back to the code profile if the lookup fails.)
DOMAIN=$("$PYTHON" -c "from evaluation.datasets import dataset_domain; print(dataset_domain('$DATASET'))" 2>/dev/null || echo code)
if [ "$DOMAIN" = "general" ]; then
  WORKERS="${WORKERS:-8}"; JOB_MEM="${JOB_MEM:-256g}"
else
  WORKERS="${WORKERS:-16}"; JOB_MEM="${JOB_MEM:-128g}"
fi
# DEFAULT submit walltime = the h24 partition's 24h MAXIMUM, so jobs are NOT killed mid-run
# (earlier per-dataset estimates were too low and got cancelled). Runs resume per-instance, so if
# a big corpus still needs more than one 24h job, just resubmit and it continues where it stopped.
# Override with JOB_TIME=, or for a single longer job use a longer partition:
#   GPU_PARTITION=<long-partition> JOB_TIME=48:00:00 ...
# NOTE: a 2h partition (h2gpu) caps jobs at 2h regardless of JOB_TIME — keep GPU_PARTITION=h24gpu
# for the long agent runs.
JOB_TIME="${JOB_TIME:-24:00:00}"
LEVEL="${LEVEL:-function}"
REPO_CACHE="${REPO_CACHE:-data/repos}"
INDEX_ROOT="${INDEX_ROOT:-indexes}"
LIMIT="${LIMIT:-}"
CORPUS_LIMIT="${CORPUS_LIMIT:-}"
# agent runs are stochastic -> pin a SINGLE fixed seed (42) for a reproducible run. SEED
# overrides the value; SEEDS="0 1 2" (space-separated) opts into a seed-to-seed variance band
# (each = its own seed=<N> run dir). TEMPERATURE sets the sampling temperature.
SEEDS="${SEEDS:-42}"
TEMPERATURE="${TEMPERATURE:-0.6}"
if [ -n "${RUNS_DIR:-}" ]; then
  :
elif [[ "$DATASET" == fixture || "$DATASET" == *_fixture ]]; then
  RUNS_DIR="runs/debug"
else
  RUNS_DIR="runs"
fi

run_eval_args () {  # $1 = retriever -> fills RE_ARGS (must match the run call EXACTLY)
  RE_ARGS=(--dataset "$DATASET" --retriever "$1"
           --policy llm --backend api --api-base "$API_BASE"
           --model "$MODEL" ${DENSE_MODEL:+--dense-model $DENSE_MODEL}
           --max-steps "${MAX_STEPS:-50}" --temperature "$TEMPERATURE"
           --workers "$WORKERS" --repo-cache "$REPO_CACHE" --index-root "$INDEX_ROOT"
           --level "$LEVEL" --k 1 3 5 10 --runs-dir "$RUNS_DIR")
  # SEED forces one value; else SEEDS (default '42' = one fixed seed; e.g. '0 1 2' for a band)
  if [ -n "${SEED:-}" ]; then
    RE_ARGS+=(--seed "$SEED")
  else
    RE_ARGS+=(--seeds "$(echo "$SEEDS" | tr ' ' ',')")
  fi
  if [ -n "$LIMIT" ]; then RE_ARGS+=(--limit "$LIMIT"); fi
  if [ -n "$CORPUS_LIMIT" ]; then RE_ARGS+=(--corpus-limit "$CORPUS_LIMIT"); fi
  return 0   # a bare trailing [ -n ] guard would sink the function under set -e
}

# --- finished settings never rerun: ONE python call checks all conditions -----
# (per-condition calls paid an interpreter start + dataset load each = slow submit)
run_eval_args "$(echo $AGENTS | tr ' ' ',')"
if [ "${SKIP_CHECK:-0}" = "1" ]; then
  # FAST SUBMIT: skip the --check-complete pre-check, which loads the whole dataset (the big
  # document corpora especially) on every submit just to count finished instances. Safe because
  # the child job resumes per-instance — an already-complete condition re-reports from rows.jsonl
  # and exits in seconds. Use for a fresh sweep; drop it once runs exist to avoid resubmitting done
  # conditions. Fabricate an all-PENDING report so the parser below queues every condition.
  echo ">> SKIP_CHECK=1: submitting all conditions WITHOUT the per-submit completeness check"
  CHECK_OUT=$(for a in $AGENTS; do echo "$a PENDING 0/0"; done)
else
  CHECK_OUT=$("$PYTHON" -m evaluation.run_eval "${RE_ARGS[@]}" --check-complete) || CHECK_RC=$?
  CHECK_RC=${CHECK_RC:-0}
  if [ -z "$CHECK_OUT" ] || { [ "$CHECK_RC" != "0" ] && [ "$CHECK_RC" != "3" ]; }; then
    echo "ERROR: completeness pre-check failed (rc=$CHECK_RC) — refusing to guess." >&2
    exit 1
  fi
fi
echo "$CHECK_OUT" | sed 's/^/>> /'
# --check-complete prints one line per condition x seed:
#   "<name> [seed=<N>] <STATUS> <done>/<total>"   (the seed=<N> column appears only when seeded)
# so STATUS is NOT a fixed column. Take the name (field 1) and look for PENDING in the rest,
# deduping by name (run.sh loops the seeds itself, so a condition is queued at most once).
PENDING_AGENTS=()
seen_pending=" "
while read -r name rest; do
  case " $rest " in
    *" PENDING "*)
      case "$seen_pending" in
        *" $name "*) : ;;                                  # already queued (another seed)
        *) PENDING_AGENTS+=("$name"); seen_pending="${seen_pending}${name} " ;;
      esac
      ;;
  esac
done <<< "$CHECK_OUT"
if [ ${#PENDING_AGENTS[@]} -eq 0 ]; then
  echo ">> all conditions already complete; nothing to run."
  exit 0
fi

# ── fan-out: `sbatch scripts/eval_agent_suite.sh` (or SUBMIT=1 bash ...) submits
# ONE gpu:1 job per pending condition — each child serves its OWN vLLM, so the
# four conditions run in parallel instead of consecutively. Distinct ports keep
# two children safe if they ever share a node.
if [ "${SUBMIT:-0}" = "1" ] || [ "${SLURM_JOB_NAME:-}" = "agent-search-suite" ]; then
  # ONE job array: tasks JOBID_0..JOBID_N map to the pending conditions
  # (same setup => same array; `scancel JOBID` cancels the whole family,
  # `scancel JOBID_2` just one task). Each task serves its own vLLM on
  # port 8101+task, so co-scheduled tasks on one node never collide.
  n=$(( ${#PENDING_AGENTS[@]} - 1 ))
  LOGDIR="slurm_logs/agent_runs/$DATASET"; mkdir -p "$LOGDIR"   # logs by type (agent_runs) then dataset
  # --- QOS: express schedules fast but has a tight per-user CPU-min cap (MaxTRESRunMins); once it
  # saturates, express jobs PEND with reason 'MaxCpuRunMins'. The --qos passed here OVERRIDES
  # run.sh's `#SBATCH --qos=express` header for these children. QOS controls the policy:
  #   QOS=split (default): exact ~50/50 between express and FALLBACK_QOS (normal) by alternating a
  #     per-user counter across submits — half the load on fast express, half on normal, so express's
  #     CPU-min cap absorbs only half the jobs and far fewer PEND on MaxCpu. (qos only affects
  #     SCHEDULING, not results, so it need not be stable per dataset.)
  #   QOS=auto: use express while it has headroom for ME, else fall back to FALLBACK_QOS (reactive).
  #   QOS=<name>: force one qos (e.g. express for the small code sets, normal for the big wiki sets).
  # (this user's qos: express,normal — `sacctmgr -n show assoc user=$USER format=qos`)
  QOS="${QOS:-split}"
  FALLBACK_QOS="${FALLBACK_QOS:-normal}"
  case "$QOS" in
    split)
      _qf="${TMPDIR:-/tmp}/${USER}-assuite-qos.count"       # alternate normal,express,normal,... = 50/50
      _n=$(( $(cat "$_qf" 2>/dev/null || echo 0) + 1 )); echo "$_n" > "$_qf" 2>/dev/null || true
      if [ $(( _n % 2 )) -eq 0 ]; then CHOSEN_QOS="express"; else CHOSEN_QOS="$FALLBACK_QOS"; fi
      ;;
    auto)
      # PROACTIVE guard (primary): would THIS submission itself push $USER over the
      # express cap? footprint = n_tasks x JOB_CPUS x JOB_TIME (this submission);
      # usage = current express usage per qos_budget.sh's formula (skipped, not
      # crashed, if squeue is unavailable — e.g. a non-cluster dev machine).
      EXPRESS_CAP_CPUMIN="${EXPRESS_CAP_CPUMIN:-54000}"
      _n_tasks=${#PENDING_AGENTS[@]}
      _job_cpus="${JOB_CPUS:-4}"
      _footprint=$(footprint_cpumin "$_n_tasks" "$_job_cpus" "$JOB_TIME")
      _fallback_reason=""
      if _usage=$(express_usage_cpumin); then
        if [ "$(decide_qos_budget "$_usage" "$_footprint" "$EXPRESS_CAP_CPUMIN")" = "fallback" ]; then
          _fallback_reason="express usage=${_usage} + this submission's footprint=${_footprint} > cap=${EXPRESS_CAP_CPUMIN} cpu-min"
        fi
      fi
      # REACTIVE guard (second line of defense; also the ONLY guard when squeue was
      # unavailable above): a pending job of mine already shows the cap-hit reason.
      if [ -z "$_fallback_reason" ] && squeue -u "$USER" -h -t PENDING -o "%r" 2>/dev/null | grep -qiE "MaxCpuRunMins|MaxTRESRunMins"; then
        _fallback_reason="express saturated for $USER (MaxCpuRunMins pending)"
      fi
      if [ -n "$_fallback_reason" ]; then
        CHOSEN_QOS="$FALLBACK_QOS"
        echo ">> $_fallback_reason -> submitting on qos=$CHOSEN_QOS"
      else
        CHOSEN_QOS="express"
      fi
      ;;
    *) CHOSEN_QOS="$QOS" ;;
  esac
  echo ">> qos=$CHOSEN_QOS (policy QOS=$QOS) for $DATASET / ${MODEL##*/}"
  # cpus-per-task=4: the job is GPU-bound (vLLM on the H100) and the eval workers are
  # GIL-bound Python threads waiting on vLLM HTTP, so few cores suffice. Lower CPUs also
  # shrink each job's MaxTRESRunMins (cpus x walltime) footprint, so more run concurrently
  # under the per-user cap. Bump with JOB_CPUS= if vLLM token streaming becomes CPU-bound.
  # job name carries the model (basename) so you can cancel/grep per model + dataset, e.g.
  #   squeue -u $USER -o "%i %j" | awk '/AgentWorld/{print $1}' | xargs scancel
  MODEL_TAG="${MODEL##*/}"   # e.g. Qwen-AgentWorld-35B-A3B vs Tongyi-DeepResearch-30B-A3B
  jid=$(sbatch --parsable --array=0-${n} --time="$JOB_TIME" \
    --partition="${GPU_PARTITION:-h24gpu}" --qos="$CHOSEN_QOS" \
    --cpus-per-task="${JOB_CPUS:-4}" --mem="${JOB_MEM:-128g}" \
    --job-name="as-suite-${DATASET}-${MODEL_TAG}" \
    --output="$LOGDIR/%x-%A_%a.out" --error="$LOGDIR/%x-%A_%a.err" \
    --export=ALL,AGENT_SEARCH_CONDS="${PENDING_AGENTS[*]}",DATASET=$DATASET,MODEL=$MODEL,DENSE_MODEL=$DENSE_MODEL,TP=$TP,WORKERS=$WORKERS,LEVEL=$LEVEL,LIMIT=$LIMIT,CORPUS_LIMIT=$CORPUS_LIMIT,MAX_STEPS=${MAX_STEPS:-50},SEEDS="$SEEDS",SEED=${SEED:-},TEMPERATURE=$TEMPERATURE,REPO_CACHE=$REPO_CACHE,INDEX_ROOT=$INDEX_ROOT,RUNS_DIR=$RUNS_DIR \
    scripts/run.sh)
  for i in $(seq 0 $n); do
    echo ">> ${jid}_${i} -> ${PENDING_AGENTS[$i]} (vLLM on :$((8101 + i)))"
  done
  echo ">> cancel all: scancel $jid ; watch: squeue -u \$USER"
  exit 0
fi

# --- start the server in the background, unless one is already up -----------
if curl -sf "${API_BASE}/models" >/dev/null 2>&1; then
  echo ">> reusing the server already running at ${API_BASE}"
else
  if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: vllm is not on PATH. Activate the cluster env or install vllm before running agent_* retrievers." >&2
    exit 127
  fi
  # 0.85 cap: leaves ~14GB on a 96GB H100 for the co-resident embedder (see
  # run.sh for the arithmetic). User VLLM_ARGS placed later -> overrides ours.
  SERVE_ARGS=(--tensor-parallel-size "$TP" --port "$PORT"
              --gpu-memory-utilization "${GPU_UTIL:-0.9}")
  # quantized / serving knobs (unset => vLLM default; see run.sh for the full list).
  [ -n "${QUANT:-}" ]          && SERVE_ARGS+=(--quantization "$QUANT")
  [ -n "${DTYPE:-}" ]          && SERVE_ARGS+=(--dtype "$DTYPE")
  [ -n "${KV_CACHE_DTYPE:-}" ] && SERVE_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  [ -n "${MAX_MODEL_LEN:-}" ]  && SERVE_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
  [ -n "${LOAD_FORMAT:-}" ]    && SERVE_ARGS+=(--load-format "$LOAD_FORMAT")
  if [ "${EAGER:-0}" = "1" ]; then
    SERVE_ARGS+=(--enforce-eager)                      # fast smoke: no compile at all
  else
    SERVE_ARGS+=(--compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH:-PIECEWISE}\"}")
  fi
  [ -n "${VLLM_ARGS:-}" ] && SERVE_ARGS+=($VLLM_ARGS)     # e.g. VLLM_ARGS="--max-model-len 65536"
  echo ">> starting vLLM (${MODEL}, tp=${TP}${EAGER:+, eager}) on :${PORT} ..."
  vllm serve "$MODEL" "${SERVE_ARGS[@]}" &
  VLLM_PID=$!
  trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT   # capture NOW; bare $! at exit time may name a different job
  echo ">> waiting for the server to be ready ..."
  until curl -sf "${API_BASE}/models" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "ERROR: vLLM exited during startup (OOM? bad model path? see log above)." >&2
      echo "       Fallbacks: EAGER=1, VLLM_ARGS='--gpu-memory-utilization 0.8'," >&2
      echo "       or MODEL=Qwen/Qwen2.5-Coder-7B-Instruct." >&2
      exit 1
    fi
    sleep 5
  done
  echo ">> server up."
fi

# agent_dense: the (137M) embedder co-resides in the GPU_UTIL headroom on the
# GPU. Still OOM on a smaller card? AGENT_SEARCH_DENSE_DEVICE=cpu (embeddings are
# cached per corpus, so CPU encoding is a one-time cost).
[ -n "${AGENT_SEARCH_DENSE_DEVICE:-}" ] && export AGENT_SEARCH_DENSE_DEVICE

# --- run each one-tool condition --------------------------------------------
for retriever in "${PENDING_AGENTS[@]}"; do
  echo ">> agent condition: ${retriever}"
  run_eval_args "$retriever"
  "$PYTHON" -m evaluation.run_eval "${RE_ARGS[@]}"
done

echo ">> done. compare: python scripts/summarize_runs.py  (runs land in $RUNS_DIR/agent/)"
