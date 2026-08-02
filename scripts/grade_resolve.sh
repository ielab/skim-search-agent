#!/usr/bin/env bash
#SBATCH --job-name=swe-resolve
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --gres=gpu:0
#SBATCH --partition=h24
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --output=slurm_logs/resolve/%x-%j.out
#SBATCH --error=slurm_logs/resolve/%x-%j.err
# Grade a patch-mode run's SWE-bench RESOLVE rate via the self-hosted apptainer harness.
# CPU-only (no GPU): applies each model_patch in the cached image, runs FAIL_TO_PASS/PASS_TO_PASS.
# Compute nodes are OFFLINE, so images must be pre-warmed first (scripts/warm_swebench_images.py
# on the login node) — this runs with --no-pull and grades only already-warmed instances.
#
#   RUN=runs/agent/swebench_lite/<model>/agent_codefix_patch SUBSET=lite sbatch scripts/grade_resolve.sh
#   RUN=<run_dir|preds.jsonl> SUBSET=verified WORKERS=12 sbatch scripts/grade_resolve.sh
# Env knobs: RUN (required), SUBSET=lite|verified, WORKERS=8, TIMEOUT=1800, NOPULL=1 (0 to allow
# pulling — only on a node WITH internet, e.g. interactive on the login node).
set -euo pipefail
if [ -n "${SLURM_JOB_ID:-}" ]; then
  cd "${SLURM_SUBMIT_DIR:-.}"; [ -d agent_search ] || cd ..
else
  cd "$(dirname "$0")/.."
fi
module load apptainer/1.5.0 2>/dev/null || module load apptainer 2>/dev/null || true
export APPTAINER_CACHEDIR="$PWD/.cache/apptainer" SINGULARITY_CACHEDIR="$PWD/.cache/apptainer"
export APPTAINER_TMPDIR="/tmp/${USER}-apptainer-build" SINGULARITY_TMPDIR="/tmp/${USER}-apptainer-build"
# compute nodes are OFFLINE: load the SWE-bench metadata (F2P/P2P/test_patch) from the HF cache
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p slurm_logs/resolve "$APPTAINER_TMPDIR"

RUN="${RUN:?set RUN=<run_dir or preds.jsonl>}"
SUBSET="${SUBSET:-lite}"
WORKERS="${WORKERS:-8}"
TIMEOUT="${TIMEOUT:-1800}"
NOPULL="${NOPULL:-1}"
PY="${PY:-.venv_swebench/bin/python}"
[ -x "$PY" ] || { echo "!! $PY missing — create it: envs/bin/python -m venv .venv_swebench && .venv_swebench/bin/pip install swebench" >&2; exit 1; }

# pre-flight: how many of this run's committed instances already have a warmed SIF?
"$PY" - "$RUN" "$SUBSET" <<'PY' || true
import json, os, sys
run, subset = sys.argv[1], sys.argv[2]
src = run if os.path.isfile(run) else (os.path.join(run,"preds.jsonl") if os.path.isfile(os.path.join(run,"preds.jsonl")) else os.path.join(run,"rows.jsonl"))
ids = [json.loads(l)["instance_id"] for l in open(src) if l.strip() and json.loads(l).get("model_patch")]
sif = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__ if False else "."))) , ".cache/swebench_sif")
sif = ".cache/swebench_sif"
warm = sum(os.path.exists(os.path.join(sif, f"{i}.sif")) for i in set(ids))
print(f">> {len(set(ids))} committed patches; {warm} have a warmed SIF, {len(set(ids))-warm} missing (will be image_missing)")
PY

EXTRA=(); [ "$NOPULL" = "1" ] && EXTRA=(--no-pull)
echo ">> grading RUN=$RUN SUBSET=$SUBSET WORKERS=$WORKERS NOPULL=$NOPULL"
echo ">> warmed SIFs on disk: $(ls .cache/swebench_sif/*.sif 2>/dev/null | wc -l)"
"$PY" -m evaluation.swebench_apptainer "$RUN" \
  --subset "$SUBSET" --workers "$WORKERS" --timeout "$TIMEOUT" "${EXTRA[@]}"
echo ">> report: $([ -d "$RUN" ] && echo "$RUN/apptainer_report.json" || echo "next to $RUN")"
