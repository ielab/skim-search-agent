#!/usr/bin/env bash
# Unit tests for the QOS=auto budget library in scripts/eval_agent_suite.sh.
# Pure bash, no SLURM required. Loads the script's library functions via
# `QOS_LIB_ONLY=1 source scripts/eval_agent_suite.sh` (which returns before any
# of the script's real work — cd, module load, mkdir, python, sbatch — runs),
# then unit-tests the functions directly, with a mocked `squeue` on PATH for
# the parts that shell out.
#
# Run: bash tests/test_qos_auto.sh   (exits 0 on all-pass, 1 on any failure)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

fail=0
check () {  # check DESC ACTUAL EXPECTED
  if [ "$2" = "$3" ]; then
    echo "ok   - $1"
  else
    echo "FAIL - $1 (got [$2], want [$3])"
    fail=1
  fi
}

# --- load library functions only; no SLURM submission happens -------------
QOS_LIB_ONLY=1 source "$REPO_ROOT/scripts/eval_agent_suite.sh"

# 1. time parsing: HH:MM:SS (also covers JOB_TIME's own default "24:00:00")
check "parse HH:MM:SS (JOB_TIME default)" "$(parse_time_to_minutes "24:00:00")" "1440"
check "parse HH:MM:SS with minutes"        "$(parse_time_to_minutes "01:30:15")" "90"

# 2. time parsing: D-HH:MM:SS
check "parse D-HH:MM:SS"           "$(parse_time_to_minutes "2-00:00:00")" "2880"
check "parse D-HH:MM:SS remainder" "$(parse_time_to_minutes "1-02:30:00")" "1590"

# 3. footprint arithmetic (mirrors the real 2026-07-12 incident: 17 tasks x
#    4 cpus x 24h = 97920 cpu-min, well over the 54000 cap on its own)
check "footprint = n x cpus x mins" "$(footprint_cpumin 17 4 "24:00:00")" "97920"

# 4. decision: fits in headroom -> express
check "fits headroom -> express" "$(decide_qos_budget 0 5760 54000)" "express"
check "exactly at cap -> express" "$(decide_qos_budget 0 54000 54000)" "express"

# 5. decision: exceeds cap -> fallback
check "exceeds cap -> fallback" "$(decide_qos_budget 50000 5760 54000)" "fallback"
check "incident totals -> fallback" "$(decide_qos_budget 0 97920 54000)" "fallback"

# 6. express_usage_cpumin, with a mocked squeue on PATH mimicking:
#      squeue -u $USER -r -h -O "QOS,cpus-per-task,TimeLimit"
MOCKBIN="$(mktemp -d)"
cat > "$MOCKBIN/squeue" <<'EOF'
#!/usr/bin/env bash
cat <<'ROWS'
express 4 24:00:00
express 4 12:00:00
normal 8 24:00:00
ROWS
EOF
chmod +x "$MOCKBIN/squeue"
# express rows only: 4*1440 + 4*720 = 5760 + 2880 = 8640
usage="$(PATH="$MOCKBIN:$PATH" express_usage_cpumin)"
check "express_usage_cpumin sums express rows only" "$usage" "8640"

# end-to-end: this usage (8640) + a small footprint (5760) fits under 54000
check "end-to-end fits -> express" \
  "$(decide_qos_budget "$usage" 5760 54000)" "express"
# end-to-end: this usage (8640) + the incident's footprint (97920) exceeds 54000
check "end-to-end incident footprint -> fallback" \
  "$(decide_qos_budget "$usage" 97920 54000)" "fallback"

rm -rf "$MOCKBIN"

# 7. squeue missing -> express_usage_cpumin fails cleanly (nonzero, no crash);
#    this is what makes the caller fall back to the reactive-only guard.
EMPTYBIN="$(mktemp -d)"
if out=$(PATH="$EMPTYBIN" express_usage_cpumin); then
  echo "FAIL - express_usage_cpumin should fail when squeue is unavailable"
  fail=1
else
  echo "ok   - squeue-missing: express_usage_cpumin returns nonzero (reactive fallback path), no crash"
fi
rmdir "$EMPTYBIN" 2>/dev/null || true

if [ "$fail" = 0 ]; then
  echo "ALL PASS"
  exit 0
else
  echo "SOME FAILED"
  exit 1
fi
