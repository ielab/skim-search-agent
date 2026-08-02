#!/usr/bin/env bash
# Persistent login-node INCREMENTAL judge daemon. Every JUDGE_INTERVAL seconds (default 1200 = 20min):
#   1. judge_cells.py --datasets all  — the hash-keyed judge_cache.jsonl skips every already-judged
#      row (EM short-circuit + cached llm verdicts), so each pass only spends on NEW answers that
#      landed since last time. Finished/complete cells cost ~0 (all their rows are already cached).
#   2. compare_cells.py — refresh comparison_result.md so the table's judge% is always current.
# Robust: a failed judge pass (transient OpenAI error, etc.) is logged and the loop continues.
# Needs OPENAI_API_KEY in the environment at launch (inherited; persists for the process's life).
# Launch (survives the interactive session):
#   setsid nohup bash scripts/judge_daemon.sh >> analysis/judge_daemon.out 2>&1 < /dev/null &
# Stop:  pkill -f judge_daemon.sh
cd "$(dirname "$0")/.."
LOG=analysis/judge_daemon.log
INTERVAL="${JUDGE_INTERVAL:-1200}"
echo "$(date '+%F %T') judge-daemon START (interval=${INTERVAL}s, key=${OPENAI_API_KEY:+present})" >> "$LOG"
while true; do
  ts=$(date '+%F %T')
  if PYTHONPATH=. envs/bin/python scripts/judge_cells.py --datasets browsecomp --workers 16 >> "$LOG" 2>&1; then
    pend=$(PYTHONPATH=. envs/bin/python scripts/judge_cells.py --datasets browsecomp --dry-run 2>/dev/null \
           | grep -oE "pending=[0-9]+" | grep -oE "[0-9]+" | awk '{s+=$1}END{print s+0}')
    envs/bin/python scripts/compare_cells.py > /dev/null 2>&1
    echo "$ts judged+refreshed; pending_after=${pend}" >> "$LOG"
  else
    # judge failed (e.g. OpenAI quota 429) — still refresh the table so EM%/n/token
    # columns stay current while judging is blocked; judge% just stays stale.
    envs/bin/python scripts/compare_cells.py > /dev/null 2>&1
    echo "$ts judge pass FAILED (see above) — table refreshed anyway, continuing" >> "$LOG"
  fi
  sleep "$INTERVAL"
done
