#!/usr/bin/env bash
# Express QOS budget calculator (user formula: sum over MY express jobs of cpus × TimeLimit-minutes ≤ 54000).
# RUN THIS IMMEDIATELY BEFORE AND AFTER every QOS=express submission.
# Counts RUNNING + PENDING express jobs across ALL projects of $USER (the cap is per-user, not per-project).
# Prints usage, headroom, and how many standard as-suite jobs (4 cpu × 1440 min = 5760) still fit.
set -uo pipefail
CAP=54000
USED=$(squeue -u "$USER" -r -h -O "QOS,cpus-per-task,TimeLimit" | awk '
  $1=="express" {
    split($3, t, "[-:]")
    if (length(t)==4)      mins = t[1]*1440 + t[2]*60 + t[3]   # D-HH:MM:SS
    else if (length(t)==3) mins = t[1]*60 + t[2]                # HH:MM:SS
    else                   mins = t[1]                          # MM:SS fallback
    sum += $2 * mins
  } END { print sum+0 }')
HEADROOM=$((CAP - USED))
FIT=$((HEADROOM / 5760))
echo "express used: $USED / $CAP cpu-min   headroom: $HEADROOM   standard as-suite jobs that fit: $FIT"
STUCK=$(squeue -u "$USER" -r -h -o "%A %r" | awk '$2=="MaxCpuRunMinsPerUser"{print $1}')
if [ -n "$STUCK" ]; then
  echo "STUCK express jobs (cancel + resubmit QOS=normal per user rule):"
  echo "$STUCK"
fi
