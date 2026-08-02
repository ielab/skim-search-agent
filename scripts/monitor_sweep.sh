#!/usr/bin/env bash
cd ${REPO_ROOT:-.}
export PYTHONPATH=${REPO_ROOT:-.} PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p runs/_monitor
while true; do
  envs/bin/python scripts/monitor_sweep.py >> runs/_monitor/monitor.log 2>&1
  [ -f runs/_monitor/DONE ] && break
  sleep 1200
done
echo "$(date) monitor exiting (DONE)" >> runs/_monitor/monitor.log
