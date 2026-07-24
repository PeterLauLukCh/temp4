#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 TRAIN_PID RUN_NAME" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PID="$1"
RUN_NAME="$2"
RUN_PARENT="${ROOT}/runs/connect3"
GPU_CSV="${RUN_PARENT}/${RUN_NAME}.resource.csv"
CPU_LOG="${RUN_PARENT}/${RUN_NAME}.resource.log"

printf '%s\n' \
  'timestamp,index,name,gpu_util_percent,memory_used_mib,memory_total_mib,temperature_c,power_w' \
  > "${GPU_CSV}"
echo "monitor_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${CPU_LOG}"
echo "training_pid=${TRAIN_PID}" >> "${CPU_LOG}"

sample() {
  local now
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader,nounits >> "${GPU_CSV}" 2>> "${CPU_LOG}" || true
  {
    echo
    echo "[${now}]"
    ps -o pid,ppid,etime,pcpu,pmem,rss,stat,cmd \
      -p "${TRAIN_PID}" --ppid "${TRAIN_PID}" 2>&1 || true
  } >> "${CPU_LOG}"
}

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sample
  sleep 30
done

sample
echo "monitor_finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${CPU_LOG}"

