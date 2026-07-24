#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-az_resnet_w64_d3_s64_a16_run0}"
RUN_PARENT="${ROOT}/runs/connect3"
RUN_DIR="${RUN_PARENT}/${RUN_NAME}"
CONSOLE="${RUN_PARENT}/${RUN_NAME}.console.log"
PID_FILE="${RUN_PARENT}/${RUN_NAME}.pid"

echo "checked_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "run_dir=${RUN_DIR}"

if [[ -r "${PID_FILE}" ]]; then
  PID="$(tr -d '[:space:]' < "${PID_FILE}")"
  echo "training_pid=${PID}"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "training_status=running"
    ps -o pid,ppid,etime,pcpu,pmem,rss,stat,cmd \
      -p "${PID}" --ppid "${PID}" || true
  else
    echo "training_status=not_running"
  fi
else
  echo "training_status=pid_file_missing"
fi

echo
echo "[gpu]"
nvidia-smi \
  --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader || true

echo
echo "[latest learner rows]"
if [[ -s "${RUN_DIR}/learner.jsonl" ]]; then
  tail -n 3 "${RUN_DIR}/learner.jsonl"
else
  echo "learner.jsonl not written yet"
fi

echo
echo "[actor games]"
if compgen -G "${RUN_DIR}/log-actor-*.txt" >/dev/null; then
  for log in "${RUN_DIR}"/log-actor-*.txt; do
    printf '%s games=' "$(basename "${log}")"
    grep -c 'Game [0-9][0-9]*: Returns:' "${log}" || true
  done
else
  echo "actor logs not written yet"
fi

echo
echo "[checkpoints]"
find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print 2>/dev/null \
  | sort -V || true

echo
echo "[error scan]"
if [[ -r "${CONSOLE}" ]]; then
  grep -Ein \
    'Traceback|Exception caught|out of memory|CUDA_ERROR|FAILED_PRECONDITION|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' \
    "${CONSOLE}" "${RUN_DIR}"/log-*.txt 2>/dev/null | tail -n 30 \
    || echo "no matched error patterns"
else
  echo "console log not written yet"
fi
