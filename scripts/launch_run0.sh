#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
RUN_NAME="${RUN_NAME:-az_resnet_w64_d3_s64_run0}"
RUN_PARENT="${ROOT}/runs/connect3"
RUN_DIR="${RUN_PARENT}/${RUN_NAME}"
CONSOLE="${RUN_PARENT}/${RUN_NAME}.console.log"
PID_FILE="${RUN_PARENT}/${RUN_NAME}.pid"
LAUNCH_FILE="${RUN_PARENT}/${RUN_NAME}.launch.txt"
MONITOR_CONSOLE="${RUN_PARENT}/${RUN_NAME}.monitor.console.log"
MONITOR_PID_FILE="${RUN_PARENT}/${RUN_NAME}.monitor.pid"

cd "${ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; run scripts/bootstrap_gpu.sh first." >&2
  exit 1
fi

for path in \
  "${RUN_DIR}" "${CONSOLE}" "${PID_FILE}" "${LAUNCH_FILE}" \
  "${MONITOR_CONSOLE}" "${MONITOR_PID_FILE}" \
  "${RUN_PARENT}/${RUN_NAME}.resource.csv" \
  "${RUN_PARENT}/${RUN_NAME}.resource.log"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite existing path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${RUN_PARENT}"

COMMAND=(
  "${PYTHON}" benchmark/connect3/train_critic.py
  --run-dir "${RUN_DIR}"
  --max-steps 200
  --checkpoint-freq 20
  --actors 4
  --evaluators 0
  --max-simulations 64
  --train-batch-size 512
  --replay-buffer-size 8192
  --replay-buffer-reuse 4
  --nn-model resnet
  --nn-width 64
  --nn-depth 3
)

{
  echo "launch_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "project_root=${ROOT}"
  echo "run_dir=${RUN_DIR}"
  printf 'command='
  printf '%q ' env \
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONUNBUFFERED=1 \
    "${COMMAND[@]}"
  echo
} > "${LAUNCH_FILE}"

nohup env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 \
  "${COMMAND[@]}" > "${CONSOLE}" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "${TRAIN_PID}" > "${PID_FILE}"
echo "training_pid=${TRAIN_PID}" >> "${LAUNCH_FILE}"

nohup bash "${ROOT}/scripts/monitor_run0.sh" "${TRAIN_PID}" "${RUN_NAME}" \
  > "${MONITOR_CONSOLE}" 2>&1 &
MONITOR_PID=$!
printf '%s\n' "${MONITOR_PID}" > "${MONITOR_PID_FILE}"
echo "monitor_pid=${MONITOR_PID}" >> "${LAUNCH_FILE}"

sleep 5
if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
  echo "Training exited during startup. Console follows:" >&2
  tail -n 100 "${CONSOLE}" >&2 || true
  exit 1
fi

echo "run_dir=${RUN_DIR}"
echo "training_pid=${TRAIN_PID}"
echo "monitor_pid=${MONITOR_PID}"
echo "console=${CONSOLE}"
echo "Use: bash scripts/status_run0.sh"
