#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/connect3/az_resnet_w64_d3_s64_a16_run0}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-60}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_PARENT="${ROOT}/runs/connect3/comparisons"
OUT_NAME="${OUT_NAME:-exploratory-checkpoint-${CHECKPOINT_STEP}-${STAMP}}"
OUT_DIR="${OUT_PARENT}/${OUT_NAME}"
CONSOLE="${OUT_DIR}.console.log"
PID_FILE="${OUT_DIR}.pid"
LAUNCH_FILE="${OUT_DIR}.launch.txt"
LATEST_FILE="${ROOT}/runs/connect3/latest_checkpoint_comparison.txt"

cd "${ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; run scripts/bootstrap_gpu.sh first." >&2
  exit 1
fi
if [[ ! -d "${RUN_DIR}/checkpoint-${CHECKPOINT_STEP}" ]]; then
  echo "Missing checkpoint: ${RUN_DIR}/checkpoint-${CHECKPOINT_STEP}" >&2
  exit 1
fi
for path in "${OUT_DIR}" "${CONSOLE}" "${PID_FILE}" "${LAUNCH_FILE}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite existing path: ${path}" >&2
    exit 1
  fi
done
mkdir -p "${OUT_PARENT}"

read -r -a ROOT_PLIES <<< "${COMPARE_ROOT_PLIES:-6 7 8 9 10}"
read -r -a CALIBRATION_PLIES <<< \
  "${COMPARE_CALIBRATION_PLIES:-6 7 8 9 10 11 12 13}"

COMMAND=(
  nice -n 10
  "${PYTHON}" benchmark/connect3/compare_checkpoint.py
  --run-dir "${RUN_DIR}"
  --checkpoint-step "${CHECKPOINT_STEP}"
  --out-dir "${OUT_DIR}"
  --root-plies "${ROOT_PLIES[@]}"
  --root-count "${COMPARE_ROOTS:-3}"
  --planning-depth "${COMPARE_DEPTH:-3}"
  --max-terminal-leaf-fraction "${COMPARE_MAX_TERMINAL_FRACTION:-1.0}"
  --min-nonterminal-leaves "${COMPARE_MIN_NONTERMINAL_LEAVES:-0}"
  --calibration-plies "${CALIBRATION_PLIES[@]}"
  --calibration-states "${COMPARE_CALIBRATION_STATES:-1000}"
  --envelope-margin "${COMPARE_ENVELOPE_MARGIN:-0.05}"
  --slow-simulations "${COMPARE_SLOW_SIMULATIONS:-16}"
  --uct-c 1.41
  --delta 0.05
  --epsilon 0
  --replicates "${COMPARE_REPLICATES:-1}"
  --max-rounds "${COMPARE_MAX_ROUNDS:-1000}"
  --seed 20260724
)

{
  echo "launch_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "project_root=${ROOT}"
  echo "run_dir=${RUN_DIR}"
  echo "checkpoint_step=${CHECKPOINT_STEP}"
  echo "out_dir=${OUT_DIR}"
  echo "execution_device=cpu"
  printf 'command='
  printf '%q ' env \
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONUNBUFFERED=1 \
    "${COMMAND[@]}"
  echo
} > "${LAUNCH_FILE}"

nohup env \
  JAX_PLATFORMS=cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 \
  "${COMMAND[@]}" > "${CONSOLE}" 2>&1 &
PID=$!
printf '%s\n' "${PID}" > "${PID_FILE}"
echo "pid=${PID}" >> "${LAUNCH_FILE}"
printf '%s\n' "${OUT_DIR}" > "${LATEST_FILE}"

sleep 5
if ! kill -0 "${PID}" 2>/dev/null; then
  if [[ ! -s "${OUT_DIR}/summary.json" ]]; then
    echo "Comparison exited during startup without a summary. Console follows:" >&2
    tail -n 100 "${CONSOLE}" >&2 || true
    exit 1
  fi
fi

echo "comparison_pid=${PID}"
echo "out_dir=${OUT_DIR}"
echo "console=${CONSOLE}"
echo "execution_device=cpu (nice level 10)"
echo "Use: bash scripts/status_checkpoint_comparison.sh"
