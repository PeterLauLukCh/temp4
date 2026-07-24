#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
RUN_NAME="${RUN_NAME:-az_resnet_w64_d3_s64_run0}"
RUN_PARENT="${ROOT}/runs/connect3"
RUN_DIR="${RUN_PARENT}/${RUN_NAME}"
PID_FILE="${RUN_PARENT}/${RUN_NAME}.pid"

cd "${ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; run scripts/bootstrap_gpu.sh first." >&2
  exit 1
fi
if [[ -r "${PID_FILE}" ]]; then
  PID="$(tr -d '[:space:]' < "${PID_FILE}")"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "Training PID ${PID} is still running; verification must wait." >&2
    exit 1
  fi
fi

for step in 20 100 200; do
  checkpoint="${RUN_DIR}/checkpoint-${step}"
  result="${RUN_DIR}/verification-checkpoint-${step}.json"
  console="${RUN_PARENT}/${RUN_NAME}.verification-checkpoint-${step}.console.log"
  [[ -d "${checkpoint}" ]] || {
    echo "Missing ${checkpoint}" >&2
    exit 1
  }
  if [[ -e "${result}" || -e "${console}" ]]; then
    echo "Refusing to overwrite existing verification output for checkpoint ${step}." >&2
    exit 1
  fi
done

if [[ -e "${RUN_DIR}/training_states.json" ||
      -e "${RUN_PARENT}/${RUN_NAME}.extract-training-states.console.log" ]]; then
  echo "Refusing to overwrite existing training-state output." >&2
  exit 1
fi

for step in 20 100 200; do
  echo "Verifying checkpoint ${step}"
  env \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONUNBUFFERED=1 \
    "${PYTHON}" benchmark/connect3/verify_critic.py \
      --run-dir "${RUN_DIR}" \
      --checkpoint-step "${step}" \
      --plies 6 7 8 9 10 \
      --max-states 1000 \
    2>&1 | tee \
      "${RUN_PARENT}/${RUN_NAME}.verification-checkpoint-${step}.console.log"
done

"${PYTHON}" benchmark/connect3/extract_training_states.py \
  --run-dir "${RUN_DIR}" \
  2>&1 | tee "${RUN_PARENT}/${RUN_NAME}.extract-training-states.console.log"

"${PYTHON}" scripts/summarize_results.py --run-dir "${RUN_DIR}"

