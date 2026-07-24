#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="gpu-smoke-${STAMP}"
RUN_PARENT="${ROOT}/runs/connect3"
RUN_DIR="${RUN_PARENT}/${RUN_NAME}"
CONSOLE="${RUN_PARENT}/${RUN_NAME}.console.log"

cd "${ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; run scripts/bootstrap_gpu.sh first." >&2
  exit 1
fi
if [[ -e "${RUN_DIR}" || -e "${CONSOLE}" ]]; then
  echo "Smoke output already exists; wait one second and rerun." >&2
  exit 1
fi

mkdir -p "${RUN_PARENT}"
printf '%s\n' "${RUN_DIR}" > "${RUN_PARENT}/latest_gpu_smoke.txt"

echo "Starting GPU smoke: ${RUN_DIR}"
env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 \
  "${PYTHON}" benchmark/connect3/train_critic.py \
    --run-dir "${RUN_DIR}" \
    --max-steps 2 \
    --checkpoint-freq 1 \
    --actors 1 \
    --evaluators 0 \
    --max-simulations 4 \
    --train-batch-size 8 \
    --replay-buffer-size 32 \
    --replay-buffer-reuse 4 \
    --nn-model resnet \
    --nn-width 16 \
    --nn-depth 1 \
  2>&1 | tee "${CONSOLE}"

[[ -d "${RUN_DIR}/checkpoint-2" ]] || {
  echo "Missing checkpoint-2" >&2
  exit 1
}
[[ -s "${RUN_DIR}/learner.jsonl" ]] || {
  echo "Missing or empty learner.jsonl" >&2
  exit 1
}
compgen -G "${RUN_DIR}/log-actor-*.txt" >/dev/null || {
  echo "No actor log was written" >&2
  exit 1
}
grep -q 'Game [0-9][0-9]*: Returns:' "${RUN_DIR}"/log-actor-*.txt || {
  echo "Actor log contains no completed self-play game" >&2
  exit 1
}

"${PYTHON}" - "${RUN_DIR}/learner.jsonl" <<'PY'
import json
import math
import pathlib
import sys

rows = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines()]
assert rows and rows[-1]["step"] == 2, rows[-1] if rows else None
for row in rows:
    for name, value in row["loss"].items():
        if not math.isfinite(float(value)):
            raise SystemExit(f"Non-finite smoke loss at step {row['step']}: {name}={value}")
print(f"Smoke learner steps: {len(rows)}; final step: {rows[-1]['step']}")
PY

echo "GPU smoke passed: ${RUN_DIR}"

