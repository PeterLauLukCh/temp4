#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEST_FILE="${ROOT}/runs/connect3/latest_checkpoint_comparison.txt"

if [[ ! -r "${LATEST_FILE}" ]]; then
  echo "No comparison launch recorded at ${LATEST_FILE}" >&2
  exit 1
fi

OUT_DIR="$(tr -d '\r\n' < "${LATEST_FILE}")"
CONSOLE="${OUT_DIR}.console.log"
PID_FILE="${OUT_DIR}.pid"

echo "checked_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "out_dir=${OUT_DIR}"
if [[ -r "${PID_FILE}" ]]; then
  PID="$(tr -d '[:space:]' < "${PID_FILE}")"
  echo "comparison_pid=${PID}"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "comparison_status=running"
    ps -o pid,ppid,ni,etime,pcpu,pmem,rss,stat,cmd -p "${PID}" || true
  elif [[ -s "${OUT_DIR}/summary.json" ]]; then
    echo "comparison_status=complete"
  else
    echo "comparison_status=not_running_without_summary"
  fi
else
  echo "comparison_status=pid_file_missing"
fi

echo
echo "[completed method/root runs]"
if [[ -r "${CONSOLE}" ]]; then
  grep -E '^\{\"correct\":' "${CONSOLE}" | tail -n 12 || true
else
  echo "console log not written yet"
fi

echo
echo "[error scan]"
if [[ -r "${CONSOLE}" ]]; then
  grep -Ein \
    'Traceback|Exception|out of memory|CUDA_ERROR|FAILED_PRECONDITION|(^|[^[:alpha:]])nan([^[:alpha:]]|$)' \
    "${CONSOLE}" | tail -n 30 || echo "no matched error patterns"
else
  echo "console log not written yet"
fi

echo
echo "[summary]"
if [[ -s "${OUT_DIR}/summary.json" ]]; then
  "${ROOT}/.venv/bin/python" - "${OUT_DIR}/summary.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
print(json.dumps({
    "status": payload["status"],
    "exploratory": payload["exploratory"],
    "adapter_revision": payload.get("adapter_revision"),
    "checkpoint_step": payload["checkpoint_step"],
    "jax_backend": payload["jax_backend"],
    "terminal_payoffs": payload["config"].get("terminal_payoffs"),
    "root_selection": payload.get("root_selection"),
    "roots": payload["roots"],
    "envelope_diagnostics": payload["envelope_diagnostics_on_comparison_nodes"],
    "summary": payload["summary"],
    "wall_seconds": payload["wall_seconds"],
}, indent=2, sort_keys=True))
PY
else
  echo "summary.json not written yet"
fi
