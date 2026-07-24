#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/node_environment.txt}"

{
  echo "captured_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "project_root=${ROOT}"
  echo "hostname=$(hostname)"
  echo "user=$(id -un)"
  echo
  echo "[os]"
  uname -a
  if [[ -r /etc/os-release ]]; then
    grep -E '^(PRETTY_NAME|VERSION_ID|ID)=' /etc/os-release
  fi
  echo
  echo "[python]"
  command -v python3 || true
  python3 --version 2>&1 || true
  echo
  echo "[nvidia]"
  nvidia-smi || true
  nvidia-smi \
    --query-gpu=index,name,uuid,driver_version,memory.total,compute_cap \
    --format=csv,noheader || true
  echo
  echo "[cpu]"
  lscpu | grep -E \
    '^(Architecture|CPU\\(s\\)|On-line CPU|Model name|Thread|Core|Socket|NUMA)' \
    || true
  echo
  echo "[memory]"
  free -h || true
  echo
  echo "[disk]"
  df -h "${ROOT}" || true
  echo
  echo "[open_spiel]"
  if [[ -r "${ROOT}/third_party/open_spiel/.source_commit" ]]; then
    printf "source_commit="
    cat "${ROOT}/third_party/open_spiel/.source_commit"
  else
    echo "not_installed"
  fi
} | tee "${OUT}"

echo "Wrote ${OUT}"

