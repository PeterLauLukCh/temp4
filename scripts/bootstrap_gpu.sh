#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROXY_PREFIX="${GH_PROXY_PREFIX:-https://gh-proxy.com/}"

OPEN_SPIEL_COMMIT="112b77704631fc2ce7ad8e4581f6ca09798ce15a"
PYBIND11_COMMIT="85198b598fd4e369dd9d766598713ee2d9c8d0b3"
DDS_COMMIT="091ea94358a4016d4fb6069dea5c452cdc98d0bd"
ABSEIL_REF="20250814.1"
JSON_COMMIT="9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03"
PYBIND11_JSON_COMMIT="d0bf434be9d287d73a963ff28745542daf02c08f"
PYBIND11_ABSEIL_COMMIT="73992b54e7d40dfada5e5cd998cad60917f0b3d1"

cd "${ROOT}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup requires Linux." >&2
  exit 1
fi

if [[ "${SKIP_APT:-0}" != "1" ]]; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get was not found. Install the prerequisites manually and rerun with SKIP_APT=1." >&2
    exit 1
  fi
  if [[ "$(id -u)" -eq 0 ]]; then
    SUDO=()
  elif command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
  else
    echo "sudo is unavailable. Install prerequisites manually and rerun with SKIP_APT=1." >&2
    exit 1
  fi
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y \
    build-essential clang cmake curl git ninja-build time unzip virtualenv wget \
    python3-dev python3-pip python3-setuptools python3-tk python3-venv python3-wheel
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] <= (3, 13)):
    raise SystemExit(
        f"Python 3.11--3.13 is required; found {sys.version.split()[0]}. "
        "Set PYTHON_BIN to a supported interpreter."
    )
print("Python:", sys.version)
PY

if [[ -e .venv ]]; then
  echo "${ROOT}/.venv already exists. Use a fresh checkout; refusing to reuse it." >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv .venv
PYTHON="${ROOT}/.venv/bin/python"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel

DOWNLOAD_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${DOWNLOAD_DIR}"
}
trap cleanup EXIT

download_archive() {
  local name="$1"
  local upstream_url="$2"
  local target="$3"
  local archive="${DOWNLOAD_DIR}/${name}.zip"
  local unpack="${DOWNLOAD_DIR}/${name}"
  local source_dir

  if [[ -e "${target}" ]]; then
    echo "Refusing to replace existing dependency path: ${target}" >&2
    exit 1
  fi

  echo "Downloading ${name}"
  wget --progress=dot:giga -O "${archive}" "${PROXY_PREFIX}${upstream_url}"
  mkdir -p "${unpack}"
  unzip -q "${archive}" -d "${unpack}"
  source_dir="$(find "${unpack}" -mindepth 1 -maxdepth 1 -type d -print -quit)"
  if [[ -z "${source_dir}" ]]; then
    echo "Archive ${name} did not contain one top-level directory." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${target}")"
  mv "${source_dir}" "${target}"
}

mkdir -p third_party
download_archive \
  open_spiel \
  "https://github.com/google-deepmind/open_spiel/archive/${OPEN_SPIEL_COMMIT}.zip" \
  third_party/open_spiel
printf '%s\n' "${OPEN_SPIEL_COMMIT}" > third_party/open_spiel/.source_commit

download_archive \
  pybind11 \
  "https://github.com/pybind/pybind11/archive/${PYBIND11_COMMIT}.zip" \
  third_party/open_spiel/pybind11
download_archive \
  dds \
  "https://github.com/jblespiau/dds/archive/${DDS_COMMIT}.zip" \
  third_party/open_spiel/open_spiel/games/bridge/double_dummy_solver
download_archive \
  abseil_cpp \
  "https://github.com/abseil/abseil-cpp/archive/refs/tags/${ABSEIL_REF}.zip" \
  third_party/open_spiel/open_spiel/abseil-cpp
download_archive \
  nlohmann_json \
  "https://github.com/nlohmann/json/archive/${JSON_COMMIT}.zip" \
  third_party/open_spiel/open_spiel/json
download_archive \
  pybind11_json \
  "https://github.com/pybind/pybind11_json/archive/${PYBIND11_JSON_COMMIT}.zip" \
  third_party/open_spiel/open_spiel/pybind11_json
download_archive \
  pybind11_abseil \
  "https://github.com/pybind/pybind11_abseil/archive/${PYBIND11_ABSEIL_COMMIT}.zip" \
  third_party/open_spiel/open_spiel/pybind11_abseil

(
  cd third_party/open_spiel
  DEFAULT_OPTIONAL_DEPENDENCY=OFF bash ./install.sh "${PYTHON}"
)

CXX=clang++ "${PYTHON}" -m pip install ./third_party/open_spiel
"${PYTHON}" -m pip install "jax[cuda12]==0.8.1"
"${PYTHON}" -m pip install -r benchmark/connect3/requirements-alpha-zero.txt
"${PYTHON}" -m pip check

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"${PYTHON}" - <<'PY'
import json
import jax
import pyspiel

game = pyspiel.load_game("connect_four(rows=4,columns=4,x_in_row=3)")
payload = {
    "jax_version": jax.__version__,
    "jax_backend": jax.default_backend(),
    "devices": [str(device) for device in jax.devices()],
    "observation_shape": game.observation_tensor_shape(),
    "num_actions": game.num_distinct_actions(),
    "max_game_length": game.max_game_length(),
}
print(json.dumps(payload, indent=2, sort_keys=True))
assert jax.__version__ == "0.8.1", jax.__version__
assert jax.default_backend() == "gpu", jax.devices()
assert tuple(game.observation_tensor_shape()) == (3, 4, 4)
assert game.num_distinct_actions() == 4
assert game.max_game_length() == 16
PY

"${PYTHON}" -m pip freeze > environment.freeze.txt
printf '%s\n' "${OPEN_SPIEL_COMMIT}" > .open_spiel_commit

echo "GPU environment is ready: ${ROOT}/.venv"
