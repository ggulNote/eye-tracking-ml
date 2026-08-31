#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MAC_ARCH="$(uname -m)"
if [[ "$MAC_ARCH" != "arm64" ]]; then
  echo "[ERROR] This pinned demo environment requires an Apple Silicon Mac (arm64)."
  echo "        On Apple Silicon, close any Rosetta/x86 terminal and use a native terminal."
  echo "        Intel Mac needs a separately tested legacy dependency set."
  exit 1
fi

MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
if [[ ! "$MACOS_MAJOR" =~ ^[0-9]+$ ]] || (( MACOS_MAJOR < 14 )); then
  echo "[ERROR] macOS 14 Sonoma or newer is required by the pinned PyTorch wheel."
  exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  DEMO_PYTHON="$PYTHON_BIN"
elif command -v python3.12 >/dev/null 2>&1; then
  DEMO_PYTHON="python3.12"
elif command -v python3 >/dev/null 2>&1; then
  DEMO_PYTHON="python3"
else
  echo "[ERROR] Python was not found. Install native arm64 Python 3.12 first."
  exit 1
fi

PYTHON_VERSION="$($DEMO_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.12" ]]; then
  echo "[ERROR] Python 3.12 is required, but $DEMO_PYTHON is Python $PYTHON_VERSION."
  echo "        Install it with: brew install python@3.12"
  exit 1
fi

PYTHON_ARCH="$($DEMO_PYTHON -c 'import platform; print(platform.machine())')"
if [[ "$PYTHON_ARCH" != "arm64" ]]; then
  echo "[ERROR] $DEMO_PYTHON is a $PYTHON_ARCH build. Install native arm64 Python 3.12."
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  "$DEMO_PYTHON" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip "setuptools>=77" wheel
python -m pip install -r requirements-demo.txt
python -m pip install -e . --no-deps --no-build-isolation

echo
echo "Setup complete. Next run ./verify_demo_macos.sh"
