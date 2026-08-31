#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[ERROR] Run ./setup_demo_macos.sh first."
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1

# Change 0 and 1 with --front-camera/--side-camera if macOS assigned other indices.
.venv/bin/python -m gaze_pipeline demo \
  --front-camera 0 --side-camera 1 --show-cameras "$@"
