"""Verify the Python runtime and direct pipeline dependencies."""

from __future__ import annotations

import importlib
import sys
from importlib import metadata
from pathlib import Path

DEPENDENCIES = (
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("cv2", "opencv-contrib-python"),
    ("mediapipe", "mediapipe"),
    ("PIL", "Pillow"),
    ("yaml", "PyYAML"),
    ("mlflow", "mlflow"),
    ("psutil", "psutil"),
    ("gaze_pipeline", "gaze-pipeline"),
)


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "<not-installed>"


def main() -> int:
    print(f"[OK] python executable: {Path(sys.executable).resolve()}")
    print(f"[OK] python version: {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 12):
        print("[FAIL] Python 3.12.x가 필요합니다.", file=sys.stderr)
        return 2

    failures: list[str] = []
    for module_name, distribution in DEPENDENCIES:
        version = _version(distribution)
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - environment-specific diagnostics
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
            print(f"[FAIL] import {module_name} ({version})", file=sys.stderr)
        else:
            print(f"[OK] import {module_name}: {version}")

    if failures:
        print("[FAIL] runtime dependency import errors:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2
    print("[OK] Python runtime and direct dependencies are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
