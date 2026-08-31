"""Download and verify the exact WebEyeTrack model assets used by this project."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

WEBEYETRACK_COMMIT = "14719ad861467c98890058f7c41a94638ae1db2b"
RAW_ROOT = f"https://raw.githubusercontent.com/RedForestAI/WebEyeTrack/{WEBEYETRACK_COMMIT}"
OPENCV_ZOO_COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
OPENCV_ZOO_RAW_ROOT = f"https://github.com/opencv/opencv_zoo/raw/{OPENCV_ZOO_COMMIT}"


@dataclass(frozen=True)
class Asset:
    filename: str
    sha256: str
    url: str


ASSETS = (
    Asset(
        "face_landmarker_v2_with_blendshapes.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        f"{RAW_ROOT}/python/webeyetrack/model_weights/face_landmarker_v2_with_blendshapes.task",
    ),
    Asset(
        "blazegaze_mpiifacegaze.keras",
        "5b011cfe82466896e27b1ac3e18130117cafbc02dbc964a1ad7315f62005cc05",
        f"{RAW_ROOT}/python/webeyetrack/model_weights/blazegaze_mpiifacegaze.keras",
    ),
    Asset(
        "face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        (f"{OPENCV_ZOO_RAW_ROOT}/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(asset: Asset, path: Path) -> bool:
    if not path.is_file():
        print(f"[MISSING] {path}")
        return False
    actual = _sha256(path)
    if actual != asset.sha256:
        print(f"[BAD SHA256] {path}")
        print(f"  expected: {asset.sha256}")
        print(f"  actual:   {actual}")
        return False
    print(f"[OK] {path} sha256={actual}")
    return True


def _download(asset: Asset, output_dir: Path) -> None:
    destination = output_dir / asset.filename
    if _check(asset, destination):
        return
    print(f"[DOWNLOAD] {asset.url}")
    with tempfile.NamedTemporaryFile(
        dir=output_dir,
        prefix=f".{asset.filename}.",
        suffix=".download",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        urllib.request.urlretrieve(asset.url, temporary_path)
        actual = _sha256(temporary_path)
        if actual != asset.sha256:
            raise RuntimeError(
                f"downloaded {asset.filename} SHA-256 mismatch: "
                f"expected {asset.sha256}, got {actual}"
            )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    if not _check(asset, destination):
        raise RuntimeError(f"asset verification failed after download: {destination}")


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        checks = [_check(asset, output_dir / asset.filename) for asset in ASSETS]
        return 0 if all(checks) else 1
    for asset in ASSETS:
        _download(asset, output_dir)
    print(f"[OK] WebEyeTrack commit: {WEBEYETRACK_COMMIT}")
    print(f"[OK] OpenCV Zoo commit: {OPENCV_ZOO_COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
