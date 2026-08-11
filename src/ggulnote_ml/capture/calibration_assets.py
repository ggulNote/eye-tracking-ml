from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable

from scipy.io import loadmat


CAMERA_REQUIRED = ("cameraMatrix", "distCoeffs", "retval")
MONITOR_REQUIRED = ("rvects", "tvecs")
SCREEN_REQUIRED = ("width_pixel", "height_pixel", "width_mm", "height_mm")
STEREO_REQUIRED = (
    "R_iphone_to_webcam",
    "T_iphone_to_webcam",
    "cameraMatrix_webcam",
    "distCoeffs_webcam",
    "cameraMatrix_iphone",
    "distCoeffs_iphone",
    "stereo_reprojection_error",
)

CALIBRATION_ASSET_PATHS = (
    Path("screenSize.mat"),
    Path("webcam/Camera.mat"),
    Path("webcam/monitorPose.mat"),
    Path("phonecam/Camera.mat"),
    Path("phonecam/monitorPose.mat"),
    Path("stereoCalibration.mat"),
)


def _inspect_mat(path: Path, required: Iterable[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "exists": path.is_file(), "valid": False}
    if not path.is_file():
        result["missing_variables"] = list(required)
        return result
    values = loadmat(path)
    missing = [name for name in required if name not in values]
    result["missing_variables"] = missing
    result["available_variables"] = sorted(name for name in values if not name.startswith("__"))
    result["valid"] = not missing
    return result


def inspect_calibration_assets(calibration_directory: Path) -> Dict[str, Any]:
    """Validate only the variables consumed by the downstream WebEyeTrack-style loader."""
    cameras = {}
    for directory_name, key in (("webcam", "webcam_front"), ("phonecam", "iphone_left")):
        camera_directory = calibration_directory / directory_name
        cameras[key] = {
            "camera": _inspect_mat(camera_directory / "Camera.mat", CAMERA_REQUIRED),
            "monitor_pose": _inspect_mat(camera_directory / "monitorPose.mat", MONITOR_REQUIRED),
        }
    screen = _inspect_mat(calibration_directory / "screenSize.mat", SCREEN_REQUIRED)
    stereo_path = calibration_directory / "stereoCalibration.mat"
    stereo = _inspect_mat(stereo_path, STEREO_REQUIRED)
    stereo["optional"] = True
    required_valid = screen["valid"] and all(
        item[asset]["valid"] for item in cameras.values() for asset in ("camera", "monitor_pose")
    )
    return {
        "required_assets_valid": bool(required_valid),
        "screen_size": screen,
        "cameras": cameras,
        "stereo": stereo,
    }


def copy_calibration_assets(source: Path, destination: Path) -> None:
    """Copy only final MAT assets from one fixed rig into a new participant folder."""

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        return
    for relative_path in CALIBRATION_ASSET_PATHS:
        source_path = source / relative_path
        if not source_path.is_file():
            continue
        destination_path = destination / relative_path
        if destination_path.exists():
            raise FileExistsError(
                "Participant calibration asset already exists: %s" % destination_path
            )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
