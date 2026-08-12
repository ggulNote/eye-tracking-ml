from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from scipy.io import loadmat


CAMERA_REQUIRED = (
    "cameraMatrix",
    "distCoeffs",
    "retval",
    "image_width",
    "image_height",
)
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

INTRINSICS_ASSET_PATHS = (
    Path("webcam/Camera.mat"),
    Path("phonecam/Camera.mat"),
)
FULL_CALIBRATION_ASSET_PATHS = (
    Path("screenSize.mat"),
    *INTRINSICS_ASSET_PATHS,
    Path("webcam/monitorPose.mat"),
    Path("phonecam/monitorPose.mat"),
    Path("stereoCalibration.mat"),
)


@dataclass(frozen=True)
class CameraIntrinsics:
    path: Path
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rms_error_px: float
    image_width: int
    image_height: int


def _scalar(values: Dict[str, Any], name: str, path: Path) -> float:
    try:
        value = float(np.asarray(values[name]).reshape(-1)[0])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("%s has invalid %s." % (path, name)) from error
    if not np.isfinite(value):
        raise ValueError("%s has non-finite %s." % (path, name))
    return value


def load_camera_intrinsics(path: Path) -> CameraIntrinsics:
    """Load and validate one generated Camera.mat before frame processing."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Camera calibration does not exist: %s" % resolved)
    values = loadmat(resolved)
    missing = [name for name in CAMERA_REQUIRED if name not in values]
    if missing:
        raise ValueError(
            "Camera calibration %s is missing variables: %s."
            % (resolved, ", ".join(missing))
        )
    matrix = np.asarray(values["cameraMatrix"], dtype=np.float64)
    distortion = np.asarray(values["distCoeffs"], dtype=np.float64).reshape(1, -1)
    rms = _scalar(values, "retval", resolved)
    width = int(_scalar(values, "image_width", resolved))
    height = int(_scalar(values, "image_height", resolved))
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("%s cameraMatrix must be finite with shape (3,3)." % resolved)
    if distortion.size < 4 or not np.all(np.isfinite(distortion)):
        raise ValueError("%s distCoeffs must contain at least four finite values." % resolved)
    if rms < 0 or width <= 0 or height <= 0:
        raise ValueError("%s contains invalid RMS or image dimensions." % resolved)
    return CameraIntrinsics(
        path=resolved,
        camera_matrix=matrix,
        distortion_coefficients=distortion,
        rms_error_px=rms,
        image_width=width,
        image_height=height,
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
    camera_intrinsics_valid = all(
        item["camera"]["valid"] for item in cameras.values()
    )
    return {
        "required_assets_valid": bool(required_valid),
        "camera_intrinsics_valid": bool(camera_intrinsics_valid),
        "screen_size": screen,
        "cameras": cameras,
        "stereo": stereo,
    }


def copy_calibration_assets(
    source: Path,
    destination: Path,
    relative_paths: Iterable[Path],
) -> None:
    """Copy immutable final MAT assets into one participant without overwriting."""

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        return
    for relative_path in relative_paths:
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
