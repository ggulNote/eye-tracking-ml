from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.io import savemat


@dataclass(frozen=True)
class CheckerboardSpec:
    """Physical checkerboard contract used by OpenCV calibration."""

    inner_corners_x: int
    inner_corners_y: int
    square_size_mm: float
    page_width_mm: float
    page_height_mm: float

    @property
    def pattern_size(self) -> Tuple[int, int]:
        return self.inner_corners_x, self.inner_corners_y

    @property
    def squares_x(self) -> int:
        return self.inner_corners_x + 1

    @property
    def squares_y(self) -> int:
        return self.inner_corners_y + 1

    @property
    def board_width_mm(self) -> float:
        return self.squares_x * self.square_size_mm

    @property
    def board_height_mm(self) -> float:
        return self.squares_y * self.square_size_mm

    def validate(self) -> None:
        if self.inner_corners_x < 2 or self.inner_corners_y < 2:
            raise ValueError("Checkerboard must have at least 2x2 inner corners.")
        if self.square_size_mm <= 0:
            raise ValueError("checkerboard.square_size_mm must be positive.")
        if self.page_width_mm <= 0 or self.page_height_mm <= 0:
            raise ValueError("Checkerboard page dimensions must be positive.")
        if self.board_width_mm > self.page_width_mm or self.board_height_mm > self.page_height_mm:
            raise ValueError(
                "Checkerboard %.1fx%.1f mm does not fit on the %.1fx%.1f mm page."
                % (
                    self.board_width_mm,
                    self.board_height_mm,
                    self.page_width_mm,
                    self.page_height_mm,
                )
            )


@dataclass(frozen=True)
class ScreenSpec:
    width_pixel: int
    height_pixel: int
    width_mm: float
    height_mm: float

    def validate(self) -> None:
        if self.width_pixel <= 0 or self.height_pixel <= 0:
            raise ValueError("Screen pixel dimensions must be positive.")
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise ValueError("Screen physical dimensions must be positive.")
        pixel_ratio = self.width_pixel / self.height_pixel
        physical_ratio = self.width_mm / self.height_mm
        relative_difference = abs(pixel_ratio - physical_ratio) / physical_ratio
        if relative_difference > 0.02:
            raise ValueError(
                "Screen pixel and physical aspect ratios differ by more than 2%%: "
                "%.4f vs %.4f." % (pixel_ratio, physical_ratio)
            )


@dataclass(frozen=True)
class IntrinsicCalibrationResult:
    rms_error_px: float
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rotation_vectors: Tuple[np.ndarray, ...]
    translation_vectors: Tuple[np.ndarray, ...]
    per_view_errors_px: Tuple[float, ...]
    image_size: Tuple[int, int]

    def validate(self) -> None:
        if not np.isfinite(self.rms_error_px) or self.rms_error_px < 0:
            raise ValueError("Camera calibration RMS error is invalid.")
        if self.camera_matrix.shape != (3, 3):
            raise ValueError("cameraMatrix must have shape (3, 3).")
        if self.distortion_coefficients.size < 4:
            raise ValueError("distCoeffs must contain at least four coefficients.")
        if len(self.rotation_vectors) != len(self.translation_vectors):
            raise ValueError("Camera calibration pose counts do not match.")
        if len(self.per_view_errors_px) != len(self.rotation_vectors):
            raise ValueError("Per-view error count does not match pose count.")


def checkerboard_object_points(spec: CheckerboardSpec) -> np.ndarray:
    """Return (N, 3) millimetre coordinates for checkerboard inner corners."""

    spec.validate()
    points = np.zeros((spec.inner_corners_x * spec.inner_corners_y, 3), np.float32)
    grid = np.mgrid[0 : spec.inner_corners_x, 0 : spec.inner_corners_y].T.reshape(-1, 2)
    points[:, :2] = grid.astype(np.float32) * np.float32(spec.square_size_mm)
    return points


def generate_checkerboard_svg(path: Path, spec: CheckerboardSpec) -> Path:
    """Create a vector A4 checkerboard whose dimensions are expressed in millimetres."""

    spec.validate()
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError("Checkerboard output already exists: %s" % resolved)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    offset_x = (spec.page_width_mm - spec.board_width_mm) / 2.0
    offset_y = (spec.page_height_mm - spec.board_height_mm) / 2.0
    rectangles = []
    for row in range(spec.squares_y):
        for column in range(spec.squares_x):
            if (row + column) % 2 == 0:
                rectangles.append(
                    '  <rect x="%.4f" y="%.4f" width="%.4f" height="%.4f" fill="#000"/>'
                    % (
                        offset_x + column * spec.square_size_mm,
                        offset_y + row * spec.square_size_mm,
                        spec.square_size_mm,
                        spec.square_size_mm,
                    )
                )
    svg = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" width="%.4fmm" height="%.4fmm" '
            'viewBox="0 0 %.4f %.4f">'
            % (spec.page_width_mm, spec.page_height_mm, spec.page_width_mm, spec.page_height_mm),
            '  <rect width="100%" height="100%" fill="#fff"/>',
            *rectangles,
            "</svg>",
            "",
        ]
    )
    resolved.write_text(svg, encoding="utf-8")
    return resolved


def write_screen_size_mat(
    path: Path, spec: ScreenSpec, verification_source: str = "unspecified"
) -> Path:
    """Write the MPIIGaze-compatible screen-size variables without overwriting."""

    spec.validate()
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError("Screen calibration already exists: %s" % resolved)
    if not verification_source.strip():
        raise ValueError("Screen verification source cannot be empty.")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        resolved,
        {
            "width_pixel": np.array([[spec.width_pixel]], dtype=np.int32),
            "height_pixel": np.array([[spec.height_pixel]], dtype=np.int32),
            "width_mm": np.array([[spec.width_mm]], dtype=np.float64),
            "height_mm": np.array([[spec.height_mm]], dtype=np.float64),
            "verification_source": np.array([verification_source]),
        },
    )
    return resolved


def find_checkerboard_corners(
    frame: np.ndarray, spec: CheckerboardSpec
) -> Tuple[bool, Optional[np.ndarray]]:
    """Detect and refine checkerboard corners in one BGR camera frame."""

    spec.validate()
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Calibration frame must have shape (H, W, 3) BGR.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, spec.pattern_size, flags)
    if not found or corners is None:
        return False, None
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def calibrate_camera_intrinsics(
    image_points: Sequence[np.ndarray],
    image_size: Tuple[int, int],
    spec: CheckerboardSpec,
) -> IntrinsicCalibrationResult:
    """Estimate camera intrinsics from matched checkerboard observations."""

    if len(image_points) < 3:
        raise ValueError("At least three checkerboard observations are required.")
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("Calibration image size must be positive.")
    expected_points = spec.inner_corners_x * spec.inner_corners_y
    normalized_points = []
    for index, points in enumerate(image_points):
        array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if array.shape[0] != expected_points:
            raise ValueError(
                "Observation %d has %d corners; expected %d."
                % (index, array.shape[0], expected_points)
            )
        normalized_points.append(array)
    object_template = checkerboard_object_points(spec)
    object_points = [object_template.copy() for _ in normalized_points]
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        normalized_points,
        image_size,
        None,
        None,
    )
    per_view_errors = []
    for object_view, observed, rvec, tvec in zip(
        object_points, normalized_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(
            object_view,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        error = cv2.norm(observed, projected, cv2.NORM_L2) / np.sqrt(len(projected))
        per_view_errors.append(float(error))
    result = IntrinsicCalibrationResult(
        rms_error_px=float(rms),
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        distortion_coefficients=np.asarray(distortion, dtype=np.float64),
        rotation_vectors=tuple(np.asarray(value, dtype=np.float64).reshape(3) for value in rvecs),
        translation_vectors=tuple(np.asarray(value, dtype=np.float64).reshape(3) for value in tvecs),
        per_view_errors_px=tuple(per_view_errors),
        image_size=image_size,
    )
    result.validate()
    return result


def write_camera_mat(
    path: Path,
    result: IntrinsicCalibrationResult,
    spec: CheckerboardSpec,
    max_rms_error_px: float,
) -> Path:
    """Persist a quality-gated MPIIGaze-compatible Camera.mat file."""

    result.validate()
    spec.validate()
    if max_rms_error_px <= 0:
        raise ValueError("max_rms_error_px must be positive.")
    if result.rms_error_px > max_rms_error_px:
        raise ValueError(
            "Calibration RMS %.4f px exceeds configured limit %.4f px."
            % (result.rms_error_px, max_rms_error_px)
        )
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError("Camera calibration already exists: %s" % resolved)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        resolved,
        {
            "cameraMatrix": result.camera_matrix,
            "distCoeffs": result.distortion_coefficients,
            "retval": np.array([[result.rms_error_px]], dtype=np.float64),
            "rvecs": np.asarray(result.rotation_vectors, dtype=np.float64),
            "tvecs": np.asarray(result.translation_vectors, dtype=np.float64),
            "image_width": np.array([[result.image_size[0]]], dtype=np.int32),
            "image_height": np.array([[result.image_size[1]]], dtype=np.int32),
            "checkerboard_inner_corners": np.array(
                [[spec.inner_corners_x, spec.inner_corners_y]], dtype=np.int32
            ),
            "checkerboard_square_mm": np.array([[spec.square_size_mm]], dtype=np.float64),
            "per_view_errors_px": np.asarray([result.per_view_errors_px], dtype=np.float64),
        },
    )
    return resolved


def write_calibration_summary(
    path: Path,
    camera_role: str,
    result: IntrinsicCalibrationResult,
    capture_paths: Iterable[Path],
) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError("Calibration summary already exists: %s" % resolved)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "camera_role": camera_role,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_size": {"width": result.image_size[0], "height": result.image_size[1]},
        "capture_count": len(result.rotation_vectors),
        "rms_error_px": result.rms_error_px,
        "per_view_errors_px": list(result.per_view_errors_px),
        "captures": [str(value) for value in capture_paths],
    }
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved
