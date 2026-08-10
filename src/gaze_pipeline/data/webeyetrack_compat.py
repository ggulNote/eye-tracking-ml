"""WebEyeTrack-compatible facial geometry and eye-patch preprocessing.

The formulas and landmark conventions in this module follow the MIT-licensed
WebEyeTrack reference implementation at commit
``14719ad861467c98890058f7c41a94638ae1db2b``.  They are kept independent of
TensorFlow so the data pipeline can create the exact inputs expected by the
published Keras checkpoint while the model backend remains configurable.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

LEFT_EAR_INDICES = (362, 385, 387, 263, 373, 380)
RIGHT_EAR_INDICES = (133, 158, 160, 33, 144, 153)
LEFT_EYE_HORIZONTAL_INDICES = (362, 263)
RIGHT_EYE_HORIZONTAL_INDICES = (33, 133)
LEFT_IRIS_INDICES = (473, 475, 474, 477, 476)
RIGHT_IRIS_INDICES = (468, 470, 469, 472, 471)
LEFTMOST_FACE_INDEX = 356
RIGHTMOST_FACE_INDEX = 127
NOSE_INDEX = 4

WEBEYETRACK_FACE_QUAD_INDICES = (103, 150, 379, 332)
WEBEYETRACK_VERTICAL_CROP_INDICES = (151, 195)


class WebEyeTrackGeometryError(ValueError):
    """Raised when a sample cannot satisfy the WebEyeTrack input contract."""


def _require_landmark_count(landmarks: np.ndarray, maximum_index: int, *, name: str) -> None:
    if landmarks.ndim != 2 or landmarks.shape[1] < 2:
        raise WebEyeTrackGeometryError(f"{name} must have shape [N, >=2]")
    if len(landmarks) <= maximum_index:
        raise WebEyeTrackGeometryError(
            f"{name} requires landmark index {maximum_index}, but only {len(landmarks)} exist"
        )


def compute_ear(
    landmarks_xy: np.ndarray,
    eye: str,
    *,
    indices: Sequence[int] | None = None,
) -> float:
    """Compute the six-point eye-aspect ratio used by WebEyeTrack."""

    points = np.asarray(landmarks_xy, dtype=np.float64)
    resolved_indices = (
        tuple(int(index) for index in indices)
        if indices is not None
        else LEFT_EAR_INDICES
        if str(eye).lower() == "left"
        else RIGHT_EAR_INDICES
    )
    if len(resolved_indices) != 6:
        raise WebEyeTrackGeometryError("EAR requires exactly six landmark indices")
    _require_landmark_count(points, max(resolved_indices), name="EAR landmarks")
    p1, p2, p3, p4, p5, p6 = points[np.asarray(resolved_indices, dtype=int), :2]
    denominator = 2.0 * float(np.linalg.norm(p1 - p4))
    if denominator <= 1e-8:
        raise WebEyeTrackGeometryError(f"{eye} EAR horizontal eyelid distance is zero")
    value = (float(np.linalg.norm(p2 - p6)) + float(np.linalg.norm(p3 - p5))) / denominator
    if not np.isfinite(value):
        raise WebEyeTrackGeometryError(f"{eye} EAR is non-finite")
    return value


def visible_eye_scores(
    landmarks_xy: np.ndarray,
    *,
    presence: np.ndarray | None = None,
) -> dict[str, float]:
    """Score projected eye visibility without assuming a camera mounting side.

    The projected canthus distance is the primary signal.  MediaPipe presence,
    when available, scales the score instead of replacing the geometric cue.
    """

    points = np.asarray(landmarks_xy, dtype=np.float64)
    required = max(*LEFT_EYE_HORIZONTAL_INDICES, *RIGHT_EYE_HORIZONTAL_INDICES)
    _require_landmark_count(points, required, name="eye selection landmarks")
    widths: dict[str, float] = {}
    for eye, indices in {
        "left": LEFT_EYE_HORIZONTAL_INDICES,
        "right": RIGHT_EYE_HORIZONTAL_INDICES,
    }.items():
        widths[eye] = float(np.linalg.norm(points[indices[0], :2] - points[indices[1], :2]))
    maximum_width = max(widths.values())
    if maximum_width <= 1e-8:
        raise WebEyeTrackGeometryError("both projected eye widths are zero")

    scores = {eye: width / maximum_width for eye, width in widths.items()}
    if presence is not None:
        presence_array = np.asarray(presence, dtype=np.float64).reshape(-1)
        # Some FaceLandmarker model bundles leave the optional field at its
        # protobuf default (all zeros). In that case projected geometry remains
        # the only usable visibility signal.
        if np.isfinite(presence_array).any() and float(np.nanmax(presence_array)) > 0:
            for eye, indices in {
                "left": LEFT_EYE_HORIZONTAL_INDICES,
                "right": RIGHT_EYE_HORIZONTAL_INDICES,
            }.items():
                if len(presence_array) > max(indices):
                    scores[eye] *= max(
                        0.0,
                        float(np.nanmean(presence_array[list(indices)])),
                    )
    return scores


def select_eye(
    landmarks_xy: np.ndarray,
    *,
    mode: str = "best_visible",
    target_eye: str | None = None,
    presence: np.ndarray | None = None,
) -> tuple[str, dict[str, float]]:
    """Return ``left``, ``right``, or ``both`` and the visibility scores."""

    normalized_mode = str(mode).lower()
    scores = visible_eye_scores(landmarks_xy, presence=presence)
    if normalized_mode in {"both", "binocular", "dual_eye"}:
        return "both", scores
    if normalized_mode in {"fixed", "left", "right"}:
        selected = normalized_mode if normalized_mode in {"left", "right"} else str(target_eye)
        selected = selected.lower()
        if selected not in {"left", "right"}:
            raise WebEyeTrackGeometryError(
                "fixed eye selection requires target_eye='left' or 'right'"
            )
        return selected, scores
    if normalized_mode in {"best_visible", "auto", "auto_visible"}:
        return max(scores, key=scores.__getitem__), scores
    raise WebEyeTrackGeometryError(
        f"unsupported eye selection mode {mode!r}; use both, fixed, or best_visible"
    )


def webeyetrack_eye_patch(
    image: np.ndarray,
    landmarks_xy: np.ndarray,
    *,
    radial_padding_xy: Sequence[float] = (0.4, 0.2),
    face_crop_size: int = 512,
    output_size_hw: Sequence[int] = (128, 512),
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create the official two-eye WebEyeTrack patch and source transform."""

    import cv2  # type: ignore

    frame = np.asarray(image)
    points = np.asarray(landmarks_xy, dtype=np.float32)
    required = max(*WEBEYETRACK_FACE_QUAD_INDICES, *WEBEYETRACK_VERTICAL_CROP_INDICES, 4)
    _require_landmark_count(points, required, name="WebEyeTrack landmarks")
    if len(radial_padding_xy) != 2 or len(output_size_hw) != 2:
        raise WebEyeTrackGeometryError("padding and output size must each contain two values")
    output_height, output_width = (int(output_size_hw[0]), int(output_size_hw[1]))
    if face_crop_size <= 0 or output_height <= 0 or output_width <= 0:
        raise WebEyeTrackGeometryError("WebEyeTrack crop/output sizes must be positive")

    source_quad = points[np.asarray(WEBEYETRACK_FACE_QUAD_INDICES, dtype=int), :2].copy()
    center = points[NOSE_INDEX, :2]
    source_quad += np.asarray(radial_padding_xy, dtype=np.float32) * (source_quad - center)
    destination_quad = np.asarray(
        [
            [0, 0],
            [0, face_crop_size],
            [face_crop_size, face_crop_size],
            [face_crop_size, 0],
        ],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(source_quad, destination_quad)
    if homography is None or not np.isfinite(homography).all():
        raise WebEyeTrackGeometryError("WebEyeTrack face homography could not be computed")
    warped_face = cv2.warpPerspective(frame, homography, (face_crop_size, face_crop_size))

    homogeneous = np.concatenate(
        [points[:, :2], np.ones((len(points), 1), dtype=np.float32)], axis=1
    )
    warped_h = homogeneous @ homography.T
    if np.any(np.abs(warped_h[:, 2]) < 1e-8):
        raise WebEyeTrackGeometryError("WebEyeTrack homography maps landmarks to infinity")
    warped_points = warped_h[:, :2] / warped_h[:, 2:3]
    top_index, bottom_index = WEBEYETRACK_VERTICAL_CROP_INDICES
    crop_y0 = int(warped_points[top_index, 1])
    crop_y1 = int(warped_points[bottom_index, 1])
    crop_y0, crop_y1 = sorted((crop_y0, crop_y1))
    crop_y0 = max(0, min(face_crop_size - 1, crop_y0))
    crop_y1 = max(crop_y0 + 1, min(face_crop_size, crop_y1))
    eye_band = warped_face[crop_y0:crop_y1, :]
    if eye_band.size == 0:
        raise WebEyeTrackGeometryError("WebEyeTrack vertical eye band is empty")
    patch = cv2.resize(eye_band, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    scale_x = output_width / face_crop_size
    scale_y = output_height / (crop_y1 - crop_y0)
    crop_resize = np.asarray(
        [[scale_x, 0, 0], [0, scale_y, -crop_y0 * scale_y], [0, 0, 1]],
        dtype=np.float64,
    )
    source_to_patch = crop_resize @ homography
    debug = {
        "source_quad_xy": source_quad.astype(np.float32),
        "face_homography": homography.astype(np.float32),
        "crop_y": np.asarray([crop_y0, crop_y1], dtype=np.int32),
        "warped_face": warped_face,
    }
    return np.ascontiguousarray(patch), source_to_patch, debug


def single_eye_patch(
    image: np.ndarray,
    landmarks_xy: np.ndarray,
    eye: str,
    *,
    output_size_hw: Sequence[int] = (128, 256),
    horizontal_margin_ratio: float = 0.45,
    vertical_margin_ratio: float = 1.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rectify one selected MediaPipe eye into a canonical rectangular patch."""

    import cv2  # type: ignore

    normalized_eye = str(eye).lower()
    if normalized_eye not in {"left", "right"}:
        raise WebEyeTrackGeometryError("single-eye patch requires eye='left' or 'right'")
    points = np.asarray(landmarks_xy, dtype=np.float32)
    ear_indices = LEFT_EAR_INDICES if normalized_eye == "left" else RIGHT_EAR_INDICES
    horizontal = (
        LEFT_EYE_HORIZONTAL_INDICES if normalized_eye == "left" else RIGHT_EYE_HORIZONTAL_INDICES
    )
    _require_landmark_count(points, max(*ear_indices, *horizontal), name="single-eye landmarks")
    p0, p1 = points[np.asarray(horizontal, dtype=int), :2]
    center = (p0 + p1) / 2.0
    axis_x = p1 - p0
    width = float(np.linalg.norm(axis_x))
    if width <= 1e-6:
        raise WebEyeTrackGeometryError("selected eye has zero projected width")
    axis_x /= width
    axis_y = np.asarray([-axis_x[1], axis_x[0]], dtype=np.float32)

    eyelid = points[np.asarray(ear_indices, dtype=int), :2]
    vertical_extent = float(np.ptp((eyelid - center) @ axis_y))
    half_width = width * (0.5 + float(horizontal_margin_ratio))
    half_height = max(width * 0.20, vertical_extent * (0.5 + float(vertical_margin_ratio)))
    source_quad = np.asarray(
        [
            center - axis_x * half_width - axis_y * half_height,
            center + axis_x * half_width - axis_y * half_height,
            center + axis_x * half_width + axis_y * half_height,
            center - axis_x * half_width + axis_y * half_height,
        ],
        dtype=np.float32,
    )
    output_height, output_width = (int(output_size_hw[0]), int(output_size_hw[1]))
    destination = np.asarray(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source_quad, destination)
    if not np.isfinite(matrix).all():
        raise WebEyeTrackGeometryError("selected-eye homography is non-finite")
    patch = cv2.warpPerspective(
        np.asarray(image), matrix, (output_width, output_height), borderValue=(0, 0, 0)
    )
    return np.ascontiguousarray(patch), matrix, source_quad


def rotation_matrix_to_euler_degrees(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise WebEyeTrackGeometryError("rotation matrix must be 3x3")
    value = float(np.clip(-matrix[2, 0], -1.0, 1.0))
    pitch = np.arcsin(value)
    yaw = np.arctan2(matrix[2, 1], matrix[2, 2])
    roll = np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.degrees([pitch, yaw, roll]).astype(np.float32)


def _euler_degrees_to_rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    pitch_r, yaw_r, roll_r = np.radians([pitch, yaw, roll])
    rx = np.asarray(
        [[1, 0, 0], [0, np.cos(pitch_r), -np.sin(pitch_r)], [0, np.sin(pitch_r), np.cos(pitch_r)]]
    )
    ry = np.asarray(
        [[np.cos(yaw_r), 0, np.sin(yaw_r)], [0, 1, 0], [-np.sin(yaw_r), 0, np.cos(yaw_r)]]
    )
    rz = np.asarray(
        [[np.cos(roll_r), -np.sin(roll_r), 0], [np.sin(roll_r), np.cos(roll_r), 0], [0, 0, 1]]
    )
    return rz @ ry @ rx


def head_vector_from_face_rt(face_rt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the public BlazeGaze head vector and mapped Euler angles."""

    matrix = np.asarray(face_rt, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise WebEyeTrackGeometryError(f"MediaPipe face transform must be 4x4, got {matrix.shape}")
    pitch, yaw, roll = rotation_matrix_to_euler_degrees(matrix[:3, :3])
    mapped = np.asarray([-yaw, pitch, roll], dtype=np.float64)
    pitch_r, yaw_r, roll_r = np.radians(mapped)
    vector = np.asarray(
        [
            np.cos(pitch_r) * np.sin(yaw_r),
            np.sin(pitch_r),
            -np.cos(pitch_r) * np.cos(yaw_r),
        ],
        dtype=np.float64,
    )
    cos_r, sin_r = np.cos(roll_r), np.sin(roll_r)
    vector = np.asarray([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]]) @ vector
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8 or not np.isfinite(vector).all():
        raise WebEyeTrackGeometryError("computed head vector is invalid")
    return (vector / norm).astype(np.float32), mapped.astype(np.float32)


def _perspective_matrix(aspect_ratio: float) -> np.ndarray:
    vertical_fov_degrees = 60.0
    near, far = 1.0, 10000.0
    f = 1.0 / np.tan(np.radians(vertical_fov_degrees) / 2.0)
    denominator = 1.0 / (near - far)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = f / aspect_ratio
    matrix[1, 1] = f
    matrix[2, 2] = (near + far) * denominator
    matrix[2, 3] = -1.0
    matrix[3, 2] = 2.0 * far * near * denominator
    return matrix


def _uvz_to_relative_xyz(perspective: np.ndarray, uvz: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(perspective)
    converted = []
    for u, v, z_relative in np.asarray(uvz, dtype=np.float64):
        ndc = np.asarray([2 * u - 1, 1 - 2 * v, -1.0, 1.0])
        world_h = inverse @ ndc
        world = world_h[:3] / world_h[3]
        converted.append([-world[0], world[1], z_relative])
    return np.asarray(converted, dtype=np.float64)


def _project(points_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    projected = np.asarray(points_3d, dtype=np.float64) @ np.asarray(camera_matrix).T
    depth = projected[:, 2:3]
    depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    # The reference implementation rounds each projected landmark to integer
    # pixels before its translation/depth refinement.
    return (projected[:, :2] / depth).astype(np.int32)


def estimate_metric_face_origin(
    landmarks_xyz_normalized: np.ndarray,
    face_rt: np.ndarray,
    image_size_hw: Sequence[int],
    *,
    iris_diameter_cm: float = 1.2,
    iris_mode: str = "both",
    initial_depth_cm: float = 60.0,
    max_iterations: int = 10,
    max_depth_step_cm: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Reconstruct metric face pose following WebEyeTrack radial refinement.

    Returns ``face_origin_3d_cm``, ``metric_transform``, ``metric_face``, and
    the iris-derived face-width estimate in centimetres.
    """

    landmarks = np.asarray(landmarks_xyz_normalized, dtype=np.float64)
    required = max(
        *LEFT_IRIS_INDICES,
        *RIGHT_IRIS_INDICES,
        *LEFT_EYE_HORIZONTAL_INDICES,
        *RIGHT_EYE_HORIZONTAL_INDICES,
        LEFTMOST_FACE_INDEX,
        RIGHTMOST_FACE_INDEX,
        NOSE_INDEX,
    )
    _require_landmark_count(landmarks, required, name="metric MediaPipe landmarks")
    height, width = (int(image_size_hw[0]), int(image_size_hw[1]))
    if height <= 0 or width <= 0:
        raise WebEyeTrackGeometryError("metric pose requires a positive image size")
    normalized_xy = landmarks[:, :2]
    xy_pixels = normalized_xy * np.asarray([width, height], dtype=np.float64)
    normalized_iris_mode = str(iris_mode).lower()
    if normalized_iris_mode in {"both", "binocular", "all"}:
        iris_groups = (LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES)
    elif normalized_iris_mode == "left":
        iris_groups = (LEFT_IRIS_INDICES,)
    elif normalized_iris_mode == "right":
        iris_groups = (RIGHT_IRIS_INDICES,)
    else:
        raise WebEyeTrackGeometryError("iris_mode must be both, left, or right")
    iris_widths = []
    for indices in iris_groups:
        iris_widths.append(
            float(np.linalg.norm(normalized_xy[indices[4]] - normalized_xy[indices[2]]))
        )
    face_width_px = float(
        np.linalg.norm(normalized_xy[LEFTMOST_FACE_INDEX] - normalized_xy[RIGHTMOST_FACE_INDEX])
    )
    mean_iris_width = float(np.mean(iris_widths))
    if face_width_px <= 1e-6 or mean_iris_width <= 1e-6:
        raise WebEyeTrackGeometryError("iris/face scale is degenerate")
    face_width_cm = float(iris_diameter_cm) / (mean_iris_width / face_width_px)

    perspective = _perspective_matrix(width / height)
    relative = _uvz_to_relative_xyz(perspective, landmarks[:, :3])
    relative -= relative[NOSE_INDEX]
    relative *= np.asarray([-1.0, -1.0, 1.0])
    normalized_face_width = float(
        np.linalg.norm(relative[LEFTMOST_FACE_INDEX] - relative[RIGHTMOST_FACE_INDEX])
    )
    if normalized_face_width <= 1e-8:
        raise WebEyeTrackGeometryError("normalized face mesh has zero width")
    canonical = relative / normalized_face_width * face_width_cm

    raw_rotation = np.asarray(face_rt, dtype=np.float64)[:3, :3].copy()
    pitch, yaw, roll = rotation_matrix_to_euler_degrees(np.linalg.inv(raw_rotation))
    adjusted_rotation = _euler_degrees_to_rotation(-yaw, pitch, roll)
    canonical = canonical @ np.linalg.inv(adjusted_rotation).T
    scale = float(np.linalg.norm(adjusted_rotation, axis=0).mean())
    adjusted_rotation /= max(scale, 1e-8)

    camera_matrix = np.asarray(
        [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = adjusted_rotation
    transform[:3, 3] = [0, 0, float(initial_depth_cm)]
    initial_camera_points = (transform @ np.c_[canonical, np.ones(len(canonical))].T).T[:, :3]
    initial_projection = _project(initial_camera_points, camera_matrix)
    detected = xy_pixels
    shift_2d = detected[NOSE_INDEX] - initial_projection[NOSE_INDEX]
    transform[:3, 3] += np.asarray(
        [
            shift_2d[0] * initial_depth_cm / camera_matrix[0, 0],
            shift_2d[1] * initial_depth_cm / camera_matrix[1, 1],
            0,
        ]
    )
    first_aligned = transform.copy()

    for _ in range(int(max_iterations)):
        camera_points = (transform @ np.c_[canonical, np.ones(len(canonical))].T).T[:, :3]
        projected = _project(camera_points, camera_matrix)
        detected_center = detected.mean(axis=0)
        total = 0.0
        for projected_point, detected_point in zip(projected, detected, strict=True):
            delta = detected_point - projected_point
            sign = -1.0 if float(np.dot(delta, detected_center - projected_point)) < 0 else 1.0
            total += sign * float(np.linalg.norm(delta))
        depth_delta = float(
            np.clip(
                0.1 * total / len(projected),
                -max_depth_step_cm,
                max_depth_step_cm,
            )
        )
        new_depth = float(transform[2, 3] + depth_delta)
        if abs(depth_delta) < 0.25:
            break
        ratio = new_depth / float(initial_depth_cm)
        transform[0, 3] = first_aligned[0, 3] * ratio
        transform[1, 3] = first_aligned[1, 3] * ratio
        transform[2, 3] = new_depth

    metric_face = (transform @ np.c_[canonical, np.ones(len(canonical))].T).T[:, :3]
    left_origin = metric_face[list(LEFT_EYE_HORIZONTAL_INDICES)].mean(axis=0)
    right_origin = metric_face[list(RIGHT_EYE_HORIZONTAL_INDICES)].mean(axis=0)
    face_origin = (left_origin + right_origin) / 2.0
    if not np.isfinite(face_origin).all() or not np.isfinite(transform).all():
        raise WebEyeTrackGeometryError("metric face reconstruction produced non-finite values")
    return (
        face_origin.astype(np.float32),
        transform.astype(np.float32),
        metric_face.astype(np.float32),
        face_width_cm,
    )


class MediaPipeFaceLandmarkerDetector:
    """Lazy, DataLoader-safe wrapper around MediaPipe FaceLandmarker Tasks."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        raw_path = self.config.get("model_asset_path", self.config.get("model_path"))
        if not raw_path:
            raise ValueError(
                "MediaPipe FaceLandmarker requires model_asset_path; run "
                "'make webeyetrack-assets' or set MEDIAPIPE_FACE_MODEL"
            )
        self.model_asset_path = Path(str(raw_path)).expanduser()
        self._landmarker: Any = None
        self._landmarker_pid: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "model_asset_path": self.model_asset_path,
            "_landmarker": None,
            "_landmarker_pid": None,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.config = dict(state["config"])
        self.model_asset_path = Path(state["model_asset_path"])
        self._landmarker = None
        self._landmarker_pid = None

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
        self._landmarker = None
        self._landmarker_pid = None

    def _get_landmarker(self) -> Any:
        process_id = os.getpid()
        if self._landmarker is not None and self._landmarker_pid == process_id:
            return self._landmarker
        if not self.model_asset_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe FaceLandmarker model does not exist: {self.model_asset_path}"
            )
        try:
            from mediapipe.tasks import python  # type: ignore
            from mediapipe.tasks.python import vision  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "MediaPipe FaceLandmarker requires the 'mediapipe' dependency"
            ) from exc

        options = vision.FaceLandmarkerOptions(
            # Request CPU model inference. Some MediaPipe platform builds still
            # initialize a graphics service for image preprocessing, so headless
            # deployment must be covered by an environment-specific smoke test.
            base_options=python.BaseOptions(
                model_asset_path=str(self.model_asset_path),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            output_face_blendshapes=bool(self.config.get("output_face_blendshapes", True)),
            output_facial_transformation_matrixes=True,
            num_faces=int(self.config.get("num_faces", 1)),
            min_face_detection_confidence=float(
                self.config.get("min_face_detection_confidence", 0.5)
            ),
            min_face_presence_confidence=float(
                self.config.get("min_face_presence_confidence", 0.5)
            ),
            min_tracking_confidence=float(self.config.get("min_tracking_confidence", 0.5)),
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._landmarker_pid = process_id
        return self._landmarker

    def detect(self, image: np.ndarray, _view: str) -> Mapping[str, Any]:
        frame = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise WebEyeTrackGeometryError("MediaPipe detector requires RGB HWC uint8 image")
        import mediapipe as mp  # type: ignore

        landmarker = self._get_landmarker()
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
        mirrored_retry = False
        if not result.face_landmarks and bool(self.config.get("mirror_retry", False)):
            mirrored_frame = np.ascontiguousarray(frame[:, ::-1])
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=mirrored_frame)
            )
            mirrored_retry = bool(result.face_landmarks)
        if not result.face_landmarks:
            raise WebEyeTrackGeometryError("MediaPipe FaceLandmarker found no face")
        proto = result.face_landmarks[0]
        normalized = np.asarray([[lm.x, lm.y, lm.z] for lm in proto], dtype=np.float32)
        presence = np.asarray([lm.presence for lm in proto], dtype=np.float32)
        visibility = np.asarray([lm.visibility for lm in proto], dtype=np.float32)
        height, width = frame.shape[:2]
        landmarks_xy = normalized[:, :2] * np.asarray([width, height], dtype=np.float32)
        matrices = result.facial_transformation_matrixes
        if not matrices:
            raise WebEyeTrackGeometryError("MediaPipe returned no facial transformation matrix")
        face_rt = np.asarray(matrices[0], dtype=np.float32).reshape(4, 4)
        if mirrored_retry:
            normalized[:, 0] = 1.0 - normalized[:, 0]
            mirror = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32)
            face_rt = mirror @ face_rt @ mirror
            landmarks_xy = normalized[:, :2] * np.asarray([width, height], dtype=np.float32)
        blendshapes = (
            np.asarray([item.score for item in result.face_blendshapes[0]], dtype=np.float32)
            if result.face_blendshapes
            else np.empty(0, dtype=np.float32)
        )
        minimum = landmarks_xy.min(axis=0)
        maximum = landmarks_xy.max(axis=0)
        return {
            "landmarks_xy": landmarks_xy,
            "landmarks_xyz_normalized": normalized,
            "landmark_presence": presence,
            "landmark_visibility": visibility,
            "face_rt": face_rt,
            "face_blendshapes": blendshapes,
            "landmark_kind": f"mediapipe_face_landmarker_{len(landmarks_xy)}",
            "face_bbox_xywh": np.asarray(
                [minimum[0], minimum[1], *(maximum - minimum)], dtype=np.float32
            ),
            "mirrored_retry": mirrored_retry,
        }
