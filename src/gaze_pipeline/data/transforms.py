"""Ordered, configuration-driven transforms for static gaze images.

The transforms in this module deliberately keep two coordinate systems apart:

* facial landmarks live in *image* pixels and follow every image-space warp;
* ``target_gaze_xy`` lives in *screen* coordinates and is therefore unchanged by
  crops, letterboxing, and eye-region normalization.

Only a semantic augmentation such as horizontal mirroring changes both the
image and the screen target.  A homogeneous 3x3 matrix mapping source image
pixels to the current representation is accumulated in sample metadata so that
preprocessing can be inspected and reproduced.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance

from .profile_side import ProfileSideGeometryError, preprocess_profile_side
from .webeyetrack_compat import (
    MediaPipeFaceLandmarkerDetector,
    WebEyeTrackGeometryError,
    estimate_metric_face_origin,
    head_vector_from_face_rt,
    select_eye,
    single_eye_patch,
    webeyetrack_eye_patch,
)


class GazeTransformError(RuntimeError):
    """Base error raised by the gaze preprocessing graph."""


class TransformConfigError(GazeTransformError, ValueError):
    """Raised when enabled stages have an invalid or incomplete configuration."""


class TransformDependencyError(GazeTransformError, ImportError):
    """Raised when an optional stage requires an unavailable dependency/input."""


class SampleValidationError(GazeTransformError, ValueError):
    """Raised when an image or annotation violates the configured contract."""


class SampleDroppedError(GazeTransformError):
    """Signals a configured ``on_failure: drop`` policy.

    A normal PyTorch ``Dataset`` cannot safely return ``None`` to the default
    collate function.  Callers should filter such rows while preparing a
    manifest, or catch this exception in an explicit filtering pass.
    """


_STAGES = (
    "decode",
    "exif_orientation",
    "validate",
    "face_landmarks",
    "eye_selection",
    "metric_head_pose",
    "eye_region_warp",
    "face_roi",
    "background_mask",
    "resize",
    "normalize",
    "augment",
)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TransformConfigError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _enabled(config: Mapping[str, Any]) -> bool:
    return bool(config.get("enabled", True))


def _as_rgb_triplet(value: Any, *, name: str) -> tuple[int, int, int]:
    if value is None:
        return (0, 0, 0)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 3:
        raise TransformConfigError(f"{name} must contain exactly three RGB values")
    result = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in result):
        raise TransformConfigError(f"{name} values must be in [0, 255]")
    return result


def _as_size_hw(value: Any, *, name: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 2:
        raise TransformConfigError(f"{name} must be [height, width]")
    height, width = (int(value[0]), int(value[1]))
    if height <= 0 or width <= 0:
        raise TransformConfigError(f"{name} values must be positive, got {value!r}")
    return height, width


def _opencv(stage: str) -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only in lean installs
        raise TransformDependencyError(
            f"preprocessing stage '{stage}' requires opencv-python; "
            "install the configured runtime dependencies or disable the stage"
        ) from exc
    return cv2


class OpenCVHaarFaceDetector:
    """Lazy OpenCV Haar face detector for ROI/mask fallback annotations.

    The returned points approximate the detected face hull.  They are suitable
    for a face crop or background mask, but are intentionally marked as
    ``face_bbox_hull`` rather than eye landmarks and cannot drive an eye-region
    homography.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(_mapping(config, name="face_landmarks.opencv_haar"))
        self._classifiers: dict[str, Any] = {}
        self._classifier_pid: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Exclude non-picklable OpenCV classifiers for spawn DataLoaders."""

        return {
            "config": self.config,
            "_classifiers": {},
            "_classifier_pid": None,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.config = dict(state.get("config", {}))
        self._classifiers = {}
        self._classifier_pid = None

    def _classifier(self, view: str) -> Any:
        process_id = os.getpid()
        if self._classifier_pid != process_id:
            self._classifiers = {}
            self._classifier_pid = process_id
        mode = "side" if view == "side" else "front"
        if mode in self._classifiers:
            return self._classifiers[mode]
        cv2 = _opencv("face_landmarks.opencv_haar")
        default_name = (
            "haarcascade_profileface.xml"
            if mode == "side"
            else "haarcascade_frontalface_default.xml"
        )
        configured_path = self.config.get(f"{mode}_cascade_path")
        cascade_path = (
            Path(str(configured_path)).expanduser()
            if configured_path
            else Path(cv2.data.haarcascades) / default_name
        )
        if not cascade_path.is_file():
            raise TransformDependencyError(
                f"OpenCV Haar cascade for {mode} view was not found: {cascade_path}"
            )
        classifier = cv2.CascadeClassifier(str(cascade_path))
        if classifier.empty():
            raise TransformDependencyError(
                f"OpenCV could not load Haar cascade for {mode} view: {cascade_path}"
            )
        self._classifiers[mode] = classifier
        return classifier

    def detect(self, image: np.ndarray, view: str) -> Mapping[str, Any]:
        cv2 = _opencv("face_landmarks.opencv_haar")
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise SampleValidationError("OpenCV Haar fallback requires an RGB HWC image")
        grayscale = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        if bool(self.config.get("equalize_histogram", True)):
            grayscale = cv2.equalizeHist(grayscale)

        min_size = _as_size_hw(
            self.config.get("min_size_hw", [30, 30]),
            name="face_landmarks.opencv_haar.min_size_hw",
        )
        detect_kwargs = {
            "scaleFactor": float(self.config.get("scale_factor", 1.1)),
            "minNeighbors": int(self.config.get("min_neighbors", 5)),
            "minSize": (min_size[1], min_size[0]),
        }
        if detect_kwargs["scaleFactor"] <= 1.0:
            raise TransformConfigError("opencv_haar.scale_factor must be greater than 1")
        if detect_kwargs["minNeighbors"] < 0:
            raise TransformConfigError("opencv_haar.min_neighbors must be non-negative")

        canonical_view = "side" if str(view).lower() == "side" else "front"
        classifier = self._classifier(canonical_view)
        detections = classifier.detectMultiScale(grayscale, **detect_kwargs)
        mirrored_retry = False
        if len(detections) == 0 and canonical_view == "side":
            mirrored_retry = True
            mirrored = cv2.flip(grayscale, 1)
            detections = classifier.detectMultiScale(mirrored, **detect_kwargs)
            if len(detections):
                image_width = grayscale.shape[1]
                detections = np.asarray(
                    [
                        (image_width - int(x) - int(width), y, width, height)
                        for x, y, width, height in detections
                    ],
                    dtype=np.int32,
                )
        if len(detections) == 0:
            raise SampleValidationError(f"OpenCV Haar detector found no {canonical_view} face")

        x, y, width, height = max(
            (tuple(int(value) for value in detection) for detection in detections),
            key=lambda box: box[2] * box[3],
        )
        # An ellipse-like polygon gives face_hull masking a less rectangular
        # fallback while preserving the complete detected bounding extent.
        center_x = x + (width - 1) / 2
        center_y = y + (height - 1) / 2
        radius_x = max(0.5, (width - 1) / 2)
        radius_y = max(0.5, (height - 1) / 2)
        angles = np.linspace(0, 2 * np.pi, num=12, endpoint=False)
        points = np.stack(
            [center_x + radius_x * np.cos(angles), center_y + radius_y * np.sin(angles)],
            axis=1,
        ).astype(np.float32)
        return {
            "landmarks_xy": points,
            "landmark_kind": "face_bbox_hull",
            "face_bbox_xywh": np.asarray([x, y, width, height], dtype=np.float32),
            "mirrored_retry": mirrored_retry,
        }

    def __call__(self, image: np.ndarray, view: str) -> np.ndarray:
        return np.asarray(self.detect(image, view)["landmarks_xy"], dtype=np.float32)


def _pil_resampling(name: str) -> Image.Resampling:
    choices = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    try:
        return choices[str(name).lower()]
    except KeyError as exc:
        raise TransformConfigError(
            f"unsupported resize interpolation {name!r}; choose one of {sorted(choices)}"
        ) from exc


def transform_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous transform to ``[..., 2]`` image points."""

    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise SampleValidationError(f"landmarks must have shape [N, 2], got {tuple(points.shape)}")
    hom = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    warped = hom @ np.asarray(matrix, dtype=np.float64).T
    denominator = warped[:, 2:3]
    if np.any(np.abs(denominator) < 1e-12):
        raise SampleValidationError("image transform maps one or more landmarks to infinity")
    return (warped[:, :2] / denominator).astype(np.float32)


def _get_landmarks(sample: Mapping[str, Any]) -> np.ndarray | None:
    for key in ("landmarks_xy", "facial_landmarks_xy"):
        value = sample.get(key)
        if value is not None:
            array = np.asarray(value, dtype=np.float32)
            if array.size == 0:
                return None
            if array.ndim == 1 and array.size % 2 == 0:
                array = array.reshape(-1, 2)
            if array.ndim != 2 or array.shape[1] != 2:
                raise SampleValidationError(
                    f"{key} must have shape [N, 2], got {tuple(array.shape)}"
                )
            return array
    return None


def _set_landmarks(sample: MutableMapping[str, Any], landmarks: np.ndarray) -> None:
    value = np.asarray(landmarks, dtype=np.float32)
    sample["landmarks_xy"] = value
    # Keep the canonical annotation name synchronized for callers that inspect it.
    sample["facial_landmarks_xy"] = value


def _metadata(sample: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    metadata = sample.get("metadata")
    if metadata is None:
        metadata = {}
        sample["metadata"] = metadata
    if not isinstance(metadata, MutableMapping):
        raise SampleValidationError("sample['metadata'] must be a mutable mapping")
    return metadata


def _update_geometry(sample: MutableMapping[str, Any], matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"geometry matrix must be 3x3, got {matrix.shape}")

    landmarks = _get_landmarks(sample)
    if landmarks is not None:
        _set_landmarks(sample, transform_points(landmarks, matrix))

    metadata = _metadata(sample)
    for key in (
        "visible_eye_keypoints_xy",
        "iris_center_xy",
        "profile_head_origin_xy",
        "profile_head_forward_xy",
    ):
        value = metadata.get(key)
        if value is None:
            continue
        points = np.asarray(value, dtype=np.float32)
        original_shape = points.shape
        points = points.reshape(-1, 2)
        metadata[key] = transform_points(points, matrix).reshape(original_shape)

    for key in ("visible_eye_bbox_xyxy", "face_bbox_xyxy"):
        value = metadata.get(key)
        if value is None:
            continue
        x0, y0, x1, y1 = np.asarray(value, dtype=np.float32).reshape(4)
        corners = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        warped = transform_points(corners, matrix)
        metadata[key] = np.asarray(
            [
                float(warped[:, 0].min()),
                float(warped[:, 1].min()),
                float(warped[:, 0].max()),
                float(warped[:, 1].max()),
            ],
            dtype=np.float32,
        )

    previous = np.asarray(metadata.get("image_transform", np.eye(3)), dtype=np.float64)
    metadata["image_transform"] = (matrix @ previous).astype(np.float32)


def _handle_failure(stage: str, config: Mapping[str, Any], message: str) -> None:
    policy = str(config.get("on_failure", "error")).lower()
    detail = f"preprocessing stage '{stage}' failed: {message}"
    if policy == "drop":
        raise SampleDroppedError(detail)
    if policy in {"use_full_image", "skip", "noop", "no_op"}:
        return
    if policy != "error":
        raise TransformConfigError(
            f"{stage}.on_failure={policy!r} is invalid; use error, drop, or use_full_image"
        )
    raise SampleValidationError(detail)


def _mark_gaze_invalid(sample: MutableMapping[str, Any], reason: str) -> None:
    metadata = _metadata(sample)
    metadata["gaze_valid"] = False
    reasons = metadata.setdefault("invalid_reasons", [])
    if isinstance(reasons, list) and reason not in reasons:
        reasons.append(reason)


def _handle_quality_failure(
    stage: str,
    sample: MutableMapping[str, Any],
    config: Mapping[str, Any],
    message: str,
) -> None:
    policy = str(config.get("on_failure", "mark_invalid")).lower()
    if policy in {"mark_invalid", "mask", "keep_invalid"}:
        _mark_gaze_invalid(sample, f"{stage}:{message}")
        return
    _handle_failure(stage, config, message)


def _read_exif_orientation(path: Path) -> int:
    try:
        with Image.open(path) as image:
            return int(image.getexif().get(274, 1))
    except Exception:
        # Corrupt files are rejected by decode/validate. Missing EXIF is orientation 1.
        return 1


def _orientation_operation(
    orientation: int,
) -> tuple[Image.Transpose | None, Callable[[int, int], np.ndarray]]:
    def identity(_width: int, _height: int) -> np.ndarray:
        return np.eye(3, dtype=np.float64)

    operations: dict[int, tuple[Image.Transpose | None, Callable[[int, int], np.ndarray]]] = {
        1: (None, identity),
        2: (
            Image.Transpose.FLIP_LEFT_RIGHT,
            lambda w, _h: np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], dtype=float),
        ),
        3: (
            Image.Transpose.ROTATE_180,
            lambda w, h: np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], dtype=float),
        ),
        4: (
            Image.Transpose.FLIP_TOP_BOTTOM,
            lambda _w, h: np.array([[1, 0, 0], [0, -1, h - 1], [0, 0, 1]], dtype=float),
        ),
        5: (
            Image.Transpose.TRANSPOSE,
            lambda _w, _h: np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float),
        ),
        6: (
            Image.Transpose.ROTATE_270,
            lambda _w, h: np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], dtype=float),
        ),
        7: (
            Image.Transpose.TRANSVERSE,
            lambda w, h: np.array([[0, -1, h - 1], [-1, 0, w - 1], [0, 0, 1]], dtype=float),
        ),
        8: (
            Image.Transpose.ROTATE_90,
            lambda w, _h: np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], dtype=float),
        ),
    }
    if orientation not in operations:
        return operations[1]
    return operations[orientation]


def letterbox_resize(
    image: np.ndarray,
    size_hw: tuple[int, int],
    *,
    pad_rgb: tuple[int, int, int] = (0, 0, 0),
    interpolation: str = "bilinear",
) -> tuple[np.ndarray, np.ndarray]:
    """Aspect-preserving resize and return its source-to-output matrix."""

    output_height, output_width = size_hw
    source_height, source_width = image.shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise SampleValidationError("cannot resize an empty image")

    scale = min(output_width / source_width, output_height / source_height)
    resized_width = max(1, min(output_width, int(round(source_width * scale))))
    resized_height = max(1, min(output_height, int(round(source_height * scale))))
    resized = np.asarray(
        Image.fromarray(image).resize(
            (resized_width, resized_height),
            resample=_pil_resampling(interpolation),
        )
    )
    left = (output_width - resized_width) // 2
    top = (output_height - resized_height) // 2
    canvas = np.empty((output_height, output_width, 3), dtype=np.uint8)
    canvas[...] = np.asarray(pad_rgb, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized

    sx = resized_width / source_width
    sy = resized_height / source_height
    matrix = np.array([[sx, 0, left], [0, sy, top], [0, 0, 1]], dtype=np.float64)
    return canvas, matrix


def _crop_from_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    margin_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = _bbox_from_landmarks(image, landmarks, margin_ratio)
    matrix = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    return image[y0:y1, x0:x1].copy(), matrix


def _bbox_from_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    """Return a clipped ``(x0, y0, x1, y1)`` face ROI in image pixels."""

    height, width = image.shape[:2]
    minimum = landmarks.min(axis=0)
    maximum = landmarks.max(axis=0)
    span = maximum - minimum
    if np.any(span <= 1e-6):
        raise SampleValidationError("landmarks do not span a valid face ROI")
    margin = span * float(margin_ratio)
    x0 = max(0, int(np.floor(minimum[0] - margin[0])))
    y0 = max(0, int(np.floor(minimum[1] - margin[1])))
    x1 = min(width, int(np.ceil(maximum[0] + margin[0])) + 1)
    y1 = min(height, int(np.ceil(maximum[1] + margin[1])) + 1)
    if x1 <= x0 or y1 <= y0:
        raise SampleValidationError("landmark face ROI lies outside the image")
    return x0, y0, x1, y1


def _eye_quad(landmarks: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    indices = config.get("eye_landmark_indices", config.get("source_indices", [0, 1, 2, 3]))
    if not isinstance(indices, Sequence) or len(indices) < 4:
        raise TransformConfigError(
            "eye_region_warp.eye_landmark_indices must identify at least four landmarks"
        )
    try:
        eyes = landmarks[np.asarray([int(index) for index in indices], dtype=int)]
    except IndexError as exc:
        raise SampleValidationError(
            f"eye landmark indices {list(indices)!r} exceed {len(landmarks)} annotations"
        ) from exc

    left_indices = config.get("left_eye_indices", [0, 1])
    right_indices = config.get("right_eye_indices", [2, 3])
    try:
        left_center = landmarks[np.asarray(left_indices, dtype=int)].mean(axis=0)
        right_center = landmarks[np.asarray(right_indices, dtype=int)].mean(axis=0)
    except IndexError as exc:
        raise SampleValidationError(
            "configured eye center indices exceed landmark annotations"
        ) from exc

    direction = right_center - left_center
    center_distance = float(np.linalg.norm(direction))
    if center_distance < 1e-5:
        raise SampleValidationError("left and right eye centers overlap")
    unit_x = direction / center_distance
    unit_y = np.array([-unit_x[1], unit_x[0]], dtype=np.float32)
    center = (left_center + right_center) / 2

    projected_x = (eyes - center) @ unit_x
    width = float(projected_x.max() - projected_x.min())
    width = max(width, center_distance)
    horizontal_margin = float(config.get("horizontal_margin_ratio", 0.20))
    vertical_ratio = float(config.get("height_to_width_ratio", 0.25))
    half_width = width * (0.5 + horizontal_margin)
    half_height = max(1.0, width * vertical_ratio / 2)

    top_left = center - unit_x * half_width - unit_y * half_height
    top_right = center + unit_x * half_width - unit_y * half_height
    bottom_right = center + unit_x * half_width + unit_y * half_height
    bottom_left = center - unit_x * half_width + unit_y * half_height
    return np.asarray([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _image_to_uint8_hwc(image: Any, *, stage: str) -> tuple[np.ndarray, str, torch.dtype | None]:
    if isinstance(image, np.ndarray):
        array = image
        origin = "numpy"
        torch_dtype = None
    elif torch.is_tensor(image):
        if image.ndim != 3:
            raise SampleValidationError(f"{stage} expects a 3D image tensor")
        torch_dtype = image.dtype
        array = image.detach().cpu().numpy()
        if array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))
        elif array.shape[-1] != 3:
            raise SampleValidationError(f"{stage} expects RGB image data")
        if np.nanmin(array) < 0 or np.nanmax(array) > 1:
            raise TransformConfigError(
                f"{stage} after mean/std normalization is unsupported; put 'augment' before "
                "'normalize' in preprocessing.stage_order"
            )
        array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
        origin = "tensor"
    else:
        raise SampleValidationError(
            f"{stage} received unsupported image type {type(image).__name__}"
        )
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array, origin, torch_dtype


def _restore_augmented_image(array: np.ndarray, origin: str, dtype: torch.dtype | None) -> Any:
    if origin == "numpy":
        return array
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float() / 255.0
    return tensor.to(dtype=dtype or torch.float32)


class OrderedGazePreprocessor:
    """Execute enabled preprocessing stages in configured order.

    Parameters may be either the ``preprocessing`` section itself or a full
    resolved config containing a ``preprocessing`` key.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        split: str = "train",
        task_config: Mapping[str, Any] | None = None,
        landmark_detector: Callable[[np.ndarray, str], np.ndarray] | None = None,
    ) -> None:
        root = _mapping(config, name="config")
        if "preprocessing" in root:
            self.config = _mapping(root.get("preprocessing"), name="preprocessing")
            if task_config is None:
                task_config = _mapping(root.get("task"), name="task")
        else:
            self.config = root
        self.task_config = _mapping(task_config, name="task_config")
        self.split = "validation" if split == "val" else str(split)
        self.landmark_detector = landmark_detector
        # Front and side branches may use different detector thresholds.
        # A single shared instance would freeze the config of whichever view
        # happened to be processed first.
        self._opencv_haar_detectors: dict[str, OpenCVHaarFaceDetector] = {}
        self._mediapipe_detectors: dict[str, MediaPipeFaceLandmarkerDetector] = {}

        configured_order = self.config.get("stage_order", _STAGES)
        if not isinstance(configured_order, Sequence) or isinstance(configured_order, str | bytes):
            raise TransformConfigError(
                "preprocessing.stage_order must be a sequence of stage names"
            )
        self.stage_order = tuple(str(stage) for stage in configured_order)
        unknown = [stage for stage in self.stage_order if stage not in _STAGES]
        if unknown:
            raise TransformConfigError(
                f"unknown preprocessing stages {unknown}; supported stages are {list(_STAGES)}"
            )
        if len(set(self.stage_order)) != len(self.stage_order):
            raise TransformConfigError("preprocessing.stage_order contains duplicate stage names")
        self._validate_configuration()

    @property
    def _opencv_haar_detector(self) -> OpenCVHaarFaceDetector | None:
        """Backward-compatible access to the first lazily created detector."""

        return next(iter(self._opencv_haar_detectors.values()), None)

    def _stage_config(self, stage: str, view: str) -> Mapping[str, Any]:
        if stage in self.config:
            base = deepcopy(dict(_mapping(self.config.get(stage), name=f"preprocessing.{stage}")))
        else:
            # A missing section is disabled; an explicitly present section may
            # omit ``enabled`` and is then enabled by default.
            base = {"enabled": False}
        overrides = _mapping(self.config.get("branch_overrides"), name="branch_overrides")
        branch = _mapping(overrides.get(view), name=f"branch_overrides.{view}")
        stage_override = branch.get(stage)
        if stage_override is not None:
            base.update(dict(_mapping(stage_override, name=f"branch_overrides.{view}.{stage}")))
        return base

    def _validate_configuration(self) -> None:
        known_sections = {stage for stage in _STAGES if stage in self.config}
        omitted_enabled = [
            stage
            for stage in known_sections
            if _enabled(_mapping(self.config.get(stage), name=stage))
            and stage not in self.stage_order
        ]
        if omitted_enabled:
            raise TransformConfigError(
                f"enabled stages {omitted_enabled} are missing from preprocessing.stage_order"
            )

        positions = {stage: index for index, stage in enumerate(self.stage_order)}
        for dependent in (
            "eye_selection",
            "metric_head_pose",
            "eye_region_warp",
            "face_roi",
            "background_mask",
        ):
            if dependent not in self.config:
                continue
            dependent_config = _mapping(self.config.get(dependent), name=dependent)
            if not _enabled(dependent_config):
                continue
            if dependent == "eye_region_warp" and str(
                dependent_config.get("method", "landmark_homography")
            ).lower() in {
                "profile90_annotation",
                "profile_annotation",
                "annotation_eye_bbox",
            }:
                continue
            landmark_config = _mapping(self.config.get("face_landmarks"), name="face_landmarks")
            if not _enabled(landmark_config):
                raise TransformConfigError(
                    f"'{dependent}' requires preprocessing.face_landmarks.enabled=true"
                )
            if positions.get("face_landmarks", 10**6) > positions.get(dependent, -1):
                raise TransformConfigError(
                    f"'face_landmarks' must appear before '{dependent}' in stage_order"
                )

        normalize = _mapping(self.config.get("normalize"), name="normalize")
        if (
            "normalize" in self.config
            and _enabled(normalize)
            and str(normalize.get("mode", "zero_one")).lower()
            in {
                "mean_std",
                "standardize",
            }
        ):
            mean, std = normalize.get("mean"), normalize.get("std")
            valid_mean = isinstance(mean, Sequence) and not isinstance(mean, str | bytes)
            valid_std = isinstance(std, Sequence) and not isinstance(std, str | bytes)
            if not valid_mean or not valid_std or len(mean) != 3 or len(std) != 3:
                raise TransformConfigError(
                    "normalize.mode=mean_std requires three-channel "
                    "normalize.mean and normalize.std"
                )
            if any(float(value) <= 0 for value in std):
                raise TransformConfigError("normalize.std values must be greater than zero")

        augment = _mapping(self.config.get("augment"), name="augment")
        jitter = _mapping(augment.get("color_jitter"), name="augment.color_jitter")
        normalize_mode = str(normalize.get("mode", "zero_one")).lower()
        if (
            "augment" in self.config
            and _enabled(augment)
            and jitter
            and _enabled(jitter)
            and "normalize" in positions
            and "augment" in positions
            and positions["normalize"] < positions["augment"]
            and normalize_mode in {"mean_std", "standardize", "imagenet"}
        ):
            raise TransformConfigError(
                "enabled color jitter must appear before mean/std normalization; "
                "move 'augment' before 'normalize' in preprocessing.stage_order"
            )

    def make_augmentation_context(
        self,
        view: str = "front",
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Sample decisions once so paired views can share geometric augmentation."""

        config = self._stage_config("augment", view)
        if generator is None:
            # Never consume torch's process-global RNG. Dataset callers pass a
            # sample-specific generator; direct transform use gets a stable
            # local default.
            generator = torch.Generator().manual_seed(0)

        def random_value() -> float:
            return float(torch.rand((), generator=generator).item())

        context: dict[str, Any] = {}
        probability = float(config.get("horizontal_flip_probability", 0.0))
        if not 0 <= probability <= 1:
            raise TransformConfigError("augment.horizontal_flip_probability must be in [0, 1]")
        context["horizontal_flip"] = bool(random_value() < probability)

        jitter = _mapping(config.get("color_jitter"), name="augment.color_jitter")
        if jitter and _enabled(jitter):
            for name in ("brightness", "contrast", "saturation"):
                magnitude = float(jitter.get(name, 0.0))
                if magnitude < 0:
                    raise TransformConfigError(f"augment.color_jitter.{name} must be non-negative")
                low = max(0.0, 1.0 - magnitude)
                high = 1.0 + magnitude
                context[name] = low + (high - low) * random_value()
            hue = float(jitter.get("hue", 0.0))
            if not 0 <= hue <= 0.5:
                raise TransformConfigError("augment.color_jitter.hue must be in [0, 0.5]")
            context["hue"] = (2 * random_value() - 1) * hue
        return context

    def __call__(self, sample: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Run deterministic preprocessing followed by optional augmentation."""

        processed = self.preprocess_deterministic(sample)
        return self.apply_augmentation(processed)

    def preprocess_deterministic(
        self, sample: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Run the deterministic prefix immediately before ``augment``.

        The returned mapping is the stable cache boundary. Dataset callers may
        serialize it and resume the remaining ordered stages after a cache hit,
        so stochastic training transforms are never frozen in a pickle.  When
        augmentation is disabled, all deterministic stages are executed here.
        """

        if not isinstance(sample, MutableMapping):
            raise TypeError("preprocessor expects a mutable sample mapping")
        view = str(sample.get("view", _metadata(sample).get("view", "front"))).lower()
        branch = _mapping(
            _mapping(self.config.get("branch_overrides"), name="branch_overrides").get(view),
            name=f"branch_overrides.{view}",
        )
        if branch and not bool(branch.get("enabled", True)):
            raise TransformConfigError(
                f"sample view {view!r} is disabled by preprocessing.branch_overrides.{view}.enabled"
            )

        _metadata(sample).setdefault("gaze_valid", True)
        _metadata(sample).setdefault("invalid_reasons", [])

        for stage in self.stage_order:
            stage_config = self._stage_config(stage, view)
            if not _enabled(stage_config):
                continue
            if stage == "augment":
                break
            getattr(self, f"_stage_{stage}")(sample, stage_config)

        if "image" not in sample:
            raise SampleValidationError(
                "preprocessing produced no image; enable 'decode' or provide sample['image']"
            )
        if branch.get("representation") and not self._enabled_augment_for_view(view):
            _metadata(sample)["representation"] = str(branch["representation"])
        return sample

    def apply_augmentation(self, sample: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Resume the ordered pipeline at ``augment`` for the current epoch."""

        if not isinstance(sample, MutableMapping):
            raise TypeError("preprocessor expects a mutable sample mapping")
        view = str(sample.get("view", _metadata(sample).get("view", "front"))).lower()
        if not self._enabled_augment_for_view(view):
            sample.pop("_augmentation_context", None)
            return sample
        augment_index = self.stage_order.index("augment")
        for stage in self.stage_order[augment_index:]:
            config = self._stage_config(stage, view)
            if _enabled(config):
                getattr(self, f"_stage_{stage}")(sample, config)
        branch = _mapping(
            _mapping(self.config.get("branch_overrides"), name="branch_overrides").get(view),
            name=f"branch_overrides.{view}",
        )
        if branch.get("representation"):
            _metadata(sample)["representation"] = str(branch["representation"])
        return sample

    def _enabled_augment_for_view(self, view: str) -> bool:
        """Return whether the ordered pipeline has an enabled augment stage."""

        return "augment" in self.stage_order and _enabled(self._stage_config("augment", view))

    def _stage_decode(self, sample: MutableMapping[str, Any], config: Mapping[str, Any]) -> None:
        if sample.get("image") is not None:
            image = np.asarray(sample["image"])
            if image.ndim != 3 or image.shape[2] != 3:
                raise SampleValidationError("preloaded sample image must have shape [H, W, 3]")
            sample["image"] = image
            _metadata(sample).setdefault("exif_orientation", 1)
            _metadata(sample).setdefault("image_transform", np.eye(3, dtype=np.float32))
            return

        raw_path = sample.get("image_path")
        if not raw_path:
            raise SampleValidationError("decode requires a non-empty sample['image_path']")
        path = Path(str(raw_path))
        if not path.is_file():
            raise SampleValidationError(f"image does not exist: {path}")

        backend = str(config.get("backend", "pillow")).lower()
        try:
            if backend in {"pillow", "pil"}:
                with Image.open(path) as pil_image:
                    orientation = int(pil_image.getexif().get(274, 1))
                    image = np.asarray(pil_image.convert("RGB"))
            elif backend in {"opencv", "cv2"}:
                cv2 = _opencv("decode")
                flags = cv2.IMREAD_COLOR
                if hasattr(cv2, "IMREAD_IGNORE_ORIENTATION"):
                    flags |= cv2.IMREAD_IGNORE_ORIENTATION
                decoded = cv2.imread(str(path), flags)
                if decoded is None:
                    raise SampleValidationError(f"OpenCV could not decode image: {path}")
                image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                orientation = _read_exif_orientation(path)
            else:
                raise TransformConfigError(
                    f"decode.backend={backend!r} is unsupported; use pillow or opencv"
                )
        except (GazeTransformError, TransformConfigError):
            raise
        except Exception as exc:
            raise SampleValidationError(f"failed to decode image {path}: {exc}") from exc

        if image.dtype != np.uint8:
            image = np.asarray(image, dtype=np.uint8)
        sample["image"] = np.ascontiguousarray(image)
        metadata = _metadata(sample)
        metadata["exif_orientation"] = orientation
        metadata["source_size_hw"] = np.asarray(image.shape[:2], dtype=np.int64)
        metadata["image_transform"] = np.eye(3, dtype=np.float32)

    def _stage_exif_orientation(
        self, sample: MutableMapping[str, Any], config: Mapping[str, Any]
    ) -> None:
        del config
        image = sample.get("image")
        if not isinstance(image, np.ndarray):
            raise SampleValidationError("exif_orientation requires decoded numpy RGB image")
        metadata = _metadata(sample)
        orientation = int(metadata.get("exif_orientation", 1))
        operation, matrix_factory = _orientation_operation(orientation)
        if operation is None:
            return
        height, width = image.shape[:2]
        sample["image"] = np.asarray(Image.fromarray(image).transpose(operation))
        _update_geometry(sample, matrix_factory(width, height))
        metadata["exif_orientation_applied"] = orientation

    def _stage_validate(self, sample: MutableMapping[str, Any], config: Mapping[str, Any]) -> None:
        image = sample.get("image")
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise SampleValidationError(
                "validate requires a decoded RGB image with shape [H, W, 3]"
            )
        height, width = image.shape[:2]
        minimum_width = int(config.get("min_width_px", 1))
        minimum_height = int(config.get("min_height_px", 1))
        if width < minimum_width or height < minimum_height:
            _handle_failure(
                "validate",
                config,
                f"image is {width}x{height}, below minimum {minimum_width}x{minimum_height}",
            )
        if not np.isfinite(image).all():
            _handle_failure("validate", config, "image contains non-finite pixels")

        landmarks = _get_landmarks(sample)
        if landmarks is not None and not np.isfinite(landmarks).all():
            _handle_failure("validate", config, "facial landmarks contain non-finite values")

    def _stage_face_landmarks(
        self, sample: MutableMapping[str, Any], config: Mapping[str, Any]
    ) -> None:
        source = str(config.get("source", "annotation")).lower()
        landmarks = _get_landmarks(sample)
        needs_detector = source != "annotation" or landmarks is None
        if needs_detector:
            detector_name = (
                source
                if source != "annotation"
                else str(config.get("fallback_detector", "")).lower()
            )
            if not detector_name:
                _handle_quality_failure(
                    "face_landmarks",
                    sample,
                    config,
                    "annotation landmark field is missing",
                )
                return

            try:
                # Explicitly injected detectors always take priority over the
                # built-in implementation, including for ``opencv_haar``.
                if self.landmark_detector is not None:
                    result: Any = self.landmark_detector(
                        np.asarray(sample["image"]), str(sample.get("view", "front"))
                    )
                elif detector_name in {"opencv_haar", "haar", "opencv-haar"}:
                    detector_key = (
                        "side" if str(sample.get("view", "front")).lower() == "side" else "front"
                    )
                    detector = self._opencv_haar_detectors.get(detector_key)
                    if detector is None:
                        haar_config = _mapping(
                            config.get("opencv_haar"), name="face_landmarks.opencv_haar"
                        )
                        detector = OpenCVHaarFaceDetector(haar_config)
                        self._opencv_haar_detectors[detector_key] = detector
                    result = detector.detect(
                        np.asarray(sample["image"]), str(sample.get("view", "front"))
                    )
                elif detector_name in {
                    "mediapipe_face_landmarker",
                    "mediapipe_face_mesh",
                    "mediapipe",
                }:
                    detector_key = (
                        "side" if str(sample.get("view", "front")).lower() == "side" else "front"
                    )
                    detector = self._mediapipe_detectors.get(detector_key)
                    if detector is None:
                        detector_config = dict(
                            _mapping(
                                config.get("mediapipe"),
                                name="face_landmarks.mediapipe",
                            )
                        )
                        for key in (
                            "model_asset_path",
                            "model_path",
                            "num_faces",
                            "min_face_detection_confidence",
                            "min_face_presence_confidence",
                            "min_tracking_confidence",
                            "output_face_blendshapes",
                            "mirror_retry",
                        ):
                            if key in config and key not in detector_config:
                                detector_config[key] = config[key]
                        detector = MediaPipeFaceLandmarkerDetector(detector_config)
                        self._mediapipe_detectors[detector_key] = detector
                    result = detector.detect(
                        np.asarray(sample["image"]), str(sample.get("view", "front"))
                    )
                else:
                    raise TransformDependencyError(
                        f"face landmark detector {detector_name!r} requires an injected "
                        "landmark_detector callable; built-in automatic fallback supports "
                        "'opencv_haar' and 'mediapipe_face_landmarker'"
                    )
            except TransformDependencyError:
                raise
            except (GazeTransformError, WebEyeTrackGeometryError) as exc:
                _handle_quality_failure("face_landmarks", sample, config, str(exc))
                return

            metadata = _metadata(sample)
            if isinstance(result, Mapping):
                landmarks = np.asarray(result.get("landmarks_xy"), dtype=np.float32)
                if result.get("landmark_kind"):
                    metadata["landmark_kind"] = str(result["landmark_kind"])
                if result.get("face_bbox_xywh") is not None:
                    metadata["face_bbox_xywh"] = np.asarray(
                        result["face_bbox_xywh"], dtype=np.float32
                    ).reshape(4)
                for key in (
                    "landmarks_xyz_normalized",
                    "landmark_presence",
                    "landmark_visibility",
                    "face_rt",
                    "face_blendshapes",
                ):
                    if result.get(key) is not None:
                        metadata[key] = np.asarray(result[key], dtype=np.float32)
                metadata["detector_mirrored_retry"] = bool(result.get("mirrored_retry", False))
            else:
                landmarks = np.asarray(result, dtype=np.float32)
                metadata["landmark_kind"] = "injected_detector"
            metadata["landmark_source"] = detector_name
        else:
            _metadata(sample)["landmark_kind"] = "annotation"
            _metadata(sample)["landmark_source"] = "annotation"

        if landmarks is None or landmarks.ndim != 2 or landmarks.shape[1] != 2:
            _handle_quality_failure(
                "face_landmarks",
                sample,
                config,
                "detector returned no [N, 2] landmarks",
            )
            return
        if len(landmarks) < int(config.get("min_landmarks", 4)):
            _handle_quality_failure(
                "face_landmarks",
                sample,
                config,
                f"received {len(landmarks)} landmarks; at least "
                f"{config.get('min_landmarks', 4)} required",
            )
            return
        expected_landmarks = config.get("expected_landmarks")
        if expected_landmarks is not None and len(landmarks) != int(expected_landmarks):
            _handle_quality_failure(
                "face_landmarks",
                sample,
                config,
                f"received {len(landmarks)} landmarks; expected exactly {int(expected_landmarks)}",
            )
            return
        _set_landmarks(sample, landmarks)

    def _stage_eye_selection(
        self, sample: MutableMapping[str, Any], config: Mapping[str, Any]
    ) -> None:
        landmarks = _get_landmarks(sample)
        metadata = _metadata(sample)
        if landmarks is None:
            _handle_quality_failure(
                "eye_selection", sample, config, "facial landmarks are unavailable"
            )
            return
        try:
            selected, scores = select_eye(
                landmarks,
                mode=str(config.get("mode", "best_visible")),
                target_eye=config.get("target_eye"),
                presence=metadata.get("landmark_presence"),
            )
        except WebEyeTrackGeometryError as exc:
            _handle_quality_failure("eye_selection", sample, config, str(exc))
            return
        metadata["selected_eye"] = selected
        metadata["eye_visibility_score"] = np.asarray(
            [scores["left"], scores["right"]], dtype=np.float32
        )
        minimum_visibility = float(config.get("min_visibility", 0.0))
        relevant_scores = (
            [scores["left"], scores["right"]] if selected == "both" else [scores[selected]]
        )
        selection_valid = bool(min(relevant_scores) >= minimum_visibility)
        metadata["eye_selection_valid"] = selection_valid
        if not selection_valid:
            _handle_quality_failure(
                "eye_selection",
                sample,
                config,
                f"selected eye visibility {min(relevant_scores):.3f} is below "
                f"min_visibility={minimum_visibility:.3f}",
            )

    def _stage_metric_head_pose(
        self, sample: MutableMapping[str, Any], config: Mapping[str, Any]
    ) -> None:
        metadata = _metadata(sample)
        pose_source = str(config.get("source", "reconstruct")).lower()
        if pose_source in {"precomputed", "precomputed_only", "precomputed_or_reconstruct"}:
            try:
                precomputed_head = np.asarray(
                    metadata.get("head_vector"), dtype=np.float32
                ).reshape(3)
                precomputed_origin = np.asarray(
                    metadata.get("face_origin_3d"), dtype=np.float32
                ).reshape(3)
                head_norm = float(np.linalg.norm(precomputed_head))
                orientation_valid = bool(
                    metadata.get("head_orientation_valid", np.isfinite(precomputed_head).all())
                )
                origin_valid = bool(
                    metadata.get("face_origin_valid", np.isfinite(precomputed_origin).all())
                )
                pose_valid = bool(
                    metadata.get("head_pose_valid", orientation_valid and origin_valid)
                )
                precomputed_valid = bool(
                    orientation_valid
                    and origin_valid
                    and pose_valid
                    and np.isfinite(precomputed_head).all()
                    and np.isfinite(precomputed_origin).all()
                    and head_norm > 1e-8
                )
            except (TypeError, ValueError):
                precomputed_valid = False

            if precomputed_valid:
                origin_unit = str(
                    metadata.get(
                        "face_origin_unit",
                        config.get("precomputed_face_origin_unit", "cm"),
                    )
                ).lower()
                if origin_unit in {"mm", "millimeter", "millimetre"}:
                    precomputed_origin = precomputed_origin / 10.0
                elif origin_unit not in {"cm", "centimeter", "centimetre"}:
                    raise TransformConfigError(
                        "metric_head_pose precomputed face_origin_unit must be mm or cm"
                    )
                metadata["head_vector"] = (precomputed_head / head_norm).astype(np.float32)
                metadata["face_origin_3d"] = precomputed_origin.astype(np.float32)
                metadata["face_origin_unit"] = "cm"
                metadata["face_origin_source"] = str(
                    metadata.get("face_origin_source", "precomputed_manifest")
                )
                metadata["head_orientation_valid"] = True
                metadata["face_origin_valid"] = True
                metadata["head_pose_valid"] = True
                return

            if pose_source in {"precomputed", "precomputed_only"}:
                metadata["head_vector"] = np.full(3, np.nan, dtype=np.float32)
                metadata["head_euler_degrees"] = np.full(3, np.nan, dtype=np.float32)
                metadata["face_origin_3d"] = np.full(3, np.nan, dtype=np.float32)
                metadata["head_orientation_valid"] = False
                metadata["face_origin_valid"] = False
                metadata["head_pose_valid"] = False
                _handle_quality_failure(
                    "metric_head_pose",
                    sample,
                    config,
                    "valid precomputed head_vector and face_origin_3d are unavailable",
                )
                return

        face_rt = metadata.get("face_rt")
        normalized_xyz = metadata.get("landmarks_xyz_normalized")
        if face_rt is None:
            metadata["head_vector"] = np.full(3, np.nan, dtype=np.float32)
            metadata["head_euler_degrees"] = np.full(3, np.nan, dtype=np.float32)
            metadata["face_origin_3d"] = np.full(3, np.nan, dtype=np.float32)
            metadata["head_orientation_valid"] = False
            metadata["face_origin_valid"] = False
            metadata["head_pose_valid"] = False
            _handle_quality_failure(
                "metric_head_pose",
                sample,
                config,
                "MediaPipe facial transformation matrix is unavailable",
            )
            return
        try:
            head_vector, mapped_euler = head_vector_from_face_rt(np.asarray(face_rt))
            metadata["head_vector"] = np.asarray(head_vector, dtype=np.float32)
            metadata["head_euler_degrees"] = np.asarray(mapped_euler, dtype=np.float32)
            metadata["head_orientation_valid"] = True
            face_origin_source = str(
                config.get("face_origin_source", "annotation_or_reconstruct")
            ).lower()
            annotation = metadata.get("face_center_3d")
            use_annotation = annotation is not None and face_origin_source in {
                "annotation",
                "annotation_or_reconstruct",
                "annotation_first",
            }
            metric_transform: np.ndarray | None = None
            metric_face: np.ndarray | None = None
            face_width_cm = np.nan
            if use_annotation:
                face_origin = np.asarray(annotation, dtype=np.float32).reshape(3)
                annotation_unit = str(config.get("annotation_unit", "mm")).lower()
                if annotation_unit in {"mm", "millimeter", "millimetre"}:
                    face_origin = face_origin / 10.0
                elif annotation_unit not in {"cm", "centimeter", "centimetre"}:
                    raise TransformConfigError("metric_head_pose.annotation_unit must be mm or cm")
                origin_kind = "annotation"
            else:
                if face_origin_source in {"annotation", "annotation_only"}:
                    raise WebEyeTrackGeometryError("face_center_3d annotation is unavailable")
                if normalized_xyz is None:
                    raise WebEyeTrackGeometryError(
                        "normalized MediaPipe XYZ landmarks are unavailable"
                    )
                image = np.asarray(sample["image"])
                iris_mode = str(config.get("required_irises", "both")).lower()
                if iris_mode in {"selected", "selected_eye"}:
                    iris_mode = str(metadata.get("selected_eye", "")).lower()
                face_origin, metric_transform, metric_face, face_width_cm = (
                    estimate_metric_face_origin(
                        np.asarray(normalized_xyz),
                        np.asarray(face_rt),
                        image.shape[:2],
                        iris_diameter_cm=float(config.get("iris_diameter_cm", 1.2)),
                        iris_mode=iris_mode,
                        initial_depth_cm=float(config.get("initial_depth_cm", 60.0)),
                        max_iterations=int(config.get("max_iterations", 10)),
                        max_depth_step_cm=float(config.get("max_depth_step_cm", 5.0)),
                    )
                )
                origin_kind = "webeyetrack_metric_reconstruction"

            metadata["face_origin_3d"] = np.asarray(face_origin, dtype=np.float32)
            metadata["face_origin_unit"] = "cm"
            metadata["face_origin_source"] = origin_kind
            metadata["face_origin_valid"] = True
            metadata["head_pose_valid"] = True
            if metric_transform is not None:
                metadata["metric_transform"] = np.asarray(metric_transform, dtype=np.float32)
            if metric_face is not None and bool(config.get("keep_metric_face", False)):
                metadata["metric_face"] = np.asarray(metric_face, dtype=np.float32)
            if np.isfinite(face_width_cm):
                metadata["estimated_face_width_cm"] = float(face_width_cm)
        except TransformConfigError:
            raise
        except (ValueError, WebEyeTrackGeometryError) as exc:
            metadata.setdefault("head_vector", np.full(3, np.nan, dtype=np.float32))
            metadata.setdefault("head_euler_degrees", np.full(3, np.nan, dtype=np.float32))
            metadata.setdefault("head_orientation_valid", False)
            metadata["face_origin_3d"] = np.full(3, np.nan, dtype=np.float32)
            metadata["face_origin_valid"] = False
            metadata["head_pose_valid"] = False
            _handle_quality_failure("metric_head_pose", sample, config, str(exc))

    def _stage_eye_region_warp(
        self, sample: MutableMapping[str, Any], config: Mapping[str, Any]
    ) -> None:
        method = str(config.get("method", "landmark_homography")).lower()

        def mark_invalid_warp(reason: str, default_size: Sequence[int]) -> bool:
            policy = str(config.get("on_failure", "error")).lower()
            if policy not in {"mark_invalid", "mask", "keep_invalid"}:
                return False
            height, width = _as_size_hw(
                config.get("size_hw", default_size), name="eye_region_warp.size_hw"
            )
            sample["image"] = np.zeros((height, width, 3), dtype=np.uint8)
            metadata = _metadata(sample)
            metadata["eye_region_warp_valid"] = False
            metadata["representation"] = "invalid_eye_patch"
            _mark_gaze_invalid(sample, f"eye_region_warp:{reason}")
            return True

        if method in {
            "profile90_annotation",
            "profile_annotation",
            "annotation_eye_bbox",
        }:
            metadata = _metadata(sample)
            raw_feature_extraction = config.get("feature_extraction")
            if raw_feature_extraction is None:
                feature_extraction: Mapping[str, Any] = {
                    "side_headpose": {"enabled": True, "source": "side_2d"},
                    "side_eyeangle": {"enabled": True},
                    "side_eyelidangle": {"enabled": True},
                }
            else:
                feature_extraction = _mapping(
                    raw_feature_extraction,
                    name="eye_region_warp.feature_extraction",
                )

            def feature_config(name: str) -> Mapping[str, Any]:
                value = feature_extraction.get(name, {})
                return _mapping(value, name=f"eye_region_warp.feature_extraction.{name}")

            head_feature = feature_config("side_headpose")
            eye_angle_feature = feature_config("side_eyeangle")
            iris_feature = feature_config("side_eyelidangle")
            head_feature_enabled = bool(head_feature.get("enabled", False))
            head_feature_source = str(head_feature.get("source", "side_2d")).lower()
            extract_side_head_pose = head_feature_enabled and head_feature_source == "side_2d"
            extract_eye_angles = bool(eye_angle_feature.get("enabled", False))
            extract_iris_pose = bool(iris_feature.get("enabled", False))

            def annotation_value(config_key: str, default_key: str) -> Any:
                annotation_key = str(config.get(config_key, default_key)).strip()
                if not annotation_key:
                    raise TransformConfigError(
                        f"eye_region_warp.{config_key} must name a metadata field"
                    )
                return metadata.get(annotation_key, sample.get(annotation_key))

            try:
                if metadata.get("eye_annotation_valid") is False:
                    raise ProfileSideGeometryError(
                        "eye_annotation_valid=false for strict-profile sample"
                    )
                result = preprocess_profile_side(
                    np.asarray(sample["image"]),
                    eye_bbox_xyxy=annotation_value("bbox_key", "visible_eye_bbox_xyxy"),
                    eyelid_keypoints_xy=annotation_value(
                        "eyelid_keypoints_key", "visible_eye_keypoints_xy"
                    ),
                    iris_center_xy=(
                        annotation_value("iris_center_key", "iris_center_xy")
                        if extract_iris_pose
                        else None
                    ),
                    head_origin_xy=(
                        annotation_value("head_origin_key", "profile_head_origin_xy")
                        if extract_side_head_pose
                        else None
                    ),
                    head_forward_point_xy=(
                        annotation_value("head_forward_key", "profile_head_forward_xy")
                        if extract_side_head_pose
                        else None
                    ),
                    output_size_hw=config.get("size_hw", [128, 256]),
                    crop_mode=str(config.get("crop_mode", "affine")).lower(),
                    vertical_only=bool(iris_feature.get("vertical_only", False)),
                    landmark_crop_scale_xy=config.get("landmark_crop_scale_xy", [2.4, 1.2]),
                    bbox_crop_scale_xy=config.get("bbox_scale_xy", [1.0, 1.0]),
                    border_value_rgb=config.get("pad_rgb", [0, 0, 0]),
                    eyelid_tail_indices=config.get("eyelid_tail_indices", [3, 2, 4]),
                    extract_head_pose=extract_side_head_pose,
                    extract_iris_pose=extract_iris_pose,
                    extract_eye_angles=extract_eye_angles,
                )
            except (KeyError, TypeError, ValueError, ProfileSideGeometryError) as exc:
                if extract_side_head_pose:
                    metadata["head_pose_2d"] = np.full(2, np.nan, dtype=np.float32)
                    metadata["head_pose_2d_valid"] = False
                if extract_eye_angles:
                    metadata["eye_angles"] = np.full(2, np.nan, dtype=np.float32)
                if extract_iris_pose:
                    metadata["iris_pose_2d"] = np.full(2, np.nan, dtype=np.float32)
                metadata["eye_selection_valid"] = False
                if mark_invalid_warp(str(exc), [128, 256]):
                    return
                _handle_failure("eye_region_warp", config, str(exc))
                return

            selected_eye = str(metadata.get("visible_eye", "")).lower()
            if selected_eye not in {"left", "right"}:
                reason = "visible_eye annotation must be left or right"
                if extract_side_head_pose:
                    metadata["head_pose_2d"] = np.full(2, np.nan, dtype=np.float32)
                    metadata["head_pose_2d_valid"] = False
                if extract_eye_angles:
                    metadata["eye_angles"] = np.full(2, np.nan, dtype=np.float32)
                if extract_iris_pose:
                    metadata["iris_pose_2d"] = np.full(2, np.nan, dtype=np.float32)
                metadata["eye_selection_valid"] = False
                if mark_invalid_warp(reason, [128, 256]):
                    return
                _handle_failure("eye_region_warp", config, reason)
                return

            sample["image"] = result.patch
            source_quad = np.asarray(result.source_quad_xy, dtype=np.float32)
            _update_geometry(sample, result.source_to_patch)
            metadata = _metadata(sample)
            metadata["representation"] = "profile90_selectable_features_v3"
            metadata["eye_region_warp_valid"] = True
            if result.head_pose_2d is not None:
                metadata["profile_pose_valid"] = True
                metadata["head_pose_2d"] = np.asarray(result.head_pose_2d, dtype=np.float32)
                metadata["head_pose_2d_valid"] = True
            if result.eye_pose_2d is not None:
                metadata["iris_pose_2d"] = np.asarray(result.eye_pose_2d, dtype=np.float32)
            if result.eyelid_direction_angles_normalized is not None:
                metadata["eye_angles"] = np.asarray(
                    result.eyelid_direction_angles_normalized, dtype=np.float32
                )
                metadata["eye_angles_degrees"] = np.asarray(
                    result.eyelid_direction_angles_degrees, dtype=np.float32
                )
                metadata["eye_angles_definition"] = (
                    "a0_to_a1_and_a0_to_a2_directions_in_semantic_eye_local_frame"
                )
            metadata["selected_eye"] = selected_eye
            metadata["eye_selection_valid"] = True
            metadata["eye_patch_source_quad_xy"] = source_quad
            metadata["profile_eye_size_xy"] = np.asarray(result.eye_size_xy, dtype=np.float32)
            metadata["head_pitch_proxy_degrees"] = result.head_pitch_proxy_degrees
            if result.eyelid_tail_points_xy is not None:
                metadata["eyelid_tail_points_xy"] = np.asarray(
                    result.eyelid_tail_points_xy, dtype=np.float32
                )
                metadata["eyelid_tail_vectors_px"] = np.asarray(
                    result.eyelid_tail_vectors_px, dtype=np.float32
                )
                metadata["eyelid_included_angle"] = result.eyelid_tail_angle_normalized
                metadata["eyelid_included_angle_degrees"] = result.eyelid_tail_angle_degrees

            return

        landmarks = _get_landmarks(sample)
        if landmarks is None:
            default_size = (
                [128, 256]
                if method in {"single_eye", "single_eye_homography", "selected_eye_similarity"}
                else [128, 512]
            )
            if mark_invalid_warp("facial landmarks are unavailable", default_size):
                return
            _handle_failure("eye_region_warp", config, "facial landmarks are unavailable")
            return
        if str(_metadata(sample).get("landmark_kind", "")) == "face_bbox_hull":
            if mark_invalid_warp(
                "OpenCV Haar fallback provides a face hull, not eye landmarks",
                [128, 512],
            ):
                return
            _handle_failure(
                "eye_region_warp",
                config,
                "OpenCV Haar fallback provides a face hull, not eye landmarks; "
                "supply annotated/detected eye landmarks or disable eye_region_warp",
            )
            return
        if method in {
            "webeyetrack",
            "webeyetrack_homography",
            "webeyetrack_obtain_eyepatch_v1",
        }:
            try:
                patch, matrix, debug = webeyetrack_eye_patch(
                    np.asarray(sample["image"]),
                    landmarks,
                    radial_padding_xy=config.get("radial_padding_xy", [0.4, 0.2]),
                    face_crop_size=int(config.get("face_crop_size", 512)),
                    output_size_hw=config.get("size_hw", [128, 512]),
                )
            except WebEyeTrackGeometryError as exc:
                if mark_invalid_warp(str(exc), [128, 512]):
                    return
                _handle_failure("eye_region_warp", config, str(exc))
                return
            sample["image"] = patch
            _update_geometry(sample, matrix)
            metadata = _metadata(sample)
            metadata["representation"] = "webeyetrack_binocular_eye_patch_v1"
            metadata["eye_region_warp_valid"] = True
            metadata["eye_patch_source_quad_xy"] = debug["source_quad_xy"]
            metadata["eye_patch_crop_y"] = debug["crop_y"]
            metadata["face_homography"] = debug["face_homography"]
            if bool(config.get("keep_debug_intermediates", False)):
                metadata["warped_face_debug"] = debug["warped_face"]
            return

        if method in {
            "single_eye",
            "single_eye_homography",
            "selected_eye_similarity",
        }:
            selected_eye = str(_metadata(sample).get("selected_eye", "")).lower()
            try:
                patch, matrix, source_quad = single_eye_patch(
                    np.asarray(sample["image"]),
                    landmarks,
                    selected_eye,
                    output_size_hw=config.get("size_hw", [128, 256]),
                    horizontal_margin_ratio=float(config.get("horizontal_margin_ratio", 0.45)),
                    vertical_margin_ratio=float(config.get("vertical_margin_ratio", 1.25)),
                )
            except WebEyeTrackGeometryError as exc:
                if mark_invalid_warp(str(exc), [128, 256]):
                    return
                _handle_failure("eye_region_warp", config, str(exc))
                return
            sample["image"] = patch
            _update_geometry(sample, matrix)
            metadata = _metadata(sample)
            metadata["representation"] = f"side_{selected_eye}_eye_patch"
            metadata["eye_region_warp_valid"] = True
            metadata["eye_patch_source_quad_xy"] = source_quad
            return

        if method != "landmark_homography":
            raise TransformConfigError(
                f"eye_region_warp.method={method!r} is unsupported; use "
                "landmark_homography, webeyetrack_homography, or single_eye_homography"
            )
        output_height, output_width = _as_size_hw(
            config.get("size_hw"), name="eye_region_warp.size_hw"
        )
        try:
            source_quad = _eye_quad(landmarks, config)
            destination = np.asarray(
                [
                    [0, 0],
                    [output_width - 1, 0],
                    [output_width - 1, output_height - 1],
                    [0, output_height - 1],
                ],
                dtype=np.float32,
            )
            cv2 = _opencv("eye_region_warp")
            matrix = cv2.getPerspectiveTransform(source_quad, destination)
            if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
                raise SampleValidationError("computed eye homography is singular")
            border = _as_rgb_triplet(
                config.get("pad_rgb", [0, 0, 0]), name="eye_region_warp.pad_rgb"
            )
            warped = cv2.warpPerspective(
                np.asarray(sample["image"]),
                matrix,
                (output_width, output_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border,
            )
        except (GazeTransformError, TransformConfigError) as exc:
            policy = str(config.get("on_failure", "error")).lower()
            if policy in {"use_full_image", "skip", "noop", "no_op"}:
                return
            if policy == "drop":
                raise SampleDroppedError(
                    f"preprocessing stage 'eye_region_warp' failed: {exc}"
                ) from exc
            raise
        sample["image"] = np.ascontiguousarray(warped)
        _update_geometry(sample, matrix)
        _metadata(sample)["representation"] = "eye_region_homography"

    def _stage_face_roi(self, sample: MutableMapping[str, Any], config: Mapping[str, Any]) -> None:
        landmarks = _get_landmarks(sample)
        if landmarks is None:
            if str(config.get("on_failure", "error")).lower() in {
                "mark_invalid",
                "mask",
                "keep_invalid",
            }:
                sample["image"] = np.zeros_like(np.asarray(sample["image"]), dtype=np.uint8)
                metadata = _metadata(sample)
                metadata["face_roi_valid"] = False
                metadata["representation"] = "invalid_face_roi"
                _mark_gaze_invalid(sample, "face_roi:facial landmarks are unavailable")
                return
            _handle_failure("face_roi", config, "facial landmarks are unavailable")
            return
        image = np.asarray(sample["image"])
        try:
            x0, y0, x1, y1 = _bbox_from_landmarks(
                image,
                landmarks,
                float(config.get("margin_ratio", 0.20)),
            )
        except SampleValidationError as exc:
            policy = str(config.get("on_failure", "error")).lower()
            if policy in {"mark_invalid", "mask", "keep_invalid"}:
                sample["image"] = np.zeros_like(image, dtype=np.uint8)
                metadata = _metadata(sample)
                metadata["face_roi_valid"] = False
                metadata["representation"] = "invalid_face_roi"
                _mark_gaze_invalid(sample, f"face_roi:{exc}")
                return
            if policy in {"use_full_image", "skip", "noop", "no_op"}:
                height, width = image.shape[:2]
                _metadata(sample)["face_roi_xyxy"] = np.asarray(
                    [0, 0, width, height], dtype=np.int32
                )
                return
            if policy == "drop":
                raise SampleDroppedError(f"preprocessing stage 'face_roi' failed: {exc}") from exc
            raise

        metadata = _metadata(sample)
        metadata["face_roi_valid"] = True
        metadata["face_roi_xyxy"] = np.asarray([x0, y0, x1, y1], dtype=np.int32)
        mode = str(config.get("mode", "crop")).lower()
        if mode in {"preserve_canvas", "keep_canvas", "mask_only"}:
            metadata["representation"] = "face_roi_full_canvas"
            return
        if mode not in {"crop", "crop_to_roi"}:
            raise TransformConfigError(
                f"face_roi.mode={mode!r} is unsupported; use crop or preserve_canvas"
            )

        cropped = image[y0:y1, x0:x1].copy()
        matrix = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
        sample["image"] = cropped
        _update_geometry(sample, matrix)
        metadata["representation"] = "face_roi"

    def _stage_background_mask(
        self, sample: MutableMapping[str, Any], config: Mapping[str, Any]
    ) -> None:
        method = str(config.get("method", "face_hull")).lower()
        if method in {"none", "noop", "no_op", "already_masked", "existing_black_canvas"}:
            return
        landmarks = _get_landmarks(sample)
        if landmarks is None:
            _handle_failure("background_mask", config, "facial landmarks are unavailable")
            return
        image = np.asarray(sample["image"])
        height, width = image.shape[:2]
        cv2 = _opencv("background_mask")
        mask = np.zeros((height, width), dtype=np.uint8)
        if method == "face_hull":
            hull = cv2.convexHull(np.rint(landmarks).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
        elif method == "bbox":
            minimum = np.floor(landmarks.min(axis=0)).astype(int)
            maximum = np.ceil(landmarks.max(axis=0)).astype(int)
            x0, y0 = np.maximum(minimum, 0)
            x1, y1 = np.minimum(maximum + 1, [width, height])
            if x1 <= x0 or y1 <= y0:
                _handle_failure("background_mask", config, "landmark bounding box is empty")
                return
            mask[y0:y1, x0:x1] = 255
        elif method in {"face_roi_bbox", "roi_bbox"}:
            roi = _metadata(sample).get("face_roi_xyxy")
            if roi is None:
                _handle_failure(
                    "background_mask",
                    config,
                    "face_roi_bbox requires face_roi.mode=preserve_canvas before masking",
                )
                return
            x0, y0, x1, y1 = np.asarray(roi, dtype=int).reshape(4)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(width, x1), min(height, y1)
            if x1 <= x0 or y1 <= y0:
                _handle_failure("background_mask", config, "face ROI bounding box is empty")
                return
            mask[y0:y1, x0:x1] = 255
        else:
            raise TransformConfigError(
                f"background_mask.method={method!r} is unsupported; "
                "use face_hull, bbox, face_roi_bbox, or already_masked"
            )
        feather = int(config.get("feather_px", 0))
        if feather > 0:
            kernel = feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel, kernel), sigmaX=0)
        alpha = mask.astype(np.float32)[..., None] / 255.0
        fill = np.asarray(
            _as_rgb_triplet(config.get("fill_rgb", [0, 0, 0]), name="background_mask.fill_rgb"),
            dtype=np.float32,
        )
        masked = image.astype(np.float32) * alpha + fill * (1.0 - alpha)
        sample["image"] = np.rint(np.clip(masked, 0, 255)).astype(np.uint8)

    def _stage_resize(self, sample: MutableMapping[str, Any], config: Mapping[str, Any]) -> None:
        size = _as_size_hw(config.get("size_hw"), name="resize.size_hw")
        image = np.asarray(sample["image"])
        interpolation_key = "interpolation_train" if self.split == "train" else "interpolation_eval"
        interpolation = str(config.get(interpolation_key, config.get("interpolation", "bilinear")))
        if bool(config.get("keep_aspect_ratio", True)):
            resized, matrix = letterbox_resize(
                image,
                size,
                pad_rgb=_as_rgb_triplet(config.get("pad_rgb", [0, 0, 0]), name="resize.pad_rgb"),
                interpolation=interpolation,
            )
        else:
            output_height, output_width = size
            source_height, source_width = image.shape[:2]
            resized = np.asarray(
                Image.fromarray(image).resize(
                    (output_width, output_height), resample=_pil_resampling(interpolation)
                )
            )
            matrix = np.array(
                [
                    [output_width / source_width, 0, 0],
                    [0, output_height / source_height, 0],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
        sample["image"] = np.ascontiguousarray(resized)
        _update_geometry(sample, matrix)

    def _stage_normalize(self, sample: MutableMapping[str, Any], config: Mapping[str, Any]) -> None:
        image = sample.get("image")
        if torch.is_tensor(image):
            tensor = image.float()
            if tensor.ndim != 3:
                raise SampleValidationError("normalize expects a 3D image")
            if tensor.shape[-1] == 3 and tensor.shape[0] != 3:
                tensor = tensor.permute(2, 0, 1)
        else:
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] != 3:
                raise SampleValidationError("normalize expects RGB image shape [H, W, 3]")
            tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()

        mode = str(config.get("mode", "zero_one")).lower()
        if mode in {"zero_one", "0_1", "unit"}:
            if tensor.max().item() > 1.0:
                tensor = tensor / 255.0
        elif mode in {"mean_std", "standardize", "imagenet"}:
            if tensor.max().item() > 1.0:
                tensor = tensor / 255.0
            if mode == "imagenet":
                mean = config.get("mean") or [0.485, 0.456, 0.406]
                std = config.get("std") or [0.229, 0.224, 0.225]
            else:
                mean, std = config.get("mean"), config.get("std")
            if mean is None or std is None:
                raise TransformConfigError(
                    "mean/std normalization requires normalize.mean and normalize.std"
                )
            mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
            std_tensor = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
            if torch.any(std_tensor <= 0):
                raise TransformConfigError("normalize.std values must be greater than zero")
            tensor = (tensor - mean_tensor) / std_tensor
        elif mode in {"none", "identity", "raw"}:
            pass
        else:
            raise TransformConfigError(
                f"normalize.mode={mode!r} is unsupported; use zero_one, mean_std, imagenet, or none"
            )

        dtype_name = str(config.get("output_dtype", "float32")).lower()
        dtypes = {"float32": torch.float32, "float": torch.float32, "float16": torch.float16}
        if dtype_name not in dtypes:
            raise TransformConfigError("normalize.output_dtype must be float32 or float16")
        tensor = tensor.to(dtypes[dtype_name])
        channel_order = str(config.get("channel_order", "CHW")).upper()
        if channel_order == "HWC":
            tensor = tensor.permute(1, 2, 0)
        elif channel_order != "CHW":
            raise TransformConfigError("normalize.channel_order must be CHW or HWC")
        sample["image"] = tensor.contiguous()

    def _stage_augment(self, sample: MutableMapping[str, Any], config: Mapping[str, Any]) -> None:
        apply_to = config.get("apply_to", "train_only")
        allowed = (
            self.split == "train"
            if str(apply_to).lower() == "train_only"
            else (
                self.split in apply_to
                if isinstance(apply_to, Sequence) and not isinstance(apply_to, str)
                else True
            )
        )
        if not allowed:
            return

        context = sample.get("_augmentation_context")
        if context is None:
            context = self.make_augmentation_context(str(sample.get("view", "front")))
        if not isinstance(context, Mapping):
            raise TransformConfigError("_augmentation_context must be a mapping")

        jitter = _mapping(config.get("color_jitter"), name="augment.color_jitter")
        if jitter and _enabled(jitter):
            array, origin, original_dtype = _image_to_uint8_hwc(
                sample["image"], stage="color jitter"
            )
            image = Image.fromarray(array)
            if "brightness" in context:
                image = ImageEnhance.Brightness(image).enhance(float(context["brightness"]))
            if "contrast" in context:
                image = ImageEnhance.Contrast(image).enhance(float(context["contrast"]))
            if "saturation" in context:
                image = ImageEnhance.Color(image).enhance(float(context["saturation"]))
            hue = float(context.get("hue", 0.0))
            if hue:
                hsv = np.asarray(image.convert("HSV")).copy()
                shift = int(round(hue * 255))
                hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
                image = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")
            sample["image"] = _restore_augmented_image(np.asarray(image), origin, original_dtype)

        if bool(context.get("horizontal_flip", False)):
            image = sample["image"]
            if torch.is_tensor(image):
                channel_order = str(
                    self._stage_config("normalize", str(sample.get("view", "front"))).get(
                        "channel_order", "CHW"
                    )
                ).upper()
                width_axis = 2 if channel_order == "CHW" else 1
                width = int(image.shape[width_axis])
                sample["image"] = torch.flip(image, dims=(width_axis,))
            else:
                width = int(np.asarray(image).shape[1])
                sample["image"] = np.ascontiguousarray(np.asarray(image)[:, ::-1])
            matrix = np.array([[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]], dtype=float)
            _update_geometry(sample, matrix)
            self._flip_target_and_pose(sample, config)

        sample.pop("_augmentation_context", None)

    def _flip_target_and_pose(
        self, sample: MutableMapping[str, Any], config: Mapping[str, Any]
    ) -> None:
        metadata = _metadata(sample)
        if (
            bool(config.get("transform_target_gaze", True))
            and sample.get("target_gaze_xy") is not None
        ):
            target = torch.as_tensor(sample["target_gaze_xy"], dtype=torch.float32).clone()
            if target.numel() != 2:
                raise SampleValidationError("target_gaze_xy must contain x and y")
            coordinate = _mapping(
                self.task_config.get("coordinate_system"), name="task.coordinate_system"
            )
            name = str(coordinate.get("name", "centered_normalized_screen")).lower()
            screen_size = np.asarray(
                metadata.get("screen_size_px", [np.nan, np.nan]), dtype=np.float64
            ).reshape(-1)
            screen_target = np.asarray(
                metadata.get("gaze_screen_xy_px", [np.nan, np.nan]), dtype=np.float64
            ).reshape(-1)
            has_pixel_target = (
                len(screen_size) >= 2
                and len(screen_target) >= 2
                and np.isfinite(screen_size[0])
                and screen_size[0] > 0
                and np.isfinite(screen_target[0])
            )
            if has_pixel_target:
                screen_target = screen_target.copy()
                screen_target[0] = screen_size[0] - 1.0 - screen_target[0]
                metadata["gaze_screen_xy_px"] = screen_target.astype(np.float32)
                if "centered" in name:
                    target[0] = float(screen_target[0] / screen_size[0] - 0.5)
                elif "unit" in name or "zero_one" in name:
                    target[0] = float(screen_target[0] / screen_size[0])
                else:
                    raise TransformConfigError(
                        "pixel-consistent horizontal gaze flipping requires a centered "
                        "or unit normalized task coordinate system"
                    )
            elif "centered" in name:
                target[0] = -target[0]
            elif "unit" in name or "zero_one" in name:
                target[0] = 1.0 - target[0]
            else:
                raise TransformConfigError(
                    "horizontal gaze-label flipping requires a centered or unit "
                    "normalized task coordinate system"
                )
            sample["target_gaze_xy"] = target

        # Mirror camera-space annotations when they are present.  Image-only
        # runs do not pay the OpenCV dependency cost here.
        rotation = metadata.get("head_rotation_3d", sample.get("head_rotation_3d"))
        if rotation is not None:
            cv2 = _opencv("augment.horizontal_flip")
            rotation_vector = np.asarray(rotation, dtype=np.float64).reshape(3, 1)
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            reflection = np.diag([-1.0, 1.0, 1.0])
            mirrored, _ = cv2.Rodrigues(reflection @ rotation_matrix @ reflection)
            metadata["head_rotation_3d"] = mirrored.reshape(3).astype(np.float32)
        for key in (
            "head_translation_3d",
            "face_center_3d",
            "face_origin_3d",
            "gaze_target_3d",
            "head_vector",
        ):
            value = metadata.get(key, sample.get(key))
            if value is not None:
                mirrored = np.asarray(value, dtype=np.float32).copy().reshape(-1)
                if len(mirrored) >= 1:
                    mirrored[0] *= -1
                metadata[key] = mirrored
        for key in ("head_pose_2d", "eye_pose_2d", "iris_pose_2d"):
            value = metadata.get(key, sample.get(key))
            if value is not None:
                mirrored = np.asarray(value, dtype=np.float32).copy().reshape(2)
                mirrored[0] *= -1
                metadata[key] = mirrored
        for key in ("evaluation_eye", "selected_eye", "visible_eye"):
            eye = str(metadata.get(key, "")).lower()
            if eye == "left":
                metadata[key] = "right"
            elif eye == "right":
                metadata[key] = "left"


# Short public alias used by dataset/config integrations.
GazePreprocessor = OrderedGazePreprocessor


__all__ = [
    "GazePreprocessor",
    "GazeTransformError",
    "OpenCVHaarFaceDetector",
    "OrderedGazePreprocessor",
    "SampleDroppedError",
    "SampleValidationError",
    "TransformConfigError",
    "TransformDependencyError",
    "letterbox_resize",
    "transform_points",
]
