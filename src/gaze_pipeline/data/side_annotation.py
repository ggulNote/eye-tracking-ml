"""Geometry and quality checks for automatic strict-profile eye annotations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from gaze_pipeline.data.webeyetrack_compat import (
    LEFT_EYELID_INDICES,
    LEFT_IRIS_INDICES,
    RIGHT_EYELID_INDICES,
    RIGHT_IRIS_INDICES,
    select_eye,
)

EYE_INDICES = {
    "left": LEFT_EYELID_INDICES,
    "right": RIGHT_EYELID_INDICES,
}
IRIS_INDICES = {
    "left": LEFT_IRIS_INDICES,
    "right": RIGHT_IRIS_INDICES,
}


class SideAnnotationError(ValueError):
    """Raised when an automatic annotation cannot produce finite geometry."""


@dataclass(frozen=True, slots=True)
class SideEyeAnnotation:
    visible_eye: str
    bbox_xyxy: tuple[float, float, float, float]
    eyelid_keypoints_xy: tuple[tuple[float, float], ...] | None
    iris_center_xy: tuple[float, float] | None
    confidence: float
    valid: bool
    quality_flags: tuple[str, ...]
    method: str
    visibility_scores: dict[str, float]


def _finite_points(value: Any, *, name: str, minimum_rows: int) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < minimum_rows or points.shape[1] < 2:
        raise SideAnnotationError(f"{name} must have shape [N, >=2]")
    if not np.isfinite(points[:, :2]).all():
        raise SideAnnotationError(f"{name} contains non-finite coordinates")
    return points


def _clip_bbox(
    bbox: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, float]:
    original_width = max(0.0, float(bbox[2] - bbox[0]))
    original_height = max(0.0, float(bbox[3] - bbox[1]))
    original_area = original_width * original_height
    clipped = np.asarray(
        [
            np.clip(bbox[0], 0.0, float(image_width)),
            np.clip(bbox[1], 0.0, float(image_height)),
            np.clip(bbox[2], 0.0, float(image_width)),
            np.clip(bbox[3], 0.0, float(image_height)),
        ],
        dtype=np.float64,
    )
    clipped_area = max(0.0, float(clipped[2] - clipped[0])) * max(
        0.0, float(clipped[3] - clipped[1])
    )
    retained = clipped_area / original_area if original_area > 0 else 0.0
    return clipped, retained


def annotation_from_landmarks(
    landmarks_xy: np.ndarray,
    *,
    image_size_hw: tuple[int, int],
    presence: np.ndarray | None = None,
    visibility: np.ndarray | None = None,
    bbox_scale_xy: tuple[float, float] = (2.4, 1.2),
    min_eye_width_px: float = 12.0,
    min_eye_opening_px: float = 1.5,
    min_visibility_dominance: float = 1.05,
    min_auto_confidence: float = 0.50,
    min_landmark_presence: float = 0.35,
    min_retained_bbox_ratio: float = 0.90,
) -> SideEyeAnnotation:
    """Create a visible-eye bbox from one 478-point MediaPipe face mesh."""

    points = _finite_points(landmarks_xy, name="landmarks_xy", minimum_rows=478)
    image_height, image_width = (int(image_size_hw[0]), int(image_size_hw[1]))
    if image_height <= 0 or image_width <= 0:
        raise SideAnnotationError("image_size_hw must contain positive values")
    visible_eye, scores = select_eye(points, mode="best_visible", presence=presence)
    other_eye = "right" if visible_eye == "left" else "left"
    eyelid = points[np.asarray(EYE_INDICES[visible_eye], dtype=int), :2]
    iris = points[np.asarray(IRIS_INDICES[visible_eye], dtype=int), :2]
    iris_center = iris.mean(axis=0)
    corner_width = float(np.linalg.norm(eyelid[3] - eyelid[0]))
    opening_a = float(np.linalg.norm(eyelid[1] - eyelid[5]))
    opening_b = float(np.linalg.norm(eyelid[2] - eyelid[4]))
    eye_opening = (opening_a + opening_b) / 2.0
    center = np.vstack([eyelid, iris_center]).mean(axis=0)
    crop_width = corner_width * float(bbox_scale_xy[0])
    crop_height = corner_width * float(bbox_scale_xy[1])
    bbox = np.asarray(
        [
            center[0] - crop_width / 2.0,
            center[1] - crop_height / 2.0,
            center[0] + crop_width / 2.0,
            center[1] + crop_height / 2.0,
        ],
        dtype=np.float64,
    )
    bbox, retained_ratio = _clip_bbox(
        bbox,
        image_width=image_width,
        image_height=image_height,
    )

    selected_score = max(0.0, float(scores[visible_eye]))
    other_score = max(0.0, float(scores[other_eye]))
    dominance = selected_score / max(other_score, 1e-6)
    presence_quality = 1.0
    presence_array = None if presence is None else np.asarray(presence, dtype=np.float64)
    if presence_array is not None and len(presence_array) > max(EYE_INDICES[visible_eye]):
        selected_presence = presence_array[np.asarray(EYE_INDICES[visible_eye], dtype=int)]
        if np.isfinite(selected_presence).any() and float(np.nanmax(selected_presence)) > 0:
            presence_quality = max(0.0, float(np.nanmean(selected_presence)))
    visibility_array = None if visibility is None else np.asarray(visibility, dtype=np.float64)
    if visibility_array is not None and len(visibility_array) > max(EYE_INDICES[visible_eye]):
        selected_visibility = visibility_array[np.asarray(EYE_INDICES[visible_eye], dtype=int)]
        if np.isfinite(selected_visibility).any() and float(np.nanmax(selected_visibility)) > 0:
            presence_quality = min(
                presence_quality,
                max(0.0, float(np.nanmean(selected_visibility))),
            )

    dominance_quality = float(np.clip((dominance - 1.0) / 0.75, 0.0, 1.0))
    size_quality = float(np.clip(corner_width / (min_eye_width_px * 2.0), 0.0, 1.0))
    confidence = float(
        np.clip(
            0.55 * dominance_quality + 0.25 * size_quality + 0.20 * presence_quality,
            0.0,
            1.0,
        )
    )

    flags: list[str] = []
    if corner_width < min_eye_width_px:
        flags.append("eye_too_small")
    if eye_opening < min_eye_opening_px:
        flags.append("eye_opening_too_small")
    if dominance < min_visibility_dominance:
        flags.append("ambiguous_visible_eye")
    if presence_quality < min_landmark_presence:
        flags.append("low_landmark_presence")
    if retained_ratio < min_retained_bbox_ratio:
        flags.append("bbox_clipped")
    if not (bbox[0] <= iris_center[0] <= bbox[2] and bbox[1] <= iris_center[1] <= bbox[3]):
        flags.append("iris_outside_bbox")
    if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
        flags.append("bbox_too_small")
    if confidence < min_auto_confidence:
        flags.append("low_detection_confidence")
    return SideEyeAnnotation(
        visible_eye=visible_eye,
        bbox_xyxy=tuple(float(value) for value in bbox),
        eyelid_keypoints_xy=tuple((float(point[0]), float(point[1])) for point in eyelid),
        iris_center_xy=(float(iris_center[0]), float(iris_center[1])),
        confidence=confidence,
        valid=not flags,
        quality_flags=tuple(flags),
        method="mediapipe_face_landmarker",
        visibility_scores={key: float(value) for key, value in scores.items()},
    )


def annotation_from_bbox_candidate(
    bbox_xyxy: Any,
    *,
    image_size_hw: tuple[int, int],
    visible_eye: str,
    method: str,
    accept: bool = False,
) -> SideEyeAnnotation:
    """Validate a detector/manual bbox and mark non-approved candidates for review."""

    bbox = np.asarray(bbox_xyxy, dtype=np.float64).reshape(-1)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise SideAnnotationError("bbox_xyxy must contain four finite values")
    image_height, image_width = image_size_hw
    bbox, retained = _clip_bbox(
        bbox,
        image_width=int(image_width),
        image_height=int(image_height),
    )
    flags: list[str] = []
    if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
        flags.append("bbox_too_small")
    if retained < 0.90:
        flags.append("bbox_clipped")
    if not accept:
        flags.append("manual_review_required")
    normalized_eye = str(visible_eye).strip().lower()
    if normalized_eye not in {"left", "right"}:
        raise SideAnnotationError("visible_eye must be left or right")
    return SideEyeAnnotation(
        visible_eye=normalized_eye,
        bbox_xyxy=tuple(float(value) for value in bbox),
        eyelid_keypoints_xy=None,
        iris_center_xy=None,
        confidence=1.0 if accept and not flags else 0.25,
        valid=not flags,
        quality_flags=tuple(flags),
        method=method,
        visibility_scores={},
    )


def annotation_from_temporal_neighbors(
    *,
    target_position: int,
    image_size_hw: tuple[int, int],
    previous: tuple[int, SideEyeAnnotation] | None = None,
    following: tuple[int, SideEyeAnnotation] | None = None,
    min_auto_confidence: float = 0.60,
    max_auto_gap: int = 20,
) -> SideEyeAnnotation:
    """Interpolate a bbox between nearby detections from the same capture sequence.

    A candidate is automatically valid only when two already-valid anchors agree,
    bracket the target, and are close in sequence order. One-sided propagation is
    still useful for manual review, but is never automatically accepted.
    """

    if previous is None and following is None:
        raise SideAnnotationError("at least one temporal neighbor is required")
    anchors = [item for item in (previous, following) if item is not None]
    eyes = {annotation.visible_eye for _, annotation in anchors}
    if len(eyes) != 1:
        raise SideAnnotationError("temporal neighbors disagree on visible_eye")
    visible_eye = eyes.pop()

    auto_accept = False
    if previous is not None and following is not None:
        previous_position, previous_annotation = previous
        following_position, following_annotation = following
        span = following_position - previous_position
        if span <= 0 or not previous_position < target_position < following_position:
            raise SideAnnotationError("temporal neighbors must bracket the target")
        weight = (target_position - previous_position) / span
        previous_bbox = np.asarray(previous_annotation.bbox_xyxy, dtype=np.float64)
        following_bbox = np.asarray(following_annotation.bbox_xyxy, dtype=np.float64)
        bbox = previous_bbox * (1.0 - weight) + following_bbox * weight

        previous_size = previous_bbox[2:] - previous_bbox[:2]
        following_size = following_bbox[2:] - following_bbox[:2]
        average_size = np.maximum((previous_size + following_size) / 2.0, 1.0)
        center_delta = np.linalg.norm(
            (previous_bbox[:2] + previous_bbox[2:]) / 2.0
            - (following_bbox[:2] + following_bbox[2:]) / 2.0
        )
        normalized_motion = float(center_delta / max(float(average_size[0]), 1.0))
        size_agreement = float(
            np.prod(
                np.minimum(previous_size, following_size)
                / np.maximum(np.maximum(previous_size, following_size), 1.0)
            )
            ** 0.5
        )
        motion_quality = math.exp(-normalized_motion)
        confidence = float(
            np.clip(
                min(previous_annotation.confidence, following_annotation.confidence)
                * size_agreement
                * motion_quality,
                0.0,
                1.0,
            )
        )
        geometry_blocking_flags = {
            "eye_too_small",
            "eye_opening_too_small",
            "ambiguous_visible_eye",
            "low_landmark_presence",
            "bbox_clipped",
            "iris_outside_bbox",
            "bbox_too_small",
            "manual_review_required",
            "low_detection_confidence",
        }
        previous_geometry_ok = not (
            geometry_blocking_flags & set(previous_annotation.quality_flags)
        )
        following_geometry_ok = not (
            geometry_blocking_flags & set(following_annotation.quality_flags)
        )
        auto_accept = (
            previous_geometry_ok
            and following_geometry_ok
            and span <= max_auto_gap
            and confidence >= min_auto_confidence
        )
        method = "temporal_interpolation"
    else:
        anchor_position, anchor = anchors[0]
        bbox = np.asarray(anchor.bbox_xyxy, dtype=np.float64)
        distance = abs(target_position - anchor_position)
        confidence = float(np.clip(anchor.confidence * math.exp(-distance / 3.0) * 0.65, 0, 1))
        method = "temporal_nearest_candidate"

    candidate = annotation_from_bbox_candidate(
        bbox,
        image_size_hw=image_size_hw,
        visible_eye=visible_eye,
        method=method,
        accept=auto_accept,
    )
    return SideEyeAnnotation(
        visible_eye=candidate.visible_eye,
        bbox_xyxy=candidate.bbox_xyxy,
        eyelid_keypoints_xy=None,
        iris_center_xy=None,
        confidence=confidence,
        valid=candidate.valid,
        quality_flags=candidate.quality_flags,
        method=candidate.method,
        visibility_scores={},
    )


def bbox_as_ints(annotation: SideEyeAnnotation) -> tuple[int, int, int, int]:
    """Return a stable integer bbox using floor for starts and ceil for ends."""

    x0, y0, x1, y1 = annotation.bbox_xyxy
    return math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)
