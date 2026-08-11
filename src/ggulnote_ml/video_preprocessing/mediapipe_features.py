from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional, Sequence, Tuple

from ggulnote_ml.exceptions import ContractError
from ggulnote_ml.video_preprocessing.contracts import VIDEO_FEATURE_NAMES
from ggulnote_ml.video_preprocessing.eye_state import (
    LEFT_EYE_EAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
    EyeState,
    calculate_eye_state,
)
from ggulnote_ml.video_preprocessing.mediapipe_landmarks import (
    FaceIrisLandmarks,
    NormalizedLandmark,
)


LEFT_IRIS_CENTER_INDEX = 473
RIGHT_IRIS_CENTER_INDEX = 468


@dataclass(frozen=True)
class VideoFrameFeatures:
    face_detected: bool
    iris_detected: bool
    landmark_count: int
    eye_state: Optional[EyeState]
    values: Optional[Tuple[float, ...]]
    feature_valid: bool
    invalid_reason: str

    def __post_init__(self) -> None:
        if self.values is not None and len(self.values) != len(VIDEO_FEATURE_NAMES):
            raise ContractError("Video feature vector must contain exactly 8 values.")


def extract_video_frame_features(
    result: FaceIrisLandmarks, ear_threshold: float
) -> VideoFrameFeatures:
    if not result.face_detected:
        return VideoFrameFeatures(
            False, False, result.landmark_count, None, None, False, "face_not_detected"
        )
    if not result.iris_detected:
        return VideoFrameFeatures(
            True, False, result.landmark_count, None, None, False, "iris_not_detected"
        )
    try:
        eye_state = calculate_eye_state(result.landmarks, ear_threshold)
        values = _feature_values(result.landmarks)
    except ContractError:
        return VideoFrameFeatures(
            True,
            True,
            result.landmark_count,
            None,
            None,
            False,
            "invalid_landmark_geometry",
        )
    if eye_state.eye_closed:
        return VideoFrameFeatures(
            True, True, result.landmark_count, eye_state, values, False, "eye_closed"
        )
    return VideoFrameFeatures(
        True, True, result.landmark_count, eye_state, values, True, ""
    )


def _feature_values(
    landmarks: Sequence[NormalizedLandmark],
) -> Tuple[float, ...]:
    required = (
        *LEFT_EYE_EAR_INDICES,
        *RIGHT_EYE_EAR_INDICES,
        LEFT_IRIS_CENTER_INDEX,
        RIGHT_IRIS_CENTER_INDEX,
    )
    if len(landmarks) <= max(required):
        raise ContractError("Refined eye and iris landmarks are incomplete.")
    left_eye = _center(landmarks, LEFT_EYE_EAR_INDICES)
    right_eye = _center(landmarks, RIGHT_EYE_EAR_INDICES)
    left_iris = landmarks[LEFT_IRIS_CENTER_INDEX]
    right_iris = landmarks[RIGHT_IRIS_CENTER_INDEX]
    values = (
        left_eye[0],
        left_eye[1],
        left_iris.x,
        left_iris.y,
        right_eye[0],
        right_eye[1],
        right_iris.x,
        right_iris.y,
    )
    if not all(isfinite(value) for value in values):
        raise ContractError("Video features must be finite.")
    return values


def _center(
    landmarks: Sequence[NormalizedLandmark], indices: Sequence[int]
) -> Tuple[float, float]:
    count = float(len(indices))
    return (
        sum(landmarks[index].x for index in indices) / count,
        sum(landmarks[index].y for index in indices) / count,
    )
