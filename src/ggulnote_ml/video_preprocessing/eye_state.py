from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Sequence, Tuple

from ggulnote_ml.exceptions import ContractError
from ggulnote_ml.video_preprocessing.mediapipe_landmarks import NormalizedLandmark


# Anatomical left/right landmark order: outer, upper-outer, upper-inner,
# inner, lower-inner, lower-outer.
LEFT_EYE_EAR_INDICES = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_EAR_INDICES = (33, 160, 158, 133, 153, 144)


@dataclass(frozen=True)
class EyeState:
    left_ear: float
    right_ear: float
    left_eye_closed: bool
    right_eye_closed: bool
    eye_closed: bool


def calculate_ear(
    landmarks: Sequence[NormalizedLandmark],
    indices: Tuple[int, int, int, int, int, int],
) -> float:
    if len(landmarks) <= max(indices):
        raise ContractError("Face landmarks do not contain the required eyelid points.")
    p1, p2, p3, p4, p5, p6 = (landmarks[index] for index in indices)
    horizontal = _distance(p1, p4)
    if horizontal <= 1e-12:
        raise ContractError("Eye corner distance is zero.")
    ear = (_distance(p2, p6) + _distance(p3, p5)) / (2.0 * horizontal)
    if not isfinite(ear) or ear < 0.0:
        raise ContractError("EAR must be finite and non-negative.")
    return ear


def calculate_eye_state(
    landmarks: Sequence[NormalizedLandmark], ear_threshold: float
) -> EyeState:
    if not 0.0 < ear_threshold < 1.0:
        raise ValueError("ear_threshold must be in (0,1).")
    left_ear = calculate_ear(landmarks, LEFT_EYE_EAR_INDICES)
    right_ear = calculate_ear(landmarks, RIGHT_EYE_EAR_INDICES)
    left_closed = left_ear < ear_threshold
    right_closed = right_ear < ear_threshold
    return EyeState(
        left_ear=left_ear,
        right_ear=right_ear,
        left_eye_closed=left_closed,
        right_eye_closed=right_closed,
        eye_closed=left_closed and right_closed,
    )


def _distance(first: NormalizedLandmark, second: NormalizedLandmark) -> float:
    return hypot(first.x - second.x, first.y - second.y)
