from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from ggulnote_ml.exceptions import ContractError, OptionalDependencyError


MIN_FACE_LANDMARK_COUNT = 468
MIN_REFINED_LANDMARK_COUNT = 478


@dataclass(frozen=True)
class NormalizedLandmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class FaceIrisLandmarks:
    face_detected: bool
    iris_detected: bool
    landmarks: Tuple[NormalizedLandmark, ...]

    @property
    def landmark_count(self) -> int:
        return len(self.landmarks)


class MediaPipeFaceIrisExtractor:
    """Extract one face and refined iris landmarks from independent video frames."""

    def __init__(
        self,
        mediapipe_module: Any = None,
        min_detection_confidence: float = 0.5,
    ) -> None:
        if not 0.0 < min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be in (0,1].")
        self._mediapipe = mediapipe_module
        self._min_detection_confidence = min_detection_confidence
        self._processor: Any = None

    def __enter__(self) -> "MediaPipeFaceIrisExtractor":
        if self._mediapipe is None:
            try:
                import mediapipe as mediapipe_module
            except ImportError as exc:
                raise OptionalDependencyError(
                    "Video landmark extraction requires: "
                    "pip install -e '.[landmarks]'."
                ) from exc
            self._mediapipe = mediapipe_module
        self._processor = self._mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self._min_detection_confidence,
        )
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._processor is not None:
            self._processor.close()
            self._processor = None

    def extract(self, bgr_frame: np.ndarray) -> FaceIrisLandmarks:
        if self._processor is None:
            raise RuntimeError("MediaPipeFaceIrisExtractor must be opened first.")
        if (
            bgr_frame.dtype != np.uint8
            or bgr_frame.ndim != 3
            or bgr_frame.shape[2] != 3
        ):
            raise ContractError("MediaPipe input must be uint8 BGR [H,W,3].")
        rgb_frame = np.ascontiguousarray(bgr_frame[:, :, ::-1])
        result = self._processor.process(rgb_frame)
        faces = getattr(result, "multi_face_landmarks", None)
        if not faces:
            return FaceIrisLandmarks(False, False, ())
        raw_landmarks = getattr(faces[0], "landmark", ())
        landmarks = tuple(
            NormalizedLandmark(float(item.x), float(item.y), float(item.z))
            for item in raw_landmarks
        )
        if len(landmarks) < MIN_FACE_LANDMARK_COUNT:
            return FaceIrisLandmarks(False, False, landmarks)
        return FaceIrisLandmarks(
            face_detected=True,
            iris_detected=len(landmarks) >= MIN_REFINED_LANDMARK_COUNT,
            landmarks=landmarks,
        )
