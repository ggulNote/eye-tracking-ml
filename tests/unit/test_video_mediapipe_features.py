from types import SimpleNamespace

import numpy as np
import pytest

from ggulnote_ml.video_preprocessing.contracts import VIDEO_FEATURE_NAMES
from ggulnote_ml.video_preprocessing.eye_state import calculate_eye_state
from ggulnote_ml.video_preprocessing.mediapipe_features import (
    extract_video_frame_features,
)
from ggulnote_ml.video_preprocessing.mediapipe_landmarks import (
    FaceIrisLandmarks,
    MediaPipeFaceIrisExtractor,
    NormalizedLandmark,
)


def _landmarks(vertical=0.03):
    values = [NormalizedLandmark(0.5, 0.5, 0.0) for _ in range(478)]
    points = {
        362: (0.40, 0.50),
        385: (0.44, 0.50 - vertical),
        387: (0.52, 0.50 - vertical),
        263: (0.60, 0.50),
        373: (0.52, 0.50 + vertical),
        380: (0.44, 0.50 + vertical),
        33: (0.10, 0.50),
        160: (0.14, 0.50 - vertical),
        158: (0.22, 0.50 - vertical),
        133: (0.30, 0.50),
        153: (0.22, 0.50 + vertical),
        144: (0.14, 0.50 + vertical),
        473: (0.51, 0.52),
        468: (0.19, 0.48),
    }
    for index, (x, y) in points.items():
        values[index] = NormalizedLandmark(x, y, 0.0)
    return tuple(values)


def test_ear_and_both_eyes_closed_policy():
    opened = calculate_eye_state(_landmarks(vertical=0.03), ear_threshold=0.20)
    closed = calculate_eye_state(_landmarks(vertical=0.005), ear_threshold=0.20)

    assert opened.left_ear == pytest.approx(0.30)
    assert opened.right_ear == pytest.approx(0.30)
    assert opened.eye_closed is False
    assert closed.left_eye_closed is True
    assert closed.right_eye_closed is True
    assert closed.eye_closed is True


def test_eight_dimensional_feature_order_and_invalid_states():
    detected = FaceIrisLandmarks(True, True, _landmarks())
    features = extract_video_frame_features(detected, ear_threshold=0.20)

    assert VIDEO_FEATURE_NAMES == (
        "left_eye_center_x",
        "left_eye_center_y",
        "left_iris_center_x",
        "left_iris_center_y",
        "right_eye_center_x",
        "right_eye_center_y",
        "right_iris_center_x",
        "right_iris_center_y",
    )
    assert features.feature_valid is True
    assert features.values == pytest.approx(
        (0.4866666667, 0.5, 0.51, 0.52, 0.1866666667, 0.5, 0.19, 0.48)
    )

    missing = extract_video_frame_features(
        FaceIrisLandmarks(False, False, ()), ear_threshold=0.20
    )
    assert missing.feature_valid is False
    assert missing.invalid_reason == "face_not_detected"
    assert missing.landmark_count == 0
    assert missing.values is None

    iris_missing = extract_video_frame_features(
        FaceIrisLandmarks(True, False, _landmarks()[:468]), ear_threshold=0.20
    )
    assert iris_missing.feature_valid is False
    assert iris_missing.invalid_reason == "iris_not_detected"

    blink = extract_video_frame_features(
        FaceIrisLandmarks(True, True, _landmarks(vertical=0.005)),
        ear_threshold=0.20,
    )
    assert blink.feature_valid is False
    assert blink.invalid_reason == "eye_closed"
    assert blink.values is not None


class _FakeFaceMesh:
    def __init__(self, result):
        self.result = result
        self.received = None
        self.closed = False

    def process(self, image):
        self.received = image.copy()
        return self.result

    def close(self):
        self.closed = True


def test_mediapipe_adapter_converts_bgr_to_rgb_and_keeps_refined_landmarks():
    raw = [SimpleNamespace(x=item.x, y=item.y, z=item.z) for item in _landmarks()]
    processor = _FakeFaceMesh(
        SimpleNamespace(
            multi_face_landmarks=[SimpleNamespace(landmark=raw)]
        )
    )
    received_options = {}

    def face_mesh_factory(**options):
        received_options.update(options)
        return processor

    module = SimpleNamespace(
        solutions=SimpleNamespace(
            face_mesh=SimpleNamespace(FaceMesh=face_mesh_factory)
        )
    )
    frame = np.asarray([[[1, 2, 3]]], dtype=np.uint8)

    with MediaPipeFaceIrisExtractor(mediapipe_module=module) as extractor:
        result = extractor.extract(frame)

    assert received_options["static_image_mode"] is True
    assert received_options["refine_landmarks"] is True
    assert processor.received.tolist() == [[[3, 2, 1]]]
    assert result.face_detected is True
    assert result.iris_detected is True
    assert result.landmark_count == 478
    assert processor.closed is True
