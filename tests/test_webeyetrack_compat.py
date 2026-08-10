from __future__ import annotations

import cv2
import numpy as np
import pytest

from gaze_pipeline.data.transforms import OrderedGazePreprocessor
from gaze_pipeline.data.webeyetrack_compat import (
    compute_ear,
    head_vector_from_face_rt,
    select_eye,
    webeyetrack_eye_patch,
)


def _landmarks() -> np.ndarray:
    points = np.tile(np.asarray([[70.0, 45.0]], dtype=np.float32), (478, 1))
    points[[103, 150, 379, 332]] = np.asarray(
        [[30, 12], [28, 78], [112, 80], [110, 10]], dtype=np.float32
    )
    points[4] = [70, 44]
    points[151] = [70, 31]
    points[195] = [70, 53]
    points[[362, 385, 387, 263, 373, 380]] = np.asarray(
        [[43, 40], [47, 36], [53, 36], [59, 40], [53, 44], [47, 44]],
        dtype=np.float32,
    )
    points[[133, 158, 160, 33, 144, 153]] = np.asarray(
        [[79, 40], [83, 37], [89, 37], [95, 40], [89, 43], [83, 43]],
        dtype=np.float32,
    )
    return points


def _official_reference(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    source = landmarks[np.asarray([103, 150, 379, 332])].copy()
    source += np.asarray([0.4, 0.2], dtype=np.float32) * (source - landmarks[4])
    destination = np.asarray(
        [[0, 0], [0, 512], [512, 512], [512, 0]],
        dtype=np.float32,
    )
    matrix, _ = cv2.findHomography(source, destination)
    warped = cv2.warpPerspective(image, matrix, (512, 512))
    homogeneous = np.vstack((landmarks.T, np.ones((1, len(landmarks)))))
    warped_landmarks = matrix @ homogeneous
    warped_landmarks = (warped_landmarks[:2] / warped_landmarks[2]).T.astype(np.int32)
    band = warped[warped_landmarks[151, 1] : warped_landmarks[195, 1], :]
    return cv2.resize(band, (512, 128))


def test_eye_patch_matches_official_obtain_eyepatch_pixels() -> None:
    yy, xx = np.mgrid[:90, :140]
    image = np.stack((xx, yy, (xx + yy) % 255), axis=-1).astype(np.uint8)
    points = _landmarks()

    actual, transform, debug = webeyetrack_eye_patch(image, points)
    expected = _official_reference(image, points)

    assert actual.shape == (128, 512, 3)
    assert np.array_equal(actual, expected)
    assert transform.shape == (3, 3)
    assert debug["source_quad_xy"].shape == (4, 2)


def test_ear_and_visible_eye_selection() -> None:
    points = _landmarks()

    assert compute_ear(points, "left") == pytest.approx(0.5)
    assert compute_ear(points, "right") == pytest.approx(0.375)
    selected, scores = select_eye(points, mode="best_visible")

    assert selected == "left"
    assert scores["left"] == pytest.approx(1.0)
    assert scores["right"] == pytest.approx(1.0)


def test_identity_face_transform_points_forward() -> None:
    vector, mapped_euler = head_vector_from_face_rt(np.eye(4, dtype=np.float32))

    assert vector.tolist() == pytest.approx([0.0, 0.0, -1.0])
    assert mapped_euler.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_closed_selected_eye_marks_side_gaze_invalid() -> None:
    points = _landmarks()
    # Collapse the right-eye vertical distances below the 0.20 threshold.
    points[[158, 160, 144, 153], 1] = 40.0
    config = {
        "stage_order": ["decode", "face_landmarks", "eye_selection", "eye_state"],
        "decode": {"enabled": True},
        "face_landmarks": {"enabled": True, "source": "annotation"},
        "eye_selection": {
            "enabled": True,
            "mode": "fixed",
            "target_eye": "right",
        },
        "eye_state": {
            "enabled": True,
            "threshold": 0.20,
            "required_eye_policy": "selected_eye_open",
            "on_closed": "mark_invalid",
        },
    }
    sample = {
        "image": np.zeros((90, 140, 3), dtype=np.uint8),
        "view": "side",
        "facial_landmarks_xy": points,
        "metadata": {},
    }

    output = OrderedGazePreprocessor(config, split="validation")(sample)

    assert output["metadata"]["selected_eye"] == "right"
    assert output["metadata"]["eye_open_mask"].tolist() == [True, False]
    assert output["metadata"]["gaze_valid"] is False
    assert "eye_state:required_eye_closed" in output["metadata"]["invalid_reasons"]
