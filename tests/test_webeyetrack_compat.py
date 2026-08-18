from __future__ import annotations

import cv2
import numpy as np
import pytest

from gaze_pipeline.data.webeyetrack_compat import (
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


def test_visible_eye_selection() -> None:
    points = _landmarks()

    selected, scores = select_eye(points, mode="best_visible")

    assert selected == "left"
    assert scores["left"] == pytest.approx(1.0)
    assert scores["right"] == pytest.approx(1.0)


def test_identity_face_transform_points_forward() -> None:
    vector, mapped_euler = head_vector_from_face_rt(np.eye(4, dtype=np.float32))

    assert vector.tolist() == pytest.approx([0.0, 0.0, -1.0])
    assert mapped_euler.tolist() == pytest.approx([0.0, 0.0, 0.0])
