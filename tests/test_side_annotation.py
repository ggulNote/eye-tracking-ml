from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gaze_pipeline.data.side_annotation import (
    annotation_from_bbox_candidate,
    annotation_from_landmarks,
    annotation_from_temporal_neighbors,
    bbox_as_ints,
)
from gaze_pipeline.data.webeyetrack_compat import (
    LEFT_EYELID_INDICES,
    LEFT_IRIS_INDICES,
    RIGHT_EYELID_INDICES,
    RIGHT_IRIS_INDICES,
)


def _landmarks() -> np.ndarray:
    points = np.zeros((478, 2), dtype=np.float32)
    right = np.asarray(
        [[30, 50], [42, 45], [58, 45], [70, 50], [58, 55], [42, 55]],
        dtype=np.float32,
    )
    left = np.asarray(
        [[115, 50], [118, 48], [122, 48], [125, 50], [122, 52], [118, 52]],
        dtype=np.float32,
    )
    points[np.asarray(RIGHT_EYELID_INDICES)] = right
    points[np.asarray(LEFT_EYELID_INDICES)] = left
    points[np.asarray(RIGHT_IRIS_INDICES)] = [50, 50]
    points[np.asarray(LEFT_IRIS_INDICES)] = [120, 50]
    return points


def test_visible_profile_eye_generates_valid_aspect_preserving_bbox() -> None:
    annotation = annotation_from_landmarks(
        _landmarks(),
        image_size_hw=(100, 200),
        presence=np.ones(478, dtype=np.float32),
        visibility=np.ones(478, dtype=np.float32),
    )

    assert annotation.visible_eye == "right"
    assert annotation.valid is True
    assert annotation.quality_flags == ()
    assert bbox_as_ints(annotation) == (2, 26, 98, 74)
    assert annotation.iris_center_xy == pytest.approx((50.0, 50.0))
    assert len(annotation.eyelid_keypoints_xy or ()) == 6


def test_detector_bbox_requires_review_until_explicitly_accepted() -> None:
    candidate = annotation_from_bbox_candidate(
        [10, 20, 90, 60],
        image_size_hw=(100, 200),
        visible_eye="right",
        method="opencv_haar_profile_eye",
    )
    accepted = annotation_from_bbox_candidate(
        [10, 20, 90, 60],
        image_size_hw=(100, 200),
        visible_eye="right",
        method="manual_override",
        accept=True,
    )

    assert candidate.valid is False
    assert candidate.quality_flags == ("manual_review_required",)
    assert accepted.valid is True
    assert accepted.quality_flags == ()


def test_low_confidence_landmarks_are_sent_to_review() -> None:
    points = _landmarks()
    points[np.asarray(LEFT_EYELID_INDICES)] = np.asarray(
        [[82, 50], [92, 45], [110, 45], [120, 50], [110, 55], [92, 55]],
        dtype=np.float32,
    )
    points[np.asarray(LEFT_IRIS_INDICES)] = [101, 50]

    annotation = annotation_from_landmarks(
        points,
        image_size_hw=(100, 200),
        presence=np.ones(478, dtype=np.float32),
        visibility=np.ones(478, dtype=np.float32),
    )

    assert annotation.valid is False
    assert "low_detection_confidence" in annotation.quality_flags


def test_temporal_interpolation_auto_accepts_close_agreeing_anchors() -> None:
    previous = annotation_from_bbox_candidate(
        [10, 20, 90, 60],
        image_size_hw=(100, 200),
        visible_eye="right",
        method="manual_override",
        accept=True,
    )
    following = annotation_from_bbox_candidate(
        [12, 20, 92, 60],
        image_size_hw=(100, 200),
        visible_eye="right",
        method="manual_override",
        accept=True,
    )

    annotation = annotation_from_temporal_neighbors(
        target_position=2,
        image_size_hw=(100, 200),
        previous=(0, previous),
        following=(4, following),
    )

    assert annotation.valid is True
    assert annotation.method == "temporal_interpolation"
    assert annotation.bbox_xyxy == pytest.approx((11, 20, 91, 60))


def test_one_sided_temporal_candidate_still_requires_review() -> None:
    anchor = annotation_from_bbox_candidate(
        [10, 20, 90, 60],
        image_size_hw=(100, 200),
        visible_eye="right",
        method="manual_override",
        accept=True,
    )

    annotation = annotation_from_temporal_neighbors(
        target_position=1,
        image_size_hw=(100, 200),
        previous=(0, anchor),
    )

    assert annotation.valid is False
    assert annotation.quality_flags == ("manual_review_required",)


def test_temporal_bbox_can_use_a_photometrically_invalid_anchor() -> None:
    anchor = annotation_from_bbox_candidate(
        [10, 20, 90, 60],
        image_size_hw=(100, 200),
        visible_eye="right",
        method="manual_override",
        accept=True,
    )
    blurry_anchor = replace(anchor, valid=False, quality_flags=("low_sharpness",))

    annotation = annotation_from_temporal_neighbors(
        target_position=3,
        image_size_hw=(100, 200),
        previous=(0, anchor),
        following=(15, blurry_anchor),
    )

    assert annotation.valid is True
