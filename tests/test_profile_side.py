from __future__ import annotations

import cv2
import numpy as np
import pytest

from gaze_pipeline.data.profile_side import (
    ProfileSideGeometryError,
    preprocess_profile_side,
    selected_eye_ear,
)


def _image() -> np.ndarray:
    yy, xx = np.mgrid[:100, :160]
    return np.stack((xx, yy, (xx + yy) % 255), axis=-1).astype(np.uint8)


def _eyelid_points() -> np.ndarray:
    return np.asarray(
        [
            [80, 50],
            [90, 44],
            [110, 44],
            [120, 50],
            [110, 56],
            [90, 56],
        ],
        dtype=np.float32,
    )


def test_affine_profile_preprocessing_returns_expected_geometry() -> None:
    result = preprocess_profile_side(
        _image(),
        eyelid_keypoints_xy=_eyelid_points(),
        iris_center_xy=[102, 46],
        head_origin_xy=[30, 80],
        head_forward_point_xy=[30, 40],
    )

    assert result.patch.shape == (128, 256, 3)
    assert result.patch.dtype == np.uint8
    assert result.source_to_patch.shape == (3, 3)
    assert result.source_quad_xy.shape == (4, 2)
    assert result.head_pose_2d.tolist() == pytest.approx([0.0, -1.0])
    assert result.eyelid_center_xy.tolist() == pytest.approx([100.0, 50.0])
    assert result.eye_size_xy.tolist() == pytest.approx([40.0, 12.0])
    assert result.eye_pose_2d.tolist() == pytest.approx([0.05, -1.0 / 3.0])
    assert result.ear == pytest.approx(0.3)
    np.testing.assert_allclose(
        result.eyelid_tail_points_xy,
        [[120.0, 50.0], [110.0, 44.0], [110.0, 56.0]],
    )
    np.testing.assert_allclose(
        result.eyelid_tail_vectors_px,
        [[-10.0, -6.0], [-10.0, 6.0]],
    )
    expected_angle_degrees = float(np.degrees(np.arccos(64.0 / 136.0)))
    expected_direction_degrees = float(np.degrees(np.arctan2(6.0, 10.0)))
    assert result.eyelid_direction_angles_degrees.tolist() == pytest.approx(
        [-expected_direction_degrees, expected_direction_degrees]
    )
    assert result.eyelid_direction_angles_normalized.tolist() == pytest.approx(
        [-expected_direction_degrees / 180.0, expected_direction_degrees / 180.0]
    )
    assert result.eyelid_tail_angle_degrees == pytest.approx(expected_angle_degrees)
    assert result.eyelid_tail_angle_normalized == pytest.approx(expected_angle_degrees / 180.0)
    assert result.head_pitch_proxy_degrees == pytest.approx(90.0)

    source_h = np.c_[result.source_quad_xy, np.ones(4, dtype=np.float32)]
    mapped_h = source_h @ result.source_to_patch.T
    mapped = mapped_h[:, :2] / mapped_h[:, 2:3]
    np.testing.assert_allclose(
        mapped,
        [[0, 0], [255, 0], [255, 127], [0, 127]],
        atol=2e-4,
    )


def test_vertical_only_preserves_vertical_magnitude_and_zeroes_horizontal() -> None:
    result = preprocess_profile_side(
        _image(),
        eyelid_keypoints_xy=_eyelid_points(),
        iris_center_xy=[104, 53],
        head_origin_xy=[20, 80],
        head_forward_point_xy=[40, 60],
        vertical_only=True,
    )

    assert result.eye_pose_2d.tolist() == pytest.approx([0.0, 0.25])
    unit = 1.0 / np.sqrt(2.0)
    assert result.head_pose_2d.tolist() == pytest.approx([unit, -unit])


def test_eye_local_direction_angles_are_invariant_to_horizontal_mirror() -> None:
    points = _eyelid_points()
    mirrored_points = points.copy()
    mirrored_points[:, 0] = 159.0 - mirrored_points[:, 0]
    original = preprocess_profile_side(
        _image(),
        eyelid_keypoints_xy=points,
        iris_center_xy=[102, 46],
        head_origin_xy=[30, 80],
        head_forward_point_xy=[30, 40],
    )
    mirrored = preprocess_profile_side(
        np.ascontiguousarray(_image()[:, ::-1]),
        eyelid_keypoints_xy=mirrored_points,
        iris_center_xy=[57, 46],
        head_origin_xy=[129, 80],
        head_forward_point_xy=[129, 40],
    )

    assert mirrored.eyelid_direction_angles_normalized.tolist() == pytest.approx(
        original.eyelid_direction_angles_normalized.tolist()
    )


def test_eye_local_direction_angles_are_invariant_to_in_plane_rotation() -> None:
    points = _eyelid_points()
    center = np.asarray([80.0, 50.0], dtype=np.float64)
    angle = 0.713
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )
    rotated_points = (points - center) @ rotation.T + center

    original = preprocess_profile_side(
        _image(),
        eyelid_keypoints_xy=points,
        extract_head_pose=False,
        extract_iris_pose=False,
    )
    rotated = preprocess_profile_side(
        _image(),
        eyelid_keypoints_xy=rotated_points,
        extract_head_pose=False,
        extract_iris_pose=False,
    )

    assert rotated.eyelid_direction_angles_normalized.tolist() == pytest.approx(
        original.eyelid_direction_angles_normalized.tolist(), abs=1e-6
    )


def test_eye_local_direction_angles_support_semantically_reordered_tail_indices() -> None:
    points = _eyelid_points()
    reordered = points[[3, 2, 1, 0, 5, 4]]
    original = preprocess_profile_side(
        _image(),
        eyelid_keypoints_xy=points,
        extract_head_pose=False,
        extract_iris_pose=False,
    )
    remapped = preprocess_profile_side(
        _image(),
        eyelid_keypoints_xy=reordered,
        eyelid_tail_indices=[0, 1, 5],
        extract_head_pose=False,
        extract_iris_pose=False,
    )

    assert remapped.eyelid_direction_angles_normalized.tolist() == pytest.approx(
        original.eyelid_direction_angles_normalized.tolist()
    )


def test_bbox_only_letterbox_returns_pose_but_not_ear() -> None:
    result = preprocess_profile_side(
        _image(),
        eye_bbox_xyxy=[20, 20, 60, 60],
        iris_center_xy=[45, 35],
        head_origin_xy=[10, 80],
        head_forward_point_xy=[30, 60],
        crop_mode="letterbox",
        extract_eye_angles=False,
    )

    assert result.patch.shape == (128, 256, 3)
    assert result.ear is None
    assert result.eyelid_center_xy.tolist() == pytest.approx([40.0, 40.0])
    assert result.eye_size_xy.tolist() == pytest.approx([40.0, 40.0])
    assert result.eye_pose_2d.tolist() == pytest.approx([0.125, -0.125])
    assert result.eyelid_tail_points_xy is None
    assert result.eyelid_tail_vectors_px is None
    assert result.eyelid_direction_angles_degrees is None
    assert result.eyelid_direction_angles_normalized is None
    assert result.eyelid_tail_angle_degrees is None
    assert result.eyelid_tail_angle_normalized is None

    # A square source centered into a 2:1 output leaves black side padding.
    assert np.count_nonzero(result.patch[:, :50]) == 0
    assert np.count_nonzero(result.patch[:, -50:]) == 0


def test_bbox_accepts_exclusive_image_boundary() -> None:
    result = preprocess_profile_side(
        _image(),
        eye_bbox_xyxy=[0, 0, 160, 100],
        iris_center_xy=[80, 50],
        head_origin_xy=[10, 80],
        head_forward_point_xy=[20, 70],
        crop_mode="letterbox",
        extract_eye_angles=False,
    )

    assert result.patch.shape == (128, 256, 3)


def test_eye_roi_is_still_created_when_all_optional_features_are_disabled() -> None:
    result = preprocess_profile_side(
        _image(),
        eyelid_keypoints_xy=_eyelid_points(),
        extract_head_pose=False,
        extract_iris_pose=False,
        extract_eye_angles=False,
    )

    assert result.patch.shape == (128, 256, 3)
    assert result.head_pose_2d is None
    assert result.eye_pose_2d is None
    assert result.iris_center_xy is None
    assert result.eyelid_direction_angles_normalized is None


def test_selected_eye_ear_is_independent_of_face_side() -> None:
    points = _eyelid_points()
    reversed_points = points[[3, 4, 5, 0, 1, 2]]

    assert selected_eye_ear(points) == pytest.approx(0.3)
    assert selected_eye_ear(reversed_points) == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"eyelid_keypoints_xy": None}, "provide eye_bbox_xyxy"),
        ({"head_forward_point_xy": [30, 80]}, "head pose vector has zero length"),
        ({"iris_center_xy": [999, 10]}, "lies outside"),
        ({"output_size_hw": [128, 0]}, "at least 2"),
        ({"crop_mode": "unknown"}, "crop_mode"),
        (
            {"eyelid_tail_indices": [3, 1, 4]},
            "corner followed by its upper/lower neighbors",
        ),
    ],
)
def test_invalid_profile_annotations_raise_clear_errors(
    overrides: dict[str, object], match: str
) -> None:
    arguments: dict[str, object] = {
        "eyelid_keypoints_xy": _eyelid_points(),
        "iris_center_xy": [102, 46],
        "head_origin_xy": [30, 80],
        "head_forward_point_xy": [30, 40],
    }
    arguments.update(overrides)

    with pytest.raises(ProfileSideGeometryError, match=match):
        preprocess_profile_side(_image(), **arguments)  # type: ignore[arg-type]


def test_degenerate_eyelid_coordinates_are_rejected() -> None:
    points = _eyelid_points()
    points[3] = points[0]

    with pytest.raises(ProfileSideGeometryError, match="zero corner-to-corner width"):
        preprocess_profile_side(
            _image(),
            eyelid_keypoints_xy=points,
            iris_center_xy=[102, 46],
            head_origin_xy=[30, 80],
            head_forward_point_xy=[30, 40],
        )


def test_source_quad_must_intersect_image_after_crop_scaling() -> None:
    # The annotation bbox is valid, but an extreme scale centered at the edge
    # must still overlap the source; this verifies the explicit quad guard.
    result = preprocess_profile_side(
        _image(),
        eye_bbox_xyxy=[0, 0, 5, 5],
        iris_center_xy=[2, 2],
        head_origin_xy=[10, 10],
        head_forward_point_xy=[11, 10],
        bbox_crop_scale_xy=[4, 4],
        extract_eye_angles=False,
    )
    assert result.patch.shape == (128, 256, 3)


def test_transform_places_iris_at_normalized_patch_location() -> None:
    result = preprocess_profile_side(
        _image(),
        eyelid_keypoints_xy=_eyelid_points(),
        iris_center_xy=[100, 50],
        head_origin_xy=[20, 80],
        head_forward_point_xy=[20, 60],
    )
    iris_h = np.asarray([100.0, 50.0, 1.0])
    mapped_h = result.source_to_patch @ iris_h
    mapped = mapped_h[:2] / mapped_h[2]

    assert mapped == pytest.approx([127.5, 63.5], abs=1e-4)


def test_letterbox_transform_matches_opencv_projection() -> None:
    result = preprocess_profile_side(
        _image(),
        eye_bbox_xyxy=[20, 20, 100, 40],
        iris_center_xy=[60, 30],
        head_origin_xy=[10, 80],
        head_forward_point_xy=[20, 70],
        crop_mode="letterbox",
        extract_eye_angles=False,
    )
    projected = cv2.perspectiveTransform(result.source_quad_xy[None, :, :], result.source_to_patch)[
        0
    ]

    assert projected[:, 0].min() == pytest.approx(0.0, abs=1e-4)
    assert projected[:, 0].max() == pytest.approx(255.0, abs=1e-4)
    assert projected[:, 1].min() > 0.0
    assert projected[:, 1].max() < 127.0
