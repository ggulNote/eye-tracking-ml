from pathlib import Path

import pytest

from gaze_pipeline.data.records import CanonicalRecord, DataContractError, ScreenCalibration


def make_record(**changes: object) -> CanonicalRecord:
    values: dict[str, object] = {
        "sample_id": "p01/day01/0005",
        "subject_id": "p01",
        "session_id": "day01",
        "view": "front",
        "image_path": Path("/data/p01/day01/0005.jpg"),
        "image_relative_path": "day01/0005.jpg",
        "image_width_px": 1280,
        "image_height_px": 720,
        "image_channels": 3,
        "gaze_screen_xy_px": (720.0, 450.0),
        "screen": ScreenCalibration(
            width_px=1440,
            height_px=900,
            width_mm=286.4,
            height_mm=179.0,
        ),
        "facial_landmarks_xy": (
            (100.0, 100.0),
            (110.0, 100.0),
            (140.0, 100.0),
            (150.0, 100.0),
            (115.0, 140.0),
            (140.0, 140.0),
        ),
    }
    values.update(changes)
    return CanonicalRecord(**values)  # type: ignore[arg-type]


def test_centered_screen_target_uses_screen_dimensions() -> None:
    record = make_record()

    assert record.gaze_screen_xy_normalized == pytest.approx((0.0, 0.0))
    row = record.to_manifest_row(split="train")
    assert row["target_x_normalized"] == pytest.approx(0.0)
    assert row["target_y_normalized"] == pytest.approx(0.0)
    assert row["screen_width_px"] == 1440


def test_target_outside_screen_is_rejected() -> None:
    with pytest.raises(DataContractError, match="outside screen"):
        make_record(gaze_screen_xy_px=(1440.0, 0.0))


def test_three_dimensional_gaze_direction_uses_target_minus_face_center() -> None:
    record = make_record(
        face_center_3d=(0.0, 0.0, 10.0),
        gaze_target_3d=(0.0, 3.0, 6.0),
    )

    assert record.gaze_vector_3d == pytest.approx((0.0, 3.0, -4.0))
    assert record.gaze_unit_3d == pytest.approx((0.0, 0.6, -0.8))


def test_generic_landmark_sets_accept_four_or_more_points() -> None:
    four_points = (
        (10.0, 10.0),
        (20.0, 10.0),
        (10.0, 20.0),
        (20.0, 20.0),
    )

    assert make_record(facial_landmarks_xy=four_points).facial_landmarks_xy == four_points
    with pytest.raises(DataContractError, match="at least 4"):
        make_record(facial_landmarks_xy=four_points[:3])


def test_profile_eye_annotations_are_validated_and_written_to_manifest() -> None:
    keypoints = (
        (100.0, 120.0),
        (105.0, 115.0),
        (115.0, 115.0),
        (120.0, 120.0),
        (115.0, 125.0),
        (105.0, 125.0),
    )
    record = make_record(
        view="side",
        visible_eye="subject_left",
        visible_eye_bbox_xyxy=(90.0, 105.0, 130.0, 135.0),
        visible_eye_keypoints_xy=keypoints,
        iris_center_xy=(110.0, 120.0),
        profile_head_origin_xy=(80.0, 160.0),
        profile_head_forward_xy=(80.0, 100.0),
        eye_annotation_valid=True,
    )

    assert record.visible_eye == "left"
    row = record.to_manifest_row()
    assert row["visible_eye"] == "left"
    assert row["visible_eye_bbox_xyxy"] == "[90.0,105.0,130.0,135.0]"
    assert row["visible_eye_keypoints_xy"] == (
        "[[100.0,120.0],[105.0,115.0],[115.0,115.0],[120.0,120.0],[115.0,125.0],[105.0,125.0]]"
    )
    assert row["iris_center_xy"] == "[110.0,120.0]"
    assert row["profile_head_origin_xy"] == "[80.0,160.0]"
    assert row["profile_head_forward_xy"] == "[80.0,100.0]"
    assert row["eye_annotation_valid"] == "true"


def test_profile_eye_annotations_reject_wrong_shape_and_degenerate_bbox() -> None:
    with pytest.raises(DataContractError, match="exactly 6 points"):
        make_record(
            view="side",
            visible_eye_keypoints_xy=((1.0, 2.0),) * 5,
        )

    with pytest.raises(DataContractError, match="x1 > x0"):
        make_record(
            view="side",
            visible_eye_bbox_xyxy=(10.0, 10.0, 10.0, 20.0),
        )
