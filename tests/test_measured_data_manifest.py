from __future__ import annotations

import csv
import json
import unicodedata
from pathlib import Path

import pytest
from PIL import Image

from gaze_pipeline.data.pipeline import read_generic_csv_manifest
from scripts.create_measured_data_manifest import (
    FIELDNAMES,
    MeasuredManifestError,
    _IndexedRow,
    _side_annotations,
    create_manifest,
)

MASTER_FIELDS = (
    "sample_id",
    "subject_id",
    "head_pose",
    "view",
    "image_path",
    "pair_id",
    "target_x_px",
    "target_y_px",
    "screen_width_px",
    "screen_height_px",
    "collection_split",
    "protocol",
    "source_frame",
)
INPUT_FIELDS = (
    "schema_version",
    "sample_id",
    "participant",
    "head_pose",
    "pair_id",
    "collection_split",
    "protocol",
    "source_image_path",
    "target_x_centered",
    "target_y_centered",
    "head_vector_x",
    "head_vector_y",
    "head_vector_z",
    "face_origin_x_cm",
    "face_origin_y_cm",
    "face_origin_z_cm",
    "left_ear",
    "right_ear",
    "valid",
    "invalid_reason",
)
FEATURE_FIELDS = ("sample_id", "participant", "head_pose", "pair_id", "image_path")
SIDE_FIELDS = (
    *FEATURE_FIELDS,
    "visible_eye",
    "visible_eye_bbox_xyxy",
    "visible_eye_keypoints_xy",
    "iris_center_xy",
    "profile_head_origin_xy",
    "profile_head_forward_xy",
    "eye_annotation_valid",
)


def test_bbox_only_side_annotation_is_a_valid_roi(tmp_path: Path) -> None:
    item = _IndexedRow(
        row={
            "visible_eye": "left",
            "visible_eye_bbox_xyxy": "[10,12,50,36]",
            "eye_annotation_valid": "true",
        },
        source=tmp_path / "side_roi.csv",
        line=2,
    )

    result = _side_annotations(item)

    assert result["eye_annotation_valid"] == "true"
    assert result["visible_eye_bbox_xyxy"] == "[10.0,12.0,50.0,36.0]"
    assert result["visible_eye_keypoints_xy"] == ""


def test_external_bbox_annotations_override_empty_phone_feature_table(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _session(root, "person", "neutral")
    annotation_csv = tmp_path / "generated_side_roi.csv"
    pair_id = "person_neutral_s000000"
    _write_csv(
        annotation_csv,
        (
            "sample_id",
            "participant",
            "head_pose",
            "pair_id",
            "visible_eye",
            "visible_eye_bbox_xyxy",
            "eye_annotation_valid",
        ),
        [
            {
                "sample_id": f"{pair_id}_phonecam",
                "participant": "person",
                "head_pose": "neutral",
                "pair_id": pair_id,
                "visible_eye": "left",
                "visible_eye_bbox_xyxy": "[8,9,52,38]",
                "eye_annotation_valid": "true",
            }
        ],
    )

    result = create_manifest(
        root,
        tmp_path / "manifest.csv",
        sessions=["neutral"],
        side_annotations=annotation_csv,
        require_side_annotations=True,
    )

    assert result.side_annotation_valid_rows == 1
    with result.path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    side = next(row for row in rows if row["view"] == "phonecam")
    assert side["visible_eye_bbox_xyxy"] == "[8.0,9.0,52.0,38.0]"


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _session(
    root: Path,
    subject_nfd: str,
    session: str,
    *,
    include_pose: bool = True,
    include_side_annotations: bool = False,
    input_valid: bool = True,
) -> None:
    subject_nfc = unicodedata.normalize("NFC", subject_nfd)
    feature_root = root / subject_nfd / session / "feature_maps"
    pair_id = f"{subject_nfc}_{session}_s000000"
    front_id = f"{pair_id}_webcam"
    side_id = f"{pair_id}_phonecam"
    front_filename_nfd = unicodedata.normalize("NFD", f"{front_id}_frame.png")
    side_filename_nfd = unicodedata.normalize("NFD", f"{side_id}_frame.png")
    front_image = feature_root / "web" / "frames" / front_filename_nfd
    side_image = feature_root / "phone" / "frames" / side_filename_nfd
    front_image.parent.mkdir(parents=True)
    side_image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 48), color=(30, 60, 90)).save(front_image)
    Image.new("RGB", (64, 48), color=(90, 60, 30)).save(side_image)

    master_rows = [
        {
            "sample_id": front_id,
            "subject_id": subject_nfc,
            "head_pose": session,
            "view": "webcam",
            # The CSV uses NFC while the actual test filename is NFD.
            "image_path": f"web/frames/{unicodedata.normalize('NFC', front_filename_nfd)}",
            "pair_id": pair_id,
            "target_x_px": 320,
            "target_y_px": 240,
            "screen_width_px": 640,
            "screen_height_px": 480,
            "collection_split": "training",
            "protocol": "train_static",
            "source_frame": 11,
        },
        {
            "sample_id": side_id,
            "subject_id": subject_nfc,
            "head_pose": session,
            "view": "phonecam",
            "image_path": f"phone/frames/{unicodedata.normalize('NFC', side_filename_nfd)}",
            "pair_id": pair_id,
            "target_x_px": 320,
            "target_y_px": 240,
            "screen_width_px": 640,
            "screen_height_px": 480,
            "collection_split": "training",
            "protocol": "train_static",
            "source_frame": 11,
        },
    ]
    _write_csv(feature_root / "training.csv", MASTER_FIELDS, master_rows)
    _write_csv(feature_root / "evaluation.csv", MASTER_FIELDS, [])
    _write_csv(
        feature_root / "web" / "features.csv",
        FEATURE_FIELDS,
        [
            {
                "sample_id": front_id,
                "participant": subject_nfc,
                "head_pose": session,
                "pair_id": pair_id,
                "image_path": master_rows[0]["image_path"],
            }
        ],
    )
    side_row: dict[str, object] = {
        "sample_id": side_id,
        "participant": subject_nfc,
        "head_pose": session,
        "pair_id": pair_id,
        "image_path": master_rows[1]["image_path"],
    }
    if include_side_annotations:
        side_row.update(
            {
                "visible_eye": "left",
                "visible_eye_bbox_xyxy": "[10,10,50,35]",
                "visible_eye_keypoints_xy": json.dumps(
                    [[12, 22], [20, 17], [35, 17], [48, 22], [35, 27], [20, 27]]
                ),
                "iris_center_xy": "[30,22]",
                "profile_head_origin_xy": "[55,35]",
                "profile_head_forward_xy": "[40,20]",
                "eye_annotation_valid": "true",
            }
        )
    _write_csv(
        feature_root / "phone" / "features.csv",
        SIDE_FIELDS if include_side_annotations else FEATURE_FIELDS,
        [side_row] if include_side_annotations else [],
    )
    input_rows = []
    if include_pose:
        input_rows.append(
            {
                "schema_version": "webeyetrack_input_v1",
                "sample_id": front_id,
                "participant": subject_nfc,
                "head_pose": session,
                "pair_id": pair_id,
                "collection_split": "training",
                "protocol": "train_static",
                "source_image_path": str(front_image),
                "target_x_centered": 0.0,
                "target_y_centered": 0.0,
                "head_vector_x": 0.1 if input_valid else "",
                "head_vector_y": -0.2 if input_valid else "",
                "head_vector_z": -0.97 if input_valid else "",
                "face_origin_x_cm": 1.0 if input_valid else "",
                "face_origin_y_cm": -2.0 if input_valid else "",
                "face_origin_z_cm": 50.0 if input_valid else "",
                # Deliberately nonnumeric: the manifest builder must not calculate
                # or inspect producer diagnostics. Only the upstream valid flag is authoritative.
                "left_ear": "not-read",
                "right_ear": "not-read",
                "valid": 1 if input_valid else 0,
                "invalid_reason": "" if input_valid else "eye_closed",
            }
        )
    _write_csv(feature_root / "webeyetrack" / "inputs.csv", INPUT_FIELDS, input_rows)
    # This tempting precomputed ROI must never become the manifest image.
    old_roi = feature_root / "webeyetrack" / "eye_roi" / f"{pair_id}.png"
    old_roi.parent.mkdir(parents=True)
    Image.new("RGB", (512, 128), color=(255, 0, 0)).save(old_roi)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_named_nfd_subject_sessions_use_corrected_frames_and_join_front_pose(
    tmp_path: Path,
) -> None:
    root = tmp_path / "measured"
    subject_nfd = unicodedata.normalize("NFD", "홍길동")
    for session in ("head_down", "neutral"):
        _session(root, subject_nfd, session)
    manifest = tmp_path / "generated" / "manifest.csv"

    result = create_manifest(root, manifest)
    rows = _rows(manifest)

    assert result.subjects == ("홍길동",)
    assert result.sessions == ("head_down", "neutral")
    assert result.front_input_rows == 2
    assert result.source_pairs == 2
    assert result.output_pairs == 2
    assert result.dropped_invalid_pairs == 0
    assert result.output_rows == 4
    assert result.front_pose_valid_rows == 2
    assert result.front_pose_missing_rows == 0
    assert result.side_annotation_valid_rows == 0
    assert result.side_annotation_missing_rows == 2
    assert tuple(rows[0]) == FIELDNAMES
    assert {row["subject_id"] for row in rows} == {"홍길동"}
    assert {row["session_id"] for row in rows} == {"head_down", "neutral"}
    assert all("feature_maps/webeyetrack/eye_roi" not in row["image_path"] for row in rows)
    assert all("/frames/" in row["image_path"] for row in rows)
    front = next(row for row in rows if row["view"] == "webcam")
    side = next(row for row in rows if row["view"] == "phonecam")
    assert json.loads(front["head_vector"]) == pytest.approx([0.1, -0.2, -0.97])
    assert json.loads(front["face_origin_3d"]) == pytest.approx([1.0, -2.0, 50.0])
    assert front["head_pose_valid"] == "true"
    assert front["input_csv_path"].endswith("feature_maps/webeyetrack/inputs.csv")
    assert front["input_csv_line"] == "2"
    assert side["head_pose_valid"] == "false"
    assert side["input_csv_path"] == ""
    assert side["eye_annotation_valid"] == "false"

    records = read_generic_csv_manifest(
        dataset_root=root,
        data_config={
            "views": {"source_to_branch": {"webcam": "front", "phonecam": "side"}},
            "pairing": {"enabled": True},
        },
        reader_config={"manifest_path": str(manifest)},
        base_dir=tmp_path,
        dataset_name="measured",
    )
    front_record = next(record for record in records if record.view == "front")
    assert front_record.head_vector == pytest.approx((0.1, -0.2, -0.97))
    assert front_record.face_origin_3d == pytest.approx((1.0, -2.0, 50.0))
    assert front_record.head_pose_valid is True


def test_inputs_csv_is_mandatory_and_side_annotations_remain_independently_strict(
    tmp_path: Path,
) -> None:
    root = tmp_path / "measured"
    _session(root, "person", "head_down", include_pose=False)
    with pytest.raises(MeasuredManifestError, match="authoritative inputs.csv row is missing"):
        create_manifest(
            root,
            tmp_path / "missing-input.csv",
            sessions=("head_down",),
        )

    complete_root = tmp_path / "complete"
    _session(complete_root, "person", "head_down")
    with pytest.raises(MeasuredManifestError, match="Side annotations are missing for 1/1"):
        create_manifest(
            complete_root,
            tmp_path / "side-strict.csv",
            sessions=("head_down",),
            require_side_annotations=True,
        )


def test_valid_zero_drops_complete_pair_before_manifest_and_pairing(tmp_path: Path) -> None:
    root = tmp_path / "measured"
    _session(root, "person", "head_down", input_valid=True)
    _session(root, "person", "neutral", input_valid=False)
    manifest = tmp_path / "manifest.csv"

    result = create_manifest(root, manifest)
    rows = _rows(manifest)

    assert result.front_input_rows == 2
    assert result.source_pairs == 2
    assert result.output_pairs == 1
    assert result.dropped_invalid_pairs == 1
    assert result.output_rows == 2
    assert {row["session_id"] for row in rows} == {"head_down"}
    assert {row["view"] for row in rows} == {"webcam", "phonecam"}
    assert len({row["pair_id"] for row in rows}) == 1
    assert all(row["head_pose_valid"] == "true" for row in rows if row["view"] == "webcam")


def test_precomputed_side_roi_replaces_phone_image_and_skips_missing_sessions(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "participants"
    side_roi_root = tmp_path / "new"
    _session(source_root, "person", "head_down")
    _session(source_root, "person", "neutral")
    source_side = next(
        (source_root / "person" / "head_down" / "feature_maps" / "phone" / "frames").iterdir()
    )
    roi_dir = side_roi_root / "person" / "head_down" / "side_eye_roi"
    roi_dir.mkdir(parents=True)
    Image.new("RGB", (128, 128), color=(12, 34, 56)).save(roi_dir / source_side.name)
    manifest = tmp_path / "generated" / "manifest.csv"

    result = create_manifest(
        source_root,
        manifest,
        dataset_root=tmp_path,
        side_roi_root=side_roi_root,
    )
    rows = _rows(manifest)

    assert result.subjects == ("person",)
    assert result.output_pairs == 1
    assert result.skipped_missing_side_roi_sessions == 1
    assert result.dropped_missing_side_roi_pairs == 0
    assert {row["session_id"] for row in rows} == {"head_down"}
    side = next(row for row in rows if row["view"] == "phonecam")
    assert side["image_path"].startswith("new/person/head_down/side_eye_roi/")
    assert side["eye_annotation_valid"] == "true"
    assert side["visible_eye_bbox_xyxy"] == ""

    records = read_generic_csv_manifest(
        dataset_root=tmp_path,
        data_config={
            "views": {"source_to_branch": {"webcam": "front", "phonecam": "side"}},
            "pairing": {"enabled": True},
        },
        reader_config={"manifest_path": str(manifest)},
        base_dir=tmp_path,
        dataset_name="precomputed-side-roi",
    )
    side_record = next(record for record in records if record.view == "side")
    assert side_record.image_path == (roi_dir / source_side.name).resolve()
    assert side_record.eye_annotation_valid is True


@pytest.mark.parametrize("bad_value", ["nan", "inf", "not-a-number"])
def test_valid_input_rejects_nonfinite_or_malformed_pose(tmp_path: Path, bad_value: str) -> None:
    root = tmp_path / "measured"
    _session(root, "person", "neutral")
    inputs = root / "person" / "neutral" / "feature_maps" / "webeyetrack" / "inputs.csv"
    input_rows = _rows(inputs)
    input_rows[0]["head_vector_y"] = bad_value
    _write_csv(inputs, INPUT_FIELDS, input_rows)

    with pytest.raises(MeasuredManifestError, match="head_vector_y"):
        create_manifest(root, tmp_path / "manifest.csv", sessions=("neutral",))
    assert not (tmp_path / "manifest.csv").exists()


def test_future_phone_feature_annotations_are_copied_without_dummy_geometry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "measured"
    _session(root, "person", "neutral", include_side_annotations=True)
    manifest = tmp_path / "manifest.csv"

    result = create_manifest(
        root,
        manifest,
        sessions=("neutral",),
        require_side_annotations=True,
    )
    side = next(row for row in _rows(manifest) if row["view"] == "phonecam")

    assert result.side_annotation_valid_rows == 1
    assert side["eye_annotation_valid"] == "true"
    assert json.loads(side["visible_eye_bbox_xyxy"]) == [10.0, 10.0, 50.0, 35.0]
    assert len(json.loads(side["visible_eye_keypoints_xy"])) == 6
    assert side["feature_csv_line"] == "2"
