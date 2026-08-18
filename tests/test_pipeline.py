import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from gaze_pipeline.data.pipeline import (
    DataPreparationError,
    prepare_data,
    read_generic_csv_manifest,
)
from gaze_pipeline.data.split import validate_explicit_pairs

GENERIC_COLUMNS = (
    "sample_id",
    "subject_id",
    "view",
    "image_path",
    "pair_id",
    "target_x_px",
    "target_y_px",
    "screen_width_px",
    "screen_height_px",
)
UNPAIRED_GENERIC_COLUMNS = tuple(column for column in GENERIC_COLUMNS if column != "pair_id")


def test_prepare_data_writes_hashed_leakage_safe_manifests_and_oob_flag(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    rows: list[dict[str, str]] = []
    for index, subject_id in enumerate(("p00", "p01", "p02")):
        image_path = data_root / subject_id / "webcam" / "0001.png"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (32, 24), color=(index * 20, 0, 0)).save(image_path)
        rows.append(
            {
                "sample_id": f"{subject_id}-front",
                "subject_id": subject_id,
                "view": "front",
                "image_path": image_path.relative_to(data_root).as_posix(),
                # p00 is deliberately outside a 100-pixel-wide screen.  The
                # source value must remain 100 rather than being clamped.
                "target_x_px": "100" if subject_id == "p00" else "50",
                "target_y_px": "50",
                "screen_width_px": "100",
                "screen_height_px": "100",
            }
        )

    source_manifest = tmp_path / "source.csv"
    with source_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=UNPAIRED_GENERIC_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest_dir = tmp_path / "prepared"
    config = {
        "experiment": {"seed": 42, "access_token": "do-not-store"},
        "mlflow": {
            "tracking_uri": (
                "https://alice:very-secret@example.test/mlflow"
                "?token=query-secret&project=demo#credential=fragment-secret"
            )
        },
        "data": {
            "dataset_name": "synthetic_dual_view_ready",
            "dataset_root": str(data_root),
            "reader": {
                "type": "generic_csv",
                "manifest_path": str(source_manifest),
                "target_bounds_policy": "keep_flagged",
            },
            "views": {
                "directory_to_branch": {
                    "webcam": "front",
                    "phonecam": "side",
                }
            },
            "pairing": {
                "enabled": False,
                "unpaired_policy": "branch_only",
            },
            "split": {
                "strategy": "grouped_ratio",
                "group_key": "subject_id",
                "ratios": {
                    "train": 0.70,
                    "validation": 0.15,
                    "test": 0.15,
                },
                "seed": 42,
                "shuffle_groups": True,
                "manifest_dir": str(manifest_dir),
            },
        },
    }

    result = prepare_data(config)

    assert dict(result.split_counts) == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    expected_csv_names = {
        "dataset_manifest.csv",
        "split_manifest.csv",
        "train.csv",
        "validation.csv",
        "test.csv",
    }
    assert {path.name for path in result.manifest_paths.values()} == expected_csv_names
    assert result.dataset_manifest_hash == result.artifact_hashes["dataset"]
    assert result.dataset_hash == result.dataset_manifest_hash
    assert result.resolved_config_path == manifest_dir.parent / "resolved_config.yaml"
    resolved_config_digest = hashlib.sha256(result.resolved_config_path.read_bytes()).hexdigest()
    assert result.resolved_config_hash == resolved_config_digest
    assert result.resolved_config_hash_path.read_text(encoding="utf-8") == (
        f"{resolved_config_digest}  resolved_config.yaml\n"
    )
    resolved_config = yaml.safe_load(result.resolved_config_path.read_text(encoding="utf-8"))
    assert resolved_config["experiment"]["access_token"] == "<redacted>"
    assert resolved_config["mlflow"]["tracking_uri"] == (
        "https://example.test/mlflow?token=%3Credacted%3E&project=demo#<redacted>"
    )
    snapshot_text = result.resolved_config_path.read_text(encoding="utf-8")
    assert "very-secret" not in snapshot_text
    assert "query-secret" not in snapshot_text
    assert "fragment-secret" not in snapshot_text

    for artifact_name, artifact_path in result.manifest_paths.items():
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert digest == result.artifact_hashes[artifact_name]
        sidecar_path = result.hash_paths[artifact_name]
        assert sidecar_path.read_text(encoding="utf-8") == (f"{digest}  {artifact_path.name}\n")

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    summary_digest = hashlib.sha256(result.summary_path.read_bytes()).hexdigest()
    assert summary_digest == result.artifact_hashes["summary"]
    assert result.hash_paths["summary"].read_text(encoding="utf-8") == (
        f"{summary_digest}  summary.json\n"
    )
    assert summary["raw_images_copied"] is False
    assert summary["dataset_manifest_hash"] == result.dataset_manifest_hash
    assert summary["resolved_config"] == {
        "path": str(result.resolved_config_path),
        "sha256": result.resolved_config_hash,
        "sensitive_values_redacted": True,
    }
    assert summary["dataset"]["counts"]["out_of_screen_target_records"] == 1
    assert summary["dataset"]["counts"]["out_of_screen_target_records_by_subject"] == {
        "p00": 1,
        "p01": 0,
        "p02": 0,
    }
    assert summary["dataset"]["target_bounds"]["metric_filter"] == (
        "target_in_screen_bounds == true"
    )

    with result.manifest_paths["dataset"].open("r", encoding="utf-8", newline="") as handle:
        dataset_rows = {row["subject_id"]: row for row in csv.DictReader(handle)}
    assert dataset_rows["p00"]["target_x_px"] == "100.0"
    assert dataset_rows["p00"]["target_in_screen_bounds"] == "false"
    assert dataset_rows["p01"]["target_in_screen_bounds"] == "true"

    subjects_by_split: dict[str, set[str]] = {}
    for split_name in ("train", "validation", "test"):
        with result.manifest_paths[split_name].open("r", encoding="utf-8", newline="") as handle:
            split_rows = list(csv.DictReader(handle))
        subjects_by_split[split_name] = {row["subject_id"] for row in split_rows}
        assert all(row["split"] == split_name for row in split_rows)

    assert subjects_by_split["train"].isdisjoint(subjects_by_split["validation"])
    assert subjects_by_split["train"].isdisjoint(subjects_by_split["test"])
    assert subjects_by_split["validation"].isdisjoint(subjects_by_split["test"])
    assert not list(manifest_dir.rglob("*.png"))


def test_session_selection_precedes_subject_wise_split_without_leakage(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    rows: list[dict[str, str]] = []
    fieldnames = (*GENERIC_COLUMNS, "session_id")
    for subject_index in range(5):
        subject_id = f"p{subject_index:02d}"
        for session_id in ("head_down", "neutral", "head_up"):
            pair_id = f"{subject_id}-{session_id}-pair"
            for view in ("front", "side"):
                image_path = data_root / subject_id / session_id / view / "0001.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 24), color=(subject_index * 20, 0, 0)).save(image_path)
                rows.append(
                    {
                        "sample_id": f"{pair_id}-{view}",
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "view": view,
                        "image_path": image_path.relative_to(data_root).as_posix(),
                        "pair_id": pair_id,
                        "target_x_px": "50",
                        "target_y_px": "50",
                        "screen_width_px": "100",
                        "screen_height_px": "100",
                    }
                )

    source_manifest = tmp_path / "source.csv"
    with source_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "experiment": {"seed": 42},
        "data": {
            "dataset_name": "selected_sessions",
            "dataset_root": str(data_root),
            "reader": {"type": "generic_csv", "manifest_path": str(source_manifest)},
            "selection": {"session_ids": ["head_down", "neutral"]},
            "pairing": {
                "enabled": True,
                "require_same_subject": True,
                "require_same_target": True,
                "unpaired_policy": "error",
            },
            "split": {
                "strategy": "grouped_ratio",
                "group_key": "subject_id",
                "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
                "seed": 42,
                "shuffle_groups": True,
                "manifest_dir": str(tmp_path / "prepared"),
            },
        },
    }

    result = prepare_data(config)

    assert len(result.records) == 20
    assert {record.session_id for record in result.records} == {"head_down", "neutral"}
    assert len(result.pair_validation.complete_pair_ids) == 10
    assert dict(result.split_counts) == {"train": 12, "validation": 4, "test": 4}
    subjects = {
        split_name: {record.subject_id for record in result.splits[split_name]}
        for split_name in ("train", "validation", "test")
    }
    assert subjects == {
        "train": {"p00", "p03", "p04"},
        "validation": {"p02"},
        "test": {"p01"},
    }
    assert subjects["train"].isdisjoint(subjects["validation"] | subjects["test"])
    assert subjects["validation"].isdisjoint(subjects["test"])
    assert result.summary["selection"] == {
        "enabled": True,
        "session_ids": ["head_down", "neutral"],
        "selected_records": 20,
        "dropped_records": 10,
    }
    assert result.summary["split"]["configured_ratios"] == {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
    }
    assert result.summary["split"]["total_groups"] == 5
    assert {
        name: partition["realized_group_ratio"]
        for name, partition in result.summary["split"]["partitions"].items()
    } == {"train": 0.6, "validation": 0.2, "test": 0.2}


def test_session_selection_rejects_unknown_session(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    image_path = data_root / "p00" / "neutral" / "front" / "0001.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(image_path)
    source_manifest = tmp_path / "source.csv"
    with source_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(*UNPAIRED_GENERIC_COLUMNS, "session_id"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "p00-neutral-front",
                "subject_id": "p00",
                "session_id": "neutral",
                "view": "front",
                "image_path": image_path.relative_to(data_root).as_posix(),
                "target_x_px": "50",
                "target_y_px": "50",
                "screen_width_px": "100",
                "screen_height_px": "100",
            }
        )

    config = {
        "data": {
            "dataset_root": str(data_root),
            "reader": {"type": "generic_csv", "manifest_path": str(source_manifest)},
            "selection": {"session_ids": ["head_down"]},
            "pairing": {"enabled": False},
            "split": {
                "strategy": "grouped_ratio",
                "ratios": {"train": 1.0, "validation": 0.0, "test": 0.0},
            },
        }
    }

    with pytest.raises(DataPreparationError, match="absent from the manifest"):
        prepare_data(config)


def test_generic_csv_maps_source_and_custom_pair_key_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    image_paths: list[Path] = []
    for directory in ("webcam", "phonecam"):
        image_path = data_root / "person-01" / directory / "capture.png"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (40, 30), color=(10, 20, 30)).save(image_path)
        image_paths.append(image_path)

    optional_columns = (
        "session_id",
        "facial_landmarks_xy",
        "head_rotation_3d",
        "head_translation_3d",
        "face_center_3d",
        "gaze_target_3d",
        "evaluation_eye",
    )
    pair_key = "capture_group"
    fieldnames = (
        tuple(pair_key if column == "pair_id" else column for column in GENERIC_COLUMNS)
        + optional_columns
    )
    metadata = {
        "session_id": "session-A",
        "facial_landmarks_xy": json.dumps([[1, 2], [3, 4], [5, 6], [7, 8]]),
        "head_rotation_3d": "[0.1,0.2,0.3]",
        "head_translation_3d": "[1,2,3]",
        "face_center_3d": "[0,0,10]",
        "gaze_target_3d": "[0,3,6]",
        "evaluation_eye": "right",
    }
    rows = []
    for view_source, image_path in zip(("webcam", "phonecam"), image_paths, strict=True):
        rows.append(
            {
                "sample_id": f"person-01-{view_source}",
                "subject_id": "person-01",
                "view": view_source,
                "image_path": image_path.relative_to(data_root).as_posix(),
                pair_key: "event-007",
                "target_x_px": "50",
                "target_y_px": "50",
                "screen_width_px": "100",
                "screen_height_px": "100",
                **(metadata if view_source == "webcam" else {}),
            }
        )

    source_manifest = tmp_path / "custom-pairs.csv"
    with source_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    data_config = {
        "views": {"source_to_branch": {"webcam": "front", "phonecam": "side"}},
        "pairing": {"enabled": True, "pair_id_key": pair_key},
    }
    records = read_generic_csv_manifest(
        dataset_root=data_root,
        data_config=data_config,
        reader_config={
            "manifest_path": str(source_manifest),
            "target_bounds_policy": "error",
        },
        base_dir=tmp_path,
        dataset_name="custom",
    )

    records_by_view = {record.view: record for record in records}
    assert set(records_by_view) == {"front", "side"}
    assert {record.pair_id for record in records} == {"event-007"}
    paired = validate_explicit_pairs(records, enabled=True)
    assert paired.complete_pair_ids == {"event-007"}
    front = records_by_view["front"]
    assert front.session_id == "session-A"
    assert front.facial_landmarks_xy == (
        (1.0, 2.0),
        (3.0, 4.0),
        (5.0, 6.0),
        (7.0, 8.0),
    )
    assert front.head_rotation_3d == pytest.approx((0.1, 0.2, 0.3))
    assert front.head_translation_3d == pytest.approx((1.0, 2.0, 3.0))
    assert front.face_center_3d == pytest.approx((0.0, 0.0, 10.0))
    assert front.gaze_target_3d == pytest.approx((0.0, 3.0, 6.0))
    assert front.evaluation_eye == "right"
    assert json.loads(str(front.to_manifest_row()["facial_landmarks_xy"])) == [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ]


def test_generic_csv_requires_the_configured_pair_id_header(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    image_path = data_root / "p00" / "webcam" / "0001.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(image_path)
    source_manifest = tmp_path / "wrong-pair-column.csv"
    row = {
        "sample_id": "p00-front",
        "subject_id": "p00",
        "view": "front",
        "image_path": image_path.relative_to(data_root).as_posix(),
        "pair_id": "event-1",
        "target_x_px": "50",
        "target_y_px": "50",
        "screen_width_px": "100",
        "screen_height_px": "100",
    }
    with source_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERIC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(DataPreparationError, match="capture_group"):
        read_generic_csv_manifest(
            dataset_root=data_root,
            data_config={"pairing": {"enabled": True, "pair_id_key": "capture_group"}},
            reader_config={"manifest_path": str(source_manifest)},
            base_dir=tmp_path,
        )


def test_generic_side_profile_annotations_survive_canonical_manifest_round_trip(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    image_path = data_root / "p90" / "phonecam" / "0001.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (320, 240), color=(20, 30, 40)).save(image_path)

    profile_columns = (
        "visible_eye",
        "visible_eye_bbox_xyxy",
        "visible_eye_keypoints_xy",
        "iris_center_xy",
        "profile_head_origin_xy",
        "profile_head_forward_xy",
        "eye_annotation_valid",
    )
    row = {
        "sample_id": "p90-side-0001",
        "subject_id": "p90",
        "view": "side",
        "image_path": image_path.relative_to(data_root).as_posix(),
        "target_x_px": "60",
        "target_y_px": "40",
        "screen_width_px": "100",
        "screen_height_px": "100",
        "visible_eye": "subject_right",
        "visible_eye_bbox_xyxy": "[90,80,170,140]",
        "visible_eye_keypoints_xy": json.dumps(
            [[100, 110], [115, 100], [140, 100], [160, 110], [140, 120], [115, 120]]
        ),
        "iris_center_xy": "[132,110]",
        "profile_head_origin_xy": "[120,180]",
        "profile_head_forward_xy": "[120,100]",
        "eye_annotation_valid": "true",
    }
    source_manifest = tmp_path / "profile-source.csv"
    with source_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=UNPAIRED_GENERIC_COLUMNS + profile_columns,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(row)

    records = read_generic_csv_manifest(
        dataset_root=data_root,
        data_config={"pairing": {"enabled": False}},
        reader_config={"manifest_path": str(source_manifest)},
        base_dir=tmp_path,
        dataset_name="profile90",
    )

    assert len(records) == 1
    record = records[0]
    assert record.visible_eye == "right"
    assert record.visible_eye_bbox_xyxy == pytest.approx((90.0, 80.0, 170.0, 140.0))
    assert record.visible_eye_keypoints_xy[0] == pytest.approx((100.0, 110.0))
    assert record.iris_center_xy == pytest.approx((132.0, 110.0))
    assert record.profile_head_origin_xy == pytest.approx((120.0, 180.0))
    assert record.profile_head_forward_xy == pytest.approx((120.0, 100.0))
    assert record.eye_annotation_valid is True

    canonical_row = record.to_manifest_row()
    assert canonical_row["visible_eye"] == "right"
    assert json.loads(str(canonical_row["visible_eye_bbox_xyxy"])) == [90.0, 80.0, 170.0, 140.0]
    assert json.loads(str(canonical_row["visible_eye_keypoints_xy"])) == [
        [100.0, 110.0],
        [115.0, 100.0],
        [140.0, 100.0],
        [160.0, 110.0],
        [140.0, 120.0],
        [115.0, 120.0],
    ]
    assert canonical_row["eye_annotation_valid"] == "true"


@pytest.mark.parametrize(
    ("section", "option", "unsupported_value"),
    (
        ("reader", "fail_on_bad_row", False),
        ("split", "reuse_existing_manifest", True),
        ("split", "stratify_by", "view"),
    ),
)
def test_prepare_rejects_declared_but_unimplemented_options(
    tmp_path: Path,
    section: str,
    option: str,
    unsupported_value: object,
) -> None:
    config = {
        "data": {
            "dataset_root": str(tmp_path / "not-read"),
            "reader": {"type": "generic_csv"},
            "split": {"strategy": "grouped_ratio"},
        }
    }
    config["data"][section][option] = unsupported_value  # type: ignore[index]

    with pytest.raises(DataPreparationError, match=option):
        prepare_data(config)
