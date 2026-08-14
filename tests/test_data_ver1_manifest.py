from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.create_data_ver1_manifest import ManifestBuildError, create_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
FRONT_PROFILE = PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml"
SIDE_PROFILE = PROJECT_ROOT / "configs" / "profiles" / "side_profile_90.yaml"
SMOKE_PROFILE = PROJECT_ROOT / "configs" / "profiles" / "data_ver1_smoke.yaml"
SOURCE_COLUMNS = (
    "sample",
    "participant",
    "pair",
    "webcam_image",
    "phonecam_image",
    "x_px",
    "y_px",
)


def _write_subject(root: Path, subject: str, *, pair: int) -> None:
    subject_dir = root / subject
    webcam = subject_dir / "images" / "webcam" / "s000000.jpg"
    phonecam = subject_dir / "images" / "phonecam" / "s000000.jpg"
    webcam.parent.mkdir(parents=True)
    phonecam.parent.mkdir(parents=True)
    Image.new("RGB", (1280, 720), color=(100, 120, 140)).save(webcam)
    Image.new("RGB", (1280, 720), color=(140, 120, 100)).save(phonecam)
    (subject_dir / "participant.json").write_text(
        json.dumps(
            {
                "screen": {
                    "canvas_width_pixel": 1920,
                    "canvas_height_pixel": 1080,
                }
            }
        ),
        encoding="utf-8",
    )
    labels = subject_dir / "labels"
    labels.mkdir()
    with (labels / "image_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "sample": "s000000",
                "participant": subject,
                "pair": pair,
                "webcam_image": "images/webcam/s000000.jpg",
                "phonecam_image": "images/phonecam/s000000.jpg",
                "x_px": 960,
                "y_px": 766,
            }
        )


@pytest.fixture
def data_ver1_root(tmp_path: Path) -> Path:
    root = tmp_path / "data(ver1)"
    _write_subject(root, "p00", pair=72)
    _write_subject(root, "p03", pair=82)
    return root


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_manifest_expands_each_source_pair_without_dummy_annotations(
    data_ver1_root: Path, tmp_path: Path
) -> None:
    manifest = tmp_path / "generated" / "manifest.csv"

    result = create_manifest(data_ver1_root, manifest)
    rows = _read_rows(manifest)

    assert result.source_pairs == 2
    assert result.output_rows == 4
    assert [row["sample_id"] for row in rows] == [
        "p00_s000000_front",
        "p00_s000000_side",
        "p03_s000000_front",
        "p03_s000000_side",
    ]
    assert [row["pair_id"] for row in rows] == [
        "p00_pair_72",
        "p00_pair_72",
        "p03_pair_82",
        "p03_pair_82",
    ]
    assert [row["view"] for row in rows] == ["webcam", "phonecam", "webcam", "phonecam"]
    assert rows[0]["image_path"] == "p00/images/webcam/s000000.jpg"
    assert rows[1]["image_path"] == "p00/images/phonecam/s000000.jpg"
    assert {(row["target_x_px"], row["target_y_px"]) for row in rows} == {("960", "766")}
    assert {(row["screen_width_px"], row["screen_height_px"]) for row in rows} == {("1920", "1080")}
    assert {row["eye_annotation_valid"] for row in rows} == {"false"}
    assert {row["annotation_source"] for row in rows} == {""}
    assert all(not row["visible_eye_keypoints_xy"] for row in rows)


def test_dummy_side_annotations_are_opt_in_subject_specific_and_marked(
    data_ver1_root: Path, tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.csv"

    create_manifest(data_ver1_root, manifest, dummy_side_annotations=True)
    rows = _read_rows(manifest)
    by_id = {row["sample_id"]: row for row in rows}

    for subject in ("p00", "p03"):
        front = by_id[f"{subject}_s000000_front"]
        side = by_id[f"{subject}_s000000_side"]
        assert front["eye_annotation_valid"] == "false"
        assert front["annotation_source"] == ""
        assert side["eye_annotation_valid"] == "true"
        assert side["annotation_source"] == "dummy_smoke"
        assert side["visible_eye"] == "left"
        assert len(json.loads(side["visible_eye_keypoints_xy"])) == 6

    assert json.loads(by_id["p00_s000000_side"]["iris_center_xy"]) == [710, 310]
    assert json.loads(by_id["p03_s000000_side"]["iris_center_xy"]) == [480, 300]
    assert (
        by_id["p00_s000000_side"]["visible_eye_bbox_xyxy"]
        != by_id["p03_s000000_side"]["visible_eye_bbox_xyxy"]
    )


def test_smoke_profile_assigns_p00_train_and_p03_validation(
    data_ver1_root: Path, tmp_path: Path
) -> None:
    from gaze_pipeline.config import load_and_validate_config
    from gaze_pipeline.data.pipeline import read_generic_csv_manifest
    from gaze_pipeline.data.split import deterministic_group_split

    manifest = tmp_path / "manifest.csv"
    create_manifest(data_ver1_root, manifest)
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(FRONT_PROFILE, SIDE_PROFILE, SMOKE_PROFILE),
        environ={
            "GAZE_DATA_ROOT": str(data_ver1_root),
            "GAZE_OUTPUT_ROOT": str(tmp_path / "outputs"),
            "DATA_VER1_MANIFEST": str(manifest),
        },
    )
    records = read_generic_csv_manifest(
        dataset_root=data_ver1_root,
        data_config=config["data"],
        reader_config=config["data"]["reader"],
        base_dir=PROJECT_ROOT,
        dataset_name=config["data"]["dataset_name"],
    )

    splits = deterministic_group_split(
        records,
        ratios=config["data"]["split"]["ratios"],
        seed=config["data"]["split"]["seed"],
        group_key=config["data"]["split"]["group_key"],
        shuffle_groups=config["data"]["split"]["shuffle_groups"],
    )

    assert {record.subject_id for record in splits["train"]} == {"p00"}
    assert {record.subject_id for record in splits["validation"]} == {"p03"}
    assert splits["test"] == []

    front = config["model"]["front"]
    side = config["model"]["side"]
    assert front["entrypoint"] == "gaze_pipeline.models.webeyetrack_front:create_model"
    assert front["init_args"]["unfreeze_encoder"] is False
    assert front["pretrained"]["path"] is None
    assert side["entrypoint"] == "gaze_pipeline.models.simple_side:create_model"
    assert side["init_args"]["feature_keys"] == [
        "side_head_pose_2d",
        "side_eye_angles",
        "side_iris_pose_2d",
    ]
    assert side["init_args"]["embedding_dim"] == 32
    assert config["training"]["max_epochs"] == 1
    assert config["mlflow"]["enabled"] is True


def test_output_manifest_must_stay_outside_raw_source(data_ver1_root: Path) -> None:
    with pytest.raises(ManifestBuildError, match="outside the read-only source root"):
        create_manifest(data_ver1_root, data_ver1_root / "manifest.csv")
