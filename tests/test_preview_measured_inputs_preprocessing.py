from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import preview_measured_inputs_preprocessing as preview  # noqa: E402


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_session(root: Path, subject: str, session: str = "head_down") -> None:
    session_root = root / subject / session
    web_root = session_root / "feature_maps/web/frames"
    phone_root = session_root / "feature_maps/phone/frames"
    eye_root = session_root / "feature_maps/webeyetrack/eye_roi"
    web_root.mkdir(parents=True)
    phone_root.mkdir(parents=True)
    eye_root.mkdir(parents=True)

    pair_id = f"{subject}_{session}_s000001"
    front_name = f"{pair_id}_web_frame_00000012.png"
    side_name = f"{pair_id}_phone_frame_00000012.png"
    roi_name = f"{pair_id}.png"
    Image.fromarray(np.full((90, 140, 3), (70, 110, 150), dtype=np.uint8)).save(
        web_root / front_name
    )
    Image.fromarray(np.full((120, 80, 3), (140, 100, 60), dtype=np.uint8)).save(
        phone_root / side_name
    )
    Image.fromarray(np.full((128, 512, 3), (90, 70, 50), dtype=np.uint8)).save(eye_root / roi_name)

    master_fields = ["sample_id", "pair_id", "view", "image_path"]
    _write_csv(
        session_root / "feature_maps/training.csv",
        master_fields,
        [
            {
                "sample_id": f"{pair_id}_webcam",
                "pair_id": pair_id,
                "view": "webcam",
                "image_path": f"web/frames/{front_name}",
            },
            {
                "sample_id": f"{pair_id}_phonecam",
                "pair_id": pair_id,
                "view": "phonecam",
                "image_path": f"phone/frames/{side_name}",
            },
        ],
    )
    _write_csv(session_root / "feature_maps/evaluation.csv", master_fields, [])

    input_fields = [
        "sample_id",
        "participant",
        "head_pose",
        "pair_id",
        "source_image_path",
        "eye_patch_path",
        "head_vector_x",
        "head_vector_y",
        "head_vector_z",
        "face_origin_x_cm",
        "face_origin_y_cm",
        "face_origin_z_cm",
        "quality_score",
        "valid",
    ]
    _write_csv(
        session_root / "feature_maps/webeyetrack/inputs.csv",
        input_fields,
        [
            {
                "sample_id": f"{subject}_closed",
                "participant": subject,
                "head_pose": session,
                "pair_id": f"{subject}_closed",
                "source_image_path": "/old/machine/closed.png",
                "eye_patch_path": "eye_roi/closed.png",
                "head_vector_x": "0",
                "head_vector_y": "0",
                "head_vector_z": "-1",
                "face_origin_x_cm": "0",
                "face_origin_y_cm": "0",
                "face_origin_z_cm": "50",
                "quality_score": "0.1",
                "valid": "0",
            },
            {
                "sample_id": f"{pair_id}_webcam",
                "participant": subject,
                "head_pose": session,
                "pair_id": pair_id,
                # The stale prefix must be ignored; only the corrected-frame basename is used.
                "source_image_path": f"/old/machine/raw/{front_name}",
                "eye_patch_path": f"eye_roi/{roi_name}",
                "head_vector_x": "0.1",
                "head_vector_y": "-0.2",
                "head_vector_z": "-0.97",
                "face_origin_x_cm": "1.5",
                "face_origin_y_cm": "-2.5",
                "face_origin_z_cm": "48.0",
                "quality_score": "0.8",
                "valid": "1",
            },
        ],
    )
    phone_feature_fields = [
        "sample_id",
        "pair_id",
        "eye_annotation_valid",
        "visible_eye",
        *preview.STRICT_SIDE_FIELDS,
    ]
    _write_csv(session_root / "feature_maps/phone/features.csv", phone_feature_fields, [])


def test_path_guard_rejects_raw_overlap(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    preview.validate_paths(raw, tmp_path / "outputs")
    with pytest.raises(preview.PreviewError, match="overlap"):
        preview.validate_paths(raw, raw / "generated")


def test_discovery_selects_only_inputs_valid_rows_and_corrected_frames(tmp_path: Path) -> None:
    _write_session(tmp_path, "person_b")
    _write_session(tmp_path, "person_a")
    # Stored eye_patch_path files are deliberately unavailable: discovery must
    # rely on corrected frames because the current pipeline regenerates ROI.
    for stored_roi in tmp_path.glob("*/head_down/feature_maps/webeyetrack/eye_roi/*.png"):
        stored_roi.unlink()

    samples = preview.discover_samples(tmp_path, count=2)

    assert [sample.subject for sample in samples] == ["person_a", "person_b"]
    assert all(sample.valid for sample in samples)
    assert all("closed" not in sample.sample_id for sample in samples)
    assert all("feature_maps/web/frames" in sample.front_path.as_posix() for sample in samples)
    assert all("feature_maps/phone/frames" in sample.side_path.as_posix() for sample in samples)
    assert all(sample.side_annotation is None for sample in samples)


def test_render_writes_privacy_safe_stages_and_marks_side_not_trainable(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    _write_session(raw, "private_person_a")
    _write_session(raw, "private_person_b", session="neutral")
    samples = preview.discover_samples(raw, count=2)
    assert {sample.session for sample in samples} == {"head_down", "neutral"}
    output = tmp_path / "outputs"

    generated_roi = Image.fromarray(np.full((128, 512, 3), (20, 40, 60), dtype=np.uint8))
    panel = preview.render_preview(
        samples,
        output,
        front_roi_factory=lambda _sample: generated_roi,
    )

    assert panel.is_file()
    assert Image.open(panel).size[0] > 1000
    expected_stages = {
        f"sample_{index:02d}_{stage}.png"
        for index in (1, 2)
        for stage in (
            "front_corrected",
            "side_corrected",
            "front_pose",
            "front_webeyetrack_roi",
            "side_profile_roi",
        )
    }
    assert expected_stages <= {path.name for path in output.glob("*.png")}
    for index in (1, 2):
        model_roi = output / f"sample_{index:02d}_front_webeyetrack_roi.png"
        assert Image.open(model_roi).size == (512, 128)

    summary_text = (output / "preview_summary.json").read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    assert summary["contract"]["source"] == "feature_maps/webeyetrack/inputs.csv"
    assert summary["contract"]["ignored_field"].startswith("eye_patch_path")
    assert summary["contract"]["sample_filter"] == "only rows accepted by inputs.csv valid=1"
    assert all(sample["inputs_csv_accepted"] is True for sample in summary["samples"])
    assert all(
        sample["front_model_image_shape_hwc"] == [128, 512, 3] for sample in summary["samples"]
    )
    assert all(sample["side_training_eligible"] is False for sample in summary["samples"])
    assert "private_person" not in summary_text


def test_side_annotation_accepts_real_bbox_and_keeps_geometry_optional(tmp_path: Path) -> None:
    _write_session(tmp_path, "person_a")
    session_root = tmp_path / "person_a/head_down"
    feature_path = session_root / "feature_maps/phone/features.csv"
    pair_id = "person_a_head_down_s000001"
    optional_fields = [
        "visible_eye_keypoints_xy",
        "iris_center_xy",
        "profile_head_origin_xy",
        "profile_head_forward_xy",
    ]
    fields = [
        "sample_id",
        "pair_id",
        "eye_annotation_valid",
        "visible_eye",
        "visible_eye_bbox_xyxy",
        *optional_fields,
    ]
    incomplete = {
        "sample_id": f"{pair_id}_phonecam",
        "pair_id": pair_id,
        "eye_annotation_valid": "true",
        "visible_eye": "left",
        "visible_eye_bbox_xyxy": "[10, 20, 60, 60]",
    }
    _write_csv(feature_path, fields, [incomplete])
    annotation = preview._side_annotation(session_root, pair_id)
    assert annotation is not None
    assert annotation["visible_eye_bbox_xyxy"] == [10.0, 20.0, 60.0, 60.0]
    assert all(annotation[field] is None for field in optional_fields)

    complete = {
        **incomplete,
        "visible_eye_keypoints_xy": "[[15,30],[25,25],[45,25],[55,30],[45,40],[25,40]]",
        "iris_center_xy": "[35,32]",
        "profile_head_origin_xy": "[30,60]",
        "profile_head_forward_xy": "[70,60]",
    }
    _write_csv(feature_path, fields, [complete])
    complete_annotation = preview._side_annotation(session_root, pair_id)
    assert complete_annotation is not None
    assert complete_annotation["visible_eye_keypoints_xy"] is not None
