from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from scripts.create_process_data_manifest import create_manifest

FIELDS = (
    "sample_id",
    "participant",
    "head_pose",
    "pair_id",
    "eye_patch_path",
    "source_image_path",
    "target_x_centered",
    "target_y_centered",
    "head_vector_x",
    "head_vector_y",
    "head_vector_z",
    "face_origin_x_cm",
    "face_origin_y_cm",
    "face_origin_z_cm",
    "valid",
    "invalid_reason",
    "collection_split",
    "protocol",
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _session(root: Path, *, valid_second: bool = False) -> None:
    session = root / "person" / "head_down"
    front = session / "eye_roi"
    side = session / "side_eye_roi"
    front.mkdir(parents=True)
    side.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for index, valid in enumerate((True, valid_second)):
        pair_id = f"person_head_down_s{index:06d}"
        front_name = f"{pair_id}.png"
        side_name = f"{pair_id}_phone_frame_{index:08d}.png"
        if valid:
            Image.new("RGB", (512, 128), color=(index, 20, 30)).save(front / front_name)
        Image.new("RGB", (128, 128), color=(30, 20, index)).save(side / side_name)
        rows.append(
            {
                "sample_id": f"{pair_id}_webcam",
                "participant": "person",
                "head_pose": "head_down",
                "pair_id": pair_id,
                "eye_patch_path": f"eye_roi/{front_name}",
                "source_image_path": f"/raw/{pair_id}_web_frame.png",
                "target_x_centered": 0.1,
                "target_y_centered": -0.2,
                "head_vector_x": 0.0 if valid else "",
                "head_vector_y": 0.0 if valid else "",
                "head_vector_z": -1.0 if valid else "",
                "face_origin_x_cm": 0.0 if valid else "",
                "face_origin_y_cm": 0.0 if valid else "",
                "face_origin_z_cm": 60.0 if valid else "",
                "valid": int(valid),
                "invalid_reason": "" if valid else "eye_closed",
                "collection_split": "training",
                "protocol": "test",
            }
        )
    _write_csv(session / "inputs.csv", rows)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_process_manifest_uses_only_valid_complete_pairs(tmp_path: Path) -> None:
    root = tmp_path / "process_data"
    _session(root)

    result = create_manifest(
        root,
        tmp_path / "manifest.csv",
        sessions=("head_down", "neutral"),
    )

    rows = _rows(result.path)
    rejections = _rows(result.rejection_path)
    assert result.input_rows == 2
    assert result.upstream_invalid_pairs == 1
    assert result.valid_incomplete_pairs == 0
    assert result.output_pairs == 1
    assert result.output_rows == 2
    assert {row["view"] for row in rows} == {"front", "side"}
    assert all(row["pair_id"] == "person_head_down_s000000" for row in rows)
    assert rows[0]["target_x_px"] == "600000.00000000"
    assert rows[0]["target_y_px"] == "300000.00000000"
    assert any(row["exclusion_reason"] == "inputs_csv_valid_0:eye_closed" for row in rejections)
    assert any(row["exclusion_reason"] == "missing_inputs_csv" for row in rejections)


def test_quality_exclusion_removes_complete_session(tmp_path: Path) -> None:
    root = tmp_path / "process_data"
    _session(root, valid_second=True)
    exclusions = tmp_path / "quality_exclusions.csv"
    with exclusions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("subject_id", "session_id", "exclusion_reason"))
        writer.writeheader()
        writer.writerow(
            {
                "subject_id": "person",
                "session_id": "head_down",
                "exclusion_reason": "side_roi_session_quality_outlier",
            }
        )

    result = create_manifest(
        root,
        tmp_path / "manifest.csv",
        quality_exclusions=exclusions,
        sessions=("head_down",),
    )

    assert result.quality_excluded_pairs == 2
    assert result.output_pairs == 0
    assert _rows(result.path) == []
    assert all(
        row["exclusion_reason"].startswith("quality_excluded_session:")
        for row in _rows(result.rejection_path)
    )
