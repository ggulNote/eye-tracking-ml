from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.generate_side_eye_annotations import (
    GeneratedRow,
    SideGenerationError,
    _annotation_row,
    discover_phone_samples,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _side_source(root: Path, *, second_valid: str = "0") -> Path:
    feature_root = root / "person" / "neutral" / "feature_maps"
    frames = feature_root / "phone" / "frames"
    frames.mkdir(parents=True)
    (frames / "open.png").touch()
    (frames / "closed.png").touch()
    _write_csv(
        feature_root / "training.csv",
        ["sample_id", "view", "pair_id", "source_frame", "image_path"],
        [
            {
                "sample_id": "open_side",
                "view": "phonecam",
                "pair_id": "pair_open",
                "source_frame": 10,
                "image_path": "phone/frames/open.png",
            },
            {
                "sample_id": "closed_side",
                "view": "phonecam",
                "pair_id": "pair_closed",
                "source_frame": 20,
                "image_path": "phone/frames/closed.png",
            },
        ],
    )
    _write_csv(
        feature_root / "evaluation.csv",
        ["sample_id", "view", "pair_id", "source_frame", "image_path"],
        [],
    )
    _write_csv(
        feature_root / "webeyetrack" / "inputs.csv",
        ["sample_id", "pair_id", "valid", "invalid_reason"],
        [
            {
                "sample_id": "open_front",
                "pair_id": "pair_open",
                "valid": "1",
                "invalid_reason": "",
            },
            {
                "sample_id": "closed_front",
                "pair_id": "pair_closed",
                "valid": second_valid,
                "invalid_reason": "eye_closed",
            },
        ],
    )
    return feature_root


def test_inputs_valid_maps_one_to_open_and_zero_to_closed(tmp_path: Path) -> None:
    _side_source(tmp_path)

    samples = discover_phone_samples(tmp_path, ("neutral",))

    assert [sample.input_valid for sample in samples] == [True, False]
    assert [sample.input_invalid_reason for sample in samples] == ["", "eye_closed"]

    excluded = _annotation_row(
        GeneratedRow(samples[1], annotation=None, reason="eye_closed"),
        tmp_path.resolve(),
    )
    assert excluded["input_valid"] == "0"
    assert excluded["eye_state"] == "closed"
    assert excluded["eye_annotation_valid"] == "false"
    assert excluded["detection_method"] == "excluded_inputs_valid_0"
    assert excluded["review_status"] == "excluded"


def test_inputs_valid_rejects_values_other_than_exact_zero_or_one(tmp_path: Path) -> None:
    _side_source(tmp_path, second_valid="true")

    with pytest.raises(SideGenerationError, match="valid must be exactly 0 or 1"):
        discover_phone_samples(tmp_path, ("neutral",))


def test_missing_inputs_are_unknown_and_excluded_without_stopping_other_people(
    tmp_path: Path,
) -> None:
    feature_root = _side_source(tmp_path)
    (feature_root / "webeyetrack" / "inputs.csv").unlink()

    samples = discover_phone_samples(tmp_path, ("neutral",))

    assert [sample.input_valid for sample in samples] == [None, None]
    excluded = _annotation_row(
        GeneratedRow(samples[0], annotation=None, reason="missing input"),
        tmp_path.resolve(),
    )
    assert excluded["eye_state"] == "unknown"
    assert excluded["detection_method"] == "excluded_missing_input_state"
    assert excluded["review_status"] == "excluded"
