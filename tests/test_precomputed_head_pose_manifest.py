from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from gaze_pipeline.data import GazeImageDataset
from gaze_pipeline.data.records import CanonicalRecord, DataContractError, ScreenCalibration


def _config() -> dict[str, object]:
    return {
        "task": {"coordinate_system": {"name": "centered_normalized_screen"}},
        "data": {"pairing": {"enabled": False}},
        "preprocessing": {
            "stage_order": ["decode", "normalize"],
            "decode": {"enabled": True, "backend": "pillow"},
            "normalize": {"enabled": True, "mode": "zero_one", "channel_order": "CHW"},
        },
    }


def test_dataset_passes_precomputed_webeyetrack_pose_without_recomputing_roi(
    tmp_path: Path,
) -> None:
    image = tmp_path / "corrected_web_frame.png"
    Image.new("RGB", (64, 48), color=(20, 40, 60)).save(image)
    row = {
        "sample_id": "front-001",
        "subject_id": "subject-001",
        "session_id": "neutral",
        "view": "front",
        "image_path": str(image),
        "target_x_normalized": "0.0",
        "target_y_normalized": "0.1",
        "head_vector": json.dumps([0.1, -0.2, -0.97]),
        "face_origin_3d": json.dumps([1.0, -2.0, 50.0]),
        "head_pose_valid": "true",
    }

    item = GazeImageDataset([row], _config(), split="train", view="front")[0]

    assert item["front_head_vector"].tolist() == pytest.approx([0.1, -0.2, -0.97])
    assert item["front_face_origin_3d"].tolist() == pytest.approx([1.0, -2.0, 50.0])
    assert item["front_head_orientation_valid"].item() is True
    assert item["front_face_origin_valid"].item() is True
    assert item["front_head_pose_valid"].item() is True


def test_canonical_pose_requires_vector_and_origin_together() -> None:
    values = {
        "sample_id": "front-001",
        "subject_id": "subject-001",
        "session_id": "neutral",
        "view": "front",
        "image_path": Path("/data/corrected_web_frame.png"),
        "image_relative_path": "subject-001/neutral/feature_maps/web/frames/frame.png",
        "image_width_px": 64,
        "image_height_px": 48,
        "image_channels": 3,
        "gaze_screen_xy_px": (320.0, 240.0),
        "screen": ScreenCalibration(width_px=640, height_px=480),
    }

    with pytest.raises(DataContractError, match="must either both be present"):
        CanonicalRecord(**values, head_vector=(0.0, 0.0, -1.0))
    with pytest.raises(DataContractError, match="head_pose_valid=true requires"):
        CanonicalRecord(**values, head_pose_valid=True)
