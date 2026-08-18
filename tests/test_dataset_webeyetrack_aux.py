from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from gaze_pipeline.data import (
    GazeImageDataset,
    ManifestError,
    SampleValidationError,
    gaze_collate_fn,
)


def _write_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[4:20, 7:25] = color
    Image.fromarray(image).save(path)


def _row(
    sample_id: str,
    image_path: Path,
    *,
    view: str = "front",
    pair_id: str = "",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "subject_id": "p01",
        "session_id": "day01",
        "view": view,
        "image_path": str(image_path),
        "pair_id": pair_id,
        "target_x_normalized": -0.2,
        "target_y_normalized": 0.1,
        "target_in_screen_bounds": True,
    }


def _config(*, paired: bool = False) -> dict[str, Any]:
    return {
        "experiment": {"seed": 17},
        "task": {"coordinate_system": {"name": "centered_normalized_screen"}},
        "data": {
            "pairing": {
                "enabled": paired,
                "strategy": "explicit_pair_id",
                "pair_id_key": "pair_id",
                "require_same_subject": True,
                "require_same_target": True,
                "max_target_distance_normalized": 0.0,
                "unpaired_policy": "branch_only",
            }
        },
        "preprocessing": {
            "stage_order": ["decode", "normalize"],
            "decode": {"enabled": True, "backend": "pillow"},
            "normalize": {
                "enabled": True,
                "mode": "zero_one",
                "channel_order": "CHW",
            },
        },
    }


def _front3d_side_config(*, invalid_policy: str = "zero_fill_and_mask") -> dict[str, Any]:
    config = _config(paired=True)
    config["preprocessing"]["branch_overrides"] = {
        "side": {
            "eye_region_warp": {
                "feature_extraction": {
                    "side_headpose": {
                        "enabled": True,
                        "source": "front_3d",
                        "front_3d_key": "front_head_vector",
                        "front_3d_validity_key": "front_head_orientation_valid",
                        "front_3d_invalid_policy": invalid_policy,
                    }
                }
            }
        }
    }
    return config


class _AuxiliaryPreprocessor:
    """Small deterministic stand-in for the landmark-dependent stages."""

    stage_order = ("eye_selection", "metric_head_pose")

    @staticmethod
    def _stage_config(_stage: str, _view: str) -> Mapping[str, Any]:
        return {"enabled": True}

    @staticmethod
    def make_augmentation_context(
        _view: str,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        del generator
        return {}

    @staticmethod
    def __call__(sample: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        image = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
        view = str(sample["view"])
        metadata = sample["metadata"]
        metadata.update(
            {
                "source_size_hw": np.asarray(image.shape[:2], dtype=np.int64),
                "image_transform": np.eye(3, dtype=np.float32),
                "head_vector": np.asarray(
                    [0.1, 0.2, 0.3] if view == "front" else [-0.4, 0.5, 0.6],
                    dtype=np.float32,
                ),
                "face_origin_3d": np.asarray(
                    [1.0, 2.0, 3.0] if view == "front" else [np.nan, np.nan, np.nan],
                    dtype=np.float32,
                ),
                "face_rt": np.eye(4, dtype=np.float32),
                "metric_transform": np.eye(4, dtype=np.float32) * 2,
                "head_euler_degrees": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                "eye_visibility_score": np.asarray([0.9, 0.8], dtype=np.float32),
                "face_origin_unit": "cm",
                "face_origin_source": "test",
                "invalid_reasons": [],
                "head_orientation_valid": True,
                "face_origin_valid": view == "front",
                "head_pose_valid": view == "front",
                "gaze_valid": view == "front",
                "selected_eye": "both" if view == "front" else "right",
                "eye_selection_valid": True,
            }
        )
        sample["image"] = image
        return sample


class _FailedAuxiliaryPreprocessor(_AuxiliaryPreprocessor):
    """Models mark-invalid failures that cannot produce numeric auxiliaries."""

    @staticmethod
    def __call__(sample: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        image = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
        sample["metadata"].update(
            {
                "source_size_hw": np.asarray(image.shape[:2], dtype=np.int64),
                "image_transform": np.eye(3, dtype=np.float32),
                "head_orientation_valid": False,
                "face_origin_valid": False,
                "head_pose_valid": False,
                "eye_selection_valid": False,
                "gaze_valid": False,
            }
        )
        sample["image"] = image
        return sample


class _InvalidFrontOrientationPreprocessor(_AuxiliaryPreprocessor):
    @staticmethod
    def __call__(sample: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        output = _AuxiliaryPreprocessor.__call__(sample)
        metadata = output["metadata"]
        metadata["gaze_valid"] = True
        if output["view"] == "front":
            metadata["head_vector"] = np.full(3, np.nan, dtype=np.float32)
            metadata["head_orientation_valid"] = False
            metadata["head_pose_valid"] = False
        return output


class _OrientationOnlyFrontPreprocessor(_AuxiliaryPreprocessor):
    @staticmethod
    def __call__(sample: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        output = _AuxiliaryPreprocessor.__call__(sample)
        metadata = output["metadata"]
        metadata["gaze_valid"] = True
        if output["view"] == "front":
            metadata["face_origin_3d"] = np.full(3, np.nan, dtype=np.float32)
            metadata["face_origin_valid"] = False
            metadata["head_pose_valid"] = False
        return output


class _Profile2DPreprocessor:
    """Stand-in for the later annotation-driven 90-degree profile transform."""

    stage_order: tuple[str, ...] = ()

    @staticmethod
    def make_augmentation_context(
        _view: str,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        del generator
        return {}

    @staticmethod
    def __call__(sample: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        source = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
        image = (
            np.asarray(
                Image.fromarray(source).resize((256, 128)),
                dtype=np.float32,
            )
            / 255.0
        )
        metadata = sample["metadata"]
        metadata.update(
            {
                "source_size_hw": np.asarray(source.shape[:2], dtype=np.int64),
                "image_transform": np.eye(3, dtype=np.float32),
                "head_pose_2d": np.asarray([0.0, -1.0], dtype=np.float32),
                "head_pose_2d_valid": True,
                "eye_angles": np.asarray([-0.2, 0.3], dtype=np.float32),
                "iris_pose_2d": np.asarray([0.0, -0.6], dtype=np.float32),
                "selected_eye": "left",
                "eye_selection_valid": True,
                "gaze_valid": True,
            }
        )
        sample["image"] = image
        return sample


@pytest.mark.parametrize(
    ("view", "image_key", "prefix", "selected_eye", "selected_index"),
    [
        ("front", "front_image", "front", "both", 2),
        ("side", "side_image", "side", "right", 1),
    ],
)
def test_single_view_exposes_branch_prefixed_webeyetrack_auxiliary_tensors(
    tmp_path: Path,
    view: str,
    image_key: str,
    prefix: str,
    selected_eye: str,
    selected_index: int,
) -> None:
    image_path = tmp_path / f"{view}.png"
    _write_image(image_path, color=(120, 80, 200))
    dataset = GazeImageDataset([_row(view, image_path, view=view)], _config(), split="train")
    dataset.preprocessor = _AuxiliaryPreprocessor()

    item = dataset[0]

    assert item[image_key].shape == (3, 24, 32)
    assert item[f"{prefix}_head_vector"].shape == (3,)
    assert item[f"{prefix}_head_vector"].dtype == torch.float32
    assert item[f"{prefix}_face_origin_3d"].shape == (3,)
    assert item[f"{prefix}_head_orientation_valid"].item() is True
    assert item[f"{prefix}_face_origin_valid"].item() is (view == "front")
    assert item[f"{prefix}_head_pose_valid"].dtype == torch.bool
    assert item[f"{prefix}_gaze_valid"].dtype == torch.bool
    assert item[f"{prefix}_selected_eye"] == selected_eye
    # Stable categorical encoding: left=0, right=1, both=2, unavailable=-1.
    assert item[f"{prefix}_selected_eye_index"].item() == selected_index
    assert item[f"{prefix}_eye_selection_valid"].item() is True
    assert item["metadata"]["face_rt"].shape == (4, 4)
    assert item["metadata"]["metric_transform"].shape == (4, 4)
    assert item["metadata"]["head_euler_degrees"].tolist() == [1.0, 2.0, 3.0]
    assert item["metadata"]["eye_visibility_score"].tolist() == pytest.approx([0.9, 0.8])
    assert item["metadata"]["face_origin_unit"] == "cm"
    assert item["metadata"]["invalid_reasons"] == []

    other_prefix = "side" if prefix == "front" else "front"
    assert not any(key.startswith(f"{other_prefix}_") for key in item)


def test_paired_dataset_and_collate_expose_both_auxiliary_branches(tmp_path: Path) -> None:
    front_path = tmp_path / "front.png"
    side_path = tmp_path / "side.png"
    _write_image(front_path, color=(180, 120, 90))
    _write_image(side_path, color=(80, 150, 210))
    rows = [
        _row("front", front_path, view="front", pair_id="pair-01"),
        _row("side", side_path, view="side", pair_id="pair-01"),
    ]
    dataset = GazeImageDataset(rows, _config(paired=True), split="train")
    dataset.preprocessor = _AuxiliaryPreprocessor()

    item = dataset[0]
    batch = gaze_collate_fn([item])

    assert item["front_selected_eye"] == "both"
    assert item["side_selected_eye"] == "right"
    assert batch["front_head_vector"].shape == (1, 3)
    assert batch["side_face_origin_3d"].shape == (1, 3)
    assert batch["front_selected_eye"] == ["both"]
    assert batch["side_selected_eye_index"].tolist() == [1]
    assert batch["front_head_pose_valid"].tolist() == [True]
    assert batch["side_head_pose_valid"].tolist() == [False]
    assert batch["front_head_orientation_valid"].tolist() == [True]
    assert batch["side_head_orientation_valid"].tolist() == [True]
    assert batch["front_face_origin_valid"].tolist() == [True]
    assert batch["side_face_origin_valid"].tolist() == [False]
    assert batch["front_gaze_valid"].tolist() == [True]
    assert batch["side_gaze_valid"].tolist() == [False]


def test_paired_front_3d_invalid_orientation_is_zero_filled_and_masks_side(
    tmp_path: Path,
) -> None:
    front_path = tmp_path / "front.png"
    side_path = tmp_path / "side.png"
    _write_image(front_path, color=(180, 120, 90))
    _write_image(side_path, color=(80, 150, 210))
    rows = [
        _row("front", front_path, view="front", pair_id="pair-01"),
        _row("side", side_path, view="side", pair_id="pair-01"),
    ]
    dataset = GazeImageDataset(rows, _front3d_side_config(), split="train")
    dataset.preprocessor = _InvalidFrontOrientationPreprocessor()

    item = dataset[0]
    batch = gaze_collate_fn([item])

    assert item["front_head_vector"].tolist() == [0.0, 0.0, 0.0]
    assert item["front_head_orientation_valid"].item() is False
    assert item["side_gaze_valid"].item() is False
    assert item["metadata"]["side"]["front_3d_head_input_valid"] is False
    assert "side_headpose:front_3d_unavailable" in item["metadata"]["side"]["invalid_reasons"]
    assert torch.isfinite(batch["front_head_vector"]).all()


def test_paired_front_3d_uses_orientation_even_if_face_origin_is_invalid(
    tmp_path: Path,
) -> None:
    front_path = tmp_path / "front.png"
    side_path = tmp_path / "side.png"
    _write_image(front_path, color=(180, 120, 90))
    _write_image(side_path, color=(80, 150, 210))
    rows = [
        _row("front", front_path, view="front", pair_id="pair-01"),
        _row("side", side_path, view="side", pair_id="pair-01"),
    ]
    dataset = GazeImageDataset(rows, _front3d_side_config(), split="train")
    dataset.preprocessor = _OrientationOnlyFrontPreprocessor()

    item = dataset[0]

    assert item["front_head_vector"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert item["front_head_orientation_valid"].item() is True
    assert item["front_head_pose_valid"].item() is False
    assert item["side_gaze_valid"].item() is True
    assert item["metadata"]["side"]["front_3d_head_input_valid"] is True


def test_paired_front_3d_invalid_orientation_can_fail_fast(tmp_path: Path) -> None:
    front_path = tmp_path / "front.png"
    side_path = tmp_path / "side.png"
    _write_image(front_path, color=(180, 120, 90))
    _write_image(side_path, color=(80, 150, 210))
    rows = [
        _row("front", front_path, view="front", pair_id="pair-01"),
        _row("side", side_path, view="side", pair_id="pair-01"),
    ]
    dataset = GazeImageDataset(
        rows,
        _front3d_side_config(invalid_policy="error"),
        split="train",
    )
    dataset.preprocessor = _InvalidFrontOrientationPreprocessor()

    with pytest.raises(SampleValidationError, match="front_3d_unavailable"):
        dataset[0]


def test_enabled_auxiliary_stages_keep_shapes_when_detection_is_invalid(tmp_path: Path) -> None:
    image_path = tmp_path / "invalid.png"
    _write_image(image_path, color=(80, 80, 80))
    dataset = GazeImageDataset([_row("invalid", image_path)], _config(), split="train")
    dataset.preprocessor = _FailedAuxiliaryPreprocessor()

    item = dataset[0]

    assert torch.isnan(item["front_head_vector"]).all()
    assert torch.isnan(item["front_face_origin_3d"]).all()
    assert item["front_head_orientation_valid"].item() is False
    assert item["front_face_origin_valid"].item() is False
    assert item["front_head_pose_valid"].item() is False
    assert item["front_selected_eye"] == "unknown"
    assert item["front_selected_eye_index"].item() == -1
    assert item["front_eye_selection_valid"].item() is False
    assert item["front_gaze_valid"].item() is False


def test_collate_fills_missing_optional_auxiliary_values_safely() -> None:
    base = {
        "front_image": torch.zeros(3, 8, 8),
        "target_gaze_xy": torch.zeros(2),
        "target_in_screen_bounds": torch.tensor(True),
        "metadata": {},
    }
    with_aux = {
        **base,
        "front_head_vector": torch.tensor([0.1, 0.2, 0.3]),
        "front_head_pose_valid": torch.tensor(True),
        "front_selected_eye": "left",
        "front_selected_eye_index": torch.tensor(0),
    }

    batch = gaze_collate_fn([with_aux, dict(base)])

    assert batch["front_head_vector"].shape == (2, 3)
    assert torch.isnan(batch["front_head_vector"][1]).all()
    assert batch["front_head_pose_valid"].tolist() == [True, False]
    assert batch["front_selected_eye"] == ["left", "unknown"]
    assert batch["front_selected_eye_index"].tolist() == [0, -1]


def test_legacy_preprocessing_config_keeps_original_dataset_keys(tmp_path: Path) -> None:
    image_path = tmp_path / "front.png"
    _write_image(image_path, color=(180, 120, 90))
    dataset = GazeImageDataset([_row("front", image_path)], _config(), split="train")

    item = dataset[0]
    batch = gaze_collate_fn([item])

    assert set(item) == {
        "front_image",
        "target_gaze_xy",
        "target_in_screen_bounds",
        "metadata",
    }
    assert set(batch) == set(item)


def test_side_profile_annotations_and_derived_2d_pose_reach_model_batch(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "profile.png"
    _write_image(image_path, color=(100, 140, 180))
    row = {
        **_row("profile", image_path, view="side"),
        "visible_eye": "subject_left",
        "visible_eye_bbox_xyxy": [8, 6, 26, 18],
        "visible_eye_keypoints_xy": [
            [9, 12],
            [12, 9],
            [18, 9],
            [24, 12],
            [18, 15],
            [12, 15],
        ],
        "iris_center_xy": [17, 11],
        "profile_head_origin_xy": [16, 20],
        "profile_head_forward_xy": [16, 8],
        "eye_annotation_valid": True,
    }
    dataset = GazeImageDataset([row], _config(), split="train")
    dataset.preprocessor = _Profile2DPreprocessor()

    item = dataset[0]
    metadata = item["metadata"]

    assert metadata["visible_eye"] == "left"
    assert metadata["visible_eye_bbox_xyxy"].shape == (4,)
    assert metadata["visible_eye_keypoints_xy"].shape == (6, 2)
    assert metadata["iris_center_xy"].tolist() == [17.0, 11.0]
    assert metadata["profile_head_origin_xy"].tolist() == [16.0, 20.0]
    assert metadata["profile_head_forward_xy"].tolist() == [16.0, 8.0]
    assert metadata["eye_annotation_valid"].item() is True
    forward_keys = (
        "side_image",
        "side_head_pose_2d",
        "side_eye_angles",
        "side_iris_pose_2d",
    )
    model_forward = {key: item[key] for key in forward_keys}
    assert tuple(model_forward) == forward_keys
    assert model_forward["side_image"].shape == (3, 128, 256)
    assert model_forward["side_image"].dtype == torch.float32
    assert 0.0 <= model_forward["side_image"].min().item()
    assert model_forward["side_image"].max().item() <= 1.0
    assert model_forward["side_head_pose_2d"].shape == (2,)
    assert model_forward["side_head_pose_2d"].dtype == torch.float32
    assert model_forward["side_head_pose_2d"].tolist() == [0.0, -1.0]
    assert torch.linalg.vector_norm(model_forward["side_head_pose_2d"]).item() == pytest.approx(1.0)
    assert model_forward["side_eye_angles"].shape == (2,)
    assert model_forward["side_eye_angles"].dtype == torch.float32
    assert model_forward["side_eye_angles"].tolist() == pytest.approx([-0.2, 0.3])
    assert model_forward["side_iris_pose_2d"].shape == (2,)
    assert model_forward["side_iris_pose_2d"].dtype == torch.float32
    assert model_forward["side_iris_pose_2d"].tolist() == pytest.approx([0.0, -0.6])

    # Validity remains outside the four SideModel.forward inputs.
    assert item["side_gaze_valid"].item() is True
    assert "side_gaze_valid" not in model_forward

    without_derived_pose = dict(item)
    without_derived_pose.pop("side_head_pose_2d")
    without_derived_pose.pop("side_eye_angles")
    without_derived_pose.pop("side_iris_pose_2d")
    batch = gaze_collate_fn([item, without_derived_pose])

    assert batch["side_head_pose_2d"].shape == (2, 2)
    assert batch["side_eye_angles"].shape == (2, 2)
    assert batch["side_iris_pose_2d"].shape == (2, 2)
    assert batch["side_image"].shape == (2, 3, 128, 256)
    assert torch.isnan(batch["side_head_pose_2d"][1]).all()
    assert torch.isnan(batch["side_eye_angles"][1]).all()
    assert torch.isnan(batch["side_iris_pose_2d"][1]).all()


def test_side_profile_keypoints_require_exactly_six_points(tmp_path: Path) -> None:
    image_path = tmp_path / "profile.png"
    _write_image(image_path, color=(100, 140, 180))
    row = {
        **_row("profile", image_path, view="side"),
        "visible_eye_keypoints_xy": [[1, 2]] * 5,
    }
    dataset = GazeImageDataset([row], _config(), split="train")

    with pytest.raises(ManifestError, match=r"shape \[6, 2\]"):
        dataset[0]
