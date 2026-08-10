import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from gaze_pipeline.data import (
    GazeImageDataset,
    ManifestError,
    OpenCVHaarFaceDetector,
    build_dataloader,
)


def _write_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    image[10:52, 18:63] = color
    Image.fromarray(image).save(path)


def _row(
    sample_id: str,
    path: Path,
    *,
    view: str = "front",
    pair_id: str = "",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "subject_id": "p00",
        "session_id": "day01",
        "view": view,
        "image_path": str(path),
        "pair_id": pair_id,
        "target_x_normalized": -0.25,
        "target_y_normalized": 0.10,
        "target_x_px": 360,
        "target_y_px": 540,
        "screen_width_px": 1440,
        "screen_height_px": 900,
        "screen_width_mm": 286.4,
        "screen_height_mm": 179.0,
        "target_in_screen_bounds": True,
        "facial_landmarks_xy": ("[[25,25],[32,25],[48,25],[55,25],[31,43],[49,43]]"),
    }


def _config(*, paired: bool = False) -> dict[str, object]:
    return {
        "experiment": {"seed": 123},
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
            },
            "dataloader": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                # These must not be passed to PyTorch when num_workers is zero.
                "persistent_workers": True,
                "prefetch_factor": 4,
                "train_shuffle": True,
                "validation_shuffle": False,
                "test_shuffle": False,
                "drop_last_train": False,
            },
        },
        "preprocessing": {
            "stage_order": [
                "decode",
                "exif_orientation",
                "validate",
                "face_landmarks",
                "eye_region_warp",
                "face_roi",
                "background_mask",
                "resize",
                "normalize",
                "augment",
            ],
            "decode": {"enabled": True, "backend": "pillow"},
            "exif_orientation": {"enabled": True},
            "validate": {"enabled": True, "min_width_px": 32, "min_height_px": 32},
            "face_landmarks": {"enabled": True, "source": "annotation"},
            "eye_region_warp": {"enabled": False},
            "face_roi": {"enabled": False},
            # MPII-like test images already contain a black canvas: no-op masking.
            "background_mask": {"enabled": False},
            "resize": {
                "enabled": True,
                "size_hw": [48, 64],
                "keep_aspect_ratio": True,
                "pad_rgb": [0, 0, 0],
            },
            "normalize": {
                "enabled": True,
                "mode": "zero_one",
                "channel_order": "CHW",
            },
            "augment": {"enabled": False},
        },
    }


def test_single_view_dataset_and_zero_worker_dataloader_contract(tmp_path: Path) -> None:
    image_path = tmp_path / "front.png"
    _write_image(image_path, color=(180, 120, 90))
    config = _config()
    dataset = GazeImageDataset([_row("front-1", image_path)], config, split="train")

    item = dataset[0]
    assert set(item) == {
        "front_image",
        "target_gaze_xy",
        "target_in_screen_bounds",
        "metadata",
    }
    assert item["front_image"].shape == (3, 48, 64)
    assert item["front_image"].dtype == torch.float32
    assert item["target_gaze_xy"].tolist() == [-0.25, 0.10000000149011612]
    assert item["target_in_screen_bounds"].item() is True
    assert item["metadata"]["target_in_screen_bounds"].item() is True
    assert item["metadata"]["landmarks_xy"].shape == (6, 2)

    loader = build_dataloader(dataset, config, split="train")
    batch = next(iter(loader))
    assert batch["front_image"].shape == (1, 3, 48, 64)
    assert batch["target_gaze_xy"].shape == (1, 2)
    assert batch["target_in_screen_bounds"].tolist() == [True]
    assert isinstance(batch["metadata"], list)
    assert loader.persistent_workers is False
    assert loader.prefetch_factor is None


def test_explicit_pair_returns_two_images_and_branch_only_stays_single_view(
    tmp_path: Path,
) -> None:
    front_path = tmp_path / "front.png"
    side_path = tmp_path / "side.png"
    orphan_path = tmp_path / "orphan.png"
    _write_image(front_path, color=(180, 120, 90))
    _write_image(side_path, color=(100, 160, 200))
    _write_image(orphan_path, color=(130, 130, 130))
    rows = [
        _row("pair-front", front_path, view="front", pair_id="pair-001"),
        _row("pair-side", side_path, view="side", pair_id="pair-001"),
        _row("front-orphan", orphan_path, view="front"),
    ]
    config = _config(paired=True)

    paired = GazeImageDataset(rows, config, split="train")
    assert len(paired) == 1
    item = paired[0]
    assert item["front_image"].shape == (3, 48, 64)
    assert item["side_image"].shape == (3, 48, 64)
    assert item["pair_mask"].item() is True
    assert item["target_in_screen_bounds"].item() is True
    assert item["metadata"]["front"]["target_in_screen_bounds"].item() is True
    assert item["metadata"]["side"]["target_in_screen_bounds"].item() is True
    batch = next(iter(build_dataloader(paired, config, split="train")))
    assert batch["front_image"].shape == (1, 3, 48, 64)
    assert batch["side_image"].shape == (1, 3, 48, 64)
    assert batch["pair_mask"].shape == (1,)
    assert batch["target_in_screen_bounds"].tolist() == [True]
    assert isinstance(batch["metadata"], list)

    # branch_only rows are consumed through an explicitly filtered branch Dataset.
    front_only = GazeImageDataset(
        rows,
        config,
        split="train",
        paired=False,
        view="front",
    )
    assert len(front_only) == 2
    assert "front_image" in front_only[1]


def test_dataloader_uses_split_shuffle_policy_and_reproducible_seed(tmp_path: Path) -> None:
    image_path = tmp_path / "front.png"
    _write_image(image_path, color=(180, 120, 90))
    rows = [_row(f"front-{index}", image_path) for index in range(6)]
    config = _config()
    config["data"]["dataloader"]["drop_last_train"] = True
    dataset = GazeImageDataset(rows, config, split="train")

    train_first = build_dataloader(dataset, config, split="train")
    train_second = build_dataloader(dataset, config, split="train")
    validation = build_dataloader(dataset, config, split="validation")
    test = build_dataloader(dataset, config, split="test")

    assert isinstance(train_first.sampler, RandomSampler)
    assert isinstance(validation.sampler, SequentialSampler)
    assert isinstance(test.sampler, SequentialSampler)
    assert list(train_first.sampler) == list(train_second.sampler)
    assert train_first.drop_last is True
    assert validation.drop_last is False
    assert test.drop_last is False


def test_pair_id_is_global_and_cannot_join_different_subjects(tmp_path: Path) -> None:
    front_path = tmp_path / "front.png"
    side_path = tmp_path / "side.png"
    _write_image(front_path, color=(180, 120, 90))
    _write_image(side_path, color=(100, 160, 200))
    front = _row("front", front_path, view="front", pair_id="global-pair")
    side = _row("side", side_path, view="side", pair_id="global-pair")
    side["subject_id"] = "p01"

    with pytest.raises(ManifestError, match="different subjects"):
        GazeImageDataset([front, side], _config(paired=True), split="train")


def test_horizontal_flip_keeps_pixel_and_normalized_screen_targets_consistent(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "front.png"
    _write_image(image_path, color=(180, 120, 90))
    config = _config()
    config["preprocessing"]["augment"] = {
        "enabled": True,
        "apply_to": "train_only",
        "horizontal_flip_probability": 1.0,
    }
    dataset = GazeImageDataset([_row("front", image_path)], config, split="train")

    item = dataset[0]
    expected_x_px = 1440 - 1 - 360
    expected_x_normalized = expected_x_px / 1440 - 0.5
    assert item["metadata"]["gaze_screen_xy_px"].tolist() == pytest.approx([expected_x_px, 540])
    assert item["target_gaze_xy"].tolist() == pytest.approx([expected_x_normalized, 0.1])
    assert item["target_in_screen_bounds"].item() is True


def test_augmentation_is_stable_per_seed_sample_and_epoch(tmp_path: Path) -> None:
    image_path = tmp_path / "front.png"
    _write_image(image_path, color=(180, 120, 90))
    config = _config()
    config["preprocessing"]["augment"] = {
        "enabled": True,
        "apply_to": "train_only",
        "horizontal_flip_probability": 0.5,
        "color_jitter": {
            "enabled": True,
            "brightness": 0.4,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        },
    }
    first = GazeImageDataset([_row("stable", image_path)], config, split="train")
    second = GazeImageDataset([_row("stable", image_path)], config, split="train")
    first.set_epoch(3)
    second.set_epoch(3)

    torch.manual_seed(1)
    first_item = first[0]
    torch.manual_seed(999999)
    second_item = second[0]
    assert torch.equal(first_item["front_image"], second_item["front_image"])
    assert torch.equal(first_item["target_gaze_xy"], second_item["target_gaze_xy"])
    assert (
        first_item["metadata"]["augmentation_seed"] == second_item["metadata"]["augmentation_seed"]
    )

    first_loader = build_dataloader(first, config, split="train")
    second_loader = build_dataloader(second, config, split="train")
    assert torch.equal(
        next(iter(first_loader))["front_image"],
        next(iter(second_loader))["front_image"],
    )

    first.set_epoch(4)
    next_epoch = first[0]
    assert (
        next_epoch["metadata"]["augmentation_seed"] != first_item["metadata"]["augmentation_seed"]
    )
    assert not torch.equal(next_epoch["front_image"], first_item["front_image"])


def test_generated_pair_id_falls_back_from_custom_source_key(tmp_path: Path) -> None:
    front_path = tmp_path / "front.png"
    side_path = tmp_path / "side.png"
    _write_image(front_path, color=(180, 120, 90))
    _write_image(side_path, color=(100, 160, 200))
    rows = [
        _row("front", front_path, view="front", pair_id="canonical-pair"),
        _row("side", side_path, view="side", pair_id="canonical-pair"),
    ]
    config = _config(paired=True)
    config["data"]["pairing"]["pair_id_key"] = "source_capture_id"

    dataset = GazeImageDataset(rows, config, split="train")
    assert len(dataset) == 1
    assert dataset[0]["metadata"]["pair_id"] == "canonical-pair"


def test_collate_keeps_mixed_optional_metadata_as_sample_list(tmp_path: Path) -> None:
    image_path = tmp_path / "front.png"
    _write_image(image_path, color=(180, 120, 90))
    first = _row("with-annotations", image_path)
    first["head_rotation_3d"] = "[0.1,0.2,0.3]"
    second = _row("without-annotations", image_path)
    second.pop("facial_landmarks_xy")
    second["target_in_screen_bounds"] = False
    config = _config()
    config["data"]["dataloader"]["train_shuffle"] = False
    config["preprocessing"]["face_landmarks"]["enabled"] = False

    dataset = GazeImageDataset([first, second], config, split="train")
    batch = next(iter(build_dataloader(dataset, config, split="train")))

    assert batch["front_image"].shape == (2, 3, 48, 64)
    assert batch["target_in_screen_bounds"].tolist() == [True, False]
    assert isinstance(batch["metadata"], list)
    assert "landmarks_xy" in batch["metadata"][0]
    assert "head_rotation_3d" in batch["metadata"][0]
    assert "landmarks_xy" not in batch["metadata"][1]
    assert "head_rotation_3d" not in batch["metadata"][1]


def test_lazy_opencv_haar_fallback_is_pickle_safe_and_supports_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "side.png"
    _write_image(image_path, color=(100, 160, 200))
    calls: list[str] = []

    def fake_detect(
        _detector: OpenCVHaarFaceDetector, image: np.ndarray, view: str
    ) -> dict[str, object]:
        calls.append(view)
        assert image.shape == (60, 80, 3)
        return {
            "landmarks_xy": np.asarray(
                [[18, 10], [40, 9], [62, 10], [65, 30], [62, 51], [40, 53], [18, 51], [15, 30]],
                dtype=np.float32,
            ),
            "landmark_kind": "face_bbox_hull",
            "face_bbox_xywh": np.asarray([15, 9, 51, 45], dtype=np.float32),
            "mirrored_retry": True,
        }

    monkeypatch.setattr(OpenCVHaarFaceDetector, "detect", fake_detect)
    row = _row("side-no-landmarks", image_path, view="side")
    row.pop("facial_landmarks_xy")
    config = _config()
    config["preprocessing"]["face_landmarks"].update(
        {"fallback_detector": "opencv_haar", "on_failure": "error"}
    )
    config["preprocessing"]["face_roi"]["enabled"] = True
    config["preprocessing"]["background_mask"] = {
        "enabled": True,
        "method": "face_hull",
        "fill_rgb": [0, 0, 0],
    }
    dataset = GazeImageDataset([row], config, split="train", paired=False, view="side")
    assert dataset.preprocessor._opencv_haar_detector is None

    item = dataset[0]
    assert calls == ["side"]
    assert item["metadata"]["landmark_kind"] == "face_bbox_hull"
    assert item["metadata"]["detector_mirrored_retry"].item() is True
    detector = dataset.preprocessor._opencv_haar_detector
    assert detector is not None
    detector._classifiers["side"] = lambda: None
    restored = pickle.loads(pickle.dumps(dataset))
    restored_detector = restored.preprocessor._opencv_haar_detector
    assert restored_detector is not None
    assert restored_detector._classifiers == {}


def test_injected_detector_takes_priority_over_opencv_haar(tmp_path: Path) -> None:
    image_path = tmp_path / "front.png"
    _write_image(image_path, color=(180, 120, 90))
    row = _row("front-no-landmarks", image_path)
    row.pop("facial_landmarks_xy")
    config = _config()
    config["preprocessing"]["face_landmarks"].update(
        {"fallback_detector": "opencv_haar", "on_failure": "error"}
    )
    calls: list[str] = []

    def injected(_image: np.ndarray, view: str) -> np.ndarray:
        calls.append(view)
        return np.asarray(
            [[25, 25], [32, 25], [48, 25], [55, 25], [31, 43], [49, 43]],
            dtype=np.float32,
        )

    dataset = GazeImageDataset([row], config, split="train", landmark_detector=injected)
    item = dataset[0]
    assert calls == ["front"]
    assert dataset.preprocessor._opencv_haar_detector is None
    assert item["metadata"]["landmark_kind"] == "injected_detector"
