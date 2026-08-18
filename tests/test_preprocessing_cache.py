from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from gaze_pipeline.data.dataset import GazeImageDataset
from gaze_pipeline.data.preprocessing_cache import (
    PreprocessingCacheConfigurationError,
    PreprocessingCacheCorruptionError,
    PreprocessingCacheSecurityError,
    PreprocessingCacheSourceChangedError,
    TrustedLocalPreprocessingCache,
)


def _cache(tmp_path: Path, config: dict[str, object] | None = None):
    return TrustedLocalPreprocessingCache(
        tmp_path / "cache",
        config or {"eye_region_warp": {"size_hw": [128, 256], "crop_mode": "stretch"}},
        trusted_local=True,
    )


def _image(tmp_path: Path, content: bytes = b"image-v1") -> Path:
    path = tmp_path / "phone.png"
    path.write_bytes(content)
    return path


def _identity(cache: TrustedLocalPreprocessingCache, image: Path):
    return cache.identity(
        row={"sample_id": "sample-1", "visible_eye_bbox_xyxy": "[1,2,30,20]"},
        view="side",
        image_path=image,
    )


def test_cache_requires_explicit_trusted_local_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(PreprocessingCacheSecurityError, match="trusted_local=true"):
        TrustedLocalPreprocessingCache(tmp_path / "cache", {}, trusted_local=False)


def test_identity_is_stable_for_mapping_order_and_tracks_all_inputs(tmp_path: Path) -> None:
    image = _image(tmp_path)
    first = _cache(
        tmp_path,
        {"normalize": {"mode": "zero_one"}, "resize": {"size_hw": [128, 256]}},
    )
    reordered = _cache(
        tmp_path / "other",
        {"resize": {"size_hw": [128, 256]}, "normalize": {"mode": "zero_one"}},
    )
    row_a = {"sample_id": "a", "pair_id": "p"}
    row_b = {"pair_id": "p", "sample_id": "a"}

    identity_a = first.identity(row=row_a, view="side", image_path=image)
    identity_b = reordered.identity(row=row_b, view="side", image_path=image)

    assert identity_a.key == identity_b.key
    changed_row = first.identity(
        row={"sample_id": "a", "pair_id": "changed"}, view="side", image_path=image
    )
    assert changed_row.key != identity_a.key
    changed_config = _cache(tmp_path / "changed-config", {"resize": [64, 128]}).identity(
        row=row_a, view="side", image_path=image
    )
    assert changed_config.key != identity_a.key
    front_identity = first.identity(row=row_a, view="front", image_path=image)
    assert front_identity.key != identity_a.key

    image.write_bytes(b"image-v2")
    changed_source = first.identity(row=row_a, view="side", image_path=image)
    assert changed_source.key != identity_a.key
    assert changed_source.source_sha256 != identity_a.source_sha256


def test_round_trip_preserves_plain_data_numpy_and_torch(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    identity = _identity(cache, _image(tmp_path))
    payload = {
        "image": torch.arange(12, dtype=torch.float32).reshape(3, 2, 2),
        "target": np.asarray([0.1, -0.2], dtype=np.float32),
        "metadata": {"sample_id": "sample-1", "valid": True, "optional": None},
    }

    shard = cache.store(identity, payload)
    restored = cache.load(identity)

    assert shard.suffix == ".pkl"
    assert shard.is_file()
    assert cache.entry_paths(identity).checksum.is_file()
    assert torch.equal(restored["image"], payload["image"])
    np.testing.assert_array_equal(restored["target"], payload["target"])
    assert restored["metadata"] == payload["metadata"]


def test_missing_entry_returns_none_but_partial_entry_is_corrupt(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    identity = _identity(cache, _image(tmp_path))
    assert cache.load(identity) is None

    paths = cache.entry_paths(identity)
    paths.shard.parent.mkdir(parents=True)
    paths.shard.write_bytes(b"partial")
    with pytest.raises(PreprocessingCacheCorruptionError, match="incomplete"):
        cache.load(identity)


@pytest.mark.parametrize("target", ["shard", "checksum"])
def test_checksum_or_shard_tampering_is_rejected(tmp_path: Path, target: str) -> None:
    cache = _cache(tmp_path)
    identity = _identity(cache, _image(tmp_path))
    cache.store(identity, {"value": 1})
    paths = cache.entry_paths(identity)
    path = getattr(paths, target)
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(PreprocessingCacheCorruptionError):
        cache.load(identity)


def test_envelope_identity_mismatch_is_rejected_even_with_updated_sidecar(
    tmp_path: Path,
) -> None:
    import hashlib
    import pickle

    cache = _cache(tmp_path)
    identity = _identity(cache, _image(tmp_path))
    cache.store(identity, {"value": 1})
    paths = cache.entry_paths(identity)
    envelope = pickle.loads(paths.shard.read_bytes())
    envelope["view"] = "front"
    changed = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    paths.shard.write_bytes(changed)
    paths.checksum.write_text(hashlib.sha256(changed).hexdigest() + "\n", encoding="ascii")

    with pytest.raises(PreprocessingCacheCorruptionError, match="field 'view'"):
        cache.load(identity)


def test_source_change_between_identity_and_store_is_rejected(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    image = _image(tmp_path, b"same-size-a")
    identity = _identity(cache, image)
    image.write_bytes(b"same-size-b")

    with pytest.raises(PreprocessingCacheSourceChangedError, match="content changed"):
        cache.store(identity, {"value": 1})


def test_payload_rejects_custom_python_objects(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    identity = _identity(cache, _image(tmp_path))

    with pytest.raises(PreprocessingCacheConfigurationError, match="unsupported value type"):
        cache.store(identity, {"unsafe": object()})


def test_forged_path_traversal_identity_is_rejected(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    identity = _identity(cache, _image(tmp_path))
    forged = dataclasses.replace(identity, key="../" + identity.key[3:])

    with pytest.raises(PreprocessingCacheSecurityError, match="not a SHA-256"):
        cache.entry_paths(forged)


def test_symlink_cache_root_and_shard_are_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(PreprocessingCacheSecurityError, match="non-symlink directory"):
        TrustedLocalPreprocessingCache(linked_root, {}, trusted_local=True)

    cache = _cache(tmp_path / "entry")
    image = _image(tmp_path / "entry")
    identity = _identity(cache, image)
    paths = cache.entry_paths(identity)
    paths.shard.parent.mkdir(parents=True)
    external = tmp_path / "external.pkl"
    external.write_bytes(b"external")
    paths.shard.symlink_to(external)
    paths.checksum.write_text("0" * 64 + "\n", encoding="ascii")

    with pytest.raises(PreprocessingCacheSecurityError, match="non-symlink regular file"):
        cache.load(identity)


def test_symlink_managed_subdirectory_cannot_escape_root(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    identity = _identity(cache, _image(tmp_path))
    paths = cache.entry_paths(identity)
    namespace = paths.shard.parents[1]
    namespace.mkdir(parents=True)
    external = tmp_path / "external-directory"
    external.mkdir()
    paths.shard.parent.symlink_to(external, target_is_directory=True)

    with pytest.raises(PreprocessingCacheSecurityError):
        cache.store(identity, {"value": 1})
    assert list(external.iterdir()) == []


def test_store_uses_atomic_replace_and_leaves_no_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _cache(tmp_path)
    identity = _identity(cache, _image(tmp_path))
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]):
        calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", recording_replace)
    cache.store(identity, {"value": 1})

    paths = cache.entry_paths(identity)
    assert [destination for _, destination in calls] == [paths.shard, paths.checksum]
    assert not list(paths.shard.parent.glob("*.tmp"))


def test_identity_from_another_config_or_schema_is_rejected(tmp_path: Path) -> None:
    image = _image(tmp_path)
    first = _cache(tmp_path / "first", {"resize": [128, 256]})
    second = _cache(tmp_path / "second", {"resize": [64, 128]})
    identity = _identity(first, image)

    with pytest.raises(PreprocessingCacheSecurityError, match="config digest"):
        second.entry_paths(identity)

    other_schema = dataclasses.replace(identity, schema_version=2)
    with pytest.raises(PreprocessingCacheSecurityError, match="schema_version"):
        first.entry_paths(other_schema)


def _dataset_config(
    cache_root: Path,
    *,
    resize_hw: list[int] | None = None,
    cache_mode: str = "read_write",
) -> dict[str, object]:
    return {
        "experiment": {"seed": 42},
        "data": {"pairing": {"enabled": False}},
        "task": {
            "coordinate_system": {"name": "centered_normalized_screen"},
        },
        "preprocessing": {
            "stage_order": ["decode", "resize", "normalize", "augment"],
            "decode": {"enabled": True, "backend": "opencv"},
            "resize": {
                "enabled": True,
                "size_hw": resize_hw or [12, 20],
                "keep_aspect_ratio": False,
            },
            "normalize": {
                "enabled": True,
                "mode": "zero_one",
                "channel_order": "CHW",
            },
            "augment": {
                "enabled": True,
                "apply_to": "train_only",
                "horizontal_flip_probability": 0.0,
                "color_jitter": {"enabled": True, "brightness": 0.4},
            },
            "cache": {
                "enabled": True,
                "dir": str(cache_root),
                "mode": cache_mode,
                "format": "pickle",
                "schema_version": 1,
                "implementation_id": "dataset-cache-integration-v1",
                "trusted_local": True,
            },
            "branch_overrides": {
                "front": {"enabled": True},
                "side": {"enabled": False},
            },
        },
    }


def _dataset_row(image_path: Path) -> dict[str, object]:
    return {
        "sample_id": "front-1",
        "subject_id": "subject-1",
        "session_id": "neutral",
        "view": "front",
        "image_path": str(image_path),
        "target_x_px": 50,
        "target_y_px": 25,
        "screen_width_px": 100,
        "screen_height_px": 50,
    }


def test_dataset_reuses_deterministic_cache_and_reapplies_epoch_augmentation(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "source.png"
    Image.fromarray(np.full((18, 30, 3), 160, dtype=np.uint8)).save(image_path)
    cache_root = tmp_path / "dataset-cache"
    dataset = GazeImageDataset(
        [_dataset_row(image_path)],
        _dataset_config(cache_root),
        split="train",
        view="front",
    )

    dataset.set_epoch(0)
    first = dataset[0]
    dataset.set_epoch(1)
    second = dataset[0]

    assert first["metadata"]["preprocessing_cache_hit"] is False
    assert second["metadata"]["preprocessing_cache_hit"] is True
    assert int(first["metadata"]["augmentation_epoch"]) == 0
    assert int(second["metadata"]["augmentation_epoch"]) == 1
    assert int(first["metadata"]["augmentation_seed"]) != int(
        second["metadata"]["augmentation_seed"]
    )
    assert first["front_image"].shape == (3, 12, 20)
    assert second["front_image"].shape == (3, 12, 20)
    assert not torch.equal(first["front_image"], second["front_image"])
    assert len(list(cache_root.rglob("*.pkl"))) == 1


def test_dataset_cache_invalidates_when_preprocessing_or_source_changes(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.fromarray(np.full((18, 30, 3), 80, dtype=np.uint8)).save(image_path)
    cache_root = tmp_path / "dataset-cache"
    row = _dataset_row(image_path)

    first_dataset = GazeImageDataset(
        [row], _dataset_config(cache_root), split="validation", view="front"
    )
    assert first_dataset[0]["metadata"]["preprocessing_cache_hit"] is False
    assert first_dataset[0]["metadata"]["preprocessing_cache_hit"] is True

    resized_dataset = GazeImageDataset(
        [row],
        _dataset_config(cache_root, resize_hw=[10, 16]),
        split="validation",
        view="front",
    )
    resized = resized_dataset[0]
    assert resized["metadata"]["preprocessing_cache_hit"] is False
    assert resized["front_image"].shape == (3, 10, 16)

    Image.fromarray(np.full((18, 30, 3), 180, dtype=np.uint8)).save(image_path)
    changed_source = GazeImageDataset(
        [row], _dataset_config(cache_root), split="validation", view="front"
    )[0]
    assert changed_source["metadata"]["preprocessing_cache_hit"] is False
    assert len(list(cache_root.rglob("*.pkl"))) == 3


def test_dataset_cache_separates_train_and_validation_namespaces(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.fromarray(np.full((18, 30, 3), 80, dtype=np.uint8)).save(image_path)
    cache_root = tmp_path / "dataset-cache"
    row = _dataset_row(image_path)

    train = GazeImageDataset([row], _dataset_config(cache_root), split="train", view="front")
    validation = GazeImageDataset(
        [row], _dataset_config(cache_root), split="validation", view="front"
    )

    assert train[0]["metadata"]["preprocessing_cache_hit"] is False
    assert validation[0]["metadata"]["preprocessing_cache_hit"] is False
    assert train[0]["metadata"]["preprocessing_cache_hit"] is True
    assert validation[0]["metadata"]["preprocessing_cache_hit"] is True
    assert len(list(cache_root.rglob("*.pkl"))) == 2


def test_dataset_cache_separates_different_augmentation_boundaries(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.fromarray(np.full((18, 30, 3), 120, dtype=np.uint8)).save(image_path)
    cache_root = tmp_path / "dataset-cache"
    row = _dataset_row(image_path)

    enabled_config = _dataset_config(cache_root)
    enabled_preprocessing = enabled_config["preprocessing"]
    assert isinstance(enabled_preprocessing, dict)
    enabled_preprocessing["stage_order"] = ["decode", "resize", "augment", "normalize"]

    disabled_config = _dataset_config(cache_root)
    disabled_preprocessing = disabled_config["preprocessing"]
    assert isinstance(disabled_preprocessing, dict)
    disabled_preprocessing["stage_order"] = ["decode", "resize", "augment", "normalize"]
    disabled_augment = disabled_preprocessing["augment"]
    assert isinstance(disabled_augment, dict)
    disabled_augment["enabled"] = False

    enabled = GazeImageDataset([row], enabled_config, split="validation", view="front")[0]
    disabled = GazeImageDataset([row], disabled_config, split="validation", view="front")[0]

    assert enabled["metadata"]["preprocessing_cache_hit"] is False
    assert disabled["metadata"]["preprocessing_cache_hit"] is False
    assert (
        enabled["metadata"]["preprocessing_cache_key"]
        != disabled["metadata"]["preprocessing_cache_key"]
    )
    assert enabled["front_image"].dtype == torch.float32
    assert disabled["front_image"].dtype == torch.float32
    assert len(list(cache_root.rglob("*.pkl"))) == 2


def test_dataset_read_only_and_refresh_cache_modes(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.fromarray(np.full((18, 30, 3), 100, dtype=np.uint8)).save(image_path)
    cache_root = tmp_path / "dataset-cache"
    row = _dataset_row(image_path)

    read_only_miss = GazeImageDataset(
        [row],
        _dataset_config(cache_root, cache_mode="read_only"),
        split="validation",
        view="front",
    )[0]
    assert read_only_miss["metadata"]["preprocessing_cache_hit"] is False
    assert not list(cache_root.rglob("*.pkl"))

    writer = GazeImageDataset([row], _dataset_config(cache_root), split="validation", view="front")
    assert writer[0]["metadata"]["preprocessing_cache_hit"] is False
    read_only_hit = GazeImageDataset(
        [row],
        _dataset_config(cache_root, cache_mode="read_only"),
        split="validation",
        view="front",
    )[0]
    assert read_only_hit["metadata"]["preprocessing_cache_hit"] is True

    refresh = GazeImageDataset(
        [row],
        _dataset_config(cache_root, cache_mode="refresh"),
        split="validation",
        view="front",
    )
    assert refresh[0]["metadata"]["preprocessing_cache_hit"] is False
    assert refresh[0]["metadata"]["preprocessing_cache_hit"] is True
    assert len(list(cache_root.rglob("*.pkl"))) == 1
