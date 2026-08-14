from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from gaze_pipeline.model_runtime import ModelContractError, build_model_runtime
from gaze_pipeline.models.webeyetrack_front import (
    OFFICIAL_BLAZEGAZE_MPIIFACEGAZE_SHA256,
    WebEyeTrackFrontError,
    _keras_with_torch_backend,
    _load_safe_keras_model,
    create_model,
    verify_keras_asset,
)
from gaze_pipeline.training import _Components, _load_weights_checkpoint, _save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ASSET = PROJECT_ROOT / "models" / "blazegaze_mpiifacegaze.keras"


@pytest.fixture(scope="module")
def keras_torch() -> Any:
    os.environ.setdefault("KERAS_BACKEND", "torch")
    keras = pytest.importorskip("keras")
    if keras.config.backend() != "torch":
        pytest.skip("WebEyeTrack Keras tests require KERAS_BACKEND=torch")
    return keras


def _branch_config(weights_path: Path) -> dict[str, Any]:
    return {
        "entrypoint": "gaze_pipeline.models.webeyetrack_front:create_model",
        "adapter_entrypoint": None,
        "init_args": {
            "weights_path": str(weights_path),
            "expected_sha256": OFFICIAL_BLAZEGAZE_MPIIFACEGAZE_SHA256,
        },
        "pretrained": {"path": None, "sha256": None, "strict": True},
        "input_contract": {
            "image_key": "front_image",
            "shape": ["B", 3, 128, 512],
            "dtype": "float32",
            "value_range": [0.0, 1.0],
            "auxiliary_keys": {
                "head_vector": {
                    "key": "front_head_vector",
                    "shape": ["B", 3],
                    "dtype": "float32",
                },
                "face_origin_3d": {
                    "key": "front_face_origin_3d",
                    "shape": ["B", 3],
                    "dtype": "float32",
                },
            },
        },
        "output_contract": {
            "gaze_key": "gaze_xy",
            "gaze_shape": ["B", 2],
        },
    }


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "front_image": torch.rand(batch_size, 3, 128, 512),
        "front_head_vector": torch.rand(batch_size, 3),
        "front_face_origin_3d": torch.rand(batch_size, 3),
    }


def _official_model(keras_torch: Any) -> nn.Module:
    if not OFFICIAL_ASSET.is_file():
        pytest.skip(f"official WebEyeTrack test asset is missing: {OFFICIAL_ASSET}")
    return create_model(OFFICIAL_ASSET)


def test_verify_asset_accepts_official_hash_and_rejects_tampering(tmp_path: Path) -> None:
    if not OFFICIAL_ASSET.is_file():
        pytest.skip(f"official WebEyeTrack test asset is missing: {OFFICIAL_ASSET}")

    verified, digest = verify_keras_asset(OFFICIAL_ASSET)

    assert verified == OFFICIAL_ASSET.resolve()
    assert digest == OFFICIAL_BLAZEGAZE_MPIIFACEGAZE_SHA256

    corrupted = tmp_path / "corrupted.keras"
    corrupted.write_bytes(OFFICIAL_ASSET.read_bytes() + b"tampered")
    with pytest.raises(WebEyeTrackFrontError, match="SHA-256 mismatch"):
        verify_keras_asset(corrupted)


def test_factory_rejects_an_already_initialized_non_torch_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_keras = SimpleNamespace(config=SimpleNamespace(backend=lambda: "tensorflow"))
    monkeypatch.setitem(sys.modules, "keras", fake_keras)

    with pytest.raises(WebEyeTrackFrontError, match="fresh process with KERAS_BACKEND=torch"):
        _keras_with_torch_backend()


def test_keras_import_defaults_to_cpu_for_device_consistency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_keras = SimpleNamespace(config=SimpleNamespace(backend=lambda: "torch"))
    monkeypatch.delitem(sys.modules, "keras", raising=False)
    monkeypatch.delenv("KERAS_TORCH_DEVICE", raising=False)
    original_import = __import__("importlib").import_module

    def fake_import(name: str) -> Any:
        return fake_keras if name == "keras" else original_import(name)

    monkeypatch.setattr(
        "gaze_pipeline.models.webeyetrack_front.importlib.import_module",
        fake_import,
    )

    assert _keras_with_torch_backend() is fake_keras
    assert os.environ["KERAS_TORCH_DEVICE"] == "cpu"


def test_keras_archive_is_loaded_without_compile_and_in_safe_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = tmp_path / "verified.keras"
    asset.write_bytes(b"fixture")
    sentinel = object()
    calls: list[tuple[Path, bool, bool]] = []

    def load_model(path: Path, *, compile: bool, safe_mode: bool) -> object:
        calls.append((path, compile, safe_mode))
        return sentinel

    fake_keras = SimpleNamespace(saving=SimpleNamespace(load_model=load_model))
    monkeypatch.setattr(
        "gaze_pipeline.models.webeyetrack_front._keras_with_torch_backend",
        lambda: fake_keras,
    )
    monkeypatch.setattr(
        "gaze_pipeline.models.webeyetrack_front._validate_loaded_model",
        lambda model: None,
    )

    assert _load_safe_keras_model(asset) is sentinel
    assert calls == [(asset, False, True)]


def test_official_factory_preserves_frozen_encoder_and_maps_chw_inputs(
    keras_torch: Any,
) -> None:
    model = _official_model(keras_torch)
    batch = _batch()
    canonical_output = model(**batch)["gaze_xy"]
    direct_output = model.keras_model(
        {
            "image": batch["front_image"].permute(0, 2, 3, 1).contiguous(),
            "head_vector": batch["front_head_vector"],
            "face_origin_3d": batch["front_face_origin_3d"],
        },
        training=True,
    )

    assert isinstance(model, nn.Module)
    assert model.weights_sha256 == OFFICIAL_BLAZEGAZE_MPIIFACEGAZE_SHA256
    assert model.encoder.trainable is False
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) == 156_018
    assert (
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        == 8_610
    )
    assert canonical_output.shape == (2, 2)
    assert torch.allclose(canonical_output, direct_output)

    canonical_output.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.gaze_mlp.parameters())


def test_encoder_can_be_unfrozen_explicitly(keras_torch: Any) -> None:
    model = create_model(OFFICIAL_ASSET, unfreeze_encoder=True)

    assert model.encoder.trainable is True
    assert any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert (
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        == 155_826
    )


def test_factory_integrates_with_default_adapter_and_validates_canonical_contract(
    keras_torch: Any,
) -> None:
    runtime = build_model_runtime("front", _branch_config(OFFICIAL_ASSET))
    runtime.eval()

    output = runtime(_batch())

    assert output["gaze_xy"].shape == (2, 2)
    with pytest.raises(ModelContractError, match="front_head_vector.*shape"):
        invalid = _batch()
        invalid["front_head_vector"] = torch.rand(2, 2)
        runtime(invalid)


def test_runtime_state_dict_is_safe_and_round_trips_checkpoint_resume(
    tmp_path: Path,
    keras_torch: Any,
) -> None:
    runtime = build_model_runtime("front", _branch_config(OFFICIAL_ASSET))
    runtime.eval()
    batch = _batch(1)

    trainable = next(parameter for parameter in runtime.parameters() if parameter.requires_grad)
    with torch.no_grad():
        trainable.add_(0.125)
    expected_output = runtime(batch)["gaze_xy"].detach().clone()
    expected_state = {key: value.detach().clone() for key, value in runtime.state_dict().items()}
    checkpoint = tmp_path / "front_state.pt"
    _save_checkpoint(
        checkpoint,
        components=_Components(front=runtime, side=None, fusion=None),
        config={"model": {"backend": "pytorch"}},
        epoch=4,
        metrics={"loss": 0.25},
        optimizer=None,
        scheduler=None,
        weights_only=True,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["epoch"] == 4
    restored = build_model_runtime("front", _branch_config(OFFICIAL_ASSET))
    _load_weights_checkpoint(
        checkpoint,
        _Components(front=restored, side=None, fusion=None),
        device=torch.device("cpu"),
    )
    restored.eval()

    assert expected_state
    assert all(torch.is_tensor(value) for value in expected_state.values())
    assert set(restored.state_dict()) == set(expected_state)
    for key, value in restored.state_dict().items():
        assert torch.equal(value, expected_state[key])
    assert torch.allclose(restored(batch)["gaze_xy"], expected_output)


def test_custom_expected_hash_is_enforced_before_keras_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = tmp_path / "small.keras"
    asset.write_bytes(b"not a keras archive")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    called = False

    def fail_if_called(path: Path) -> Any:
        nonlocal called
        called = True
        raise AssertionError(path)

    monkeypatch.setattr(
        "gaze_pipeline.models.webeyetrack_front._load_safe_keras_model",
        fail_if_called,
    )
    with pytest.raises(WebEyeTrackFrontError, match="SHA-256 mismatch"):
        create_model(asset, expected_sha256="0" * 64)
    assert called is False

    with pytest.raises(AssertionError):
        create_model(asset, expected_sha256=digest)
    assert called is True
