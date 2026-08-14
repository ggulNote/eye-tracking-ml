from __future__ import annotations

import importlib
from collections.abc import Callable
from inspect import signature
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from gaze_pipeline.config import load_and_validate_config, resolve_model_forward_keys
from gaze_pipeline.model_runtime import build_model_runtime
from gaze_pipeline.models.side import SideAuxiliaryProjector, SideEncoderHead
from gaze_pipeline.models.side.components import validate_and_sanitize_side_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
BLAZEGAZE_PROFILE = PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml"
SIDE_PROFILE = PROJECT_ROOT / "configs" / "profiles" / "side_profile_90.yaml"
BLAZEGAZE_MODEL = PROJECT_ROOT / "configs" / "models" / "side_blazegaze_transfer.yaml"
MOBILENET_MODEL = PROJECT_ROOT / "configs" / "models" / "side_mobilenet_v4.yaml"


def _import_entrypoint_for_test(entrypoint: str) -> Callable[..., Any]:
    """Resolve a config entrypoint independently of the product loader."""

    module_name, separator, attribute_path = entrypoint.partition(":")
    assert separator and module_name and attribute_path
    value: Any = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    assert callable(value)
    return value


def _load_side_model_config(model_config: Path) -> dict[str, Any]:
    return load_and_validate_config(
        BASE_CONFIG,
        profiles=(BLAZEGAZE_PROFILE, SIDE_PROFILE, model_config),
    )


def _side_image(batch_size: int = 2) -> torch.Tensor:
    return torch.zeros(batch_size, 3, 128, 256, dtype=torch.float32)


def test_side_inputs_zero_fill_non_finite_auxiliary_without_validity_output() -> None:
    head = torch.tensor([[float("nan"), 1.0], [float("inf"), -float("inf")]])
    eye = torch.tensor([[0.1, 0.2], [0.3, float("nan")]])

    validated = validate_and_sanitize_side_inputs(
        _side_image(),
        side_head_pose_2d=head,
        side_eye_angles=eye,
    )

    assert validated.side_head_pose_2d is not None
    assert validated.side_eye_angles is not None
    assert torch.isfinite(validated.side_head_pose_2d).all()
    assert torch.isfinite(validated.side_eye_angles).all()
    assert validated.side_head_pose_2d.tolist() == [[0.0, 1.0], [0.0, 0.0]]
    torch.testing.assert_close(
        validated.side_eye_angles,
        torch.tensor([[0.1, 0.2], [0.3, 0.0]]),
    )
    assert torch.isnan(head[0, 0])
    assert not hasattr(validated, "side_gaze_valid")


def test_side_head_sources_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_and_sanitize_side_inputs(
            _side_image(),
            side_head_pose_2d=torch.zeros(2, 2),
            front_head_vector=torch.zeros(2, 3),
        )


@pytest.mark.parametrize(
    ("name", "value", "error", "match"),
    [
        ("side_head_pose_2d", torch.zeros(2, 3), ValueError, r"\[B,2\]"),
        ("front_head_vector", torch.zeros(1, 3), ValueError, "B=2"),
        ("side_eye_angles", torch.zeros(2, 2, dtype=torch.float64), TypeError, "float32"),
        ("side_iris_pose_2d", [[0.0, 0.0]], TypeError, "torch.Tensor"),
    ],
)
def test_optional_auxiliary_contract_is_validated(
    name: str, value: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        validate_and_sanitize_side_inputs(_side_image(), **{name: value})  # type: ignore[arg-type]


def test_optional_auxiliary_device_must_match_side_image() -> None:
    meta_eye = torch.empty(2, 2, dtype=torch.float32, device="meta")

    with pytest.raises(ValueError, match="side_image.device"):
        validate_and_sanitize_side_inputs(_side_image(), side_eye_angles=meta_eye)


@pytest.mark.parametrize(
    ("image", "error", "match"),
    [
        (torch.zeros(2, 3, 128, 255), ValueError, r"\[B,3,128,256\]"),
        (torch.zeros(2, 3, 128, 256, dtype=torch.float64), TypeError, "float32"),
        (torch.full((2, 3, 128, 256), 1.1), ValueError, r"\[0,1\]"),
    ],
)
def test_required_side_image_contract_is_validated(
    image: torch.Tensor, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        validate_and_sanitize_side_inputs(image)


def test_auxiliary_projector_has_fixed_head_eye_iris_slots() -> None:
    projector = SideAuxiliaryProjector(projection_dim=8)
    image = _side_image()

    empty = projector(image)
    eye_only = projector(image, side_eye_angles=torch.ones(2, 2))
    front_head = projector(image, front_head_vector=torch.ones(2, 3))

    assert projector.output_dim == 24
    assert empty.shape == (2, 24)
    assert empty.dtype == torch.float32
    assert torch.count_nonzero(empty) == 0
    assert torch.count_nonzero(eye_only[:, :8]) == 0
    assert torch.count_nonzero(eye_only[:, 16:]) == 0
    assert torch.count_nonzero(front_head[:, 8:]) == 0


def test_common_head_returns_only_standard_residual_side_outputs() -> None:
    torch.manual_seed(7)
    head = SideEncoderHead(16, 12, embedding_dim=256)

    outputs = head(torch.randn(3, 16), torch.randn(3, 12))

    assert head.delta_y_head.out_features == 1
    assert set(outputs) == {"delta_y_side", "side_embedding", "quality"}
    assert outputs["delta_y_side"].shape == (3, 1)
    assert outputs["side_embedding"].shape == (3, 256)
    assert outputs["quality"].shape == (3, 1)
    assert all(value.dtype == torch.float32 for value in outputs.values())
    assert torch.all((outputs["quality"] >= 0.0) & (outputs["quality"] <= 1.0))
    assert "gaze_xy" not in outputs


def test_public_component_signature_excludes_pipeline_only_inputs() -> None:
    parameters = signature(SideAuxiliaryProjector.forward).parameters

    assert "side_image" in parameters
    assert "side_gaze_valid" not in parameters
    assert "target_gaze_xy" not in parameters
    assert "front_prediction" not in parameters


def test_model_overrides_resolve_to_importable_factories() -> None:
    blazegaze = _load_side_model_config(BLAZEGAZE_MODEL)
    mobilenet = _load_side_model_config(MOBILENET_MODEL)

    for config in (blazegaze, mobilenet):
        side = config["model"]["side"]
        assert _import_entrypoint_for_test(side["entrypoint"])
        assert side["output_contract"]["gaze_key"] is None
        assert side["output_contract"]["gaze_shape"] is None
        assert side["output_contract"]["delta_y_key"] == "delta_y_side"
        assert side["output_contract"]["delta_y_shape"] == ["B", 1]
        assert side["output_contract"]["embedding_shape"] == ["B", 256]
        assert side["output_contract"]["quality_shape"] == ["B", 1]

    assert (
        blazegaze["model"]["side"]["output_contract"]
        == mobilenet["model"]["side"]["output_contract"]
    )


@pytest.mark.parametrize("model_config", (BLAZEGAZE_MODEL, MOBILENET_MODEL))
def test_product_runtime_loads_residual_side_factory(model_config: Path) -> None:
    config = _load_side_model_config(model_config)
    side_config = config["model"]["side"]
    runtime = build_model_runtime(
        "side",
        side_config,
        forward_keys=resolve_model_forward_keys(config, "side"),
    ).eval()
    batch = {
        "side_image": torch.rand(2, 3, 128, 256),
        "side_head_pose_2d": torch.rand(2, 2),
        "side_eye_angles": torch.rand(2, 2),
        "side_iris_pose_2d": torch.rand(2, 2),
        "side_gaze_valid": torch.ones(2, dtype=torch.bool),
    }

    with torch.inference_mode():
        outputs = runtime(batch)

    assert set(outputs) == {"delta_y_side", "side_embedding", "quality"}
    assert outputs["delta_y_side"].shape == (2, 1)
    assert outputs["side_embedding"].shape == (2, 256)
    assert outputs["quality"].shape == (2, 1)
    assert all(torch.isfinite(value).all() for value in outputs.values())
    assert torch.all((outputs["quality"] >= 0.0) & (outputs["quality"] <= 1.0))


def test_model_yaml_files_are_small_model_side_overrides() -> None:
    expected_keys = {
        BLAZEGAZE_MODEL: {"entrypoint", "init_args", "output_contract"},
        MOBILENET_MODEL: {"entrypoint", "init_args", "initialization", "output_contract"},
    }

    for path, side_keys in expected_keys.items():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(raw) == {"model"}
        assert set(raw["model"]) == {"side"}
        assert set(raw["model"]["side"]) == side_keys


def test_mobilenet_override_replaces_webeyetrack_initialization_metadata() -> None:
    config = _load_side_model_config(MOBILENET_MODEL)
    initialization = config["model"]["side"]["initialization"]

    assert initialization == {
        "mode": "mobilenet_v4",
        "source_framework": "pytorch",
        "path": None,
        "load_scope": "timm_feature_extractor",
        "reinitialize": ["side_embedding", "delta_y_side", "quality"],
        "importer_entrypoint": None,
    }


def test_mobilenet_config_uses_exact_offline_safe_timm_model() -> None:
    config = _load_side_model_config(MOBILENET_MODEL)
    side = config["model"]["side"]

    assert side["entrypoint"] == "gaze_pipeline.models.side.mobilenet_v4:create_model"
    assert side["init_args"] == {
        "model_id": "mobilenetv4_conv_small.e2400_r224_in1k",
        "pretrained": False,
        "embedding_dim": 256,
        "auxiliary_dim": 32,
        "normalize_imagenet": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "dropout": 0.0,
    }


def test_blazegaze_config_uses_random_init_transfer_factory() -> None:
    config = _load_side_model_config(BLAZEGAZE_MODEL)
    side = config["model"]["side"]

    assert side["entrypoint"] == "gaze_pipeline.models.side.blazegaze_transfer:create_model"
    assert side["init_args"] == {
        "encoder_weights_path": None,
        "image_feature_dim": 256,
        "embedding_dim": 256,
        "auxiliary_dim": 32,
        "dropout": 0.0,
    }
