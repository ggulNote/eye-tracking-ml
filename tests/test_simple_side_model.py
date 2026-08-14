from __future__ import annotations

from typing import Any

import pytest
import torch

from gaze_pipeline.model_runtime import build_model_runtime
from gaze_pipeline.models.simple_side import (
    DEFAULT_FEATURE_KEYS,
    SimpleSideModelError,
    create_model,
)


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "side_image": torch.rand(batch_size, 3, 32, 64),
        "side_head_pose_2d": torch.rand(batch_size, 2),
        "side_eye_angles": torch.rand(batch_size, 2),
        "side_iris_pose_2d": torch.rand(batch_size, 2),
    }


def _runtime_config() -> dict[str, Any]:
    return {
        "source_dir": None,
        "entrypoint": "gaze_pipeline.models.simple_side:create_model",
        "adapter_entrypoint": None,
        "init_args": {
            "image_key": "side_image",
            "feature_keys": list(DEFAULT_FEATURE_KEYS),
            "embedding_dim": 32,
        },
        "pretrained": {"path": None, "sha256": None, "strict": True},
        "input_contract": {
            "forward_keys": ["side_image", *DEFAULT_FEATURE_KEYS],
            "image_key": "side_image",
            "shape": ["B", 3, 32, 64],
            "dtype": "float32",
            "value_range": [0.0, 1.0],
            "auxiliary_keys": {
                "head_pose": {
                    "key": "side_head_pose_2d",
                    "shape": ["B", 2],
                    "dtype": "float32",
                },
                "eye_angles": {
                    "key": "side_eye_angles",
                    "shape": ["B", 2],
                    "dtype": "float32",
                },
                "iris_pose": {
                    "key": "side_iris_pose_2d",
                    "shape": ["B", 2],
                    "dtype": "float32",
                },
            },
        },
        "output_contract": {
            "gaze_key": None,
            "gaze_shape": None,
            "delta_y_key": "delta_y_side",
            "delta_y_shape": ["B", 1],
            "embedding_key": "side_embedding",
            "embedding_shape": ["B", 32],
        },
    }


def test_factory_returns_residual_and_fused_embedding() -> None:
    model = create_model()

    output = model(**_batch(3))

    assert model.forward_keys == ("side_image", *DEFAULT_FEATURE_KEYS)
    assert output["delta_y_side"].shape == (3, 1)
    assert output["side_embedding"].shape == (3, 32)
    assert torch.isfinite(output["delta_y_side"]).all()
    assert torch.isfinite(output["side_embedding"]).all()


def test_selected_features_reach_embedding_and_residual_gradients() -> None:
    model = create_model(feature_keys=["side_eye_angles"], embedding_dim=8)
    with torch.no_grad():
        model.embedding_projection.weight.fill_(0.05)
        model.embedding_projection.bias.zero_()
        model.residual_head.weight.fill_(0.1)
        model.residual_head.bias.zero_()
    feature = torch.tensor([[0.2, -0.1], [-0.3, 0.4]], requires_grad=True)

    output = model(side_image=torch.rand(2, 3, 16, 16), side_eye_angles=feature)
    output["delta_y_side"].sum().backward()

    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()
    assert feature.grad.abs().sum().item() > 0


def test_empty_feature_selection_remains_a_valid_image_only_model() -> None:
    model = create_model(feature_keys=[], embedding_dim=7)

    output = model(side_image=torch.rand(2, 3, 12, 20))

    assert model.forward_keys == ("side_image",)
    assert output["delta_y_side"].shape == (2, 1)
    assert output["side_embedding"].shape == (2, 7)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"image_key": ""}, "image_key"),
        ({"feature_keys": "side_eye_angles"}, "sequence"),
        ({"feature_keys": ["side_eye_angles", "side_eye_angles"]}, "duplicates"),
        ({"feature_keys": ["unknown_feature"]}, "unsupported"),
        ({"image_key": "side_eye_angles", "feature_keys": ["side_eye_angles"]}, "also"),
        ({"embedding_dim": 0}, "positive integer"),
        ({"embedding_dim": True}, "positive integer"),
    ],
)
def test_factory_rejects_ambiguous_contracts(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(SimpleSideModelError, match=match):
        create_model(**kwargs)


def test_forward_rejects_missing_and_unexpected_keys() -> None:
    model = create_model(feature_keys=["side_eye_angles"])
    image = torch.rand(2, 3, 16, 16)

    with pytest.raises(SimpleSideModelError, match="missing.*side_eye_angles"):
        model(side_image=image)
    with pytest.raises(SimpleSideModelError, match="unexpected.*side_gaze_valid"):
        model(
            side_image=image,
            side_eye_angles=torch.rand(2, 2),
            side_gaze_valid=torch.ones(2, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    ("feature", "match"),
    [
        (torch.rand(2), r"shape \[B,2\]"),
        (torch.rand(2, 3), r"shape \[B,2\]"),
        (torch.rand(3, 2), r"B=2"),
        (torch.rand(2, 2, dtype=torch.float64), "torch.float32"),
        (torch.tensor([[0.0, torch.nan], [0.0, 0.0]]), "NaN or Inf"),
        (torch.tensor([[0.0, torch.inf], [0.0, 0.0]]), "NaN or Inf"),
    ],
)
def test_forward_strictly_validates_feature_tensor(feature: torch.Tensor, match: str) -> None:
    model = create_model(feature_keys=["side_eye_angles"])

    with pytest.raises(SimpleSideModelError, match=match):
        model(side_image=torch.rand(2, 3, 16, 16), side_eye_angles=feature)


def test_forward_requires_feature_and_image_on_same_device() -> None:
    model = create_model(feature_keys=["side_eye_angles"])
    feature = torch.empty(2, 2, dtype=torch.float32, device="meta")

    with pytest.raises(SimpleSideModelError, match="must be on cpu"):
        model(side_image=torch.rand(2, 3, 16, 16), side_eye_angles=feature)


@pytest.mark.parametrize(
    ("image", "match"),
    [
        (torch.rand(2, 16, 16), r"shape \[B,3,H,W\]"),
        (torch.rand(2, 1, 16, 16), r"shape \[B,3,H,W\]"),
        (torch.rand(2, 3, 16, 16, dtype=torch.float64), "torch.float32"),
        (torch.full((2, 3, 16, 16), torch.nan), "NaN or Inf"),
        (torch.empty(0, 3, 16, 16), "non-empty"),
    ],
)
def test_forward_strictly_validates_image_tensor(image: torch.Tensor, match: str) -> None:
    model = create_model(feature_keys=[])

    with pytest.raises(SimpleSideModelError, match=match):
        model(side_image=image)


def test_factory_integrates_with_default_runtime_adapter() -> None:
    runtime = build_model_runtime(
        "side",
        _runtime_config(),
        forward_keys=("side_image", *DEFAULT_FEATURE_KEYS),
    )

    output = runtime(_batch(2))

    assert output["delta_y_side"].shape == (2, 1)
    assert output["side_embedding"].shape == (2, 32)


def test_front_head_vector_is_supported_for_front_3d_feature_mode() -> None:
    model = create_model(
        feature_keys=["front_head_vector", "side_eye_angles"],
        embedding_dim=5,
    )

    output = model(
        side_image=torch.rand(4, 3, 16, 16),
        front_head_vector=torch.rand(4, 3),
        side_eye_angles=torch.rand(4, 2),
    )

    assert output["delta_y_side"].shape == (4, 1)
    assert output["side_embedding"].shape == (4, 5)
