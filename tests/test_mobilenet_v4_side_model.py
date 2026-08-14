from __future__ import annotations

from io import BytesIO

import pytest
import torch
from torch import nn

from gaze_pipeline.models.side.mobilenet_v4 import (
    MOBILENET_V4_CONV_S_MODEL_ID,
    MobileNetV4SideEncoder,
    create_model,
)


@pytest.fixture(scope="module")
def model() -> MobileNetV4SideEncoder:
    torch.manual_seed(17)
    created = create_model(pretrained=False)
    assert isinstance(created, MobileNetV4SideEncoder)
    return created.eval()


def _image(batch_size: int) -> torch.Tensor:
    return torch.rand(batch_size, 3, 128, 256, dtype=torch.float32)


def _optional_inputs(batch_size: int, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    widths = {
        "side_head_pose_2d": 2,
        "front_head_vector": 3,
        "side_eye_angles": 2,
        "side_iris_pose_2d": 2,
    }
    return {name: torch.randn(batch_size, widths[name]) for name in names}


def _assert_output_contract(outputs: dict[str, torch.Tensor], batch_size: int) -> None:
    assert set(outputs) == {"delta_y_side", "side_embedding", "quality"}
    assert outputs["delta_y_side"].shape == (batch_size, 1)
    assert outputs["side_embedding"].shape == (batch_size, 256)
    assert outputs["quality"].shape == (batch_size, 1)
    assert all(value.dtype == torch.float32 for value in outputs.values())
    assert all(torch.isfinite(value).all() for value in outputs.values())
    assert torch.all((outputs["quality"] >= 0.0) & (outputs["quality"] <= 1.0))
    assert "gaze_xy" not in outputs


def test_factory_creates_exact_timm_mobilenet_v4_feature_extractor(
    model: MobileNetV4SideEncoder,
) -> None:
    assert isinstance(model, nn.Module)
    assert model.model_id == MOBILENET_V4_CONV_S_MODEL_ID
    assert model.pretrained is False
    assert model.image_feature_dim == 960
    assert isinstance(model.global_pool, nn.AdaptiveAvgPool2d)
    assert model.auxiliary_projector.output_dim == 96
    assert "image_mean" in dict(model.named_buffers())
    assert "image_std" in dict(model.named_buffers())


@pytest.mark.parametrize(
    ("batch_size", "names"),
    [
        (1, ()),
        (2, ("side_head_pose_2d",)),
        (1, ("front_head_vector",)),
        (2, ("side_eye_angles",)),
        (1, ("side_iris_pose_2d",)),
        (2, ("side_eye_angles", "side_iris_pose_2d")),
        (1, ("front_head_vector", "side_eye_angles", "side_iris_pose_2d")),
    ],
)
def test_non_square_forward_supports_optional_input_combinations(
    model: MobileNetV4SideEncoder,
    batch_size: int,
    names: tuple[str, ...],
) -> None:
    side_image = _image(batch_size)

    with torch.inference_mode():
        outputs = model(side_image, **_optional_inputs(batch_size, names))

    assert side_image.shape[-2:] == (128, 256)
    _assert_output_contract(outputs, batch_size)


def test_two_head_sources_are_rejected_before_image_forward(
    model: MobileNetV4SideEncoder,
) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        model(
            _image(2),
            side_head_pose_2d=torch.zeros(2, 2),
            front_head_vector=torch.zeros(2, 3),
        )


def test_invalid_side_image_shape_is_rejected(model: MobileNetV4SideEncoder) -> None:
    with pytest.raises(ValueError, match=r"\[B,3,128,256\]"):
        model(torch.zeros(2, 3, 224, 224))


def test_state_dict_strict_cpu_round_trip(model: MobileNetV4SideEncoder) -> None:
    payload = BytesIO()
    torch.save(model.state_dict(), payload)
    payload.seek(0)
    state_dict = torch.load(payload, map_location="cpu", weights_only=True)

    restored = create_model(pretrained=False)
    restored.load_state_dict(state_dict, strict=True)
    restored.eval()

    side_image = _image(1)
    inputs = _optional_inputs(1, ("side_head_pose_2d", "side_eye_angles"))
    with torch.inference_mode():
        expected = model(side_image, **inputs)
        actual = restored(side_image, **inputs)

    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
