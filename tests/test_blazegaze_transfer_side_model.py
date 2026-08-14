from __future__ import annotations

from io import BytesIO

import pytest
import torch
from torch import nn

from gaze_pipeline.models.side.blazegaze_transfer import (
    SIDE_ENCODER_FEATURE_SHAPE,
    SOURCE_COMMIT,
    SOURCE_FUNCTION,
    SOURCE_LICENSE,
    BlazeGazeTransferSideEncoder,
    DoubleBlazeBlock,
    TensorFlowSameConv2d,
    create_model,
    tensorflow_same_padding_2d,
)


@pytest.fixture(scope="module")
def model() -> BlazeGazeTransferSideEncoder:
    torch.manual_seed(23)
    created = create_model(encoder_weights_path=None)
    assert isinstance(created, BlazeGazeTransferSideEncoder)
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


def test_source_lineage_is_pinned_to_official_encoder() -> None:
    assert SOURCE_COMMIT == "14719ad861467c98890058f7c41a94638ae1db2b"
    assert SOURCE_FUNCTION == "python/webeyetrack/blazegaze.py:get_cnn_encoder"
    assert SOURCE_LICENSE == "MIT"


def test_tensorflow_same_stride_two_padding_is_dynamic_and_asymmetric() -> None:
    assert tensorflow_same_padding_2d((128, 256), 5, stride=2) == (1, 2, 1, 2)
    assert tensorflow_same_padding_2d((127, 255), 5, stride=2) == (2, 2, 2, 2)

    convolution = TensorFlowSameConv2d(3, 24, 5, stride=2)
    assert convolution(torch.zeros(1, 3, 128, 256)).shape == (1, 24, 64, 128)


def test_official_encoder_stage_shapes_for_side_width(
    model: BlazeGazeTransferSideEncoder,
) -> None:
    expected = {
        "first_conv": (1, 24, 64, 128),
        "single_1": (1, 24, 64, 128),
        "single_2": (1, 24, 64, 128),
        "single_3": (1, 48, 32, 64),
        "single_4": (1, 48, 32, 64),
        "single_5": (1, 48, 32, 64),
        "double_1": (1, 96, 16, 32),
        "double_2": (1, 96, 16, 32),
        "double_3": (1, 96, 16, 32),
        "double_4": (1, 96, 8, 16),
        "double_5": (1, 96, 8, 16),
        "double_6": (1, 96, 8, 16),
        "squeeze_1": (1, 64, 4, 8),
        "squeeze_2": (1, 32, 2, 4),
    }
    observed: dict[str, tuple[int, ...]] = {}
    handles = []
    for name in expected:
        module = getattr(model.encoder, name)
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, stage=name: observed.__setitem__(
                    stage, tuple(output.shape)
                )
            )
        )
    try:
        with torch.inference_mode():
            final = model.encoder(_image(1))
    finally:
        for handle in handles:
            handle.remove()

    assert observed == expected
    assert tuple(final.shape[1:]) == SIDE_ENCODER_FEATURE_SHAPE
    assert model.flatten_projection[1].in_features == 32 * 2 * 4


def test_official_conv_bias_residual_and_squeeze_order(
    model: BlazeGazeTransferSideEncoder,
) -> None:
    assert model.encoder.first_conv[0].conv.bias is not None

    stride_two_single = model.encoder.single_3
    assert stride_two_single.depthwise.conv.bias is not None
    assert stride_two_single.pointwise.conv.bias is not None
    assert isinstance(stride_two_single.residual_pool, nn.MaxPool2d)
    assert isinstance(stride_two_single.residual_projection, TensorFlowSameConv2d)
    assert stride_two_single.residual_projection.conv.bias is not None

    stride_two_double = model.encoder.double_1
    assert isinstance(stride_two_double, DoubleBlazeBlock)
    assert stride_two_double.depthwise_1.conv.bias is not None
    assert stride_two_double.pointwise_1.conv.bias is not None
    assert stride_two_double.depthwise_2.conv.bias is not None
    assert stride_two_double.pointwise_2.conv.bias is not None
    assert stride_two_double.residual_projection.conv.bias is not None

    for squeeze in (model.encoder.squeeze_1, model.encoder.squeeze_2):
        assert isinstance(squeeze[0], TensorFlowSameConv2d)
        assert squeeze[0].conv.bias is not None
        assert isinstance(squeeze[1], nn.ReLU)
        assert isinstance(squeeze[2], nn.BatchNorm2d)
        assert squeeze[2].eps == pytest.approx(1e-3)
        assert squeeze[2].momentum == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("batch_size", "names"),
    [
        (1, ()),
        (2, ("side_head_pose_2d",)),
        (1, ("front_head_vector",)),
        (2, ("side_eye_angles",)),
        (1, ("side_iris_pose_2d",)),
        (2, ("side_eye_angles", "side_iris_pose_2d")),
        (1, ("side_head_pose_2d", "side_eye_angles", "side_iris_pose_2d")),
    ],
)
def test_random_init_forward_supports_public_optional_contract(
    model: BlazeGazeTransferSideEncoder,
    batch_size: int,
    names: tuple[str, ...],
) -> None:
    with torch.inference_mode():
        outputs = model(_image(batch_size), **_optional_inputs(batch_size, names))

    _assert_output_contract(outputs, batch_size)


def test_two_head_sources_are_rejected(model: BlazeGazeTransferSideEncoder) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        model(
            _image(2),
            side_head_pose_2d=torch.zeros(2, 2),
            front_head_vector=torch.zeros(2, 3),
        )


def test_eval_forward_is_deterministic(model: BlazeGazeTransferSideEncoder) -> None:
    side_image = _image(2)
    inputs = _optional_inputs(2, ("front_head_vector", "side_eye_angles"))

    with torch.inference_mode():
        first = model(side_image, **inputs)
        second = model(side_image, **inputs)

    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0.0, atol=0.0)


def test_state_dict_strict_cpu_round_trip(model: BlazeGazeTransferSideEncoder) -> None:
    payload = BytesIO()
    torch.save(model.state_dict(), payload)
    payload.seek(0)
    state_dict = torch.load(payload, map_location="cpu", weights_only=True)

    restored = create_model(encoder_weights_path=None)
    restored.load_state_dict(state_dict, strict=True)
    restored.eval()

    side_image = _image(1)
    inputs = _optional_inputs(1, ("side_head_pose_2d", "side_iris_pose_2d"))
    with torch.inference_mode():
        expected = model(side_image, **inputs)
        actual = restored(side_image, **inputs)

    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
