"""PyTorch Side model based on the pinned official BlazeGaze CNN encoder.

Source lineage:
    repository: https://github.com/RedForestAI/WebEyeTrack
    commit: 14719ad861467c98890058f7c41a94638ae1db2b
    source: python/webeyetrack/blazegaze.py:get_cnn_encoder
    license: MIT

Only the convolution encoder topology is reproduced here.  The Side-specific
flatten projection and prediction heads are newly initialized PyTorch layers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from gaze_pipeline.models.side.components import SideAuxiliaryProjector, SideEncoderHead

SOURCE_REPOSITORY = "https://github.com/RedForestAI/WebEyeTrack"
SOURCE_COMMIT = "14719ad861467c98890058f7c41a94638ae1db2b"
SOURCE_FUNCTION = "python/webeyetrack/blazegaze.py:get_cnn_encoder"
SOURCE_LICENSE = "MIT"
ENCODER_STATE_DICT_FORMAT = "gaze_pipeline.blazegaze_encoder.v1"
SIDE_ENCODER_FEATURE_SHAPE = (32, 2, 4)
SIDE_ENCODER_FLAT_DIM = math.prod(SIDE_ENCODER_FEATURE_SHAPE)


def tensorflow_same_padding_2d(
    input_hw: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
) -> tuple[int, int, int, int]:
    """Return TensorFlow SAME padding as ``(left, right, top, bottom)``.

    TensorFlow defines the output size as ``ceil(input / stride)`` and assigns
    an odd total padding amount asymmetrically, with the extra element on the
    right or bottom.  This differs from static ``kernel_size // 2`` padding for
    stride-two convolutions on even-sized inputs.
    """

    height, width = input_hw
    kernel_height, kernel_width = _pair(kernel_size)
    stride_height, stride_width = _pair(stride)
    dilation_height, dilation_width = _pair(dilation)
    top, bottom = _same_padding_1d(height, kernel_height, stride_height, dilation_height)
    left, right = _same_padding_1d(width, kernel_width, stride_width, dilation_width)
    return left, right, top, bottom


class TensorFlowSameConv2d(nn.Module):
    """Conv2d preceded by dynamic TensorFlow-compatible SAME padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply dynamic SAME padding followed by convolution."""

        padding = tensorflow_same_padding_2d(
            (inputs.shape[-2], inputs.shape[-1]),
            self.kernel_size,
            self.stride,
            self.dilation,
        )
        if any(padding):
            inputs = functional.pad(inputs, padding)
        return self.conv(inputs)


class BlazeBlock(nn.Module):
    """Official single Blaze block with depthwise/pointwise residual path."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError("BlazeBlock stride must be 1 or 2")
        if stride == 1 and in_channels != out_channels:
            raise ValueError("stride-one BlazeBlock requires matching channel counts")
        self.depthwise = TensorFlowSameConv2d(
            in_channels,
            in_channels,
            5,
            stride=stride,
            groups=in_channels,
            bias=True,
        )
        self.pointwise = TensorFlowSameConv2d(in_channels, out_channels, 1, bias=True)
        self.residual_pool = nn.MaxPool2d(2, stride=2) if stride == 2 else nn.Identity()
        self.residual_projection = (
            TensorFlowSameConv2d(in_channels, out_channels, 1, bias=True)
            if stride == 2
            else nn.Identity()
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        """Return the ReLU-activated sum of main and residual paths."""

        main = self.pointwise(self.depthwise(inputs))
        residual = self.residual_projection(self.residual_pool(inputs))
        return self.activation(main + residual)


class DoubleBlazeBlock(nn.Module):
    """Official double Blaze block with two depthwise/pointwise pairs."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError("DoubleBlazeBlock stride must be 1 or 2")
        if stride == 1 and in_channels != out_channels:
            raise ValueError("stride-one DoubleBlazeBlock requires matching channel counts")
        self.depthwise_1 = TensorFlowSameConv2d(
            in_channels,
            in_channels,
            5,
            stride=stride,
            groups=in_channels,
            bias=True,
        )
        self.pointwise_1 = TensorFlowSameConv2d(in_channels, hidden_channels, 1, bias=True)
        self.intermediate_activation = nn.ReLU()
        self.depthwise_2 = TensorFlowSameConv2d(
            hidden_channels,
            hidden_channels,
            5,
            groups=hidden_channels,
            bias=True,
        )
        self.pointwise_2 = TensorFlowSameConv2d(hidden_channels, out_channels, 1, bias=True)
        self.residual_pool = nn.MaxPool2d(2, stride=2) if stride == 2 else nn.Identity()
        self.residual_projection = (
            TensorFlowSameConv2d(in_channels, out_channels, 1, bias=True)
            if stride == 2
            else nn.Identity()
        )
        self.output_activation = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        """Return the ReLU-activated sum of double main and residual paths."""

        main = self.depthwise_1(inputs)
        main = self.intermediate_activation(self.pointwise_1(main))
        main = self.pointwise_2(self.depthwise_2(main))
        residual = self.residual_projection(self.residual_pool(inputs))
        return self.output_activation(main + residual)


class BlazeGazeConvEncoder(nn.Module):
    """Pinned official BlazeGaze convolution sequence in NCHW PyTorch form."""

    def __init__(self) -> None:
        super().__init__()
        self.first_conv = nn.Sequential(
            TensorFlowSameConv2d(3, 24, 5, stride=2, bias=True),
            nn.ReLU(),
        )
        self.single_1 = BlazeBlock(24, 24)
        self.single_2 = BlazeBlock(24, 24)
        self.single_3 = BlazeBlock(24, 48, stride=2)
        self.single_4 = BlazeBlock(48, 48)
        self.single_5 = BlazeBlock(48, 48)
        self.double_1 = DoubleBlazeBlock(48, 24, 96, stride=2)
        self.double_2 = DoubleBlazeBlock(96, 24, 96)
        self.double_3 = DoubleBlazeBlock(96, 24, 96)
        self.double_4 = DoubleBlazeBlock(96, 24, 96, stride=2)
        self.double_5 = DoubleBlazeBlock(96, 24, 96)
        self.double_6 = DoubleBlazeBlock(96, 24, 96)
        self.squeeze_1 = nn.Sequential(
            TensorFlowSameConv2d(96, 64, 3, stride=2, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
        )
        self.squeeze_2 = nn.Sequential(
            TensorFlowSameConv2d(64, 32, 3, stride=2, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return encoder feature map ``[B,32,2,4]`` for Side input width 256."""

        outputs = self.first_conv(inputs)
        outputs = self.single_1(outputs)
        outputs = self.single_2(outputs)
        outputs = self.single_3(outputs)
        outputs = self.single_4(outputs)
        outputs = self.single_5(outputs)
        outputs = self.double_1(outputs)
        outputs = self.double_2(outputs)
        outputs = self.double_3(outputs)
        outputs = self.double_4(outputs)
        outputs = self.double_5(outputs)
        outputs = self.double_6(outputs)
        outputs = self.squeeze_1(outputs)
        return self.squeeze_2(outputs)


class BlazeGazeTransferSideEncoder(nn.Module):
    """BlazeGaze-convolution Side model with newly initialized Side heads.

    Input and output tensors follow the same public contract as the MobileNetV4
    Side model.  ``encoder_weights_path=None`` is the random-initialization
    default and performs no file access.
    """

    def __init__(
        self,
        *,
        encoder_weights_path: str | Path | None = None,
        image_feature_dim: int = 256,
        embedding_dim: int = 256,
        auxiliary_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_feature_dim <= 0:
            raise ValueError("image_feature_dim must be positive")
        self.encoder = BlazeGazeConvEncoder()
        self.flatten_projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(SIDE_ENCODER_FLAT_DIM, image_feature_dim),
            nn.ReLU(),
        )
        self.auxiliary_projector = SideAuxiliaryProjector(projection_dim=auxiliary_dim)
        self.output_head = SideEncoderHead(
            image_feature_dim,
            self.auxiliary_projector.output_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        if encoder_weights_path is not None:
            load_encoder_state_dict(self.encoder, encoder_weights_path)

    def forward(
        self,
        side_image: Tensor,
        *,
        side_head_pose_2d: Tensor | None = None,
        front_head_vector: Tensor | None = None,
        side_eye_angles: Tensor | None = None,
        side_iris_pose_2d: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return independent ``gaze_xy``, ``side_embedding``, and ``quality``."""

        auxiliary_features = self.auxiliary_projector(
            side_image,
            side_head_pose_2d=side_head_pose_2d,
            front_head_vector=front_head_vector,
            side_eye_angles=side_eye_angles,
            side_iris_pose_2d=side_iris_pose_2d,
        )
        image_features = self.flatten_projection(self.encoder(side_image))
        return self.output_head(image_features, auxiliary_features)


def create_model(
    *,
    encoder_weights_path: str | Path | None = None,
    image_feature_dim: int = 256,
    embedding_dim: int = 256,
    auxiliary_dim: int = 32,
    dropout: float = 0.0,
) -> nn.Module:
    """Create a random-init or encoder-weight-initialized BlazeGaze Side model."""

    return BlazeGazeTransferSideEncoder(
        encoder_weights_path=encoder_weights_path,
        image_feature_dim=image_feature_dim,
        embedding_dim=embedding_dim,
        auxiliary_dim=auxiliary_dim,
        dropout=dropout,
    )


def load_encoder_state_dict(encoder: BlazeGazeConvEncoder, path: str | Path) -> None:
    """Strictly load the future converter's encoder-only, CPU-safe payload."""

    payload: Any = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("BlazeGaze encoder weight payload must be a mapping")
    if payload.get("format") != ENCODER_STATE_DICT_FORMAT:
        raise ValueError(f"BlazeGaze encoder weight format must be {ENCODER_STATE_DICT_FORMAT!r}")
    if payload.get("source_commit") != SOURCE_COMMIT:
        raise ValueError(f"BlazeGaze encoder source_commit must be {SOURCE_COMMIT}")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in state_dict.items()
    ):
        raise ValueError("BlazeGaze encoder state_dict must map string keys to tensors")
    encoder.load_state_dict(dict(state_dict), strict=True)


def _same_padding_1d(
    input_size: int, kernel_size: int, stride: int, dilation: int
) -> tuple[int, int]:
    output_size = math.ceil(input_size / stride)
    effective_kernel = dilation * (kernel_size - 1) + 1
    total = max((output_size - 1) * stride + effective_kernel - input_size, 0)
    before = total // 2
    return before, total - before


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError("2D convolution values must be int or length-two tuple")
    return int(value[0]), int(value[1])


__all__ = [
    "ENCODER_STATE_DICT_FORMAT",
    "SIDE_ENCODER_FEATURE_SHAPE",
    "SOURCE_COMMIT",
    "SOURCE_FUNCTION",
    "SOURCE_LICENSE",
    "SOURCE_REPOSITORY",
    "BlazeBlock",
    "BlazeGazeConvEncoder",
    "BlazeGazeTransferSideEncoder",
    "DoubleBlazeBlock",
    "TensorFlowSameConv2d",
    "create_model",
    "load_encoder_state_dict",
    "tensorflow_same_padding_2d",
]
