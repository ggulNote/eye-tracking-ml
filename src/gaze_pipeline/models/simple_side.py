"""Small Side residual model that actually consumes selected geometry features.

This model is intentionally modest: it is a trainable smoke/integration model,
not a claim about the final Side architecture.  A compact convolutional encoder
turns ``side_image`` into an image representation, selected canonical geometry
features are concatenated to it, and a fused embedding predicts one y-axis
residual.

Use the factory as the generic runtime entrypoint::

    gaze_pipeline.models.simple_side:create_model

The configured model ``forward_keys`` must be ``image_key`` followed by the
same ``feature_keys`` supplied to the factory.  Inputs are deliberately strict:
float32 tensors only, exact feature widths, a common non-empty batch and device,
and no NaN/Inf values or unexpected keyword arguments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn

DEFAULT_FEATURE_KEYS = (
    "side_head_pose_2d",
    "side_eye_angles",
    "side_iris_pose_2d",
)

# These are the canonical keys produced by the current Dataset contracts.  An
# explicit registry keeps Linear layer dimensions checkpoint-stable and rejects
# silent feature-width changes.
FEATURE_WIDTHS: Mapping[str, int] = MappingProxyType(
    {
        "side_head_pose_2d": 2,
        "front_head_vector": 3,
        "side_eye_angles": 2,
        "side_iris_pose_2d": 2,
    }
)


class SimpleSideModelError(ValueError):
    """Raised when factory arguments or forward inputs violate the model contract."""


def _non_empty_key(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SimpleSideModelError(f"{name} must be a non-empty string")
    return value.strip()


def _resolve_feature_keys(feature_keys: Sequence[str]) -> tuple[str, ...]:
    if isinstance(feature_keys, str | bytes) or not isinstance(feature_keys, Sequence):
        raise SimpleSideModelError("feature_keys must be a sequence of canonical batch keys")
    resolved = tuple(
        _non_empty_key(value, name=f"feature_keys[{index}]")
        for index, value in enumerate(feature_keys)
    )
    if len(resolved) != len(set(resolved)):
        raise SimpleSideModelError("feature_keys must not contain duplicates")
    unknown = [key for key in resolved if key not in FEATURE_WIDTHS]
    if unknown:
        raise SimpleSideModelError(
            f"unsupported feature_keys {unknown}; supported keys are {sorted(FEATURE_WIDTHS)}"
        )
    return resolved


class _SideImageEncoder(nn.Module):
    """Spatial-size-independent RGB encoder for a smoke model."""

    output_width = 32

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.network(image)


class SimpleSideResidualModel(nn.Module):
    """Encode a Side image, fuse selected features, and predict ``delta_y_side``."""

    def __init__(
        self,
        *,
        image_key: str = "side_image",
        feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.image_key = _non_empty_key(image_key, name="image_key")
        self.feature_keys = _resolve_feature_keys(feature_keys)
        if self.image_key in self.feature_keys:
            raise SimpleSideModelError("image_key must not also appear in feature_keys")
        if (
            not isinstance(embedding_dim, int)
            or isinstance(embedding_dim, bool)
            or embedding_dim <= 0
        ):
            raise SimpleSideModelError("embedding_dim must be a positive integer")
        self.embedding_dim = embedding_dim
        self.feature_widths = tuple(FEATURE_WIDTHS[key] for key in self.feature_keys)

        self.image_encoder = _SideImageEncoder()
        fused_width = self.image_encoder.output_width + sum(self.feature_widths)
        self.embedding_projection = nn.Linear(fused_width, embedding_dim)
        self.embedding_activation = nn.Tanh()
        self.residual_head = nn.Linear(embedding_dim, 1)

    @property
    def forward_keys(self) -> tuple[str, ...]:
        """Canonical kwargs expected from ``DefaultModelAdapter``."""

        return (self.image_key, *self.feature_keys)

    def _validate_image(self, value: Any) -> Tensor:
        if not isinstance(value, Tensor):
            raise SimpleSideModelError(
                f"{self.image_key} must be a torch.Tensor, got {type(value).__name__}"
            )
        if value.ndim != 4 or value.shape[1] != 3:
            raise SimpleSideModelError(
                f"{self.image_key} must have shape [B,3,H,W], got {tuple(value.shape)}"
            )
        if value.shape[0] <= 0 or value.shape[2] <= 0 or value.shape[3] <= 0:
            raise SimpleSideModelError(
                f"{self.image_key} must have non-empty batch and spatial dimensions"
            )
        if value.dtype != torch.float32:
            raise SimpleSideModelError(
                f"{self.image_key} must use torch.float32, got {value.dtype}"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise SimpleSideModelError(f"{self.image_key} contains NaN or Inf")
        return value

    def _validate_feature(
        self,
        key: str,
        value: Any,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if not isinstance(value, Tensor):
            raise SimpleSideModelError(f"{key} must be a torch.Tensor, got {type(value).__name__}")
        width = FEATURE_WIDTHS[key]
        if value.ndim != 2 or tuple(value.shape) != (batch_size, width):
            raise SimpleSideModelError(
                f"{key} must have shape [B,{width}] with B={batch_size}, got {tuple(value.shape)}"
            )
        if value.dtype != torch.float32:
            raise SimpleSideModelError(f"{key} must use torch.float32, got {value.dtype}")
        if value.device != device:
            raise SimpleSideModelError(f"{key} must be on {device}, got {value.device}")
        if not bool(torch.isfinite(value).all().item()):
            raise SimpleSideModelError(f"{key} contains NaN or Inf")
        return value

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        missing = [key for key in self.forward_keys if key not in inputs]
        if missing:
            raise SimpleSideModelError(f"missing required Side model inputs: {missing}")
        unexpected = [key for key in inputs if key not in self.forward_keys]
        if unexpected:
            raise SimpleSideModelError(f"unexpected Side model inputs: {unexpected}")

        image = self._validate_image(inputs[self.image_key])
        batch_size = int(image.shape[0])
        features = [
            self._validate_feature(
                key,
                inputs[key],
                batch_size=batch_size,
                device=image.device,
            )
            for key in self.feature_keys
        ]
        image_embedding = self.image_encoder(image)
        fused = torch.cat([image_embedding, *features], dim=1)
        embedding = self.embedding_activation(self.embedding_projection(fused))
        delta_y_side = self.residual_head(embedding)
        if not bool(torch.isfinite(embedding).all().item()):
            raise SimpleSideModelError("side_embedding contains NaN or Inf")
        if not bool(torch.isfinite(delta_y_side).all().item()):
            raise SimpleSideModelError("delta_y_side contains NaN or Inf")
        return {
            "delta_y_side": delta_y_side,
            "side_embedding": embedding,
        }


def create_model(
    image_key: str = "side_image",
    feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS,
    embedding_dim: int = 32,
) -> SimpleSideResidualModel:
    """Generic-runtime factory for the feature-consuming Side smoke model."""

    return SimpleSideResidualModel(
        image_key=image_key,
        feature_keys=feature_keys,
        embedding_dim=embedding_dim,
    )


__all__ = [
    "DEFAULT_FEATURE_KEYS",
    "FEATURE_WIDTHS",
    "SimpleSideModelError",
    "SimpleSideResidualModel",
    "create_model",
]
