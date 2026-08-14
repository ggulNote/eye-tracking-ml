"""Shared, model-independent components for strict-profile Side Encoders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

SIDE_IMAGE_SHAPE = (3, 128, 256)


@dataclass(frozen=True)
class ValidatedSideInputs:
    """Validated Side model inputs with non-finite auxiliary values zero-filled.

    Tensor contract:
        side_image: float32 ``[B,3,128,256]``, RGB in ``[0,1]``.
        side_head_pose_2d: optional float32 ``[B,2]``.
        front_head_vector: optional float32 ``[B,3]``.
        side_eye_angles: optional float32 ``[B,2]``.
        side_iris_pose_2d: optional float32 ``[B,2]``.

    This value contains no validity decision.  In particular, zero-filled
    auxiliary values must not be interpreted as a replacement for
    ``side_gaze_valid``.
    """

    side_image: Tensor
    side_head_pose_2d: Tensor | None
    front_head_vector: Tensor | None
    side_eye_angles: Tensor | None
    side_iris_pose_2d: Tensor | None


def validate_and_sanitize_side_inputs(
    side_image: Tensor,
    *,
    side_head_pose_2d: Tensor | None = None,
    front_head_vector: Tensor | None = None,
    side_eye_angles: Tensor | None = None,
    side_iris_pose_2d: Tensor | None = None,
) -> ValidatedSideInputs:
    """Validate the public Side input contract and sanitize auxiliary tensors.

    ``side_image`` must be finite float32 RGB ``[B,3,128,256]`` in ``[0,1]``.
    Every provided auxiliary tensor must be float32, share its batch and device,
    and have the documented trailing dimension.  ``side_head_pose_2d`` and
    ``front_head_vector`` are mutually exclusive.  Auxiliary NaN and infinity
    values are replaced by zero without producing or changing a validity mask.
    """

    _validate_side_image(side_image)
    if side_head_pose_2d is not None and front_head_vector is not None:
        raise ValueError(
            "side_head_pose_2d and front_head_vector are mutually exclusive head sources"
        )

    batch_size = side_image.shape[0]
    device = side_image.device
    sanitized = {
        "side_head_pose_2d": _validate_and_zero_fill_auxiliary(
            "side_head_pose_2d", side_head_pose_2d, batch_size, 2, device
        ),
        "front_head_vector": _validate_and_zero_fill_auxiliary(
            "front_head_vector", front_head_vector, batch_size, 3, device
        ),
        "side_eye_angles": _validate_and_zero_fill_auxiliary(
            "side_eye_angles", side_eye_angles, batch_size, 2, device
        ),
        "side_iris_pose_2d": _validate_and_zero_fill_auxiliary(
            "side_iris_pose_2d", side_iris_pose_2d, batch_size, 2, device
        ),
    }
    return ValidatedSideInputs(side_image=side_image, **sanitized)


class SideAuxiliaryProjector(nn.Module):
    """Project optional Side cues into fixed head/eye/iris representation slots.

    ``forward`` accepts only the public model inputs.  The returned float32
    tensor has shape ``[B, 3 * projection_dim]`` on ``side_image.device``.
    Missing head, eye, or iris inputs contribute an all-zero slot.  The two head
    sources use separate projections but occupy the same head slot.
    """

    def __init__(self, projection_dim: int = 32) -> None:
        super().__init__()
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        self.projection_dim = projection_dim
        self.side_head_projection = _make_projection(2, projection_dim)
        self.front_head_projection = _make_projection(3, projection_dim)
        self.eye_projection = _make_projection(2, projection_dim)
        self.iris_projection = _make_projection(2, projection_dim)

    @property
    def output_dim(self) -> int:
        """Return the fixed auxiliary representation width."""

        return 3 * self.projection_dim

    def forward(
        self,
        side_image: Tensor,
        *,
        side_head_pose_2d: Tensor | None = None,
        front_head_vector: Tensor | None = None,
        side_eye_angles: Tensor | None = None,
        side_iris_pose_2d: Tensor | None = None,
    ) -> Tensor:
        """Return a float32 auxiliary representation shaped ``[B, output_dim]``."""

        inputs = validate_and_sanitize_side_inputs(
            side_image,
            side_head_pose_2d=side_head_pose_2d,
            front_head_vector=front_head_vector,
            side_eye_angles=side_eye_angles,
            side_iris_pose_2d=side_iris_pose_2d,
        )
        empty = side_image.new_zeros((side_image.shape[0], self.projection_dim))
        if inputs.side_head_pose_2d is not None:
            head = self.side_head_projection(inputs.side_head_pose_2d)
        elif inputs.front_head_vector is not None:
            head = self.front_head_projection(inputs.front_head_vector)
        else:
            head = empty
        eye = (
            self.eye_projection(inputs.side_eye_angles)
            if inputs.side_eye_angles is not None
            else empty
        )
        iris = (
            self.iris_projection(inputs.side_iris_pose_2d)
            if inputs.side_iris_pose_2d is not None
            else empty
        )
        return torch.cat((head, eye, iris), dim=1)


class SideEncoderHead(nn.Module):
    """Build residual Side outputs from image and auxiliary representations.

    Inputs:
        image_features: float32 ``[B, image_feature_dim]``.
        auxiliary_features: float32 ``[B, auxiliary_feature_dim]``.

    Outputs:
        ``delta_y_side``: float32 ``[B,1]`` y-axis correction in
        centered-normalized screen-coordinate units.  Downstream fusion adds
        this residual to the Front branch's y prediction.
        ``side_embedding``: float32 ``[B,256]`` by default.
        ``quality``: float32 ``[B,1]`` in ``[0,1]`` after sigmoid.  It is not a
        validity mask and does not replace ``side_gaze_valid``.
    """

    def __init__(
        self,
        image_feature_dim: int,
        auxiliary_feature_dim: int,
        *,
        embedding_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("image_feature_dim", image_feature_dim),
            ("auxiliary_feature_dim", auxiliary_feature_dim),
            ("embedding_dim", embedding_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.image_feature_dim = image_feature_dim
        self.auxiliary_feature_dim = auxiliary_feature_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Sequential(
            nn.Linear(image_feature_dim + auxiliary_feature_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )
        self.delta_y_head = nn.Linear(embedding_dim, 1)
        self.quality_head = nn.Linear(embedding_dim, 1)

    def forward(self, image_features: Tensor, auxiliary_features: Tensor) -> dict[str, Tensor]:
        """Return ``delta_y_side``, ``side_embedding``, and sigmoid ``quality``."""

        _validate_feature_tensor(
            "image_features", image_features, expected_width=self.image_feature_dim
        )
        _validate_feature_tensor(
            "auxiliary_features",
            auxiliary_features,
            expected_width=self.auxiliary_feature_dim,
            batch_size=image_features.shape[0],
            device=image_features.device,
        )
        side_embedding = self.embedding(torch.cat((image_features, auxiliary_features), dim=1))
        return {
            "delta_y_side": self.delta_y_head(side_embedding),
            "side_embedding": side_embedding,
            "quality": torch.sigmoid(self.quality_head(side_embedding)),
        }


def _make_projection(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.GELU(),
        nn.LayerNorm(output_dim),
    )


def _validate_side_image(side_image: Tensor) -> None:
    if not isinstance(side_image, Tensor):
        raise TypeError("side_image must be a torch.Tensor")
    if side_image.dtype != torch.float32:
        raise TypeError("side_image must have dtype torch.float32")
    if side_image.ndim != 4 or tuple(side_image.shape[1:]) != SIDE_IMAGE_SHAPE:
        raise ValueError("side_image must have shape [B,3,128,256]")
    if side_image.shape[0] <= 0:
        raise ValueError("side_image batch dimension must be positive")
    if not bool(torch.isfinite(side_image).all().item()):
        raise ValueError("side_image must contain only finite values")
    if not bool(((side_image >= 0.0) & (side_image <= 1.0)).all().item()):
        raise ValueError("side_image RGB values must be in [0,1]")


def _validate_and_zero_fill_auxiliary(
    name: str,
    value: Tensor | None,
    batch_size: int,
    width: int,
    device: torch.device,
) -> Tensor | None:
    if value is None:
        return None
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if value.ndim != 2 or tuple(value.shape) != (batch_size, width):
        raise ValueError(f"{name} must have shape [B,{width}] with B={batch_size}")
    if value.device != device:
        raise ValueError(f"{name} must be on side_image.device ({device})")
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _validate_feature_tensor(
    name: str,
    value: Tensor,
    *,
    expected_width: int,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if value.ndim != 2 or value.shape[1] != expected_width:
        raise ValueError(f"{name} must have shape [B,{expected_width}]")
    if batch_size is not None and value.shape[0] != batch_size:
        raise ValueError(f"{name} batch size must match image_features")
    if device is not None and value.device != device:
        raise ValueError(f"{name} device must match image_features")


__all__ = [
    "SIDE_IMAGE_SHAPE",
    "SideAuxiliaryProjector",
    "SideEncoderHead",
    "ValidatedSideInputs",
    "validate_and_sanitize_side_inputs",
]
