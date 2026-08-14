"""MobileNetV4-Conv-S Side Encoder backed by the exact pinned timm provider."""

from __future__ import annotations

from collections.abc import Sequence

import timm
import torch
from torch import Tensor, nn

from gaze_pipeline.models.side.components import SideAuxiliaryProjector, SideEncoderHead

MOBILENET_V4_CONV_S_MODEL_ID = "mobilenetv4_conv_small.e2400_r224_in1k"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MobileNetV4SideEncoder(nn.Module):
    """Predict independent Side gaze from a non-square eye image and optional cues.

    External input contract:
        side_image: float32 RGB ``[B,3,128,256]`` in ``[0,1]``.
        side_head_pose_2d: optional float32 ``[B,2]``.
        front_head_vector: optional float32 ``[B,3]``; mutually exclusive with
            ``side_head_pose_2d``.
        side_eye_angles: optional float32 ``[B,2]``.
        side_iris_pose_2d: optional float32 ``[B,2]``.

    Output contract:
        gaze_xy: float32 ``[B,2]`` in centered-normalized screen coordinates.
        side_embedding: float32 ``[B,embedding_dim]``.
        quality: float32 ``[B,1]`` in ``[0,1]`` after sigmoid.

    ``pretrained=True`` asks timm to download or reuse cached weights for the
    exact ``model_id``.  This requires network access on a cold cache and uses
    the cache configured by timm/Hugging Face; deployments should pre-populate
    and pin that cache when offline or reproducibility-sensitive.  The default
    is ``False`` so construction and tests never download weights.
    """

    def __init__(
        self,
        *,
        model_id: str = MOBILENET_V4_CONV_S_MODEL_ID,
        pretrained: bool = False,
        embedding_dim: int = 256,
        auxiliary_dim: int = 32,
        normalize_imagenet: bool = True,
        image_mean: Sequence[float] = IMAGENET_MEAN,
        image_std: Sequence[float] = IMAGENET_STD,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty timm model identifier")
        if not isinstance(pretrained, bool):
            raise TypeError("pretrained must be bool")
        if not isinstance(normalize_imagenet, bool):
            raise TypeError("normalize_imagenet must be bool")

        mean = _channel_buffer("image_mean", image_mean, positive=False)
        std = _channel_buffer("image_std", image_std, positive=True)
        self.register_buffer("image_mean", mean, persistent=True)
        self.register_buffer("image_std", std, persistent=True)
        self.model_id = model_id
        self.pretrained = pretrained
        self.normalize_imagenet = normalize_imagenet

        self.image_encoder = timm.create_model(
            model_id,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
        )
        channels = tuple(int(value) for value in self.image_encoder.feature_info.channels())
        if len(channels) != 1 or channels[0] <= 0:
            raise ValueError(
                f"timm model {model_id!r} must expose one positive final feature width"
            )
        self.image_feature_dim = channels[0]
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.auxiliary_projector = SideAuxiliaryProjector(projection_dim=auxiliary_dim)
        self.output_head = SideEncoderHead(
            self.image_feature_dim,
            self.auxiliary_projector.output_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(
        self,
        side_image: Tensor,
        *,
        side_head_pose_2d: Tensor | None = None,
        front_head_vector: Tensor | None = None,
        side_eye_angles: Tensor | None = None,
        side_iris_pose_2d: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return ``gaze_xy``, ``side_embedding``, and sigmoid ``quality``.

        No validity mask, target, Front prediction, or residual target is
        accepted or returned by this model.
        """

        auxiliary_features = self.auxiliary_projector(
            side_image,
            side_head_pose_2d=side_head_pose_2d,
            front_head_vector=front_head_vector,
            side_eye_angles=side_eye_angles,
            side_iris_pose_2d=side_iris_pose_2d,
        )
        image_input = side_image
        if self.normalize_imagenet:
            image_input = (image_input - self.image_mean) / self.image_std
        feature_maps = self.image_encoder(image_input)
        if not isinstance(feature_maps, list | tuple) or len(feature_maps) != 1:
            raise RuntimeError("MobileNetV4 feature extractor must return one feature map")
        image_features = self.global_pool(feature_maps[0]).flatten(1)
        return self.output_head(image_features, auxiliary_features)


def create_model(
    *,
    model_id: str = MOBILENET_V4_CONV_S_MODEL_ID,
    pretrained: bool = False,
    embedding_dim: int = 256,
    auxiliary_dim: int = 32,
    normalize_imagenet: bool = True,
    image_mean: Sequence[float] = IMAGENET_MEAN,
    image_std: Sequence[float] = IMAGENET_STD,
    dropout: float = 0.0,
) -> nn.Module:
    """Create the timm MobileNetV4-Conv-S Side Encoder.

    ``pretrained=False`` is the offline-safe default.  Setting it to ``True``
    delegates weight download/cache lookup to timm for the exact ``model_id``;
    a cold cache therefore requires network access.
    """

    return MobileNetV4SideEncoder(
        model_id=model_id,
        pretrained=pretrained,
        embedding_dim=embedding_dim,
        auxiliary_dim=auxiliary_dim,
        normalize_imagenet=normalize_imagenet,
        image_mean=image_mean,
        image_std=image_std,
        dropout=dropout,
    )


def _channel_buffer(name: str, values: Sequence[float], *, positive: bool) -> Tensor:
    if isinstance(values, str | bytes):
        raise TypeError(f"{name} must contain three numbers")
    try:
        tensor = torch.as_tensor(tuple(values), dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain three numbers") from exc
    if tensor.shape != (3,):
        raise ValueError(f"{name} must contain exactly three numbers")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    if positive and not bool((tensor > 0.0).all().item()):
        raise ValueError(f"{name} values must be positive")
    return tensor.view(1, 3, 1, 1)


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MOBILENET_V4_CONV_S_MODEL_ID",
    "MobileNetV4SideEncoder",
    "create_model",
]
