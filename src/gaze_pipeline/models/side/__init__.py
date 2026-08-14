"""Side Encoder factories and shared tensor-contract components.

The two public factories are intentionally importable placeholders.  The
shared input/output contract can be tested before either image backbone is
implemented, while runtime loading remains outside this package's scope.
"""

from __future__ import annotations

from torch import nn

from gaze_pipeline.models.side.components import (
    SideAuxiliaryProjector,
    SideEncoderHead,
    ValidatedSideInputs,
    validate_and_sanitize_side_inputs,
)


def create_blazegaze_transfer_side_encoder(
    *,
    image_feature_dim: int,
    auxiliary_projection_dim: int = 32,
    embedding_dim: int = 256,
    dropout: float = 0.0,
) -> nn.Module:
    """Create the future BlazeGaze-transfer Side Encoder.

    The image backbone and Keras-weight import are deliberately not wired yet.
    The arguments already match the config contract so the entrypoint can be
    resolved and inspected without adding a runtime loader.
    """

    _ = (image_feature_dim, auxiliary_projection_dim, embedding_dim, dropout)
    raise NotImplementedError(
        "BlazeGaze-transfer Side backbone is not implemented; only shared contract "
        "components are available."
    )


def create_mobilenet_v4_side_encoder(
    *,
    image_feature_dim: int,
    auxiliary_projection_dim: int = 32,
    embedding_dim: int = 256,
    dropout: float = 0.0,
    variant: str = "small",
) -> nn.Module:
    """Create the future MobileNet-v4 Side Encoder.

    This callable is a config-facing skeleton only.  It does not instantiate a
    backbone, download weights, or register itself with a runtime loader.
    """

    _ = (image_feature_dim, auxiliary_projection_dim, embedding_dim, dropout, variant)
    raise NotImplementedError(
        "MobileNet-v4 Side backbone is not implemented; only shared contract "
        "components are available."
    )


__all__ = [
    "SideAuxiliaryProjector",
    "SideEncoderHead",
    "ValidatedSideInputs",
    "create_blazegaze_transfer_side_encoder",
    "create_mobilenet_v4_side_encoder",
    "validate_and_sanitize_side_inputs",
]
