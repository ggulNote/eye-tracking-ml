"""Side Encoder factories and shared tensor-contract components."""

from __future__ import annotations

from gaze_pipeline.models.side.blazegaze_transfer import (
    BlazeGazeTransferSideEncoder,
)
from gaze_pipeline.models.side.blazegaze_transfer import (
    create_model as create_blazegaze_transfer_side_encoder,
)
from gaze_pipeline.models.side.components import (
    SideAuxiliaryProjector,
    SideEncoderHead,
    ValidatedSideInputs,
    validate_and_sanitize_side_inputs,
)
from gaze_pipeline.models.side.mobilenet_v4 import (
    MobileNetV4SideEncoder,
)
from gaze_pipeline.models.side.mobilenet_v4 import (
    create_model as create_mobilenet_v4_side_encoder,
)

__all__ = [
    "BlazeGazeTransferSideEncoder",
    "MobileNetV4SideEncoder",
    "SideAuxiliaryProjector",
    "SideEncoderHead",
    "ValidatedSideInputs",
    "create_blazegaze_transfer_side_encoder",
    "create_mobilenet_v4_side_encoder",
    "validate_and_sanitize_side_inputs",
]
