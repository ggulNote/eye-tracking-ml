from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ggulnote_ml.exceptions import ContractError


@dataclass(frozen=True)
class RawVideoSample:
    """One labeled temporal window before resize and normalization.

    frames: uint8 RGB tensor with shape [T, H, W, 3]
    target: float tensor with shape [2], normalized x/y in [0, 1]
    """

    frames: np.ndarray
    target: np.ndarray
    participant_id: str
    session_id: str
    source: str
    sample_id: str = ""
    frame_indices: Tuple[int, ...] = ()
    timestamps_ms: Tuple[float, ...] = ()
    screen_size_px: Optional[Tuple[int, int]] = None
    screen_size_cm: Optional[Tuple[float, float]] = None
    device_id: str = ""
    camera_intrinsics_path: Optional[str] = None
    calibration_point_id: Optional[str] = None
    sample_weight: float = 1.0

    def validate(self, sequence_length: int) -> None:
        if self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            raise ContractError(
                "Raw frames must have shape [T, H, W, 3]; got %s." % (self.frames.shape,)
            )
        if self.frames.shape[0] != sequence_length:
            raise ContractError(
                "Expected %d frames per sample; got %d."
                % (sequence_length, self.frames.shape[0])
            )
        if self.frames.dtype != np.uint8:
            raise ContractError("Raw frames must use uint8 RGB values.")
        if self.target.shape != (2,):
            raise ContractError("Gaze target must have shape [2]; got %s." % (self.target.shape,))
        if not np.isfinite(self.target).all() or not ((0 <= self.target) & (self.target <= 1)).all():
            raise ContractError("Gaze target values must be finite and normalized to [0, 1].")
        if self.source not in {"webcam", "phone", "synthetic"}:
            raise ContractError("Unsupported data source: %s" % self.source)
        if not self.participant_id or not self.session_id:
            raise ContractError("participant_id and session_id must not be empty.")
        if self.frame_indices and len(self.frame_indices) != sequence_length:
            raise ContractError("frame_indices must contain one value per frame.")
        if self.timestamps_ms and len(self.timestamps_ms) != sequence_length:
            raise ContractError("timestamps_ms must contain one value per frame.")
        if self.timestamps_ms and not np.isfinite(self.timestamps_ms).all():
            raise ContractError("timestamps_ms contains NaN or infinity.")
        if self.screen_size_px is not None and any(value <= 0 for value in self.screen_size_px):
            raise ContractError("screen_size_px values must be positive.")
        if self.screen_size_cm is not None and any(value <= 0 for value in self.screen_size_cm):
            raise ContractError("screen_size_cm values must be positive.")
        if not np.isfinite(self.sample_weight) or self.sample_weight <= 0:
            raise ContractError("sample_weight must be finite and greater than zero.")


@dataclass(frozen=True)
class RawDataset:
    samples: Tuple[RawVideoSample, ...]

    def validate(self, sequence_length: int) -> None:
        if not self.samples:
            raise ContractError("A dataset must contain at least one sample.")
        for sample in self.samples:
            sample.validate(sequence_length)

    def subset(self, indices: Sequence[int]) -> "RawDataset":
        return RawDataset(samples=tuple(self.samples[index] for index in indices))


@dataclass(frozen=True)
class DatasetSplits:
    train: RawDataset
    validation: RawDataset
    test: RawDataset


@dataclass(frozen=True)
class CanonicalBatch:
    """Common full-frame tensor before any paper-model-specific processing."""

    frames: np.ndarray
    targets: np.ndarray
    participant_ids: Tuple[str, ...]
    session_ids: Tuple[str, ...]
    sources: Tuple[str, ...]
    sample_ids: Tuple[str, ...]
    frame_indices: Tuple[Tuple[int, ...], ...]
    timestamps_ms: Tuple[Tuple[float, ...], ...]
    screen_sizes_px: Tuple[Optional[Tuple[int, int]], ...]
    screen_sizes_cm: Tuple[Optional[Tuple[float, float]], ...]
    device_ids: Tuple[str, ...]
    camera_intrinsics_paths: Tuple[Optional[str], ...]
    calibration_point_ids: Tuple[Optional[str], ...]
    sample_weights: np.ndarray

    def validate(self, sequence_length: int, channels: int, height: int, width: int) -> None:
        expected_tail = (sequence_length, channels, height, width)
        if self.frames.ndim != 5 or self.frames.shape[1:] != expected_tail:
            raise ContractError(
                "Canonical frames must have shape [B, %d, %d, %d, %d]; got %s."
                % (sequence_length, channels, height, width, self.frames.shape)
            )
        if self.frames.dtype != np.float32:
            raise ContractError("Canonical frames must use float32 values.")
        if self.targets.shape != (self.frames.shape[0], 2):
            raise ContractError(
                "Targets must have shape [B, 2]; got %s." % (self.targets.shape,)
            )
        if self.targets.dtype != np.float32:
            raise ContractError("Targets must use float32 values.")
        if not np.isfinite(self.frames).all() or not np.isfinite(self.targets).all():
            raise ContractError("Canonical batch contains NaN or infinity.")
        batch_size = self.frames.shape[0]
        metadata = (
            self.participant_ids,
            self.session_ids,
            self.sources,
            self.sample_ids,
            self.frame_indices,
            self.timestamps_ms,
            self.screen_sizes_px,
            self.screen_sizes_cm,
            self.device_ids,
            self.camera_intrinsics_paths,
            self.calibration_point_ids,
        )
        if any(len(values) != batch_size for values in metadata):
            raise ContractError("Canonical metadata must contain one value per sample.")
        if self.sample_weights.shape != (batch_size,):
            raise ContractError("sample_weights must have shape [B].")
        if not np.isfinite(self.sample_weights).all() or np.any(self.sample_weights <= 0):
            raise ContractError("sample_weights must be finite and greater than zero.")


@dataclass(frozen=True)
class WebEyeTrackInputBatch:
    """Contract only: tensors that a future WebEyeTrack processor must produce.

    `image` is the combined eye-region patch, not the original camera frame.
    """

    image: np.ndarray
    head_vector: np.ndarray
    face_origin_3d: np.ndarray

    def validate(self) -> None:
        if self.image.ndim != 4 or self.image.shape[1:] != (128, 512, 3):
            raise ContractError(
                "WebEyeTrack image must have shape [B,128,512,3]; got %s."
                % (self.image.shape,)
            )
        batch_size = self.image.shape[0]
        if self.head_vector.shape != (batch_size, 3):
            raise ContractError("head_vector must have shape [B,3].")
        if self.face_origin_3d.shape != (batch_size, 3):
            raise ContractError("face_origin_3d must have shape [B,3].")
        for name, values in (
            ("image", self.image),
            ("head_vector", self.head_vector),
            ("face_origin_3d", self.face_origin_3d),
        ):
            if values.dtype != np.float32:
                raise ContractError("%s must use float32." % name)
            if not np.isfinite(values).all():
                raise ContractError("%s contains NaN or infinity." % name)
        if np.any(self.image < 0.0) or np.any(self.image > 1.0):
            raise ContractError("WebEyeTrack image values must be in [0,1].")


@dataclass(frozen=True)
class WebEyeTrackOutputBatch:
    """Screen-centered normalized gaze coordinates returned by the model."""

    gaze: np.ndarray

    def validate(self, batch_size: Optional[int] = None) -> None:
        if self.gaze.ndim != 2 or self.gaze.shape[1] != 2:
            raise ContractError("WebEyeTrack gaze must have shape [B,2].")
        if batch_size is not None and self.gaze.shape[0] != batch_size:
            raise ContractError("WebEyeTrack output batch size does not match input.")
        if self.gaze.dtype != np.float32 or not np.isfinite(self.gaze).all():
            raise ContractError("WebEyeTrack gaze must be finite float32 values.")
        if np.any(self.gaze < -0.5) or np.any(self.gaze > 0.5):
            raise ContractError("WebEyeTrack gaze values must be in [-0.5,0.5].")


@dataclass(frozen=True)
class WebEyeTrackTrainingBatch:
    """Processor-to-training boundary; no MediaPipe/model implementation included."""

    inputs: WebEyeTrackInputBatch
    targets: np.ndarray
    valid_mask: np.ndarray
    participant_ids: Tuple[str, ...]
    sample_ids: Tuple[str, ...]
    calibration_point_ids: Tuple[Optional[str], ...]
    sample_weights: np.ndarray

    def validate(self) -> None:
        self.inputs.validate()
        batch_size = self.inputs.image.shape[0]
        WebEyeTrackOutputBatch(self.targets).validate(batch_size=batch_size)
        if self.valid_mask.shape != (batch_size,) or self.valid_mask.dtype != np.bool_:
            raise ContractError("valid_mask must be bool[B].")
        if any(
            len(values) != batch_size
            for values in (
                self.participant_ids,
                self.sample_ids,
                self.calibration_point_ids,
            )
        ):
            raise ContractError("Training metadata must contain one value per sample.")
        if self.sample_weights.shape != (batch_size,):
            raise ContractError("Training sample_weights must have shape [B].")
        if self.sample_weights.dtype != np.float32:
            raise ContractError("Training sample_weights must use float32.")
        if not np.isfinite(self.sample_weights).all() or np.any(self.sample_weights <= 0):
            raise ContractError("Training sample_weights must be finite and positive.")


@dataclass(frozen=True)
class FeatureBatch:
    """Feature tensor with shape [B, T, F] and gaze targets [B, 2]."""

    values: np.ndarray
    targets: np.ndarray

    def validate(self) -> None:
        if self.values.ndim != 3:
            raise ContractError(
                "Features must have shape [B, T, F]; got %s." % (self.values.shape,)
            )
        if self.values.dtype != np.float32:
            raise ContractError("Features must use float32 values.")
        if self.targets.shape != (self.values.shape[0], 2):
            raise ContractError("Feature targets must have shape [B, 2].")
        if not np.isfinite(self.values).all():
            raise ContractError("Features contain NaN or infinity.")


def contract_summary(
    canonical: CanonicalBatch,
    features: FeatureBatch,
    predictions: np.ndarray,
) -> Dict[str, object]:
    return {
        "raw_sample": {
            "frames": "uint8[T,H,W,3] RGB",
            "target": "float32[2] top-left normalized gaze coordinate [0,1]",
            "alignment": "one manifest row = one labeled instant/window",
        },
        "preprocessing_output": {
            "shape": list(canonical.frames.shape),
            "dtype": str(canonical.frames.dtype),
            "layout": "BTCHW",
            "channels": canonical.frames.shape[2],
        },
        "feature_output": {
            "shape": list(features.values.shape),
            "dtype": str(features.values.dtype),
            "layout": "BTF",
        },
        "model_output": {
            "shape": list(predictions.shape),
            "dtype": str(predictions.dtype),
            "range": [0.0, 1.0],
        },
        "future_webeyetrack_boundary": {
            "image": "float32[B,128,512,3] RGB BHWC in [0,1]",
            "head_vector": "float32[B,3]",
            "face_origin_3d": "float32[B,3] centimeters in camera coordinates",
            "output": "float32[B,2] screen-centered normalized [-0.5,0.5]",
            "calibration_support_size": "1..9 samples per participant",
            "training_metadata": "valid_mask[B], participant_id, calibration_point_id, weight[B]",
        },
    }
