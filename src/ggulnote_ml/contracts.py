from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

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
    """Preprocessed model input with shape [B, T, 3, H, W]."""

    frames: np.ndarray
    targets: np.ndarray
    participant_ids: Tuple[str, ...]
    session_ids: Tuple[str, ...]
    sources: Tuple[str, ...]

    def validate(self, sequence_length: int, height: int, width: int) -> None:
        expected_tail = (sequence_length, 3, height, width)
        if self.frames.ndim != 5 or self.frames.shape[1:] != expected_tail:
            raise ContractError(
                "Canonical frames must have shape [B, %d, 3, %d, %d]; got %s."
                % (sequence_length, height, width, self.frames.shape)
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
            "target": "float32[2] normalized gaze coordinate",
        },
        "preprocessing_output": {
            "shape": list(canonical.frames.shape),
            "dtype": str(canonical.frames.dtype),
            "layout": "BTCHW",
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
    }

