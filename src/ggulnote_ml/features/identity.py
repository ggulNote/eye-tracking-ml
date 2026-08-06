from __future__ import annotations

import numpy as np

from ggulnote_ml.contracts import CanonicalBatch, FeatureBatch
from ggulnote_ml.features.base import FeatureExtractor


class IdentityFeatureExtractor(FeatureExtractor):
    """Flatten each frame without learning or hand-crafted feature extraction."""

    name = "identity"

    def __init__(self, version: str) -> None:
        self.version = version

    def transform(self, batch: CanonicalBatch) -> FeatureBatch:
        batch_size, sequence_length = batch.frames.shape[:2]
        values = batch.frames.reshape(batch_size, sequence_length, -1).astype(
            np.float32, copy=False
        )
        result = FeatureBatch(values=values, targets=batch.targets)
        result.validate()
        return result

