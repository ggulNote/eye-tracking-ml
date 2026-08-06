from __future__ import annotations

import numpy as np

from ggulnote_ml.contracts import CanonicalBatch, FeatureBatch
from ggulnote_ml.features.base import FeatureExtractor


class SpatialStatisticsFeatureExtractor(FeatureExtractor):
    """Small deterministic baseline: RGB moments plus intensity centroid."""

    name = "spatial_statistics"

    def __init__(self, version: str) -> None:
        self.version = version

    def transform(self, batch: CanonicalBatch) -> FeatureBatch:
        frames = batch.frames
        _, _, _, height, width = frames.shape
        channel_means = frames.mean(axis=(3, 4))
        channel_stds = frames.std(axis=(3, 4))
        intensity = frames.mean(axis=2)
        intensity = intensity - intensity.min(axis=(2, 3), keepdims=True)
        weights = intensity + np.float32(1e-6)
        x_grid = np.linspace(0.0, 1.0, width, dtype=np.float32).reshape(1, 1, 1, width)
        y_grid = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(1, 1, height, 1)
        denominator = weights.sum(axis=(2, 3))
        centroid_x = (weights * x_grid).sum(axis=(2, 3)) / denominator
        centroid_y = (weights * y_grid).sum(axis=(2, 3)) / denominator
        values = np.concatenate(
            [
                channel_means,
                channel_stds,
                centroid_x[..., None],
                centroid_y[..., None],
            ],
            axis=2,
        ).astype(np.float32, copy=False)
        result = FeatureBatch(values=values, targets=batch.targets)
        result.validate()
        return result

