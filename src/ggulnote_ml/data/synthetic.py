from __future__ import annotations

from typing import List

import numpy as np

from ggulnote_ml.config import DataConfig
from ggulnote_ml.contracts import RawDataset, RawVideoSample
from ggulnote_ml.data.base import DataSource


class SyntheticGazeDataSource(DataSource):
    """Deterministic fake data used only to exercise the complete pipeline."""

    def __init__(self, config: DataConfig, seed: int) -> None:
        self.config = config
        self.seed = seed

    def load(self) -> RawDataset:
        rng = np.random.default_rng(self.seed)
        height = self.config.raw_height
        width = self.config.raw_width
        sequence_length = self.config.sequence_length
        yy, xx = np.mgrid[0:height, 0:width]
        samples: List[RawVideoSample] = []

        for index in range(self.config.num_samples):
            target = rng.uniform(0.08, 0.92, size=2).astype(np.float32)
            frames = np.empty((sequence_length, height, width, 3), dtype=np.uint8)
            source = "webcam" if index % 2 == 0 else "phone"

            for frame_index in range(sequence_length):
                jitter = rng.normal(0.0, 0.008, size=2)
                center_x = float(np.clip(target[0] + jitter[0], 0.0, 1.0)) * (width - 1)
                center_y = float(np.clip(target[1] + jitter[1], 0.0, 1.0)) * (height - 1)
                sigma = max(1.5, min(height, width) * 0.06)
                gaze_blob = np.exp(
                    -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2)
                )
                background = rng.normal(22.0, 3.0, size=(height, width, 3))
                tint = np.array([1.0, 0.9, 0.8]) if source == "webcam" else np.array([0.8, 0.9, 1.0])
                image = background + gaze_blob[..., None] * 220.0 * tint
                frames[frame_index] = np.clip(image, 0, 255).astype(np.uint8)

            participant_number = index % self.config.num_participants
            session_number = index // self.config.num_participants
            samples.append(
                RawVideoSample(
                    frames=frames,
                    target=target,
                    participant_id="participant-%03d" % participant_number,
                    session_id="session-%03d-%03d" % (participant_number, session_number),
                    source=source,
                )
            )

        dataset = RawDataset(samples=tuple(samples))
        dataset.validate(sequence_length)
        return dataset

