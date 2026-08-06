from __future__ import annotations

import numpy as np

from ggulnote_ml.config import DataConfig, PreprocessingConfig
from ggulnote_ml.contracts import CanonicalBatch, RawDataset


class VideoPreprocessor:
    """Convert variable-resolution RGB windows into normalized BTCHW tensors."""

    version = "v1"

    def __init__(self, data_config: DataConfig, config: PreprocessingConfig) -> None:
        self.data_config = data_config
        self.config = config
        self._mean = np.asarray(config.mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.asarray(config.std, dtype=np.float32).reshape(1, 3, 1, 1)

    def transform(self, dataset: RawDataset) -> CanonicalBatch:
        dataset.validate(self.data_config.sequence_length)
        processed = []
        targets = []
        participant_ids = []
        session_ids = []
        sources = []

        for sample in dataset.samples:
            frames = _resize_nearest(
                sample.frames,
                output_height=self.config.output_height,
                output_width=self.config.output_width,
            )
            frames = frames.astype(np.float32) / np.float32(255.0)
            frames = np.transpose(frames, (0, 3, 1, 2))
            frames = (frames - self._mean) / self._std
            processed.append(frames.astype(np.float32, copy=False))
            targets.append(sample.target.astype(np.float32, copy=False))
            participant_ids.append(sample.participant_id)
            session_ids.append(sample.session_id)
            sources.append(sample.source)

        batch = CanonicalBatch(
            frames=np.stack(processed, axis=0),
            targets=np.stack(targets, axis=0).astype(np.float32, copy=False),
            participant_ids=tuple(participant_ids),
            session_ids=tuple(session_ids),
            sources=tuple(sources),
        )
        batch.validate(
            sequence_length=self.data_config.sequence_length,
            height=self.config.output_height,
            width=self.config.output_width,
        )
        return batch


def _resize_nearest(frames: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
    input_height = frames.shape[1]
    input_width = frames.shape[2]
    y_indices = np.linspace(0, input_height - 1, output_height).round().astype(int)
    x_indices = np.linspace(0, input_width - 1, output_width).round().astype(int)
    return frames[:, y_indices][:, :, x_indices]

