from __future__ import annotations

import numpy as np

from ggulnote_ml.config import DataConfig, PreprocessingConfig
from ggulnote_ml.contracts import CanonicalBatch, RawDataset


class VideoPreprocessor:
    """Convert webcam/phone RGB windows into a common normalized BTCHW tensor."""

    version = "v2"

    def __init__(self, data_config: DataConfig, config: PreprocessingConfig) -> None:
        self.data_config = data_config
        self.config = config
        self.channels = 3 if config.color_mode == "rgb" else 1
        self._mean = np.asarray(config.mean, dtype=np.float32).reshape(
            1, self.channels, 1, 1
        )
        self._std = np.asarray(config.std, dtype=np.float32).reshape(
            1, self.channels, 1, 1
        )

    def transform(self, dataset: RawDataset) -> CanonicalBatch:
        dataset.validate(self.data_config.sequence_length)
        processed = []
        targets = []
        participant_ids = []
        session_ids = []
        sources = []
        sample_ids = []
        frame_indices = []
        timestamps_ms = []
        screen_sizes_px = []
        screen_sizes_cm = []
        device_ids = []
        camera_intrinsics_paths = []
        calibration_point_ids = []
        sample_weights = []

        for sample in dataset.samples:
            frames = _resize_nearest(
                sample.frames,
                output_height=self.config.output_height,
                output_width=self.config.output_width,
            )
            if self.config.color_mode == "gray":
                frames = _rgb_to_gray(frames)
            frames = frames.astype(np.float32) / np.float32(255.0)
            frames = np.transpose(frames, (0, 3, 1, 2))
            frames = (frames - self._mean) / self._std
            processed.append(frames.astype(np.float32, copy=False))
            targets.append(sample.target.astype(np.float32, copy=False))
            participant_ids.append(sample.participant_id)
            session_ids.append(sample.session_id)
            sources.append(sample.source)
            sample_ids.append(sample.sample_id)
            frame_indices.append(sample.frame_indices)
            timestamps_ms.append(sample.timestamps_ms)
            screen_sizes_px.append(sample.screen_size_px)
            screen_sizes_cm.append(sample.screen_size_cm)
            device_ids.append(sample.device_id)
            camera_intrinsics_paths.append(sample.camera_intrinsics_path)
            calibration_point_ids.append(sample.calibration_point_id)
            sample_weights.append(sample.sample_weight)

        batch = CanonicalBatch(
            frames=np.stack(processed, axis=0),
            targets=np.stack(targets, axis=0).astype(np.float32, copy=False),
            participant_ids=tuple(participant_ids),
            session_ids=tuple(session_ids),
            sources=tuple(sources),
            sample_ids=tuple(sample_ids),
            frame_indices=tuple(frame_indices),
            timestamps_ms=tuple(timestamps_ms),
            screen_sizes_px=tuple(screen_sizes_px),
            screen_sizes_cm=tuple(screen_sizes_cm),
            device_ids=tuple(device_ids),
            camera_intrinsics_paths=tuple(camera_intrinsics_paths),
            calibration_point_ids=tuple(calibration_point_ids),
            sample_weights=np.asarray(sample_weights, dtype=np.float32),
        )
        batch.validate(
            sequence_length=self.data_config.sequence_length,
            channels=self.channels,
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


def _rgb_to_gray(frames: np.ndarray) -> np.ndarray:
    coefficients = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    gray = np.rint(frames.astype(np.float32) @ coefficients)
    return np.clip(gray, 0, 255).astype(np.uint8)[..., None]
