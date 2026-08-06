import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ggulnote_ml.config import DataConfig
from ggulnote_ml.contracts import (
    WebEyeTrackInputBatch,
    WebEyeTrackOutputBatch,
    WebEyeTrackTrainingBatch,
)
from ggulnote_ml.coordinates import (
    to_screen_centered_normalized,
    to_top_left_from_centered,
    to_top_left_normalized,
)
from ggulnote_ml.data.manifest import ManifestVideoDataSource, _window_indices
from ggulnote_ml.exceptions import ContractError


def test_coordinate_spaces_round_trip() -> None:
    top_left = np.asarray([[0.0, 0.25], [0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
    centered = to_screen_centered_normalized(top_left)
    np.testing.assert_allclose(centered, [[-0.5, -0.25], [0.0, 0.0], [0.5, 0.5]])
    np.testing.assert_allclose(to_top_left_from_centered(centered), top_left)
    np.testing.assert_allclose(
        to_top_left_normalized([960, 540], "screen_pixels", [1920, 1080]),
        [0.5, 0.5],
    )


def test_labeled_window_is_centered_and_clamped() -> None:
    indices = _window_indices(100, 5, 2, label_frame_index=1, label_timestamp_ms=None, fps=30)
    np.testing.assert_array_equal(indices, [0, 0, 1, 3, 5])
    timestamp_indices = _window_indices(
        100, 1, 1, label_frame_index=None, label_timestamp_ms=1000, fps=30
    )
    np.testing.assert_array_equal(timestamp_indices, [30])


def test_webeyetrack_model_boundary_validation() -> None:
    inputs = WebEyeTrackInputBatch(
        image=np.zeros((2, 128, 512, 3), dtype=np.float32),
        head_vector=np.zeros((2, 3), dtype=np.float32),
        face_origin_3d=np.zeros((2, 3), dtype=np.float32),
    )
    outputs = WebEyeTrackOutputBatch(gaze=np.zeros((2, 2), dtype=np.float32))
    inputs.validate()
    outputs.validate(batch_size=inputs.image.shape[0])
    WebEyeTrackTrainingBatch(
        inputs=inputs,
        targets=outputs.gaze,
        valid_mask=np.asarray([True, False], dtype=np.bool_),
        participant_ids=("p1", "p1"),
        sample_ids=("s1", "s2"),
        calibration_point_ids=("grid-01", None),
        sample_weights=np.ones(2, dtype=np.float32),
    ).validate()

    with pytest.raises(ContractError):
        WebEyeTrackInputBatch(
            image=np.zeros((2, 128, 512, 1), dtype=np.float32),
            head_vector=np.zeros((2, 3), dtype=np.float32),
            face_origin_3d=np.zeros((2, 3), dtype=np.float32),
        ).validate()


def test_manifest_reuses_video_with_synchronized_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "capture.mp4"
    video_path.touch()
    intrinsics_path = tmp_path / "intrinsics.json"
    intrinsics_path.write_text(
        '{"image_width_px":6,"image_height_px":4,'
        '"camera_matrix":[[10,0,3],[0,10,2],[0,0,1]],'
        '"distortion_coefficients":[0,0,0,0,0]}',
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "video_path,source,participant_id,session_id,label_frame_index,target_x,target_y,"
        "target_coordinate_space,screen_width_px,screen_height_px,screen_width_cm,"
        "screen_height_cm,device_id,rotation_degrees,camera_intrinsics_path\n"
        "capture.mp4,webcam,p1,s1,2,960,540,screen_pixels,1920,1080,53.1,29.8,laptop-a,0,intrinsics.json\n"
        "capture.mp4,webcam,p1,s1,4,0.25,0.75,top_left_normalized,1920,1080,53.1,29.8,laptop-a,0,intrinsics.json\n",
        encoding="utf-8",
    )
    frames = np.stack(
        [np.full((4, 6, 3), index, dtype=np.uint8) for index in range(6)], axis=0
    )

    class FakeCapture:
        def __init__(self) -> None:
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            return {1: 6.0, 2: 30.0}.get(property_id, 0.0)

        def set(self, property_id: int, value: float) -> None:
            self.index = int(value)

        def read(self):
            return True, frames[self.index]

        def release(self) -> None:
            pass

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FRAME_COUNT=1,
        CAP_PROP_FPS=2,
        CAP_PROP_POS_FRAMES=3,
        COLOR_BGR2RGB=4,
        VideoCapture=lambda _: FakeCapture(),
        cvtColor=lambda frame, _: frame[..., ::-1],
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    config = DataConfig(
        source="manifest",
        manifest_path=str(manifest_path),
        dataset_version="manifest-test-v001",
        processed_dir="processed",
        train_from_processed=False,
        num_samples=0,
        num_participants=0,
        sequence_length=1,
        frame_stride=1,
        raw_height=0,
        raw_width=0,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    dataset = ManifestVideoDataSource(config, manifest_path).load()

    assert len(dataset.samples) == 2
    assert dataset.samples[0].frame_indices == (2,)
    assert dataset.samples[1].frame_indices == (4,)
    np.testing.assert_allclose(dataset.samples[0].target, [0.5, 0.5])
    assert dataset.samples[0].screen_size_cm == (53.1, 29.8)
    assert dataset.samples[0].device_id == "laptop-a"
    assert dataset.samples[0].camera_intrinsics_path == str(intrinsics_path)
