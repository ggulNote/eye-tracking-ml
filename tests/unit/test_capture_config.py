from pathlib import Path

import pytest

from ggulnote_ml.capture.config import load_capture_config


def test_capture_config_loads_two_unique_cameras():
    config = load_capture_config(Path("configs/capture.yaml"))

    assert [(camera.name, camera.device_index) for camera in config.cameras] == [
        ("macbook", 0),
        ("iphone", 1),
    ]
    assert config.recording.output_directory.name == "captures"


def test_capture_config_rejects_duplicate_device_indices(tmp_path):
    config_path = tmp_path / "capture.yaml"
    config_path.write_text(
        """cameras:
  - {name: first, device_index: 0, backend: auto, width: 640, height: 480, fps: 30, warmup_frames: 0, mirror: false}
  - {name: second, device_index: 0, backend: auto, width: 640, height: 480, fps: 30, warmup_frames: 0, mirror: false}
preview: {enabled: false, window_name_prefix: test}
recording: {enabled: false, output_directory: output, video_codec: mp4v}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="indices must be unique"):
        load_capture_config(config_path)
