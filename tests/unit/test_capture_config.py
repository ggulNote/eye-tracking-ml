from pathlib import Path

import pytest
import yaml

from ggulnote_ml.capture.config import load_capture_config


def test_capture_config_loads_required_camera_roles_and_protocols():
    config = load_capture_config(Path("configs/capture.yaml"))

    assert [(camera.role, camera.device_index) for camera in config.cameras] == [
        ("webcam_front", 1),
        ("iphone_left", 0),
    ]
    assert config.protocols.train_static.columns == 3
    assert config.protocols.train_static.rows == 9
    assert config.protocols.train_static.repeats == 1
    assert config.protocols.train_static.confirmation_required
    assert config.protocols.train_static.minimum_fixation_ms == 400
    assert config.protocols.train_static.capture_duration_ms == 650
    assert config.protocols.evaluation_static.columns == 3
    assert config.protocols.evaluation_static.rows == 6
    assert len(config.protocols.evaluation_static.y_positions) == 6
    assert len(config.protocols.vertical_click.columns) == 3
    assert config.protocols.vertical_click.rows == 6
    assert config.protocols.vertical_click.confirmation_required
    assert config.frame_capture.enabled
    assert config.frame_capture.image_format == "jpg"
    assert config.frame_capture.jpeg_quality == 95
    assert config.frame_capture.samples_per_target == 1
    assert config.frame_capture.eye_open_weight > config.frame_capture.face_weight
    assert config.frame_capture.mediapipe_ready_weight == pytest.approx(1000.0)


def test_capture_config_rejects_duplicate_device_indices(tmp_path):
    with Path("configs/capture.yaml").open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    raw["cameras"][1]["device_index"] = raw["cameras"][0]["device_index"]
    config_path = tmp_path / "capture.yaml"
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="indices must be unique"):
        load_capture_config(config_path)
