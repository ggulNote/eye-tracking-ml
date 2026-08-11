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
    assert config.dataset.calibration_source_directory.name == "macbook_air_m5_13_iphone16"
    assert config.display.canvas_height == 1248
    assert config.protocols.train_static.columns == 3
    assert config.protocols.train_static.rows == 9
    assert config.protocols.train_static.repeats == 1
    assert config.protocols.evaluation_static.columns == 3
    assert config.protocols.evaluation_static.rows == 6
    assert len(config.protocols.evaluation_static.y_positions) == 6
    assert len(config.protocols.dynamic.columns) == 3
    assert config.protocols.dynamic.movement_duration_ms == 5000
    assert config.protocols.dynamic.edge_exclusion_ms == 400


def test_capture_config_rejects_duplicate_device_indices(tmp_path):
    with Path("configs/capture.yaml").open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    raw["cameras"][1]["device_index"] = raw["cameras"][0]["device_index"]
    config_path = tmp_path / "capture.yaml"
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="indices must be unique"):
        load_capture_config(config_path)
