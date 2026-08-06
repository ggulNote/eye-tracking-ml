from pathlib import Path

from ggulnote_ml.config import flatten_config, load_config


def test_base_config_is_valid() -> None:
    loaded = load_config(Path("configs/base.yaml"))
    assert loaded.config.project.task == "gaze_coordinate_regression"
    assert loaded.config.data.source == "synthetic"
    assert flatten_config(loaded.config)["model.name"] == "ridge"

