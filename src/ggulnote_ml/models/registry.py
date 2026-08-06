from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from ggulnote_ml.config import ModelConfig
from ggulnote_ml.exceptions import ConfigurationError
from ggulnote_ml.models.base import GazeRegressor
from ggulnote_ml.models.ridge import RidgeGazeRegressor


ModelFactory = Callable[[ModelConfig], GazeRegressor]


def _ridge(config: ModelConfig) -> GazeRegressor:
    return RidgeGazeRegressor(alpha=config.alpha, version=config.version)


MODEL_REGISTRY: Dict[str, ModelFactory] = {
    "ridge": _ridge,
}


def build_model(config: ModelConfig) -> GazeRegressor:
    try:
        factory = MODEL_REGISTRY[config.name]
    except KeyError as exc:
        raise ConfigurationError("Unknown model: %s" % config.name) from exc
    return factory(config)


def load_model(name: str, path: Path) -> GazeRegressor:
    if name == "ridge":
        return RidgeGazeRegressor.load(path)
    raise ConfigurationError("No model loader registered for: %s" % name)
