from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from ggulnote_ml.config import DataConfig
from ggulnote_ml.data.base import DataSource
from ggulnote_ml.data.manifest import ManifestVideoDataSource
from ggulnote_ml.data.synthetic import SyntheticGazeDataSource
from ggulnote_ml.exceptions import ConfigurationError


DataSourceFactory = Callable[[DataConfig, int, Path], DataSource]


def _synthetic(config: DataConfig, seed: int, project_root: Path) -> DataSource:
    del project_root
    return SyntheticGazeDataSource(config=config, seed=seed)


def _manifest(config: DataConfig, seed: int, project_root: Path) -> DataSource:
    del seed
    manifest_path = Path(config.manifest_path or "")
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    return ManifestVideoDataSource(config=config, manifest_path=manifest_path.resolve())


DATA_SOURCE_REGISTRY: Dict[str, DataSourceFactory] = {
    "synthetic": _synthetic,
    "manifest": _manifest,
}


def build_data_source(config: DataConfig, seed: int, project_root: Path) -> DataSource:
    try:
        factory = DATA_SOURCE_REGISTRY[config.source]
    except KeyError as exc:
        raise ConfigurationError("Unknown data source: %s" % config.source) from exc
    return factory(config, seed, project_root)

