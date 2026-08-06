from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

from ggulnote_ml.config import DataConfig
from ggulnote_ml.data.base import DataSource
from ggulnote_ml.data.manifest import ManifestVideoDataSource
from ggulnote_ml.data.synthetic import SyntheticGazeDataSource
from ggulnote_ml.exceptions import ConfigurationError
from ggulnote_ml.settings import resolve_data_path, resolve_data_root


DataSourceFactory = Callable[[DataConfig, int, Path, Optional[Path]], DataSource]


def _synthetic(
    config: DataConfig, seed: int, project_root: Path, data_root: Optional[Path]
) -> DataSource:
    del project_root, data_root
    return SyntheticGazeDataSource(config=config, seed=seed)


def _manifest(
    config: DataConfig, seed: int, project_root: Path, data_root: Optional[Path]
) -> DataSource:
    del seed
    root = data_root or resolve_data_root(project_root, required=True)
    assert root is not None
    manifest_path = resolve_data_path(config.manifest_path or "", root)
    return ManifestVideoDataSource(config=config, manifest_path=manifest_path)


DATA_SOURCE_REGISTRY: Dict[str, DataSourceFactory] = {
    "synthetic": _synthetic,
    "manifest": _manifest,
}


def build_data_source(
    config: DataConfig,
    seed: int,
    project_root: Path,
    data_root: Optional[Path] = None,
) -> DataSource:
    try:
        factory = DATA_SOURCE_REGISTRY[config.source]
    except KeyError as exc:
        raise ConfigurationError("Unknown data source: %s" % config.source) from exc
    return factory(config, seed, project_root, data_root)
