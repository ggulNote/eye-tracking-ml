from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ggulnote_ml.config import PipelineConfig
from ggulnote_ml.data import build_data_source, split_by_participant
from ggulnote_ml.data.processed import ProcessedDatasetStore
from ggulnote_ml.preprocessing import VideoPreprocessor
from ggulnote_ml.settings import resolve_data_path, resolve_data_root


@dataclass(frozen=True)
class PreprocessingResult:
    dataset_dir: Path
    metadata: Dict[str, object]


def run_preprocessing(
    config: PipelineConfig,
    project_root: Path,
    force: bool = False,
) -> PreprocessingResult:
    config.validate()
    data_root = resolve_data_root(project_root, required=True)
    assert data_root is not None
    data_root.mkdir(parents=True, exist_ok=True)
    source = build_data_source(
        config=config.data,
        seed=config.training.seed,
        project_root=project_root,
        data_root=data_root,
    )
    raw_dataset = source.load()
    raw_splits = split_by_participant(
        dataset=raw_dataset,
        config=config.data,
        seed=config.training.seed,
    )
    preprocessor = VideoPreprocessor(config.data, config.preprocessing)
    canonical_splits = {
        "train": preprocessor.transform(raw_splits.train),
        "validation": preprocessor.transform(raw_splits.validation),
        "test": preprocessor.transform(raw_splits.test),
    }
    manifest_path: Optional[Path] = None
    if config.data.source == "manifest":
        manifest_path = resolve_data_path(config.data.manifest_path or "", data_root)
    store = ProcessedDatasetStore(data_root, config.data, config.preprocessing)
    metadata = store.save(
        canonical_splits,
        manifest_path=manifest_path,
        preprocessor_version=preprocessor.version,
        force=force,
    )
    return PreprocessingResult(dataset_dir=store.version_dir, metadata=metadata)
