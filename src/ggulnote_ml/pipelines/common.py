from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ggulnote_ml.config import PipelineConfig
from ggulnote_ml.contracts import CanonicalBatch, DatasetSplits, FeatureBatch
from ggulnote_ml.data import build_data_source, split_by_participant
from ggulnote_ml.data.processed import ProcessedDatasetStore
from ggulnote_ml.features import build_feature_extractor
from ggulnote_ml.preprocessing import VideoPreprocessor
from ggulnote_ml.settings import resolve_data_root


@dataclass(frozen=True)
class PreparedSplit:
    canonical: CanonicalBatch
    features: FeatureBatch


@dataclass(frozen=True)
class PreparedSplits:
    train: PreparedSplit
    validation: PreparedSplit
    test: PreparedSplit
    raw: Optional[DatasetSplits]
    dataset_metadata: Dict[str, object]


def prepare_splits(config: PipelineConfig, project_root: Path) -> PreparedSplits:
    feature_extractor = build_feature_extractor(config.feature_extraction)
    if config.data.train_from_processed:
        data_root = resolve_data_root(project_root, required=True)
        assert data_root is not None
        stored = ProcessedDatasetStore(
            data_root, config.data, config.preprocessing
        ).load()

        def prepare_canonical(canonical: CanonicalBatch) -> PreparedSplit:
            return PreparedSplit(
                canonical=canonical,
                features=feature_extractor.transform(canonical),
            )

        return PreparedSplits(
            train=prepare_canonical(stored.train),
            validation=prepare_canonical(stored.validation),
            test=prepare_canonical(stored.test),
            raw=None,
            dataset_metadata=stored.metadata,
        )

    source = build_data_source(
        config=config.data,
        seed=config.training.seed,
        project_root=project_root,
    )
    raw_dataset = source.load()
    raw_splits = split_by_participant(
        dataset=raw_dataset,
        config=config.data,
        seed=config.training.seed,
    )
    preprocessor = VideoPreprocessor(config.data, config.preprocessing)

    def prepare(raw_split: object) -> PreparedSplit:
        canonical = preprocessor.transform(raw_split)
        features = feature_extractor.transform(canonical)
        return PreparedSplit(canonical=canonical, features=features)

    return PreparedSplits(
        train=prepare(raw_splits.train),
        validation=prepare(raw_splits.validation),
        test=prepare(raw_splits.test),
        raw=raw_splits,
        dataset_metadata={
            "dataset_version": config.data.dataset_version,
            "source": config.data.source,
            "storage": "in_memory",
            "sample_count": len(raw_dataset.samples),
            "participant_count": len(
                {sample.participant_id for sample in raw_dataset.samples}
            ),
            "preprocessor_version": preprocessor.version,
        },
    )
