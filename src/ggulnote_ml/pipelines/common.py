from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ggulnote_ml.config import PipelineConfig
from ggulnote_ml.contracts import CanonicalBatch, DatasetSplits, FeatureBatch
from ggulnote_ml.data import build_data_source, split_by_participant
from ggulnote_ml.features import build_feature_extractor
from ggulnote_ml.preprocessing import VideoPreprocessor


@dataclass(frozen=True)
class PreparedSplit:
    canonical: CanonicalBatch
    features: FeatureBatch


@dataclass(frozen=True)
class PreparedSplits:
    train: PreparedSplit
    validation: PreparedSplit
    test: PreparedSplit
    raw: DatasetSplits


def prepare_splits(config: PipelineConfig, project_root: Path) -> PreparedSplits:
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
    feature_extractor = build_feature_extractor(config.feature_extraction)

    def prepare(raw_split: object) -> PreparedSplit:
        canonical = preprocessor.transform(raw_split)
        features = feature_extractor.transform(canonical)
        return PreparedSplit(canonical=canonical, features=features)

    return PreparedSplits(
        train=prepare(raw_splits.train),
        validation=prepare(raw_splits.validation),
        test=prepare(raw_splits.test),
        raw=raw_splits,
    )

