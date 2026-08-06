from pathlib import Path

import numpy as np

from ggulnote_ml.data import build_data_source
from ggulnote_ml.features import build_feature_extractor
from ggulnote_ml.preprocessing import VideoPreprocessor


def test_preprocessing_and_feature_shapes(pipeline_config, tmp_path: Path) -> None:
    dataset = build_data_source(
        pipeline_config.data,
        pipeline_config.training.seed,
        tmp_path,
    ).load()
    canonical = VideoPreprocessor(
        pipeline_config.data,
        pipeline_config.preprocessing,
    ).transform(dataset)
    features = build_feature_extractor(pipeline_config.feature_extraction).transform(canonical)

    assert canonical.frames.shape == (36, 2, 3, 8, 10)
    assert canonical.frames.dtype == np.float32
    assert features.values.shape == (36, 2, 240)
    assert features.targets.shape == (36, 2)

