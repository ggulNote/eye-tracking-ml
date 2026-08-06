from __future__ import annotations

from typing import Callable, Dict

from ggulnote_ml.config import FeatureExtractionConfig
from ggulnote_ml.exceptions import ConfigurationError
from ggulnote_ml.features.base import FeatureExtractor
from ggulnote_ml.features.identity import IdentityFeatureExtractor
from ggulnote_ml.features.statistics import SpatialStatisticsFeatureExtractor


FeatureFactory = Callable[[str], FeatureExtractor]

FEATURE_REGISTRY: Dict[str, FeatureFactory] = {
    "identity": IdentityFeatureExtractor,
    "spatial_statistics": SpatialStatisticsFeatureExtractor,
}


def build_feature_extractor(config: FeatureExtractionConfig) -> FeatureExtractor:
    try:
        factory = FEATURE_REGISTRY[config.name]
    except KeyError as exc:
        raise ConfigurationError("Unknown feature extractor: %s" % config.name) from exc
    return factory(config.version)

