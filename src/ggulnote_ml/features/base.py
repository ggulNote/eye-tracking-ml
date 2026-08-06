from __future__ import annotations

from abc import ABC, abstractmethod

from ggulnote_ml.contracts import CanonicalBatch, FeatureBatch


class FeatureExtractor(ABC):
    name: str
    version: str

    @abstractmethod
    def transform(self, batch: CanonicalBatch) -> FeatureBatch:
        """Convert canonical BTCHW frames into BTF features."""

