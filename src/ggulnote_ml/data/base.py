from __future__ import annotations

from abc import ABC, abstractmethod

from ggulnote_ml.contracts import RawDataset


class DataSource(ABC):
    @abstractmethod
    def load(self) -> RawDataset:
        """Load samples that satisfy the raw video contract."""

