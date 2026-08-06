from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class GazeRegressor(ABC):
    name: str
    version: str

    @abstractmethod
    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Fit on features [B,T,F] and normalized targets [B,2]."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return normalized gaze coordinates with shape [B,2]."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist all state required for inference."""

