from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Optional, Type

import numpy as np

from ggulnote_ml.models.base import GazeRegressor


class RunTracker(ABC):
    run_id: str

    @abstractmethod
    def __enter__(self) -> "RunTracker":
        pass

    @abstractmethod
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        pass

    @abstractmethod
    def log_params(self, values: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def log_metrics(self, values: Dict[str, float], prefix: str = "") -> None:
        pass

    @abstractmethod
    def log_json(self, value: Dict[str, Any], artifact_file: str) -> None:
        pass

    @abstractmethod
    def log_artifact(self, path: Path, artifact_path: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def log_model(
        self,
        model: GazeRegressor,
        model_path: Path,
        input_example: np.ndarray,
        output_example: np.ndarray,
        registered_model_name: Optional[str],
    ) -> None:
        pass

