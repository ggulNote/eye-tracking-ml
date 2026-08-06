from __future__ import annotations

import inspect
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Optional, Type

import numpy as np

from ggulnote_ml.config import MlflowConfig
from ggulnote_ml.exceptions import OptionalDependencyError
from ggulnote_ml.models.base import GazeRegressor
from ggulnote_ml.models.ridge import RidgeGazeRegressor
from ggulnote_ml.tracking.base import RunTracker


class NullRunTracker(RunTracker):
    def __init__(self) -> None:
        self.run_id = "local-%s" % uuid.uuid4().hex[:12]

    def __enter__(self) -> "NullRunTracker":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        return None

    def log_params(self, values: Dict[str, Any]) -> None:
        del values

    def log_metrics(self, values: Dict[str, float], prefix: str = "") -> None:
        del values, prefix

    def log_json(self, value: Dict[str, Any], artifact_file: str) -> None:
        del value, artifact_file

    def log_artifact(self, path: Path, artifact_path: Optional[str] = None) -> None:
        del path, artifact_path

    def log_model(
        self,
        model: GazeRegressor,
        model_path: Path,
        input_example: np.ndarray,
        output_example: np.ndarray,
        registered_model_name: Optional[str],
    ) -> None:
        del model, model_path, input_example, output_example, registered_model_name


class MlflowRunTracker(RunTracker):
    def __init__(self, config: MlflowConfig, tags: Dict[str, str], project_root: Path) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise OptionalDependencyError(
                "MLflow tracking is enabled but mlflow is not installed. Run 'make setup'."
            ) from exc
        self.mlflow = mlflow
        self.config = config
        self.tags = tags
        self.project_root = project_root.resolve()
        self.run_id = ""

    def __enter__(self) -> "MlflowRunTracker":
        self.mlflow.set_tracking_uri(self._tracking_uri())
        self.mlflow.set_experiment(self.config.experiment_name)
        active_run = self.mlflow.start_run(run_name=self.config.run_name, tags=self.tags)
        self.run_id = active_run.info.run_id
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        status = "FAILED" if exc_type else "FINISHED"
        self.mlflow.end_run(status=status)

    def log_params(self, values: Dict[str, Any]) -> None:
        serializable = {key: "null" if value is None else value for key, value in values.items()}
        self.mlflow.log_params(serializable)

    def log_metrics(self, values: Dict[str, float], prefix: str = "") -> None:
        metrics = {
            "%s%s" % (prefix, key): float(value)
            for key, value in values.items()
        }
        self.mlflow.log_metrics(metrics)

    def log_json(self, value: Dict[str, Any], artifact_file: str) -> None:
        self.mlflow.log_dict(value, artifact_file)

    def log_artifact(self, path: Path, artifact_path: Optional[str] = None) -> None:
        self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_model(
        self,
        model: GazeRegressor,
        model_path: Path,
        input_example: np.ndarray,
        output_example: np.ndarray,
        registered_model_name: Optional[str],
    ) -> None:
        if not isinstance(model, RidgeGazeRegressor):
            self.log_artifact(model_path, artifact_path="model")
            return

        mlflow = self.mlflow

        class RidgePyFuncModel(mlflow.pyfunc.PythonModel):
            # An explicit tensor signature is provided below. Skipping MLflow's
            # list-oriented type-hint wrapper preserves ndarray inputs.
            _skip_type_hint_validation = True

            def load_context(self, context: Any) -> None:
                self._model = RidgeGazeRegressor.load(Path(context.artifacts["weights"]))

            def predict(self, context, model_input, params=None):
                del context, params
                return self._model.predict(np.asarray(model_input, dtype=np.float32))

        signature = mlflow.models.infer_signature(input_example, output_example)
        kwargs: Dict[str, Any] = {
            "python_model": RidgePyFuncModel(),
            "artifacts": {"weights": str(model_path)},
            "signature": signature,
            "input_example": input_example,
            "pip_requirements": ["numpy>=1.26,<3"],
            "code_paths": [str(self.project_root / "src")],
        }
        parameters = inspect.signature(mlflow.pyfunc.log_model).parameters
        if "name" in parameters:
            kwargs["name"] = "model"
        else:
            kwargs["artifact_path"] = "model"
        if registered_model_name:
            kwargs["registered_model_name"] = registered_model_name
        mlflow.pyfunc.log_model(**kwargs)

    def _tracking_uri(self) -> str:
        prefix = "sqlite:///"
        if not self.config.tracking_uri.startswith(prefix):
            return self.config.tracking_uri
        database_path = Path(self.config.tracking_uri[len(prefix) :])
        if database_path.is_absolute():
            return self.config.tracking_uri
        return "%s%s" % (prefix, self.project_root / database_path)
