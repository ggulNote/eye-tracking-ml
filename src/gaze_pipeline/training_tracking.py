"""MLflow tracking session for model training and evaluation.

The data-preparation tracker intentionally owns a separate MLflow run.  This
module provides the corresponding training run without importing MLflow when
tracking is disabled.  Callers explicitly choose every checkpoint/model file
that is uploaded; directories and image files are never traversed or logged.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any

from gaze_pipeline.tracking import (
    MLflowTrackingError,
    _artifact_uri,
    _boolean_flag,
    _environment_snapshot,
    _flatten_params,
    _log_params_in_batches,
    _redact_value,
    _required_text,
    _run_id,
    _string_value,
)

_TRAINING_PARAM_SECTIONS = (
    "experiment",
    "task",
    "data",
    "preprocessing",
    "model",
    "fusion",
    "training",
    "optimizer",
    "scheduler",
    "loss",
    "metrics",
    "checkpoint",
    "model_export",
)
_ARTIFACT_KIND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ARTIFACT_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RAW_IMAGE_SUFFIXES = frozenset(
    {".avif", ".bmp", ".gif", ".heic", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


@dataclass(frozen=True, slots=True)
class LoggedTrainingArtifact:
    """Metadata recorded beside an explicitly selected training artifact."""

    path: Path
    artifact_path: str
    sha256: str
    size_bytes: int


@dataclass(slots=True)
class TrainingTrackingSession:
    """One active MLflow training run, or a no-op session when disabled."""

    enabled: bool
    run_id: str | None = None
    artifact_uri: str | None = None
    _mlflow: Any = field(default=None, repr=False)
    _active: bool = field(default=True, repr=False)

    def log_epoch(
        self,
        epoch: int,
        *,
        train_loss: Real | None = None,
        val_loss: Real | None = None,
        train_metrics: Mapping[str, Real] | None = None,
        val_metrics: Mapping[str, Real] | None = None,
    ) -> None:
        """Log train/validation loss and metrics using ``epoch`` as MLflow step."""

        self._ensure_active()
        if not self.enabled:
            return

        metrics: dict[str, Real] = {}
        if train_loss is not None:
            metrics["train/loss"] = train_loss
        if val_loss is not None:
            metrics["val/loss"] = val_loss
        metrics.update(_namespaced_metrics(train_metrics, namespace="train"))
        metrics.update(_namespaced_metrics(val_metrics, namespace="val"))
        self.log_metrics(metrics, step=epoch)

    def log_metrics(self, metrics: Mapping[str, Real], *, step: int = 0) -> None:
        """Log metrics whose complete MLflow names are supplied by the caller."""

        self._ensure_active()
        if not self.enabled:
            return
        normalized_step = _metric_step(step)
        normalized_metrics = _metric_values(metrics)

        if not normalized_metrics:
            return
        try:
            self._mlflow.log_metrics(normalized_metrics, step=normalized_step)
        except Exception as exc:
            raise _tracking_operation_error("loss/metric", exc) from exc

    def log_checkpoint(
        self,
        path: str | Path,
        *,
        kind: str = "checkpoint",
    ) -> LoggedTrainingArtifact | None:
        """Upload one checkpoint file plus its SHA-256 metadata."""

        normalized_kind = _artifact_kind(kind)
        return self.log_artifact(
            path,
            f"training/checkpoints/{normalized_kind}",
            normalized_kind,
        )

    def log_model_artifact(
        self,
        path: str | Path,
        *,
        kind: str = "final",
    ) -> LoggedTrainingArtifact | None:
        """Upload one exported model file plus its SHA-256 metadata."""

        normalized_kind = _artifact_kind(kind)
        return self.log_artifact(path, f"training/model/{normalized_kind}", normalized_kind)

    def log_artifact(
        self,
        path: str | Path,
        artifact_path: str,
        kind: str = "artifact",
    ) -> LoggedTrainingArtifact | None:
        """Upload one explicitly selected non-image file to a safe artifact path."""

        self._ensure_active()

        # Disabled tracking must remain a true no-op: a training-only machine
        # can call the same hooks without MLflow or artifact files installed.
        if not self.enabled:
            return None

        normalized_kind = _artifact_kind(kind)
        normalized_artifact_path = _artifact_path(artifact_path)
        local_path = Path(path).expanduser()
        if not local_path.is_file():
            raise MLflowTrackingError(
                f"MLflow에 기록할 학습 artifact 파일이 없습니다: {local_path}"
            )
        if local_path.suffix.lower() in _RAW_IMAGE_SUFFIXES:
            raise MLflowTrackingError(
                f"원본 이미지로 보이는 파일은 training artifact로 기록할 수 없습니다: {local_path}"
            )

        metadata = LoggedTrainingArtifact(
            path=local_path,
            artifact_path=normalized_artifact_path,
            sha256=_sha256(local_path),
            size_bytes=local_path.stat().st_size,
        )
        try:
            self._mlflow.log_artifact(str(local_path), artifact_path=normalized_artifact_path)
            self._mlflow.log_dict(
                {
                    "schema_version": 1,
                    "filename": local_path.name,
                    "sha256": metadata.sha256,
                    "size_bytes": metadata.size_bytes,
                    "kind": normalized_kind,
                },
                f"{normalized_artifact_path}/{local_path.name}.metadata.json",
            )
        except Exception as exc:
            raise _tracking_operation_error("checkpoint/model artifact", exc) from exc
        return metadata

    def _ensure_active(self) -> None:
        if not self._active:
            raise MLflowTrackingError("종료된 MLflow training session에는 기록할 수 없습니다.")


@contextmanager
def training_run(
    config: Mapping[str, Any],
    *,
    extra_params: Mapping[str, Any] | None = None,
    run_type: str = "training",
    extra_tags: Mapping[str, Any] | None = None,
) -> Iterator[TrainingTrackingSession]:
    """Open a distinct MLflow run for training/evaluation work.

    ``mlflow.enabled=false`` yields a no-op session and does not import MLflow.
    Errors raised by the caller's training body are preserved; only tracking
    setup/logging/finalization failures become :class:`MLflowTrackingError`.
    """

    mlflow_config = config.get("mlflow", {})
    if not isinstance(mlflow_config, Mapping):
        raise MLflowTrackingError("config의 mlflow 항목은 YAML mapping이어야 합니다.")
    enabled = mlflow_config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise MLflowTrackingError("mlflow.enabled는 true 또는 false여야 합니다.")

    if enabled is False:
        session = TrainingTrackingSession(enabled=False)
        try:
            yield session
        finally:
            session._active = False
        return

    try:
        mlflow = importlib.import_module("mlflow")
    except ImportError as exc:
        raise MLflowTrackingError(
            "MLflow 기록이 켜져 있지만 mlflow package를 불러오지 못했습니다. "
            "requirements.txt를 설치하거나 mlflow.enabled=false로 설정하세요."
        ) from exc

    tracking_uri = _required_text(mlflow_config, "tracking_uri", "mlflow.tracking_uri")
    experiment_name = _required_text(mlflow_config, "experiment_name", "mlflow.experiment_name")
    run_name = _required_text(mlflow_config, "run_name", "mlflow.run_name")
    log_system_metrics = _boolean_flag(mlflow_config, "log_system_metrics")
    log_resolved_config = _boolean_flag(mlflow_config, "log_resolved_config")
    log_environment = _boolean_flag(mlflow_config, "log_environment")
    artifact_namespace = _tag_token(run_type, field="run_type")
    tags = _training_tags(
        mlflow_config,
        run_type=run_type,
        extra_tags=extra_tags,
    )
    params = _training_params(config, extra_params=extra_params)

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        run_manager = mlflow.start_run(
            run_name=run_name,
            tags=tags,
            log_system_metrics=log_system_metrics,
        )
        active_run = run_manager.__enter__()
    except Exception as exc:
        raise _tracking_operation_error("training run 시작", exc) from exc

    try:
        session = TrainingTrackingSession(
            enabled=True,
            run_id=_run_id(active_run),
            artifact_uri=_artifact_uri(active_run),
            _mlflow=mlflow,
        )
    except Exception as exc:
        try:
            run_manager.__exit__(*sys.exc_info())
        except Exception:
            pass
        if isinstance(exc, MLflowTrackingError):
            raise
        raise _tracking_operation_error("training run identity", exc) from exc

    try:
        if params:
            _log_params_in_batches(mlflow, params)
        if log_resolved_config:
            mlflow.log_dict(_redact_value(config), f"{artifact_namespace}/resolved_config.json")
        if log_environment:
            mlflow.log_dict(_environment_snapshot(), f"{artifact_namespace}/environment.json")
    except MLflowTrackingError:
        session._active = False
        try:
            run_manager.__exit__(*sys.exc_info())
        except Exception:
            pass
        raise
    except Exception as exc:
        session._active = False
        error = _tracking_operation_error("training parameter/config", exc)
        try:
            run_manager.__exit__(type(error), error, error.__traceback__)
        except Exception:
            pass
        raise error from exc

    try:
        yield session
    except BaseException:
        session._active = False
        caller_error = sys.exc_info()
        try:
            run_manager.__exit__(*caller_error)
        except Exception:
            # An MLflow finalization failure must not hide a model/training
            # exception raised by the caller's body.
            pass
        raise
    else:
        session._active = False
        try:
            run_manager.__exit__(None, None, None)
        except Exception as exc:
            raise _tracking_operation_error("training run 종료", exc) from exc


def _training_tags(
    mlflow_config: Mapping[str, Any],
    *,
    run_type: str,
    extra_tags: Mapping[str, Any] | None,
) -> dict[str, str]:
    normalized_run_type = _tag_token(run_type, field="run_type")
    configured = mlflow_config.get("tags", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise MLflowTrackingError("mlflow.tags는 key/value mapping이어야 합니다.")
    tags = {
        str(key): _string_value(_redact_value(value, key=str(key)))
        for key, value in configured.items()
    }
    if extra_tags is not None:
        if not isinstance(extra_tags, Mapping):
            raise MLflowTrackingError("training extra_tags는 mapping이어야 합니다.")
        tags.update(
            {
                str(key): _string_value(_redact_value(value, key=str(key)))
                for key, value in extra_tags.items()
            }
        )
    tags.update(
        {
            "pipeline_stage": normalized_run_type,
            "run_type": normalized_run_type,
            "component": "model",
            "raw_images_logged": "false",
        }
    )
    return tags


def _training_params(
    config: Mapping[str, Any],
    *,
    extra_params: Mapping[str, Any] | None,
) -> dict[str, str]:
    params: dict[str, str] = {}
    schema_version = config.get("schema_version")
    if schema_version is not None:
        params["schema_version"] = _string_value(_redact_value(schema_version))

    for section in _TRAINING_PARAM_SECTIONS:
        value = config.get(section)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise MLflowTrackingError(f"config.{section}는 mapping이어야 합니다.")
        params.update(_flatten_params(value, prefix=section))

    if extra_params is not None:
        if not isinstance(extra_params, Mapping):
            raise MLflowTrackingError("training extra_params는 mapping이어야 합니다.")
        params.update(_flatten_params(extra_params, prefix="runtime"))
    return params


def _namespaced_metrics(
    values: Mapping[str, Real] | None,
    *,
    namespace: str,
) -> dict[str, Real]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise MLflowTrackingError(f"{namespace}_metrics는 metric name/value mapping이어야 합니다.")
    result: dict[str, Real] = {}
    for raw_name, raw_value in values.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise MLflowTrackingError("MLflow metric 이름은 비어 있지 않은 문자열이어야 합니다.")
        name = raw_name.strip().strip("/")
        if not name or name == "loss":
            raise MLflowTrackingError(
                f"{namespace}_metrics의 'loss'는 예약 이름입니다. *_loss 인자를 사용하세요."
            )
        result[f"{namespace}/{name}"] = raw_value
    return result


def _metric_values(values: Mapping[str, Real]) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise MLflowTrackingError("metrics는 metric name/value mapping이어야 합니다.")
    result: dict[str, float] = {}
    for raw_name, raw_value in values.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise MLflowTrackingError("MLflow metric 이름은 비어 있지 않은 문자열이어야 합니다.")
        name = raw_name.strip().strip("/")
        if not name:
            raise MLflowTrackingError("MLflow metric 이름은 비어 있지 않은 문자열이어야 합니다.")
        result[name] = _metric_value(raw_value, name)
    return result


def _metric_step(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MLflowTrackingError("MLflow metric step은 0 이상의 정수여야 합니다.")
    return value


def _metric_value(value: Real, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise MLflowTrackingError(f"MLflow metric {name} 값은 실수여야 합니다.")
    converted = float(value)
    if not math.isfinite(converted):
        raise MLflowTrackingError(f"MLflow metric {name} 값은 유한한 실수여야 합니다.")
    return converted


def _artifact_kind(value: str) -> str:
    if not isinstance(value, str) or not _ARTIFACT_KIND_RE.fullmatch(value):
        raise MLflowTrackingError(
            "training artifact kind는 영문/숫자로 시작하는 안전한 이름이어야 합니다."
        )
    return value


def _artifact_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        raise MLflowTrackingError("MLflow artifact_path는 안전한 상대 POSIX 경로여야 합니다.")
    text = value.strip()
    raw_parts = text.split("/")
    if any(
        part in {"", ".", ".."} or not _ARTIFACT_PATH_SEGMENT_RE.fullmatch(part)
        for part in raw_parts
    ):
        raise MLflowTrackingError("MLflow artifact_path는 안전한 상대 POSIX 경로여야 합니다.")
    path = PurePosixPath(text)
    if path.is_absolute():
        raise MLflowTrackingError("MLflow artifact_path는 안전한 상대 POSIX 경로여야 합니다.")
    return path.as_posix()


def _tag_token(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _ARTIFACT_KIND_RE.fullmatch(value):
        raise MLflowTrackingError(
            f"training {field}은 영문/숫자로 시작하는 안전한 이름이어야 합니다."
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MLflowTrackingError(f"학습 artifact SHA-256을 계산하지 못했습니다: {path}") from exc
    return digest.hexdigest()


def _tracking_operation_error(operation: str, exc: BaseException) -> MLflowTrackingError:
    return MLflowTrackingError(
        f"MLflow server에 {operation} 기록을 남기지 못했습니다. "
        "tracking URI, 인증 정보와 server 상태를 확인하세요. "
        f"원인 유형: {type(exc).__name__}"
    )


__all__ = [
    "LoggedTrainingArtifact",
    "TrainingTrackingSession",
    "training_run",
]
