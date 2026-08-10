"""Minimal, privacy-conscious MLflow tracking for data preparation.

MLflow is deliberately imported lazily.  A user who sets
``mlflow.enabled=false`` can validate and prepare data without MLflow being
installed, and this module never scans a data directory or uploads an image.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_PARAM_BATCH_SIZE = 50
_SENSITIVE_KEY_MARKERS = (
    "secret",
    "token",
    "password",
    "credential",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "authorization",
    "cookie",
)
_PATH_KEY_SUFFIXES = ("_path", "_root", "_dir")
_SPLIT_ARTIFACT_NAMES = ("split", "train", "validation", "test")


class MLflowTrackingError(RuntimeError):
    """Raised when preparation succeeded but its MLflow record did not."""


@dataclass(frozen=True, slots=True)
class DataPreparationTrackingResult:
    """Identity of the MLflow run created for one preparation operation."""

    run_id: str
    artifact_uri: str | None = None


def log_data_preparation(
    config: Mapping[str, Any], result: Any
) -> DataPreparationTrackingResult | None:
    """Log one completed data-preparation result to MLflow.

    Only resolved data/preprocessing/split parameters and the exact files
    returned by the preparation layer are logged.  No directory is traversed,
    so source face images cannot be picked up as incidental artifacts.
    """

    mlflow_config = config.get("mlflow", {})
    if not isinstance(mlflow_config, Mapping):
        raise MLflowTrackingError("config의 mlflow 항목은 YAML mapping이어야 합니다.")

    enabled = mlflow_config.get("enabled", False)
    if enabled is False:
        return None
    if enabled is not True:
        raise MLflowTrackingError("mlflow.enabled는 true 또는 false여야 합니다.")

    # Do not move this import to module scope: disabled tracking is a genuine
    # no-dependency path used by preparation-only environments.
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
    log_dataset_manifest = _boolean_flag(mlflow_config, "log_dataset_manifest")
    log_split_manifest = _boolean_flag(mlflow_config, "log_split_manifest")
    log_resolved_config = _boolean_flag(mlflow_config, "log_resolved_config")
    log_environment = _boolean_flag(mlflow_config, "log_environment")
    log_system_metrics = _boolean_flag(mlflow_config, "log_system_metrics")
    tags = _build_tags(mlflow_config)
    params = _build_preparation_params(config, result)
    summary = _remote_summary(result)
    artifacts = _preparation_artifacts(
        result,
        log_dataset_manifest=log_dataset_manifest,
        log_split_manifest=log_split_manifest,
    )
    resolved_config_artifacts = _resolved_config_artifacts(result) if log_resolved_config else ()

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(
            run_name=run_name,
            tags=tags,
            log_system_metrics=log_system_metrics,
        ) as active_run:
            if params:
                _log_params_in_batches(mlflow, params)
            mlflow.log_dict(summary, "data_preparation/summary.json")
            for local_path, artifact_path in artifacts:
                mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
            if log_resolved_config:
                # Keep the exact resolved YAML locally.  The MLflow copy drops
                # credentials and machine-specific paths so a remote server
                # does not receive usernames or dataset locations.
                mlflow.log_dict(
                    _redact_value(config),
                    "data_preparation/resolved_config.json",
                )
                for local_path, artifact_path in resolved_config_artifacts:
                    mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
            if log_environment:
                mlflow.log_dict(
                    _environment_snapshot(),
                    "data_preparation/environment.json",
                )

            run_id = _run_id(active_run)
            artifact_uri = _artifact_uri(active_run)
    except MLflowTrackingError:
        raise
    except Exception as exc:
        raise MLflowTrackingError(
            "MLflow server에 data-preparation 기록을 남기지 못했습니다. "
            "tracking URI, 인증 정보와 server 상태를 확인하세요. "
            "로컬 manifest는 이미 생성되어 있으며 그대로 유지됩니다. "
            f"원인 유형: {type(exc).__name__}"
        ) from exc

    return DataPreparationTrackingResult(run_id=run_id, artifact_uri=artifact_uri)


def _build_tags(mlflow_config: Mapping[str, Any]) -> dict[str, str]:
    configured = mlflow_config.get("tags", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise MLflowTrackingError("mlflow.tags는 key/value mapping이어야 합니다.")

    tags = {
        str(key): _string_value(_redact_value(value, key=str(key)))
        for key, value in configured.items()
    }
    # These values are owned by the tracker and intentionally override a
    # conflicting user tag, so preparation runs remain easy to identify.
    tags.update(
        {
            "pipeline_stage": "data_preparation",
            "run_type": "data_preparation",
            "component": "data",
            "raw_images_logged": "false",
        }
    )
    return tags


def _build_preparation_params(config: Mapping[str, Any], result: Any) -> dict[str, str]:
    data = config.get("data", {})
    preprocessing = config.get("preprocessing", {})
    if not isinstance(data, Mapping):
        raise MLflowTrackingError("config.data는 mapping이어야 합니다.")
    if not isinstance(preprocessing, Mapping):
        raise MLflowTrackingError("config.preprocessing은 mapping이어야 합니다.")

    split = data.get("split", {})
    if not isinstance(split, Mapping):
        raise MLflowTrackingError("config.data.split은 mapping이어야 합니다.")

    data_without_split = {key: value for key, value in data.items() if key != "split"}
    params: dict[str, str] = {}
    params.update(_flatten_params(data_without_split, prefix="data"))
    params.update(_flatten_params(preprocessing, prefix="preprocessing"))
    params.update(_flatten_params(split, prefix="split"))

    dataset_manifest_hash = getattr(result, "dataset_manifest_hash", None)
    if dataset_manifest_hash is None:
        dataset_manifest_hash = getattr(result, "dataset_hash", None)
    if dataset_manifest_hash:
        params["preparation.dataset_manifest_hash"] = str(dataset_manifest_hash)
    split_counts = getattr(result, "split_counts", None)
    if isinstance(split_counts, Mapping):
        for split_name, count in sorted(split_counts.items(), key=lambda item: str(item[0])):
            params[f"preparation.split_count.{split_name}"] = str(count)
    return params


def _flatten_params(value: Mapping[str, Any], *, prefix: str) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, child in sorted(value.items(), key=lambda item: str(item[0])):
        full_key = f"{prefix}.{key}"
        if isinstance(child, Mapping):
            flattened.update(_flatten_params(child, prefix=full_key))
        else:
            normalized_key = str(key).strip().lower().replace("-", "_")
            if _is_sensitive_key(normalized_key):
                flattened[full_key] = "<redacted>"
            elif _is_local_path_key(normalized_key) and isinstance(child, str | Path):
                flattened[full_key] = "<redacted-local-path>"
            else:
                flattened[full_key] = _string_value(_redact_value(child))
    return flattened


def _log_params_in_batches(mlflow: Any, params: Mapping[str, str]) -> None:
    """Stay below tracking-store batch limits without changing parameter keys."""

    items = sorted(params.items())
    for start in range(0, len(items), _PARAM_BATCH_SIZE):
        mlflow.log_params(dict(items[start : start + _PARAM_BATCH_SIZE]))


def _string_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise MLflowTrackingError(
            f"MLflow parameter로 변환할 수 없는 config 값입니다: {type(value).__name__}"
        ) from exc


def _remote_summary(result: Any) -> dict[str, Any]:
    """Build the non-identifying summary uploaded to the tracking server."""

    summary_path = _required_result_path(result, "summary_path")
    if summary_path.name != "summary.json":
        raise MLflowTrackingError(
            "data preparation summary artifact는 이름이 summary.json이어야 합니다."
        )

    raw_summary = getattr(result, "summary", None)
    if raw_summary is None:
        try:
            raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MLflowTrackingError(
                "data preparation summary.json을 안전하게 읽지 못했습니다."
            ) from exc
    if not isinstance(raw_summary, Mapping):
        raise MLflowTrackingError("data preparation summary는 JSON mapping이어야 합니다.")

    safe = _redact_value(raw_summary)
    if not isinstance(safe, dict):  # defensive: Mapping above always produces dict
        raise MLflowTrackingError("data preparation summary 비식별화에 실패했습니다.")

    dataset = safe.get("dataset")
    if isinstance(dataset, dict):
        dataset.pop("root", None)
        counts = dataset.get("counts")
        if isinstance(counts, dict):
            # Subject IDs are useful in the local summary, but even
            # pseudonymous IDs should not be copied to a remote tracker.
            for count_key in tuple(counts):
                if count_key == "by_subject" or count_key.endswith("_by_subject"):
                    counts.pop(count_key, None)
    split = safe.get("split")
    if isinstance(split, dict):
        partitions = split.get("partitions")
        if isinstance(partitions, dict):
            for partition in partitions.values():
                if isinstance(partition, dict):
                    partition.pop("group_ids", None)
    resolved = safe.get("resolved_config")
    if isinstance(resolved, dict):
        resolved.pop("path", None)
    artifact_summary = safe.get("artifacts")
    if isinstance(artifact_summary, dict):
        for artifact in artifact_summary.values():
            if isinstance(artifact, dict):
                artifact.pop("path", None)
    safe["privacy"] = {
        "local_paths_removed": True,
        "split_group_ids_removed": True,
        "subject_breakdowns_removed": True,
        "raw_images_logged": False,
    }
    return safe


def _preparation_artifacts(
    result: Any,
    *,
    log_dataset_manifest: bool,
    log_split_manifest: bool,
) -> tuple[tuple[Path, str], ...]:
    summary_path = _required_result_path(result, "summary_path")
    artifact_root = summary_path.resolve().parent

    selected_names: list[str] = []
    if log_dataset_manifest:
        selected_names.append("dataset")
    if log_split_manifest:
        selected_names.extend(_SPLIT_ARTIFACT_NAMES)
    if not selected_names:
        return ()

    manifests = _required_path_mapping(result, "manifest_paths")
    hashes = _required_path_mapping(result, "hash_paths")
    selected: list[tuple[Path, str]] = []

    for name in selected_names:
        path = manifests.get(name)
        if path is None:
            raise MLflowTrackingError(f"manifest_paths.{name} artifact가 필요합니다.")
        _validate_generated_artifact(
            path,
            expected_root=artifact_root,
            allowed_suffixes={".csv"},
            field=f"manifest_paths.{name}",
        )
        selected.append((path, "data_preparation/manifests"))

    for name in selected_names:
        path = hashes.get(name)
        if path is None:
            raise MLflowTrackingError(f"hash_paths.{name} sidecar가 필요합니다.")
        _validate_generated_artifact(
            path,
            expected_root=artifact_root,
            allowed_suffixes={".sha256"},
            field=f"hash_paths.{name}",
        )
        selected.append((path, "data_preparation/hashes"))

    return tuple(selected)


def _resolved_config_artifacts(result: Any) -> tuple[tuple[Path, str], ...]:
    config_value = getattr(result, "resolved_config_path", None)
    hash_value = getattr(result, "resolved_config_hash_path", None)
    if config_value is None and hash_value is None:
        return ()
    if not isinstance(config_value, Path) or not isinstance(hash_value, Path):
        raise MLflowTrackingError(
            "resolved config artifact와 hash sidecar는 둘 다 Path로 제공되어야 합니다."
        )

    summary_path = _required_result_path(result, "summary_path")
    expected_root = summary_path.resolve().parent.parent
    _validate_generated_artifact(
        config_value,
        expected_root=expected_root,
        allowed_suffixes={".yaml", ".yml"},
        field="resolved_config_path",
    )
    if config_value.name not in {"resolved_config.yaml", "resolved_config.yml"}:
        raise MLflowTrackingError(
            "resolved config artifact 이름은 resolved_config.yaml이어야 합니다."
        )
    _validate_generated_artifact(
        hash_value,
        expected_root=expected_root,
        allowed_suffixes={".sha256"},
        field="resolved_config_hash_path",
    )
    if hash_value.name != f"{config_value.name}.sha256":
        raise MLflowTrackingError(
            "resolved config hash sidecar 이름이 artifact와 일치하지 않습니다."
        )
    return ((hash_value, "data_preparation/hashes"),)


def _required_result_path(result: Any, attribute: str) -> Path:
    value = getattr(result, attribute, None)
    if not isinstance(value, Path):
        raise MLflowTrackingError(f"data preparation 결과에 Path 타입의 {attribute}가 필요합니다.")
    if not value.is_file():
        raise MLflowTrackingError(f"기록할 artifact 파일이 없습니다: {value}")
    return value


def _required_path_mapping(result: Any, attribute: str) -> dict[str, Path]:
    value = getattr(result, attribute, None)
    if not isinstance(value, Mapping) or not value:
        raise MLflowTrackingError(
            f"data preparation 결과에 비어 있지 않은 {attribute} mapping이 필요합니다."
        )
    paths: dict[str, Path] = {}
    for name, path in value.items():
        if not isinstance(name, str) or not isinstance(path, Path):
            raise MLflowTrackingError(f"{attribute}는 str -> Path mapping이어야 합니다.")
        paths[name] = path
    return paths


def _validate_generated_artifact(
    path: Path,
    *,
    expected_root: Path,
    allowed_suffixes: set[str],
    field: str,
) -> None:
    resolved = path.resolve()
    if not path.is_file():
        raise MLflowTrackingError(f"{field} 파일이 없습니다: {path}")
    if resolved.parent != expected_root:
        raise MLflowTrackingError(f"{field}가 preparation artifact 폴더 밖을 가리킵니다: {path}")
    if path.suffix.lower() not in allowed_suffixes:
        raise MLflowTrackingError(
            f"{field}의 확장자는 {sorted(allowed_suffixes)} 중 하나여야 합니다: {path}"
        )


def _boolean_flag(config: Mapping[str, Any], key: str) -> bool:
    value = config.get(key, False)
    if not isinstance(value, bool):
        raise MLflowTrackingError(f"mlflow.{key}는 true 또는 false여야 합니다.")
    return value


def _redact_value(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _is_sensitive_key(key):
        return "<redacted>"
    if key is not None and _is_local_path_key(key) and isinstance(value, str | Path):
        return "<redacted-local-path>"
    if isinstance(value, Mapping):
        return {
            str(child_key): _redact_value(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_redact_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return _redact_uri_userinfo(value)
    if value is None or isinstance(value, bool | int | float):
        return value
    raise MLflowTrackingError(
        f"MLflow artifact로 안전하게 직렬화할 수 없는 값입니다: {type(value).__name__}"
    )


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(marker in normalized for marker in _SENSITIVE_KEY_MARKERS)


def _is_local_path_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return (
        normalized == "path"
        or normalized == "root"
        or normalized == "dir"
        or normalized.endswith(_PATH_KEY_SUFFIXES)
    )


def _redact_uri_userinfo(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme:
        return value
    netloc = parsed.netloc
    if "@" in netloc:
        netloc = f"<redacted>@{netloc.rsplit('@', 1)[1]}"
    query = urlencode(
        [
            (name, "<redacted>" if _is_sensitive_key(name) else query_value)
            for name, query_value in parse_qsl(parsed.query, keep_blank_values=True)
        ]
    )
    fragment = "<redacted>" if parsed.fragment else ""
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, fragment))


def _environment_snapshot() -> dict[str, Any]:
    distributions = (
        "mlflow",
        "psutil",
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "Pillow",
        "opencv-contrib-python",
        "opencv-python",
        "PyYAML",
    )
    packages: dict[str, str] = {}
    for distribution in distributions:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "schema_version": 1,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
    }


def _required_text(config: Mapping[str, Any], key: str, field: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MLflowTrackingError(f"{field}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def _run_id(active_run: Any) -> str:
    info = getattr(active_run, "info", None)
    run_id = getattr(info, "run_id", None)
    if not isinstance(run_id, str) or not run_id:
        raise MLflowTrackingError("MLflow가 생성한 run_id를 확인하지 못했습니다.")
    return run_id


def _artifact_uri(active_run: Any) -> str | None:
    info = getattr(active_run, "info", None)
    value = getattr(info, "artifact_uri", None)
    return value if isinstance(value, str) and value else None


__all__ = [
    "DataPreparationTrackingResult",
    "MLflowTrackingError",
    "log_data_preparation",
]
