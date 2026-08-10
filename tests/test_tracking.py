from __future__ import annotations

import json
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gaze_pipeline import cli
from gaze_pipeline.tracking import (
    DataPreparationTrackingResult,
    MLflowTrackingError,
    log_data_preparation,
)


class _FakeActiveRun:
    def __init__(self) -> None:
        self.info = SimpleNamespace(run_id="prepare-run-123", artifact_uri="memory://artifacts")

    def __enter__(self) -> _FakeActiveRun:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.run_kwargs: dict[str, Any] = {}
        self.params: dict[str, str] = {}
        self.artifacts: list[tuple[str, str]] = []
        self.dict_artifacts: list[tuple[dict[str, Any], str]] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    def start_run(self, **kwargs: Any) -> _FakeActiveRun:
        self.run_kwargs = kwargs
        return _FakeActiveRun()

    def log_params(self, params: dict[str, str]) -> None:
        self.params.update(params)

    def log_artifact(self, local_path: str, *, artifact_path: str) -> None:
        self.artifacts.append((local_path, artifact_path))

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        self.dict_artifacts.append((dictionary, artifact_file))


def _enabled_config() -> dict[str, Any]:
    return {
        "mlflow": {
            "enabled": True,
            "tracking_uri": "memory://tracking",
            "experiment_name": "gaze-data",
            "run_name": "prepare-001",
            "tags": {"schema_version": 1},
            "log_dataset_manifest": True,
            "log_split_manifest": True,
            "log_resolved_config": True,
            "log_environment": True,
            "log_system_metrics": True,
        },
        "data": {
            "mode": "image",
            "dataset_name": "synthetic",
            "dataset_root": "/private/face-data/subject-secret",
            "reader": {"manifest_path": "/private/face-data/input.csv"},
            "split": {
                "strategy": "grouped_ratio",
                "manifest_dir": "/private/face-data/manifests",
                "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
            },
        },
        "preprocessing": {"resize": {"enabled": True, "size_hw": [224, 224]}},
    }


def _preparation_result(tmp_path: Path) -> SimpleNamespace:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    summary = manifest_dir / "summary.json"
    summary_payload = {
        "dataset": {
            "name": "synthetic",
            "root": "/private/face-data/subject-secret",
            "counts": {
                "records": 1,
                "by_subject": {"subject-secret": 1},
                "out_of_screen_target_records_by_subject": {"subject-secret": 0},
            },
        },
        "split": {
            "partitions": {
                "train": {"records": 1, "group_ids": ["subject-secret"]},
                "validation": {"records": 0, "group_ids": []},
                "test": {"records": 0, "group_ids": []},
            }
        },
        "artifacts": {
            "dataset": {"path": "/private/face-data/subject-secret.csv", "sha256": "def"}
        },
    }
    summary.write_text(json.dumps(summary_payload), encoding="utf-8")
    manifests: dict[str, Path] = {}
    hashes: dict[str, Path] = {}
    for name in ("dataset", "split", "train", "validation", "test"):
        filename = "dataset_manifest.csv" if name == "dataset" else f"{name}.csv"
        manifest = manifest_dir / filename
        manifest.write_text("sample_id\ns001\n", encoding="utf-8")
        manifest_hash = manifest_dir / f"{filename}.sha256"
        manifest_hash.write_text(f"def  {filename}\n", encoding="utf-8")
        manifests[name] = manifest
        hashes[name] = manifest_hash
    summary_hash = manifest_dir / "summary.json.sha256"
    summary_hash.write_text("abc  summary.json\n", encoding="utf-8")
    hashes["summary"] = summary_hash
    resolved_config = tmp_path / "resolved_config.yaml"
    resolved_config.write_text("api_token: <redacted>\n", encoding="utf-8")
    resolved_config_hash = tmp_path / "resolved_config.yaml.sha256"
    resolved_config_hash.write_text("cfg  resolved_config.yaml\n", encoding="utf-8")
    return SimpleNamespace(
        summary_path=summary,
        summary=summary_payload,
        manifest_paths=manifests,
        hash_paths=hashes,
        dataset_manifest_hash="def",
        dataset_hash="legacy-alias",
        split_counts={"train": 1, "validation": 0, "test": 0},
        resolved_config_path=resolved_config,
        resolved_config_hash_path=resolved_config_hash,
        resolved_config_hash="cfg",
    )


def test_disabled_tracking_does_not_import_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_imported(name: str) -> None:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("gaze_pipeline.tracking.importlib.import_module", fail_if_imported)

    assert log_data_preparation({"mlflow": {"enabled": False}}, object()) is None


def test_tracking_logs_only_preparation_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr("gaze_pipeline.tracking.importlib.import_module", lambda name: fake_mlflow)
    result = _preparation_result(tmp_path)
    raw_face = tmp_path / "face.jpg"
    raw_face.write_bytes(b"not-an-artifact")

    tracked = log_data_preparation(_enabled_config(), result)

    assert tracked == DataPreparationTrackingResult(
        run_id="prepare-run-123", artifact_uri="memory://artifacts"
    )
    assert fake_mlflow.tracking_uri == "memory://tracking"
    assert fake_mlflow.experiment_name == "gaze-data"
    assert fake_mlflow.run_kwargs["run_name"] == "prepare-001"
    assert fake_mlflow.run_kwargs["log_system_metrics"] is True
    assert fake_mlflow.run_kwargs["tags"]["pipeline_stage"] == "data_preparation"
    assert fake_mlflow.run_kwargs["tags"]["raw_images_logged"] == "false"
    assert fake_mlflow.params["data.mode"] == "image"
    assert fake_mlflow.params["split.ratios.train"] == "0.7"
    assert fake_mlflow.params["preprocessing.resize.size_hw"] == "[224,224]"
    assert fake_mlflow.params["data.dataset_root"] == "<redacted-local-path>"
    assert fake_mlflow.params["data.reader.manifest_path"] == "<redacted-local-path>"
    assert fake_mlflow.params["split.manifest_dir"] == "<redacted-local-path>"
    assert fake_mlflow.params["preparation.dataset_manifest_hash"] == "def"
    assert "preparation.dataset_hash" not in fake_mlflow.params
    logged_paths = {Path(path) for path, _ in fake_mlflow.artifacts}
    assert logged_paths == {
        *result.manifest_paths.values(),
        *(path for name, path in result.hash_paths.items() if name != "summary"),
        result.resolved_config_hash_path,
    }
    assert raw_face not in logged_paths
    logged_dicts = {name: payload for payload, name in fake_mlflow.dict_artifacts}
    assert "data_preparation/summary.json" in logged_dicts
    assert "data_preparation/environment.json" in logged_dicts
    assert "data_preparation/resolved_config.json" in logged_dicts
    remote_summary = json.dumps(logged_dicts["data_preparation/summary.json"])
    assert "/private/face-data" not in remote_summary
    assert "subject-secret" not in remote_summary
    assert logged_dicts["data_preparation/summary.json"]["privacy"]["subject_breakdowns_removed"]
    assert json.loads(result.summary_path.read_text())["dataset"]["root"].startswith("/private")
    remote_config = json.dumps(logged_dicts["data_preparation/resolved_config.json"])
    assert "/private/face-data" not in remote_config


def test_tracking_rejects_image_disguised_as_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr("gaze_pipeline.tracking.importlib.import_module", lambda name: fake_mlflow)
    result = _preparation_result(tmp_path)
    face_path = result.summary_path.parent / "face.jpg"
    face_path.write_bytes(b"face")
    result.manifest_paths = {"dataset": face_path}

    with pytest.raises(MLflowTrackingError, match="확장자"):
        log_data_preparation(_enabled_config(), result)

    assert fake_mlflow.artifacts == []


def test_tracking_flags_disable_optional_artifacts_and_redact_fallback_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr("gaze_pipeline.tracking.importlib.import_module", lambda name: fake_mlflow)
    result = _preparation_result(tmp_path)
    del result.resolved_config_path
    del result.resolved_config_hash_path
    config = _enabled_config()
    config["mlflow"].update(
        {
            "tracking_uri": "https://alice:password@example.test/mlflow?token=secret",
            "log_dataset_manifest": False,
            "log_split_manifest": False,
            "log_environment": False,
        }
    )
    config["service"] = {
        "credential": "private-value",
        "endpoint": "https://bob:pass@example.test/api?api_key=very-secret",
    }

    log_data_preparation(config, result)

    assert fake_mlflow.artifacts == []
    logged_dicts = {name: payload for payload, name in fake_mlflow.dict_artifacts}
    assert set(logged_dicts) == {
        "data_preparation/summary.json",
        "data_preparation/resolved_config.json",
    }
    rendered = json.dumps(logged_dicts["data_preparation/resolved_config.json"])
    assert "private-value" not in rendered
    assert "alice" not in rendered
    assert "very-secret" not in rendered


def test_prepare_cli_prints_mlflow_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    config = _enabled_config()
    result = _preparation_result(tmp_path)

    monkeypatch.setattr(cli, "load_and_validate_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        cli,
        "log_data_preparation",
        lambda *args, **kwargs: DataPreparationTrackingResult("prepare-run-123"),
    )
    data_package = types.ModuleType("gaze_pipeline.data")
    data_package.__path__ = []  # type: ignore[attr-defined]
    pipeline_module = types.ModuleType("gaze_pipeline.data.pipeline")
    pipeline_module.prepare_data = lambda *args, **kwargs: result  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gaze_pipeline.data", data_package)
    monkeypatch.setitem(sys.modules, "gaze_pipeline.data.pipeline", pipeline_module)

    exit_code = cli._run_prepare(Namespace(config=config_path, profile=[], overrides=[]))

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "MLflow run_id: prepare-run-123" in output
    assert "dataset manifest hash: def" in output
    assert "dataset hash:" not in output


def test_prepare_cli_keeps_manifests_when_tracking_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    config = _enabled_config()
    result = _preparation_result(tmp_path)

    monkeypatch.setattr(cli, "load_and_validate_config", lambda *args, **kwargs: config)

    def fail_tracking(*args: Any, **kwargs: Any) -> None:
        raise MLflowTrackingError("server unavailable")

    monkeypatch.setattr(cli, "log_data_preparation", fail_tracking)
    data_package = types.ModuleType("gaze_pipeline.data")
    data_package.__path__ = []  # type: ignore[attr-defined]
    pipeline_module = types.ModuleType("gaze_pipeline.data.pipeline")
    pipeline_module.prepare_data = lambda *args, **kwargs: result  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gaze_pipeline.data", data_package)
    monkeypatch.setitem(sys.modules, "gaze_pipeline.data.pipeline", pipeline_module)

    with pytest.raises(cli.CommandError, match="생성은 완료"):
        cli._run_prepare(Namespace(config=config_path, profile=[], overrides=[]))

    assert result.summary_path.is_file()
    assert all(path.is_file() for path in result.manifest_paths.values())
    assert "데이터 준비가 완료되었습니다." in capsys.readouterr().out
