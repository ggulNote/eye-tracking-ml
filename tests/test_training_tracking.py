from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gaze_pipeline.tracking import MLflowTrackingError
from gaze_pipeline.training_tracking import LoggedTrainingArtifact, training_run


class _FakeActiveRun:
    def __init__(self) -> None:
        self.info = SimpleNamespace(
            run_id="training-run-123",
            artifact_uri="memory://training-artifacts",
        )


class _FakeRunManager:
    def __init__(self) -> None:
        self.active_run = _FakeActiveRun()
        self.exit_args: tuple[Any, Any, Any] | None = None

    def __enter__(self) -> _FakeActiveRun:
        return self.active_run

    def __exit__(self, *args: Any) -> None:
        self.exit_args = args


class _FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.run_kwargs: dict[str, Any] = {}
        self.params: dict[str, str] = {}
        self.metrics: list[tuple[dict[str, float], int]] = []
        self.artifacts: list[tuple[str, str]] = []
        self.dict_artifacts: list[tuple[dict[str, Any], str]] = []
        self.run_manager = _FakeRunManager()

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    def start_run(self, **kwargs: Any) -> _FakeRunManager:
        self.run_kwargs = kwargs
        return self.run_manager

    def log_params(self, params: dict[str, str]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        self.metrics.append((metrics, step))

    def log_artifact(self, local_path: str, *, artifact_path: str) -> None:
        self.artifacts.append((local_path, artifact_path))

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        self.dict_artifacts.append((dictionary, artifact_file))


def _enabled_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mlflow": {
            "enabled": True,
            "tracking_uri": "memory://tracking",
            "experiment_name": "gaze-training",
            "run_name": "train-001",
            "log_system_metrics": True,
            "log_resolved_config": True,
            "log_environment": True,
            "tags": {"schema_version": 1, "api_token": "do-not-upload"},
        },
        "experiment": {"seed": 42},
        "data": {"dataset_root": "/private/face-data", "split": {"seed": 42}},
        "model": {
            "front": {
                "enabled": True,
                "source_dir": "/private/model-source",
                "entrypoint": "models.front:create_model",
            }
        },
        "training": {"max_epochs": 2, "gradient_clip_norm": 1.0},
        "optimizer": {"name": "AdamW", "learning_rate": 0.001},
        "loss": {"primary": {"name": "huber_xy", "delta": 0.05}},
        "metrics": {"selection_metric": "euclidean_normalized_mean"},
        "checkpoint": {"dir": "/private/checkpoints"},
    }


def test_disabled_session_is_noop_and_does_not_import_mlflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_imported(name: str) -> None:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("gaze_pipeline.training_tracking.importlib.import_module", fail_if_imported)

    with training_run({"mlflow": {"enabled": False}}) as tracker:
        assert tracker.enabled is False
        assert tracker.run_id is None
        tracker.log_epoch(0, train_loss=1.0)
        tracker.log_metrics({"test/loss": 1.0})
        assert tracker.log_checkpoint("does-not-exist.pt", kind="last") is None
        assert tracker.log_model_artifact("does-not-exist.pt") is None
        assert (
            tracker.log_artifact(
                "does-not-exist.json",
                "evaluation/test/metrics",
                "metrics",
            )
            is None
        )


def test_training_run_logs_redacted_config_params_and_epoch_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(
        "gaze_pipeline.training_tracking.importlib.import_module", lambda name: fake_mlflow
    )

    with training_run(
        _enabled_config(),
        extra_params={"dataset_manifest_hash": "abc123", "source_path": "/private/input"},
    ) as tracker:
        assert tracker.enabled is True
        assert tracker.run_id == "training-run-123"
        assert tracker.artifact_uri == "memory://training-artifacts"
        tracker.log_epoch(
            1,
            train_loss=0.4,
            val_loss=0.5,
            train_metrics={"mae_x": 0.1},
            val_metrics={"fusion/euclidean": 0.2},
        )

    assert fake_mlflow.tracking_uri == "memory://tracking"
    assert fake_mlflow.experiment_name == "gaze-training"
    assert fake_mlflow.run_kwargs["run_name"] == "train-001"
    assert fake_mlflow.run_kwargs["log_system_metrics"] is True
    assert fake_mlflow.run_kwargs["tags"]["pipeline_stage"] == "training"
    assert fake_mlflow.run_kwargs["tags"]["run_type"] == "training"
    assert fake_mlflow.run_kwargs["tags"]["raw_images_logged"] == "false"
    assert fake_mlflow.run_kwargs["tags"]["api_token"] == "<redacted>"
    assert fake_mlflow.params["training.max_epochs"] == "2"
    assert fake_mlflow.params["optimizer.learning_rate"] == "0.001"
    assert fake_mlflow.params["data.dataset_root"] == "<redacted-local-path>"
    assert fake_mlflow.params["model.front.source_dir"] == "<redacted-local-path>"
    assert fake_mlflow.params["runtime.dataset_manifest_hash"] == "abc123"
    assert fake_mlflow.params["runtime.source_path"] == "<redacted-local-path>"
    assert fake_mlflow.metrics == [
        (
            {
                "train/loss": 0.4,
                "val/loss": 0.5,
                "train/mae_x": 0.1,
                "val/fusion/euclidean": 0.2,
            },
            1,
        )
    ]
    logged_dicts = {name: payload for payload, name in fake_mlflow.dict_artifacts}
    assert "training/resolved_config.json" in logged_dicts
    assert "training/environment.json" in logged_dicts
    rendered_config = json.dumps(logged_dicts["training/resolved_config.json"])
    assert "do-not-upload" not in rendered_config
    assert "/private/face-data" not in rendered_config
    assert fake_mlflow.run_manager.exit_args == (None, None, None)


def test_evaluation_run_supports_owned_tags_and_generic_metric_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(
        "gaze_pipeline.training_tracking.importlib.import_module", lambda name: fake_mlflow
    )

    with training_run(
        _enabled_config(),
        run_type="evaluation",
        extra_tags={
            "command": "evaluate",
            "split": "test",
            "run_type": "must-not-override-owned-tag",
            "api_token": "do-not-upload",
        },
    ) as tracker:
        tracker.log_metrics(
            {
                "test/loss": 0.25,
                "test/euclidean_normalized_mean": 0.1,
            }
        )

    tags = fake_mlflow.run_kwargs["tags"]
    assert tags["pipeline_stage"] == "evaluation"
    assert tags["run_type"] == "evaluation"
    assert tags["command"] == "evaluate"
    assert tags["split"] == "test"
    assert tags["api_token"] == "<redacted>"
    logged_dicts = {name: payload for payload, name in fake_mlflow.dict_artifacts}
    assert "evaluation/resolved_config.json" in logged_dicts
    assert "evaluation/environment.json" in logged_dicts
    assert "training/resolved_config.json" not in logged_dicts
    assert fake_mlflow.metrics == [
        (
            {
                "test/loss": 0.25,
                "test/euclidean_normalized_mean": 0.1,
            },
            0,
        )
    ]


def test_checkpoint_and_model_artifacts_include_sha256_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(
        "gaze_pipeline.training_tracking.importlib.import_module", lambda name: fake_mlflow
    )
    checkpoint = tmp_path / "best_weights.pt"
    checkpoint.write_bytes(b"checkpoint-weights")
    model = tmp_path / "final_weights.safetensors"
    model.write_bytes(b"exported-model")

    with training_run(_enabled_config()) as tracker:
        checkpoint_result = tracker.log_checkpoint(checkpoint, kind="best")
        model_result = tracker.log_model_artifact(model)

    assert checkpoint_result == LoggedTrainingArtifact(
        path=checkpoint,
        artifact_path="training/checkpoints/best",
        sha256=hashlib.sha256(b"checkpoint-weights").hexdigest(),
        size_bytes=len(b"checkpoint-weights"),
    )
    assert model_result is not None
    assert model_result.sha256 == hashlib.sha256(b"exported-model").hexdigest()
    assert fake_mlflow.artifacts == [
        (str(checkpoint), "training/checkpoints/best"),
        (str(model), "training/model/final"),
    ]
    logged_dicts = {name: payload for payload, name in fake_mlflow.dict_artifacts}
    checkpoint_metadata = logged_dicts["training/checkpoints/best/best_weights.pt.metadata.json"]
    assert checkpoint_metadata["kind"] == "best"
    assert checkpoint_metadata["sha256"] == checkpoint_result.sha256


def test_generic_artifact_uses_requested_safe_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(
        "gaze_pipeline.training_tracking.importlib.import_module", lambda name: fake_mlflow
    )
    metrics = tmp_path / "test_metrics.json"
    metrics.write_text('{"loss": 0.25}\n', encoding="utf-8")

    with training_run(_enabled_config(), run_type="evaluation") as tracker:
        result = tracker.log_artifact(
            metrics,
            "evaluation/test/metrics",
            "metrics",
        )

    assert result is not None
    assert result.artifact_path == "evaluation/test/metrics"
    assert fake_mlflow.artifacts == [(str(metrics), "evaluation/test/metrics")]
    logged_dicts = {name: payload for payload, name in fake_mlflow.dict_artifacts}
    metadata = logged_dicts["evaluation/test/metrics/test_metrics.json.metadata.json"]
    assert metadata["kind"] == "metrics"
    assert metadata["sha256"] == hashlib.sha256(metrics.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "artifact_path",
    ("/absolute/path", "evaluation/../private", "evaluation//metrics", r"evaluation\metrics"),
)
def test_generic_artifact_rejects_unsafe_remote_path(
    artifact_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(
        "gaze_pipeline.training_tracking.importlib.import_module", lambda name: fake_mlflow
    )
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}\n", encoding="utf-8")

    with training_run(_enabled_config()) as tracker:
        with pytest.raises(MLflowTrackingError, match="artifact_path"):
            tracker.log_artifact(artifact, artifact_path, "metrics")

    assert fake_mlflow.artifacts == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epoch": -1, "train_loss": 1.0}, "0 이상"),
        ({"epoch": 0, "train_loss": float("nan")}, "유한"),
        ({"epoch": 0, "train_metrics": {"loss": 1.0}}, "예약 이름"),
    ],
)
def test_epoch_metric_contract_rejects_invalid_values(
    kwargs: dict[str, Any],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(
        "gaze_pipeline.training_tracking.importlib.import_module", lambda name: fake_mlflow
    )

    with training_run(_enabled_config()) as tracker:
        with pytest.raises(MLflowTrackingError, match=message):
            tracker.log_epoch(**kwargs)


def test_training_body_exception_is_preserved_and_marks_run_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(
        "gaze_pipeline.training_tracking.importlib.import_module", lambda name: fake_mlflow
    )

    with pytest.raises(ValueError, match="model failed"):
        with training_run(_enabled_config()):
            raise ValueError("model failed")

    assert fake_mlflow.run_manager.exit_args is not None
    assert fake_mlflow.run_manager.exit_args[0] is ValueError


def test_session_rejects_logging_after_context_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(
        "gaze_pipeline.training_tracking.importlib.import_module", lambda name: fake_mlflow
    )

    with training_run(_enabled_config()) as tracker:
        pass

    with pytest.raises(MLflowTrackingError, match="종료된"):
        tracker.log_epoch(2, train_loss=0.2)


def test_image_cannot_be_logged_as_model_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(
        "gaze_pipeline.training_tracking.importlib.import_module", lambda name: fake_mlflow
    )
    image = tmp_path / "subject-face.jpg"
    image.write_bytes(b"face")

    with training_run(_enabled_config()) as tracker:
        with pytest.raises(MLflowTrackingError, match="원본 이미지"):
            tracker.log_model_artifact(image)

    assert fake_mlflow.artifacts == []


def test_image_cannot_be_logged_through_generic_artifact_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr(
        "gaze_pipeline.training_tracking.importlib.import_module", lambda name: fake_mlflow
    )
    image = tmp_path / "subject-face.png"
    image.write_bytes(b"face")

    with training_run(_enabled_config(), run_type="evaluation") as tracker:
        with pytest.raises(MLflowTrackingError, match="원본 이미지"):
            tracker.log_artifact(image, "evaluation/test/predictions", "predictions")

    assert fake_mlflow.artifacts == []
