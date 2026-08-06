from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

from ggulnote_ml.config import PipelineConfig, flatten_config
from ggulnote_ml.contracts import contract_summary
from ggulnote_ml.evaluation import gaze_metrics
from ggulnote_ml.models import build_model
from ggulnote_ml.pipelines.common import prepare_splits
from ggulnote_ml.tracking import MlflowRunTracker, NullRunTracker
from ggulnote_ml.tracking.base import RunTracker
from ggulnote_ml.utils import git_sha, write_json, write_predictions_csv


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    output_dir: Path
    model_path: Path
    metrics: Dict[str, Dict[str, float]]


def run_training(config: PipelineConfig, project_root: Path) -> TrainingResult:
    config.validate()
    prepared = prepare_splits(config=config, project_root=project_root)
    model = build_model(config.model)

    tags = {
        "task": config.project.task,
        "data_source": config.data.source,
        "feature_extractor": config.feature_extraction.name,
        "feature_version": config.feature_extraction.version,
        "model_name": config.model.name,
        "model_version": config.model.version,
        "preprocessing_version": "v2",
        "dataset_version": config.data.dataset_version,
        "git_sha": git_sha(project_root),
    }
    tracker: RunTracker
    if config.mlflow.enabled:
        tracker = MlflowRunTracker(
            config=config.mlflow,
            tags=tags,
            project_root=project_root,
        )
    else:
        tracker = NullRunTracker()

    with tracker:
        model.fit(prepared.train.features.values, prepared.train.features.targets)
        predictions = {
            "train": model.predict(prepared.train.features.values),
            "validation": model.predict(prepared.validation.features.values),
            "test": model.predict(prepared.test.features.values),
        }
        metrics = {
            "train": gaze_metrics(
                prepared.train.features.targets, predictions["train"], config.evaluation
            ),
            "validation": gaze_metrics(
                prepared.validation.features.targets,
                predictions["validation"],
                config.evaluation,
            ),
            "test": gaze_metrics(
                prepared.test.features.targets, predictions["test"], config.evaluation
            ),
        }

        output_dir = (project_root / config.training.output_dir / tracker.run_id).resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        model_path = output_dir / "model.npz"
        model.save(model_path)

        schema = contract_summary(
            prepared.train.canonical,
            prepared.train.features,
            predictions["train"],
        )
        split_manifest = _split_manifest(prepared)
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "schema.json", schema)
        write_json(output_dir / "split_manifest.json", split_manifest)
        with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)
        write_predictions_csv(
            output_dir / "test_predictions.csv",
            prepared.test.canonical.participant_ids,
            prepared.test.canonical.session_ids,
            prepared.test.features.targets,
            predictions["test"],
        )

        tracker.log_params(flatten_config(config))
        tracker.log_params(
            {
                "dataset.dataset_version": prepared.dataset_metadata.get(
                    "dataset_version", config.data.dataset_version
                ),
                "dataset.dataset_hash": prepared.dataset_metadata.get("dataset_hash"),
                "dataset.manifest_hash": prepared.dataset_metadata.get("manifest_hash"),
                "dataset.participant_count": prepared.dataset_metadata.get(
                    "participant_count"
                ),
                "dataset.sample_count": prepared.dataset_metadata.get("sample_count"),
                "dataset.preprocessor_version": prepared.dataset_metadata.get(
                    "preprocessor_version"
                ),
            }
        )
        for split_name, split_metrics in metrics.items():
            tracker.log_metrics(split_metrics, prefix="%s_" % split_name)
        tracker.log_json(schema, "contracts/schema.json")
        tracker.log_json(split_manifest, "data/split_manifest.json")
        tracker.log_json(prepared.dataset_metadata, "data/dataset.json")
        tracker.log_artifact(output_dir / "resolved_config.yaml", artifact_path="config")
        # Preserve the MLflow tensor signature without uploading real image-derived
        # feature values as the model input example.
        input_example = np.zeros_like(prepared.test.features.values[:2], dtype=np.float32)
        output_example = model.predict(input_example)
        tracker.log_model(
            model=model,
            model_path=model_path,
            input_example=input_example,
            output_example=output_example,
            registered_model_name=config.mlflow.registered_model_name,
        )

        return TrainingResult(
            run_id=tracker.run_id,
            output_dir=output_dir,
            model_path=model_path,
            metrics=metrics,
        )


def _split_manifest(prepared: object) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for name in ("train", "validation", "test"):
        if prepared.raw is not None:
            raw_split = getattr(prepared.raw, name)
            result[name] = {
                "sample_count": len(raw_split.samples),
                "participant_ids": sorted(
                    {sample.participant_id for sample in raw_split.samples}
                ),
                "sources": sorted({sample.source for sample in raw_split.samples}),
            }
        else:
            canonical = getattr(prepared, name).canonical
            result[name] = {
                "sample_count": canonical.frames.shape[0],
                "participant_ids": sorted(set(canonical.participant_ids)),
                "sources": sorted(set(canonical.sources)),
            }
    return result
