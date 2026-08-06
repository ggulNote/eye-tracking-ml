from __future__ import annotations

from pathlib import Path
from typing import Dict

from ggulnote_ml.config import PipelineConfig
from ggulnote_ml.evaluation import gaze_metrics
from ggulnote_ml.models.registry import load_model
from ggulnote_ml.pipelines.common import prepare_splits


def run_evaluation(
    config: PipelineConfig,
    project_root: Path,
    model_path: Path,
) -> Dict[str, float]:
    prepared = prepare_splits(config=config, project_root=project_root)
    model = load_model(config.model.name, model_path.resolve())
    predictions = model.predict(prepared.test.features.values)
    return gaze_metrics(prepared.test.features.targets, predictions, config.evaluation)

