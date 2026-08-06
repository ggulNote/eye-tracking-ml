from __future__ import annotations

from pathlib import Path
from typing import Dict

from ggulnote_ml.config import PipelineConfig
from ggulnote_ml.exceptions import ContractError
from ggulnote_ml.models.registry import load_model
from ggulnote_ml.pipelines.common import prepare_splits


def run_prediction(
    config: PipelineConfig,
    project_root: Path,
    model_path: Path,
    sample_index: int,
) -> Dict[str, object]:
    prepared = prepare_splits(config=config, project_root=project_root)
    sample_count = prepared.test.features.values.shape[0]
    if sample_index < 0 or sample_index >= sample_count:
        raise ContractError(
            "sample_index must be between 0 and %d; got %d."
            % (sample_count - 1, sample_index)
        )
    model = load_model(config.model.name, model_path.resolve())
    prediction = model.predict(prepared.test.features.values[sample_index : sample_index + 1])[0]
    target = prepared.test.features.targets[sample_index]
    return {
        "sample_index": sample_index,
        "participant_id": prepared.test.canonical.participant_ids[sample_index],
        "session_id": prepared.test.canonical.session_ids[sample_index],
        "target": [float(target[0]), float(target[1])],
        "prediction": [float(prediction[0]), float(prediction[1])],
    }

