from pathlib import Path

from ggulnote_ml.pipelines.evaluate import run_evaluation
from ggulnote_ml.pipelines.predict import run_prediction
from ggulnote_ml.pipelines.train import run_training


def test_training_evaluation_and_prediction_pipeline(
    pipeline_config,
    tmp_path: Path,
) -> None:
    result = run_training(pipeline_config, tmp_path)
    assert result.model_path.exists()
    assert (result.output_dir / "metrics.json").exists()
    assert (result.output_dir / "schema.json").exists()
    assert (result.output_dir / "split_manifest.json").exists()
    assert result.metrics["test"]["rmse"] >= 0.0

    metrics = run_evaluation(pipeline_config, tmp_path, result.model_path)
    prediction = run_prediction(pipeline_config, tmp_path, result.model_path, sample_index=0)
    assert metrics["rmse"] == result.metrics["test"]["rmse"]
    assert len(prediction["prediction"]) == 2
    assert all(0.0 <= value <= 1.0 for value in prediction["prediction"])

