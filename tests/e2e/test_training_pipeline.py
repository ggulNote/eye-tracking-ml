from dataclasses import replace
from pathlib import Path

from ggulnote_ml.pipelines.evaluate import run_evaluation
from ggulnote_ml.pipelines.predict import run_prediction
from ggulnote_ml.pipelines.preprocess import run_preprocessing
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


def test_local_preprocess_then_train_from_processed(
    pipeline_config,
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "local-data"
    monkeypatch.setenv("GGULNOTE_DATA_ROOT", str(data_root))
    preprocess_config = replace(
        pipeline_config,
        data=replace(
            pipeline_config.data,
            dataset_version="processed-e2e-v001",
            train_from_processed=False,
        ),
    )
    preprocessing_result = run_preprocessing(preprocess_config, tmp_path)

    assert (preprocessing_result.dataset_dir / "dataset.json").is_file()
    assert preprocessing_result.metadata["sample_count"] == 36
    assert preprocessing_result.metadata["dataset_hash"]
    repeated = run_preprocessing(preprocess_config, tmp_path, force=True)
    assert repeated.metadata["dataset_hash"] == preprocessing_result.metadata["dataset_hash"]

    training_config = replace(
        preprocess_config,
        data=replace(preprocess_config.data, train_from_processed=True),
        training=replace(preprocess_config.training, output_dir="processed-outputs"),
    )
    result = run_training(training_config, tmp_path)
    assert result.model_path.is_file()
    assert result.metrics["test"]["rmse"] >= 0.0
