from __future__ import annotations

from pathlib import Path

import pytest

from ggulnote_ml.config import (
    DataConfig,
    EvaluationConfig,
    FeatureExtractionConfig,
    MlflowConfig,
    ModelConfig,
    ModelIOConfig,
    PipelineConfig,
    PreprocessingConfig,
    ProjectConfig,
    TrainingConfig,
)


@pytest.fixture
def pipeline_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        project=ProjectConfig(name="ggulnote-test", task="gaze_coordinate_regression"),
        data=DataConfig(
            source="synthetic",
            manifest_path=None,
            dataset_version="test-v001",
            processed_dir="processed",
            train_from_processed=False,
            num_samples=36,
            num_participants=6,
            sequence_length=2,
            frame_stride=1,
            raw_height=12,
            raw_width=16,
            train_fraction=0.5,
            validation_fraction=0.25,
            test_fraction=0.25,
        ),
        preprocessing=PreprocessingConfig(
            color_mode="rgb",
            output_height=8,
            output_width=10,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        model_io=ModelIOConfig(
            profile="webeyetrack_v1",
            image_height=128,
            image_width=512,
            color_mode="rgb",
            channels=3,
            layout="BHWC",
            image_dtype="float32",
            image_range=[0.0, 1.0],
            head_vector_size=3,
            face_origin_3d_size=3,
            embedding_size=512,
            output_size=2,
            output_coordinate_space="screen_centered_normalized",
            support_size=9,
        ),
        feature_extraction=FeatureExtractionConfig(name="identity", version="test"),
        model=ModelConfig(name="ridge", version="test", alpha=1.0),
        training=TrainingConfig(seed=7, output_dir="outputs"),
        evaluation=EvaluationConfig(
            screen_width=1920,
            screen_height=1080,
            distance_thresholds=[0.05, 0.1],
        ),
        mlflow=MlflowConfig(
            enabled=False,
            tracking_uri="sqlite:///mlflow.db",
            experiment_name="test",
            registered_model_name=None,
            run_name="test",
        ),
    )
