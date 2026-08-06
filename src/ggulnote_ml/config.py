from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from ggulnote_ml.exceptions import ConfigurationError


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    task: str


@dataclass(frozen=True)
class DataConfig:
    source: str
    manifest_path: Optional[str]
    dataset_version: str
    processed_dir: str
    train_from_processed: bool
    num_samples: int
    num_participants: int
    sequence_length: int
    frame_stride: int
    raw_height: int
    raw_width: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class PreprocessingConfig:
    color_mode: str
    output_height: int
    output_width: int
    mean: List[float]
    std: List[float]


@dataclass(frozen=True)
class FeatureExtractionConfig:
    name: str
    version: str


@dataclass(frozen=True)
class ModelConfig:
    name: str
    version: str
    alpha: float


@dataclass(frozen=True)
class ModelIOConfig:
    """Stable boundary expected by a future paper-model implementation."""

    profile: str
    image_height: int
    image_width: int
    color_mode: str
    channels: int
    layout: str
    image_dtype: str
    image_range: List[float]
    head_vector_size: int
    face_origin_3d_size: int
    embedding_size: int
    output_size: int
    output_coordinate_space: str
    support_size: int


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    output_dir: str


@dataclass(frozen=True)
class EvaluationConfig:
    screen_width: int
    screen_height: int
    distance_thresholds: List[float]


@dataclass(frozen=True)
class MlflowConfig:
    enabled: bool
    tracking_uri: str
    experiment_name: str
    registered_model_name: Optional[str]
    run_name: str


@dataclass(frozen=True)
class PipelineConfig:
    project: ProjectConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    model_io: ModelIOConfig
    feature_extraction: FeatureExtractionConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    mlflow: MlflowConfig

    def validate(self) -> None:
        fractions = (
            self.data.train_fraction,
            self.data.validation_fraction,
            self.data.test_fraction,
        )
        if any(value <= 0 for value in fractions):
            raise ConfigurationError("All data split fractions must be greater than zero.")
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ConfigurationError("Data split fractions must sum to 1.0.")
        if self.data.sequence_length <= 0:
            raise ConfigurationError("data.sequence_length must be greater than zero.")
        if self.data.frame_stride <= 0:
            raise ConfigurationError("data.frame_stride must be greater than zero.")
        if self.data.source not in {"synthetic", "manifest"}:
            raise ConfigurationError("data.source must be 'synthetic' or 'manifest'.")
        if self.data.source == "manifest" and not self.data.manifest_path:
            raise ConfigurationError("data.manifest_path is required for manifest data.")
        if not self.data.dataset_version.strip():
            raise ConfigurationError("data.dataset_version must not be empty.")
        if (
            self.data.dataset_version in {".", ".."}
            or Path(self.data.dataset_version).name != self.data.dataset_version
        ):
            raise ConfigurationError("data.dataset_version must be a single directory name.")
        if not self.data.processed_dir.strip():
            raise ConfigurationError("data.processed_dir must not be empty.")
        if self.data.num_participants < 3 and self.data.source == "synthetic":
            raise ConfigurationError("Synthetic data requires at least three participants.")
        if self.preprocessing.color_mode not in {"rgb", "gray"}:
            raise ConfigurationError("preprocessing.color_mode must be 'rgb' or 'gray'.")
        preprocessing_channels = 3 if self.preprocessing.color_mode == "rgb" else 1
        if (
            len(self.preprocessing.mean) != preprocessing_channels
            or len(self.preprocessing.std) != preprocessing_channels
        ):
            raise ConfigurationError(
                "preprocessing.mean and std must match color_mode channel count (%d)."
                % preprocessing_channels
            )
        if any(value <= 0 for value in self.preprocessing.std):
            raise ConfigurationError("preprocessing.std values must be greater than zero.")
        if self.preprocessing.output_height <= 0 or self.preprocessing.output_width <= 0:
            raise ConfigurationError("Preprocessing output dimensions must be positive.")
        self._validate_model_io()
        if self.model.alpha < 0:
            raise ConfigurationError("model.alpha must be greater than or equal to zero.")
        if self.evaluation.screen_width <= 0 or self.evaluation.screen_height <= 0:
            raise ConfigurationError("Evaluation screen dimensions must be positive.")

    def _validate_model_io(self) -> None:
        io = self.model_io
        if io.profile not in {"webeyetrack_v1", "custom"}:
            raise ConfigurationError("model_io.profile must be 'webeyetrack_v1' or 'custom'.")
        expected_channels = 3 if io.color_mode == "rgb" else 1
        if io.color_mode not in {"rgb", "gray"} or io.channels != expected_channels:
            raise ConfigurationError("model_io color_mode and channels do not match.")
        if io.image_height <= 0 or io.image_width <= 0:
            raise ConfigurationError("model_io image dimensions must be positive.")
        if io.layout not in {"BHWC", "BCHW"} or io.image_dtype != "float32":
            raise ConfigurationError("model_io requires BHWC/BCHW float32 image tensors.")
        if (
            len(io.image_range) != 2
            or not all(float("-inf") < value < float("inf") for value in io.image_range)
            or io.image_range[0] >= io.image_range[1]
        ):
            raise ConfigurationError("model_io.image_range must contain finite increasing bounds.")
        if min(
            io.head_vector_size,
            io.face_origin_3d_size,
            io.embedding_size,
            io.output_size,
            io.support_size,
        ) <= 0:
            raise ConfigurationError("model_io dimensions and support_size must be positive.")
        if io.output_coordinate_space not in {
            "top_left_normalized",
            "screen_centered_normalized",
        }:
            raise ConfigurationError("Unsupported model_io.output_coordinate_space.")
        if io.profile == "custom":
            return
        if self.preprocessing.color_mode != "rgb":
            raise ConfigurationError("webeyetrack_v1 requires preprocessing.color_mode='rgb'.")
        if (
            io.image_height,
            io.image_width,
            io.color_mode,
            io.channels,
            io.layout,
            io.image_dtype,
        ) != (128, 512, "rgb", 3, "BHWC", "float32"):
            raise ConfigurationError(
                "webeyetrack_v1 image contract is float32[B,128,512,3] RGB in BHWC layout."
            )
        if io.image_range != [0.0, 1.0]:
            raise ConfigurationError("webeyetrack_v1 image_range must be [0.0, 1.0].")
        if (io.head_vector_size, io.face_origin_3d_size, io.output_size) != (3, 3, 2):
            raise ConfigurationError(
                "webeyetrack_v1 expects head_vector[3], face_origin_3d[3], and gaze[2]."
            )
        if io.embedding_size != 512:
            raise ConfigurationError("webeyetrack_v1 embedding_size must be 512.")
        if io.output_coordinate_space != "screen_centered_normalized":
            raise ConfigurationError(
                "webeyetrack_v1 output coordinates must be screen_centered_normalized."
            )
        if io.support_size <= 0 or io.support_size > 9:
            raise ConfigurationError("model_io.support_size must be between 1 and 9.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedConfig:
    config: PipelineConfig
    project_root: Path
    source_path: Path


def _require_section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError("Missing or invalid config section: %s" % name)
    return value


def load_config(path: Path) -> LoadedConfig:
    source_path = path.expanduser().resolve()
    if not source_path.exists():
        raise ConfigurationError("Config file does not exist: %s" % source_path)

    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ConfigurationError("The config root must be a mapping.")

    try:
        config = PipelineConfig(
            project=ProjectConfig(**_require_section(raw, "project")),
            data=DataConfig(**_require_section(raw, "data")),
            preprocessing=PreprocessingConfig(**_require_section(raw, "preprocessing")),
            model_io=ModelIOConfig(**_require_section(raw, "model_io")),
            feature_extraction=FeatureExtractionConfig(
                **_require_section(raw, "feature_extraction")
            ),
            model=ModelConfig(**_require_section(raw, "model")),
            training=TrainingConfig(**_require_section(raw, "training")),
            evaluation=EvaluationConfig(**_require_section(raw, "evaluation")),
            mlflow=MlflowConfig(**_require_section(raw, "mlflow")),
        )
    except TypeError as exc:
        raise ConfigurationError("Config fields do not match the expected schema: %s" % exc) from exc

    config.validate()
    project_root = _find_project_root(source_path)
    return LoadedConfig(config=config, project_root=project_root, source_path=source_path)


def _find_project_root(source_path: Path) -> Path:
    for candidate in source_path.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigurationError(
        "Could not locate project root containing pyproject.toml above %s" % source_path
    )


def flatten_config(config: PipelineConfig) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit("%s.%s" % (prefix, key) if prefix else str(key), child)
            return
        if isinstance(value, list):
            flattened[prefix] = ",".join(str(item) for item in value)
            return
        flattened[prefix] = value

    visit("", config.to_dict())
    return flattened
