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
    num_samples: int
    num_participants: int
    sequence_length: int
    raw_height: int
    raw_width: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class PreprocessingConfig:
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
        if self.data.source not in {"synthetic", "manifest"}:
            raise ConfigurationError("data.source must be 'synthetic' or 'manifest'.")
        if self.data.source == "manifest" and not self.data.manifest_path:
            raise ConfigurationError("data.manifest_path is required for manifest data.")
        if self.data.num_participants < 3 and self.data.source == "synthetic":
            raise ConfigurationError("Synthetic data requires at least three participants.")
        if len(self.preprocessing.mean) != 3 or len(self.preprocessing.std) != 3:
            raise ConfigurationError("preprocessing.mean and std must contain three RGB values.")
        if any(value <= 0 for value in self.preprocessing.std):
            raise ConfigurationError("preprocessing.std values must be greater than zero.")
        if self.model.alpha < 0:
            raise ConfigurationError("model.alpha must be greater than or equal to zero.")
        if self.evaluation.screen_width <= 0 or self.evaluation.screen_height <= 0:
            raise ConfigurationError("Evaluation screen dimensions must be positive.")

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
