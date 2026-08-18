"""Configuration loading, interpolation, and validation.

The project documents Hydra-style interpolation, but the CLI should not need
Hydra just to read a YAML file. This module resolves the small, explicit subset
used by validation, data preparation, training, and evaluation:

* ``${oc.env:NAME,default}``
* ``${now:%Y%m%d_%H%M%S}``
* references such as ``${experiment.seed}``

No arbitrary resolver or Python expression is evaluated.
"""

from __future__ import annotations

import copy
import math
import os
import re
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in an incomplete env
    yaml = None  # type: ignore[assignment]


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_PREPROCESSING_STAGES = (
    "decode",
    "exif_orientation",
    "validate",
    "face_landmarks",
    "eye_selection",
    "metric_head_pose",
    "eye_region_warp",
    "face_roi",
    "background_mask",
    "resize",
    "normalize",
    "augment",
)

_INTERPOLATION_RE = re.compile(r"\$\{([^{}]+)\}")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REFERENCE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_ENTRYPOINT_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DYNAMIC_OVERRIDE_PARENTS = frozenset({"init_args", "tags"})
_GENERIC_PAIR_KEY_RESERVED = frozenset(
    {
        "sample_id",
        "subject_id",
        "view",
        "image_path",
        "target_x_px",
        "target_y_px",
        "screen_width_px",
        "screen_height_px",
        "session_id",
        "facial_landmarks_xy",
        "head_rotation_3d",
        "head_translation_3d",
        "face_center_3d",
        "gaze_target_3d",
        "evaluation_eye",
    }
)
_MISSING = object()


class ConfigError(ValueError):
    """Base class for user-correctable configuration errors."""


class ConfigLoadError(ConfigError):
    """Raised when a YAML file or CLI override cannot be loaded."""


class ConfigResolutionError(ConfigError):
    """Raised when a safe interpolation cannot be resolved."""


class ConfigValidationError(ConfigError):
    """Raised after collecting one or more invalid config fields."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        count = len(self.issues)
        details = "\n".join(f"  - {issue}" for issue in self.issues)
        super().__init__(f"설정에서 {count}개의 문제를 발견했습니다.\n{details}")


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    profiles: Sequence[str | Path] = (),
    overrides: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Load, merge, override, and safely resolve a project config.

    Profiles are merged in the order supplied, then CLI ``key=value``
    overrides are applied.  The returned dictionary contains no supported
    interpolation tokens.
    """

    base_path = Path(config_path).expanduser()
    merged = _load_yaml_mapping(base_path, description="기준 config")

    for profile in profiles:
        profile_path = Path(profile).expanduser()
        profile_config = _load_yaml_mapping(profile_path, description="profile config")
        merged = deep_merge(merged, profile_config)

    if overrides:
        apply_overrides(merged, overrides)

    return resolve_interpolations(merged, environ=environ, now=now)


def load_and_validate_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    profiles: Sequence[str | Path] = (),
    overrides: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
    check_paths: bool = False,
    require_model_entrypoints: bool = False,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load a config and run all currently supported validation checks."""

    config = load_config(
        config_path,
        profiles=profiles,
        overrides=overrides,
        environ=environ,
        now=now,
    )
    validate_config(
        config,
        check_paths=check_paths,
        require_model_entrypoints=require_model_entrypoints,
        base_dir=base_dir,
    )
    return config


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge a profile over a base mapping without mutation."""

    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = result.get(key, _MISSING)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_overrides(config: MutableMapping[str, Any], overrides: Sequence[str]) -> None:
    """Apply dotted ``key=value`` overrides in place.

    New keys are accepted only below intentionally open mappings such as
    ``init_args`` and ``tags``.  Elsewhere a misspelled key fails loudly.
    """

    for raw_override in overrides:
        if "=" not in raw_override:
            raise ConfigLoadError(
                f"CLI override '{raw_override}'는 key=value 형식이어야 합니다. "
                "예: training.max_epochs=20"
            )
        raw_key, raw_value = raw_override.split("=", 1)
        key = raw_key.strip()
        if not _REFERENCE_RE.fullmatch(key):
            raise ConfigLoadError(
                f"CLI override key '{raw_key}'가 올바르지 않습니다. "
                "점으로 연결된 config 경로를 사용하세요."
            )
        value = _parse_override_value(raw_value)
        _set_dotted_value(config, key.split("."), value)


def resolve_interpolations(
    config: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Resolve only the interpolation forms explicitly supported above."""

    source = copy.deepcopy(dict(config))
    resolver = _SafeInterpolationResolver(
        source,
        environ=os.environ if environ is None else environ,
        now=now or datetime.now().astimezone(),
    )
    resolved = resolver.resolve()
    if not isinstance(resolved, dict):  # defensive: root is always a mapping
        raise ConfigResolutionError("config 최상위 값은 mapping이어야 합니다.")
    return resolved


def validate_config(
    config: Mapping[str, Any],
    *,
    check_paths: bool = False,
    require_model_entrypoints: bool = False,
    base_dir: str | Path | None = None,
) -> None:
    """Validate the static-image pipeline contract.

    ``prepare`` deliberately calls this with
    ``require_model_entrypoints=False``: manifest/split preparation does not
    import a model.  Model-dependent commands can set it to ``True`` later.
    """

    issues: list[str] = []
    if not isinstance(config, Mapping):
        raise ConfigValidationError(["config 최상위 값은 YAML mapping이어야 합니다."])

    _find_unresolved_tokens(config, issues)

    schema_version = config.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        issues.append(f"schema_version은 현재 1만 지원합니다 (입력값: {schema_version!r}).")

    paths = _expect_mapping(config, "paths", issues)
    for key in ("data_root", "output_root", "run_dir", "checkpoint_dir"):
        value = paths.get(key, _MISSING)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"paths.{key}는 비어 있지 않은 경로 문자열이어야 합니다.")

    data = _expect_mapping(config, "data", issues)
    _validate_image_data(data, issues)
    _validate_data_selection(data, issues)
    available_branches = _validate_views(data, issues)
    pairing_enabled = _validate_pairing(data, available_branches, issues)
    _validate_split(data, issues)

    preprocessing = _expect_mapping(config, "preprocessing", issues)
    _validate_preprocessing(preprocessing, available_branches, issues)

    model = _expect_mapping(config, "model", issues)
    enabled_models = _validate_models(
        model,
        available_branches,
        require_model_entrypoints=require_model_entrypoints,
        issues=issues,
    )
    _validate_preprocessing_model_contracts(
        preprocessing,
        model,
        enabled_models=enabled_models,
        issues=issues,
    )
    _validate_side_feature_selection(
        preprocessing,
        model,
        enabled_models=enabled_models,
        pairing_enabled=pairing_enabled,
        issues=issues,
    )

    fusion = _expect_mapping(config, "fusion", issues)
    _validate_fusion(
        fusion,
        available_branches=available_branches,
        enabled_models=enabled_models,
        pairing_enabled=pairing_enabled,
        model=model,
        require_model_entrypoints=require_model_entrypoints,
        issues=issues,
    )
    _validate_execution_config(config, issues)

    if check_paths:
        _validate_filesystem_paths(
            config,
            base_dir=Path.cwd() if base_dir is None else Path(base_dir),
            issues=issues,
        )

    if issues:
        raise ConfigValidationError(issues)


def resolve_model_forward_keys(config: Mapping[str, Any], branch: str) -> tuple[str, ...]:
    """Resolve the actual model input keys after feature on/off selection.

    A literal ``forward_keys`` list is returned unchanged. The strict-profile
    side contract uses ``forward_keys: auto`` so changing one feature's
    ``enabled`` flag is sufficient; users do not need to edit a second list.
    """

    branch = str(branch).lower()
    model = config.get("model")
    branch_model = model.get(branch) if isinstance(model, Mapping) else None
    input_contract = (
        branch_model.get("input_contract") if isinstance(branch_model, Mapping) else None
    )
    if not isinstance(input_contract, Mapping):
        raise ConfigLoadError(f"model.{branch}.input_contract mapping이 필요합니다.")

    declared = input_contract.get("forward_keys")
    if isinstance(declared, list):
        if not declared or not all(isinstance(key, str) and key.strip() for key in declared):
            raise ConfigLoadError(f"model.{branch}.input_contract.forward_keys가 잘못되었습니다.")
        keys = tuple(str(key) for key in declared)
        if len(keys) != len(set(keys)):
            raise ConfigLoadError(
                f"model.{branch}.input_contract.forward_keys에 중복 key가 있습니다."
            )
        return keys
    if declared is None:
        image_key = input_contract.get("image_key")
        if not isinstance(image_key, str) or not image_key.strip():
            raise ConfigLoadError(f"model.{branch}.input_contract.image_key가 잘못되었습니다.")
        keys = [image_key]
        auxiliary = input_contract.get("auxiliary_keys", {})
        if auxiliary is not None and not isinstance(auxiliary, Mapping):
            raise ConfigLoadError(f"model.{branch}.input_contract.auxiliary_keys가 잘못되었습니다.")
        if isinstance(auxiliary, Mapping):
            for name, contract in auxiliary.items():
                if not isinstance(contract, Mapping):
                    raise ConfigLoadError(
                        f"model.{branch}.input_contract.auxiliary_keys.{name}가 잘못되었습니다."
                    )
                key = contract.get("key")
                if not isinstance(key, str) or not key.strip():
                    raise ConfigLoadError(
                        f"model.{branch}.input_contract.auxiliary_keys.{name}.key가 필요합니다."
                    )
                keys.append(key)
        if len(keys) != len(set(keys)):
            raise ConfigLoadError(f"model.{branch}.input_contract에 중복 input key가 있습니다.")
        return tuple(keys)
    if declared != "auto":
        raise ConfigLoadError(
            f"model.{branch}.input_contract.forward_keys는 string list, "
            "'auto', 또는 생략이어야 합니다."
        )
    if branch != "side":
        raise ConfigLoadError("forward_keys: auto는 현재 strict-profile side branch만 지원합니다.")

    preprocessing = config.get("preprocessing")
    overrides = (
        preprocessing.get("branch_overrides") if isinstance(preprocessing, Mapping) else None
    )
    side = overrides.get("side") if isinstance(overrides, Mapping) else None
    warp = side.get("eye_region_warp") if isinstance(side, Mapping) else None
    features = warp.get("feature_extraction") if isinstance(warp, Mapping) else None
    if not isinstance(features, Mapping):
        raise ConfigLoadError(
            "forward_keys: auto에는 side eye_region_warp.feature_extraction mapping이 필요합니다."
        )

    image_key = input_contract.get("image_key", "side_image")
    if not isinstance(image_key, str) or not image_key.strip():
        raise ConfigLoadError("model.side.input_contract.image_key가 잘못되었습니다.")
    keys = [image_key]

    head = features.get("side_headpose")
    if isinstance(head, Mapping) and head.get("enabled") is True:
        source = str(head.get("source", "side_2d")).lower()
        key_field = "side_2d_key" if source == "side_2d" else "front_3d_key"
        key = head.get(key_field)
        if not isinstance(key, str) or not key.strip():
            raise ConfigLoadError(
                f"side_headpose.{key_field}는 비어 있지 않은 batch key여야 합니다."
            )
        keys.append(key)

    for feature_name in ("side_eyeangle", "side_eyelidangle"):
        feature = features.get(feature_name)
        if isinstance(feature, Mapping) and feature.get("enabled") is True:
            key = feature.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ConfigLoadError(f"{feature_name}.key는 비어 있지 않은 batch key여야 합니다.")
            keys.append(key)
    if len(keys) != len(set(keys)):
        raise ConfigLoadError("resolved side model forward key에 중복이 있습니다.")
    return tuple(keys)


def select_model_forward_inputs(
    batch: Mapping[str, Any], config: Mapping[str, Any], branch: str
) -> dict[str, Any]:
    """Select exactly the batch tensors declared by a branch input contract."""

    keys = resolve_model_forward_keys(config, branch)
    missing = [key for key in keys if key not in batch]
    if missing:
        raise KeyError(f"model.{str(branch).lower()} forward input이 batch에 없습니다: {missing}")
    return {key: batch[key] for key in keys}


def _load_yaml_mapping(path: Path, *, description: str) -> dict[str, Any]:
    if yaml is None:
        raise ConfigLoadError(
            "YAML을 읽으려면 PyYAML이 필요합니다. 가상환경에서 "
            "'python -m pip install -r requirements.txt'를 먼저 실행하세요."
        )
    if not path.exists():
        raise ConfigLoadError(
            f"{description} 파일을 찾을 수 없습니다: {path}. --config/--profile 경로를 확인하세요."
        )
    if not path.is_file():
        raise ConfigLoadError(f"{description} 경로가 파일이 아닙니다: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigLoadError(f"{description} 파일을 읽지 못했습니다: {path}: {exc}") from exc

    class UniqueKeyLoader(yaml.SafeLoader):  # type: ignore[misc, name-defined]
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        loader.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                line = getattr(key_node.start_mark, "line", 0) + 1
                raise ConfigLoadError(
                    f"{path}:{line}에 YAML key '{key}'가 두 번 정의되어 있습니다."
                )
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(  # type: ignore[attr-defined]
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )

    try:
        loaded = yaml.load(text, Loader=UniqueKeyLoader)
    except ConfigError:
        raise
    except yaml.YAMLError as exc:
        problem = getattr(exc, "problem", None) or str(exc)
        mark = getattr(exc, "problem_mark", None)
        location = ""
        if mark is not None:
            location = f" ({mark.line + 1}행 {mark.column + 1}열)"
        raise ConfigLoadError(f"{description} YAML 문법 오류{location}: {problem}") from exc

    if not isinstance(loaded, dict):
        raise ConfigLoadError(f"{description} 최상위 값은 YAML mapping이어야 합니다: {path}")
    _ensure_string_keys(loaded, source=path)
    return loaded


def _ensure_string_keys(value: Any, *, source: Path, location: str = "<root>") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ConfigLoadError(
                    f"{source}의 {location} 아래 key {key!r}는 문자열이어야 합니다."
                )
            child_location = key if location == "<root>" else f"{location}.{key}"
            _ensure_string_keys(child, source=source, location=child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _ensure_string_keys(child, source=source, location=f"{location}[{index}]")


def _parse_override_value(raw_value: str) -> Any:
    if raw_value == "":
        return ""
    if yaml is None:
        raise ConfigLoadError(
            "CLI override 값을 읽으려면 PyYAML이 필요합니다. requirements.txt를 설치하세요."
        )
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigLoadError(
            f"CLI override 값 '{raw_value}'를 YAML 값으로 해석하지 못했습니다: {exc}"
        ) from exc


def _set_dotted_value(config: MutableMapping[str, Any], parts: Sequence[str], value: Any) -> None:
    current: MutableMapping[str, Any] = config
    dynamic_parent_seen = False
    for index, part in enumerate(parts[:-1]):
        dynamic_parent_seen = dynamic_parent_seen or part in _DYNAMIC_OVERRIDE_PARENTS
        if part not in current:
            if dynamic_parent_seen:
                current[part] = {}
            else:
                prefix = ".".join(parts[: index + 1])
                raise ConfigLoadError(
                    f"CLI override 경로 '{prefix}'가 config에 없습니다. 오타를 확인하세요."
                )
        child = current[part]
        if not isinstance(child, MutableMapping):
            prefix = ".".join(parts[: index + 1])
            raise ConfigLoadError(
                f"CLI override 경로 '{prefix}'는 mapping이 아니어서 하위 값을 설정할 수 없습니다."
            )
        current = child

    final_key = parts[-1]
    if final_key not in current and not (
        dynamic_parent_seen or any(part in _DYNAMIC_OVERRIDE_PARENTS for part in parts[:-1])
    ):
        dotted = ".".join(parts)
        raise ConfigLoadError(
            f"CLI override 경로 '{dotted}'가 config에 없습니다. 오타를 확인하세요."
        )
    current[final_key] = value


class _SafeInterpolationResolver:
    def __init__(
        self,
        source: dict[str, Any],
        *,
        environ: Mapping[str, str],
        now: datetime,
    ) -> None:
        self.source = source
        self.environ = environ
        self.now = now
        self.cache: dict[tuple[str, ...], Any] = {}

    def resolve(self) -> dict[str, Any]:
        return {key: self._resolve_path((key,), stack=()) for key in self.source}

    def _resolve_path(self, path: tuple[str, ...], *, stack: tuple[tuple[str, ...], ...]) -> Any:
        if path in self.cache:
            return copy.deepcopy(self.cache[path])
        if path in stack:
            cycle = " -> ".join(".".join(item) for item in (*stack, path))
            raise ConfigResolutionError(f"config interpolation이 순환 참조합니다: {cycle}")

        raw = self._lookup(path)
        next_stack = (*stack, path)
        if isinstance(raw, Mapping):
            resolved: Any = {key: self._resolve_path((*path, key), stack=next_stack) for key in raw}
        elif isinstance(raw, list):
            resolved = [self._resolve_value(item, stack=next_stack) for item in raw]
        else:
            resolved = self._resolve_value(raw, stack=next_stack)
        self.cache[path] = resolved
        return copy.deepcopy(resolved)

    def _resolve_value(self, value: Any, *, stack: tuple[tuple[str, ...], ...]) -> Any:
        if isinstance(value, str):
            return self._resolve_string(value, stack=stack)
        if isinstance(value, Mapping):
            return {key: self._resolve_value(child, stack=stack) for key, child in value.items()}
        if isinstance(value, list):
            return [self._resolve_value(child, stack=stack) for child in value]
        return copy.deepcopy(value)

    def _resolve_string(self, value: str, *, stack: tuple[tuple[str, ...], ...]) -> Any:
        full_match = _INTERPOLATION_RE.fullmatch(value)
        if full_match:
            return self._resolve_token(full_match.group(1), stack=stack)

        def replace(match: re.Match[str]) -> str:
            resolved = self._resolve_token(match.group(1), stack=stack)
            if isinstance(resolved, Mapping | list):
                raise ConfigResolutionError(
                    f"문자열 '{value}' 안에는 mapping/list 참조를 넣을 수 없습니다."
                )
            return str(resolved)

        return _INTERPOLATION_RE.sub(replace, value)

    def _resolve_token(self, token: str, *, stack: tuple[tuple[str, ...], ...]) -> Any:
        if token.startswith("oc.env:"):
            payload = token[len("oc.env:") :]
            name, separator, default = payload.partition(",")
            name = name.strip()
            if not _ENV_NAME_RE.fullmatch(name):
                raise ConfigResolutionError(f"환경 변수 이름 '{name}'가 올바르지 않습니다.")
            if name in self.environ:
                return self.environ[name]
            if separator:
                return default.strip()
            raise ConfigResolutionError(
                f"필수 환경 변수 {name}가 없습니다. 환경 변수를 설정하거나 "
                f"${{oc.env:{name},기본값}} 형태로 기본값을 지정하세요."
            )

        if token.startswith("now:"):
            date_format = token[len("now:") :]
            if not date_format:
                raise ConfigResolutionError("${now:...}에 날짜 format이 비어 있습니다.")
            return self.now.strftime(date_format)

        if ":" in token:
            resolver_name = token.split(":", 1)[0]
            raise ConfigResolutionError(
                f"보안을 위해 interpolation resolver '{resolver_name}'는 지원하지 않습니다. "
                "oc.env, now 또는 dotted config reference만 사용하세요."
            )
        if not _REFERENCE_RE.fullmatch(token):
            raise ConfigResolutionError(
                f"config reference '${{{token}}}' 형식이 올바르지 않습니다."
            )
        return self._resolve_path(tuple(token.split(".")), stack=stack)

    def _lookup(self, path: tuple[str, ...]) -> Any:
        current: Any = self.source
        for part in path:
            if not isinstance(current, Mapping) or part not in current:
                dotted = ".".join(path)
                raise ConfigResolutionError(
                    f"config reference '${{{dotted}}}'가 가리키는 값이 없습니다."
                )
            current = current[part]
        return current


def _expect_mapping(parent: Mapping[str, Any], key: str, issues: list[str]) -> Mapping[str, Any]:
    value = parent.get(key, _MISSING)
    if not isinstance(value, Mapping):
        issues.append(f"{key}는 YAML mapping이어야 합니다.")
        return {}
    return value


def _validate_image_data(data: Mapping[str, Any], issues: list[str]) -> None:
    mode = data.get("mode")
    if mode != "image":
        issues.append(
            "data.mode은 현재 정지 이미지 파이프라인에서 'image'여야 합니다 "
            f"(입력값: {mode!r}). 영상/sequence 지원은 아직 구현되지 않았습니다."
        )

    dataset_root = data.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root.strip():
        issues.append("data.dataset_root는 비어 있지 않은 경로 문자열이어야 합니다.")

    extensions = data.get("image_extensions")
    if not isinstance(extensions, list) or not extensions:
        issues.append("data.image_extensions는 하나 이상의 확장자를 가진 list여야 합니다.")
    else:
        for extension in extensions:
            if not isinstance(extension, str) or not extension.startswith("."):
                issues.append(
                    "data.image_extensions의 각 값은 '.jpg'처럼 점으로 시작해야 합니다 "
                    f"(입력값: {extension!r})."
                )

    reader = data.get("reader")
    if not isinstance(reader, Mapping):
        issues.append("data.reader는 YAML mapping이어야 합니다.")
        return
    if reader.get("type") not in {
        "mpiifacegaze_text",
        "generic_csv",
        "dual_view_csv",
        "csv_manifest",
    }:
        issues.append("data.reader.type은 mpiifacegaze_text 또는 generic CSV reader여야 합니다.")
    target_bounds_policy = reader.get("target_bounds_policy", "error")
    if target_bounds_policy not in {"error", "keep_flagged"}:
        issues.append("data.reader.target_bounds_policy는 error 또는 keep_flagged여야 합니다.")
    if reader.get("fail_on_bad_row", True) is not True:
        issues.append(
            "data.reader.fail_on_bad_row=false는 현재 executor가 지원하지 않습니다. "
            "잘못된 row를 조용히 건너뛰지 않도록 true를 사용하세요."
        )


def _validate_data_selection(data: Mapping[str, Any], issues: list[str]) -> None:
    """Validate optional record selection before pairing and splitting.

    ``session_ids`` is deliberately an allow-list rather than a filename/path
    pattern.  The manifest builder owns path discovery, while the core data
    pipeline selects on the canonical ``session_id`` contract.
    """

    selection = data.get("selection")
    if selection is None:
        return
    if not isinstance(selection, Mapping):
        issues.append("data.selection은 YAML mapping이어야 합니다.")
        return

    session_ids = selection.get("session_ids")
    if session_ids is None:
        return
    if not isinstance(session_ids, list) or not session_ids:
        issues.append(
            "data.selection.session_ids는 null 또는 비어 있지 않은 session 문자열 list여야 합니다."
        )
        return

    normalized: list[str] = []
    for value in session_ids:
        if not isinstance(value, str) or not value.strip():
            issues.append(
                "data.selection.session_ids의 각 값은 비어 있지 않은 문자열이어야 합니다."
            )
            continue
        normalized.append(value.strip())
    if len(normalized) != len(set(normalized)):
        issues.append("data.selection.session_ids에 중복 session이 있습니다.")


def _validate_views(data: Mapping[str, Any], issues: list[str]) -> set[str]:
    views = data.get("views")
    if not isinstance(views, Mapping):
        issues.append("data.views는 YAML mapping이어야 합니다.")
        return set()

    available = views.get("available")
    if not isinstance(available, list) or not available:
        issues.append("data.views.available은 front/side 중 하나 이상을 가진 list여야 합니다.")
        return set()
    if not all(isinstance(branch, str) for branch in available):
        issues.append("data.views.available의 모든 branch 이름은 문자열이어야 합니다.")
        return set()

    branches = set(available)
    if len(branches) != len(available):
        issues.append("data.views.available에 같은 branch가 중복되어 있습니다.")
    unknown = branches - {"front", "side"}
    if unknown:
        issues.append(
            "data.views.available에는 front와 side만 사용할 수 있습니다 "
            f"(알 수 없는 값: {sorted(unknown)})."
        )

    for mapping_name in ("source_to_branch", "directory_to_branch"):
        branch_mapping = views.get(mapping_name, {})
        if not isinstance(branch_mapping, Mapping):
            issues.append(f"data.views.{mapping_name}는 mapping이어야 합니다.")
            continue
        for source, branch in branch_mapping.items():
            field = f"data.views.{mapping_name}"
            if not isinstance(source, str) or not source.strip():
                issues.append(f"{field}의 source key는 비어 있지 않은 문자열이어야 합니다.")
            if not isinstance(branch, str) or branch not in {"front", "side"}:
                issues.append(
                    f"{field}[{source!r}] 값은 문자열 front 또는 side여야 합니다 "
                    f"(입력값: {branch!r})."
                )
            elif mapping_name == "source_to_branch" and branch not in branches:
                issues.append(
                    f"{field}[{source!r}]={branch!r}이지만 data.views.available에 "
                    f"'{branch}'가 없습니다."
                )
    return branches & {"front", "side"}


def _validate_pairing(
    data: Mapping[str, Any], available_branches: set[str], issues: list[str]
) -> bool:
    pairing = data.get("pairing")
    if not isinstance(pairing, Mapping):
        issues.append("data.pairing은 YAML mapping이어야 합니다.")
        return False
    enabled = pairing.get("enabled")
    if not isinstance(enabled, bool):
        issues.append("data.pairing.enabled는 true 또는 false여야 합니다.")
        return False
    pair_key = pairing.get("pair_id_key")
    if not isinstance(pair_key, str) or not pair_key.strip():
        issues.append("data.pairing.pair_id_key는 비어 있지 않은 문자열이어야 합니다.")
    elif pair_key.strip() in _GENERIC_PAIR_KEY_RESERVED:
        issues.append(
            "data.pairing.pair_id_key가 generic CSV canonical column과 충돌합니다 "
            f"(입력값: {pair_key!r}). pair_id 또는 capture_group과 같은 "
            "별도 column을 사용하세요."
        )
    if not enabled:
        return False

    missing = {"front", "side"} - available_branches
    if missing:
        issues.append(
            "data.pairing.enabled=true이면 data.views.available에 front와 side가 "
            f"모두 있어야 합니다 (누락: {sorted(missing)})."
        )
    if pairing.get("unit") != "image":
        issues.append("현재 pairing 구현에서는 data.pairing.unit이 'image'여야 합니다.")
    if pairing.get("strategy") != "explicit_pair_id":
        issues.append(
            "현재 pairing 구현에서는 data.pairing.strategy가 "
            "'explicit_pair_id'여야 합니다. 파일명/순서로 pair를 추측하지 않습니다."
        )
    policy = pairing.get("unpaired_policy")
    if policy not in {"branch_only", "drop", "error"}:
        issues.append("data.pairing.unpaired_policy는 branch_only, drop, error 중 하나여야 합니다.")
    return True


def _validate_split(data: Mapping[str, Any], issues: list[str]) -> None:
    split = data.get("split")
    if not isinstance(split, Mapping):
        issues.append("data.split은 YAML mapping이어야 합니다.")
        return
    if split.get("strategy") != "grouped_ratio":
        issues.append("현재 splitter에서는 data.split.strategy가 'grouped_ratio'여야 합니다.")
    group_key = split.get("group_key")
    if not isinstance(group_key, str) or not group_key.strip():
        issues.append("data.split.group_key가 필요합니다. subject_id를 권장합니다.")
    if split.get("prevent_group_leakage") is not True:
        issues.append(
            "subject leakage를 막기 위해 data.split.prevent_group_leakage는 true여야 합니다."
        )
    if split.get("reuse_existing_manifest", False) is not False:
        issues.append(
            "data.split.reuse_existing_manifest=true는 현재 executor가 지원하지 않습니다. "
            "false로 두고 매 prepare 결과의 hash를 사용하세요."
        )
    if split.get("stratify_by") is not None:
        issues.append(
            "data.split.stratify_by는 현재 executor가 지원하지 않습니다. null을 사용하세요."
        )

    ratios = split.get("ratios")
    if not isinstance(ratios, Mapping):
        issues.append("data.split.ratios는 train/validation/test mapping이어야 합니다.")
        return
    expected = ("train", "validation", "test")
    numeric_ratios: list[float] = []
    for name in expected:
        value = ratios.get(name, _MISSING)
        if isinstance(value, bool) or not isinstance(value, int | float):
            issues.append(f"data.split.ratios.{name}은 0 이상 1 이하의 유한한 숫자여야 합니다.")
            continue
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            issues.append(
                f"data.split.ratios.{name}은 0 이상 1 이하여야 합니다 (입력값: {number})."
            )
            continue
        if name == "train" and number <= 0.0:
            issues.append("data.split.ratios.train은 0보다 커야 합니다.")
        numeric_ratios.append(number)
    extra = set(ratios) - set(expected)
    if extra:
        issues.append(f"data.split.ratios에 알 수 없는 split이 있습니다: {sorted(extra)}.")
    if len(numeric_ratios) == len(expected) and not math.isclose(
        sum(numeric_ratios), 1.0, rel_tol=0.0, abs_tol=1e-8
    ):
        issues.append(
            "data.split.ratios의 train + validation + test 합은 1이어야 합니다 "
            f"(현재 합: {sum(numeric_ratios):.10g})."
        )


def _validate_preprocessing(
    preprocessing: Mapping[str, Any],
    available_branches: set[str],
    issues: list[str],
) -> None:
    cache = preprocessing.get("cache", {})
    if not isinstance(cache, Mapping):
        issues.append("preprocessing.cache는 mapping이어야 합니다.")
    else:
        enabled = cache.get("enabled", False)
        if not isinstance(enabled, bool):
            issues.append("preprocessing.cache.enabled는 true/false여야 합니다.")
        if enabled is True:
            if str(cache.get("format", "pickle")).lower() != "pickle":
                issues.append("preprocessing.cache.format은 현재 pickle만 지원합니다.")
            _validate_enum(
                cache.get("mode", "read_write"),
                "preprocessing.cache.mode",
                {"read_write", "read_only", "refresh"},
                issues,
            )
            cache_dir = cache.get("dir")
            if not isinstance(cache_dir, str) or not cache_dir.strip():
                issues.append("preprocessing.cache.dir은 비어 있지 않은 경로 문자열이어야 합니다.")
            if cache.get("trusted_local") is not True:
                issues.append(
                    "pickle cache를 사용하려면 preprocessing.cache.trusted_local=true가 필요합니다."
                )
            schema_version = cache.get("schema_version", 1)
            if (
                isinstance(schema_version, bool)
                or not isinstance(schema_version, int)
                or schema_version <= 0
            ):
                issues.append("preprocessing.cache.schema_version은 양의 정수여야 합니다.")
            implementation_id = cache.get("implementation_id")
            if not isinstance(implementation_id, str) or not implementation_id.strip():
                issues.append(
                    "preprocessing.cache.implementation_id는 비어 있지 않은 문자열이어야 합니다."
                )
            max_entry_bytes = cache.get("max_entry_bytes", 512 * 1024 * 1024)
            if (
                isinstance(max_entry_bytes, bool)
                or not isinstance(max_entry_bytes, int)
                or max_entry_bytes <= 0
            ):
                issues.append("preprocessing.cache.max_entry_bytes는 양의 정수여야 합니다.")

    order = preprocessing.get("stage_order")
    valid_order: list[str] = []
    if not isinstance(order, list) or not order:
        issues.append("preprocessing.stage_order는 하나 이상의 stage를 가진 list여야 합니다.")
    elif not all(isinstance(stage, str) for stage in order):
        issues.append("preprocessing.stage_order의 모든 stage 이름은 문자열이어야 합니다.")
    else:
        unknown = [stage for stage in order if stage not in SUPPORTED_PREPROCESSING_STAGES]
        if unknown:
            issues.append(
                f"preprocessing.stage_order에 지원하지 않는 stage가 있습니다: {unknown}. "
                f"지원 stage: {list(SUPPORTED_PREPROCESSING_STAGES)}"
            )
        valid_order = [stage for stage in order if stage in SUPPORTED_PREPROCESSING_STAGES]
        if len(set(order)) != len(order):
            issues.append("preprocessing.stage_order에 중복 stage가 있습니다.")
        for stage in order:
            if stage not in SUPPORTED_PREPROCESSING_STAGES:
                continue
            stage_config = preprocessing.get(stage)
            if not isinstance(stage_config, Mapping):
                issues.append(
                    f"preprocessing.stage_order의 '{stage}'에 대응하는 mapping이 없습니다."
                )
            elif not isinstance(stage_config.get("enabled"), bool):
                issues.append(f"preprocessing.{stage}.enabled는 true/false여야 합니다.")
            else:
                _validate_stage_options(
                    stage,
                    stage_config,
                    field=f"preprocessing.{stage}",
                    issues=issues,
                )

        for stage in SUPPORTED_PREPROCESSING_STAGES:
            if stage not in preprocessing or stage in valid_order:
                continue
            stage_config = preprocessing.get(stage)
            if not isinstance(stage_config, Mapping):
                issues.append(f"preprocessing.{stage}는 stage mapping이어야 합니다.")
            elif stage_config.get("enabled", True) is True:
                issues.append(
                    f"preprocessing.{stage}.enabled=true이지만 stage_order에 '{stage}'가 "
                    "없습니다. executor는 이 stage를 실행하지 않습니다."
                )

    resize = preprocessing.get("resize")
    if isinstance(resize, Mapping) and resize.get("enabled") is True:
        _validate_hw(resize.get("size_hw"), "preprocessing.resize.size_hw", issues)
    eye_region_warp = preprocessing.get("eye_region_warp")
    if isinstance(eye_region_warp, Mapping) and eye_region_warp.get("enabled") is True:
        _validate_hw(
            eye_region_warp.get("size_hw"),
            "preprocessing.eye_region_warp.size_hw",
            issues,
        )

    normalize = preprocessing.get("normalize")
    if isinstance(normalize, Mapping) and normalize.get("enabled") is True:
        mean, std = normalize.get("mean"), normalize.get("std")
        if (mean is None) != (std is None):
            issues.append(
                "preprocessing.normalize.mean과 std는 둘 다 null이거나 둘 다 지정해야 합니다."
            )
        if mean is not None:
            if not (_is_three_numbers(mean) and _is_three_numbers(std)):
                issues.append("RGB normalization의 mean/std는 각각 숫자 3개의 list여야 합니다.")
            elif any(float(item) <= 0 for item in std):
                issues.append("preprocessing.normalize.std의 값은 모두 0보다 커야 합니다.")

    overrides = preprocessing.get("branch_overrides")
    if not isinstance(overrides, Mapping):
        issues.append("preprocessing.branch_overrides는 mapping이어야 합니다.")
        return
    for branch in ("front", "side"):
        branch_config = overrides.get(branch)
        if not isinstance(branch_config, Mapping):
            issues.append(f"preprocessing.branch_overrides.{branch} mapping이 필요합니다.")
            continue
        enabled = branch_config.get("enabled")
        if not isinstance(enabled, bool):
            issues.append(
                f"preprocessing.branch_overrides.{branch}.enabled는 true/false여야 합니다."
            )
        elif enabled and branch not in available_branches:
            issues.append(f"{branch} 전처리를 켰지만 data.views.available에 '{branch}'가 없습니다.")
        _validate_branch_stage_overrides(
            preprocessing,
            branch=branch,
            branch_config=branch_config,
            stage_order=valid_order,
            branch_enabled=enabled is True,
            issues=issues,
        )


def _validate_branch_stage_overrides(
    preprocessing: Mapping[str, Any],
    *,
    branch: str,
    branch_config: Mapping[str, Any],
    stage_order: Sequence[str],
    branch_enabled: bool,
    issues: list[str],
) -> None:
    """Validate a view's stage overrides and their effective dependencies."""

    effective: dict[str, Mapping[str, Any]] = {}
    for stage in SUPPORTED_PREPROCESSING_STAGES:
        global_stage = preprocessing.get(stage)
        if not isinstance(global_stage, Mapping):
            global_stage = {"enabled": False}
        override = branch_config.get(stage, _MISSING)
        field = f"preprocessing.branch_overrides.{branch}.{stage}"
        if override is _MISSING:
            effective[stage] = global_stage
        elif not isinstance(override, Mapping):
            issues.append(f"{field}는 stage override mapping이어야 합니다.")
            effective[stage] = global_stage
        else:
            if "enabled" in override and not isinstance(override.get("enabled"), bool):
                issues.append(f"{field}.enabled는 true/false여야 합니다.")
            _validate_stage_options(stage, override, field=field, issues=issues)
            effective[stage] = deep_merge(global_stage, override)

        if (
            stage not in stage_order
            and effective[stage].get("enabled", True) is True
            and (branch_enabled or override is not _MISSING)
        ):
            issues.append(
                f"{field}.enabled=true이지만 preprocessing.stage_order에 '{stage}'가 없습니다."
            )

    # These stage-specific options are useful even when the branch is disabled
    # in a base config, so validate their shape eagerly. Dependencies apply only
    # when that branch actually participates in the run.
    resize = effective.get("resize", {})
    if resize.get("enabled") is True:
        _validate_hw(
            resize.get("size_hw"),
            f"preprocessing.branch_overrides.{branch}.resize.size_hw",
            issues,
        )
    eye_region_warp = effective.get("eye_region_warp", {})
    if eye_region_warp.get("enabled") is True:
        _validate_hw(
            eye_region_warp.get("size_hw"),
            f"preprocessing.branch_overrides.{branch}.eye_region_warp.size_hw",
            issues,
        )

    face_roi = effective.get("face_roi", {})
    margin_ratio = face_roi.get("margin_ratio")
    if face_roi.get("enabled") is True and margin_ratio is not None:
        if (
            isinstance(margin_ratio, bool)
            or not isinstance(margin_ratio, int | float)
            or float(margin_ratio) < 0.0
        ):
            issues.append(
                f"preprocessing.branch_overrides.{branch}.face_roi.margin_ratio는 "
                "0 이상의 숫자여야 합니다."
            )

    background_mask = effective.get("background_mask", {})
    fill_rgb = background_mask.get("fill_rgb")
    if background_mask.get("enabled") is True and fill_rgb is not None:
        if not _is_rgb_triplet(fill_rgb):
            issues.append(
                f"preprocessing.branch_overrides.{branch}.background_mask.fill_rgb는 "
                "0~255 정수 3개의 list여야 합니다."
            )

    if not branch_enabled or not stage_order:
        return

    _require_effective_stage_dependency(
        effective,
        stage_order,
        branch=branch,
        dependent="eye_selection",
        prerequisite="face_landmarks",
        issues=issues,
    )
    _require_effective_stage_dependency(
        effective,
        stage_order,
        branch=branch,
        dependent="metric_head_pose",
        prerequisite="face_landmarks",
        issues=issues,
    )
    eye_region_warp = effective.get("eye_region_warp", {})
    warp_method = str(eye_region_warp.get("method", "landmark_homography")).lower()
    if warp_method not in {
        "profile90_annotation",
        "profile_annotation",
        "annotation_eye_bbox",
    }:
        _require_effective_stage_dependency(
            effective,
            stage_order,
            branch=branch,
            dependent="eye_region_warp",
            prerequisite="face_landmarks",
            issues=issues,
        )
    _require_effective_stage_dependency(
        effective,
        stage_order,
        branch=branch,
        dependent="face_roi",
        prerequisite="face_landmarks",
        issues=issues,
    )

    eye_selection = effective.get("eye_selection", {})
    metric_head_pose = effective.get("metric_head_pose", {})
    selection_mode = str(eye_selection.get("mode", "both")).lower()
    if eye_selection.get("enabled") is True and selection_mode == "best_visible":
        face_landmarks = effective.get("face_landmarks", {})
        if face_landmarks.get("output_visibility_presence") is not True:
            issues.append(
                f"{branch}.eye_selection.mode={selection_mode!r}이면 "
                f"{branch}.face_landmarks.output_visibility_presence=true가 필요합니다."
            )

    if eye_region_warp.get("enabled") is True and warp_method in {
        "selected_eye_similarity",
        "single_eye_similarity",
    }:
        _require_effective_stage_dependency(
            effective,
            stage_order,
            branch=branch,
            dependent="eye_region_warp",
            prerequisite="eye_selection",
            issues=issues,
        )

    pose_method = str(metric_head_pose.get("method", "webeyetrack_radial_procrustes")).lower()
    if metric_head_pose.get("enabled") is True and pose_method in {
        "webeyetrack_radial_procrustes",
        "webeyetrack_metric_face",
    }:
        face_landmarks = effective.get("face_landmarks", {})
        if face_landmarks.get("output_uvz") is not True:
            issues.append(
                f"{branch}.metric_head_pose.method={pose_method!r}이면 "
                f"{branch}.face_landmarks.output_uvz=true가 필요합니다."
            )
        if face_landmarks.get("output_face_transform") is not True:
            issues.append(
                f"{branch}.metric_head_pose.method={pose_method!r}이면 "
                f"{branch}.face_landmarks.output_face_transform=true가 필요합니다."
            )

    background_method = str(background_mask.get("method", "face_hull")).lower()
    if background_mask.get("enabled") is True and background_method not in {
        "none",
        "noop",
        "no_op",
        "already_masked",
        "existing_black_canvas",
    }:
        _require_effective_stage_dependency(
            effective,
            stage_order,
            branch=branch,
            dependent="background_mask",
            prerequisite="face_landmarks",
            issues=issues,
        )
    if background_mask.get("enabled") is True and background_method in {
        "face_roi_bbox",
        "roi_bbox",
    }:
        _require_effective_stage_dependency(
            effective,
            stage_order,
            branch=branch,
            dependent="background_mask",
            prerequisite="face_roi",
            issues=issues,
        )

    periocular = branch_config.get("periocular_roi")
    if periocular is not None and not isinstance(periocular, Mapping):
        issues.append(
            f"preprocessing.branch_overrides.{branch}.periocular_roi는 mapping이어야 합니다."
        )
    elif (
        isinstance(periocular, Mapping)
        and "enabled" in periocular
        and not isinstance(periocular.get("enabled"), bool)
    ):
        issues.append(
            f"preprocessing.branch_overrides.{branch}.periocular_roi.enabled는 "
            "true/false여야 합니다."
        )
    if isinstance(periocular, Mapping) and periocular.get("enabled") is True:
        if effective.get("face_landmarks", {}).get("enabled") is not True:
            issues.append(
                f"preprocessing.branch_overrides.{branch}.periocular_roi.enabled=true이면 "
                f"{branch}.face_landmarks도 enabled=true여야 합니다."
            )
        elif _stage_index(stage_order, "face_landmarks") is None:
            issues.append(
                f"{branch} periocular_roi를 사용하려면 face_landmarks가 "
                "preprocessing.stage_order에 있어야 합니다."
            )

    _require_enabled_stage_order(
        effective,
        stage_order,
        branch=branch,
        before="eye_selection",
        after="eye_region_warp",
        issues=issues,
    )
    _require_enabled_stage_order(
        effective,
        stage_order,
        branch=branch,
        before="metric_head_pose",
        after="eye_region_warp",
        issues=issues,
    )
    _require_enabled_stage_order(
        effective,
        stage_order,
        branch=branch,
        before="metric_head_pose",
        after="face_roi",
        issues=issues,
    )
    _require_enabled_stage_order(
        effective,
        stage_order,
        branch=branch,
        before="metric_head_pose",
        after="resize",
        issues=issues,
    )
    _require_enabled_stage_order(
        effective,
        stage_order,
        branch=branch,
        before="eye_region_warp",
        after="normalize",
        issues=issues,
    )
    _require_enabled_stage_order(
        effective,
        stage_order,
        branch=branch,
        before="face_roi",
        after="background_mask",
        issues=issues,
    )
    _require_enabled_stage_order(
        effective,
        stage_order,
        branch=branch,
        before="face_roi",
        after="resize",
        issues=issues,
    )
    _require_enabled_stage_order(
        effective,
        stage_order,
        branch=branch,
        before="background_mask",
        after="resize",
        issues=issues,
    )
    _require_enabled_stage_order(
        effective,
        stage_order,
        branch=branch,
        before="resize",
        after="normalize",
        issues=issues,
    )


def _require_effective_stage_dependency(
    effective: Mapping[str, Mapping[str, Any]],
    stage_order: Sequence[str],
    *,
    branch: str,
    dependent: str,
    prerequisite: str,
    issues: list[str],
) -> None:
    if effective.get(dependent, {}).get("enabled") is not True:
        return
    if effective.get(prerequisite, {}).get("enabled") is not True:
        issues.append(
            f"preprocessing.branch_overrides.{branch}.{dependent}.enabled=true이면 "
            f"{branch}.{prerequisite}도 enabled=true여야 합니다."
        )
        return
    prerequisite_index = _stage_index(stage_order, prerequisite)
    dependent_index = _stage_index(stage_order, dependent)
    if prerequisite_index is None or dependent_index is None:
        issues.append(
            f"{branch}.{dependent}를 사용하려면 {prerequisite}와 {dependent}가 "
            "preprocessing.stage_order에 있어야 합니다."
        )
    elif prerequisite_index >= dependent_index:
        issues.append(
            f"preprocessing.stage_order에서 {prerequisite}가 {dependent}보다 "
            f"먼저 실행되어야 합니다 ({branch} branch)."
        )


def _require_enabled_stage_order(
    effective: Mapping[str, Mapping[str, Any]],
    stage_order: Sequence[str],
    *,
    branch: str,
    before: str,
    after: str,
    issues: list[str],
) -> None:
    if not (
        effective.get(before, {}).get("enabled") is True
        and effective.get(after, {}).get("enabled") is True
    ):
        return
    before_index = _stage_index(stage_order, before)
    after_index = _stage_index(stage_order, after)
    if before_index is not None and after_index is not None and before_index >= after_index:
        issues.append(
            f"preprocessing.stage_order에서 {before}가 {after}보다 먼저 실행되어야 합니다 "
            f"({branch} branch)."
        )


def _stage_index(stage_order: Sequence[str], stage: str) -> int | None:
    try:
        return stage_order.index(stage)
    except ValueError:
        return None


def _validate_stage_options(
    stage: str,
    stage_config: Mapping[str, Any],
    *,
    field: str,
    issues: list[str],
) -> None:
    if "on_failure" in stage_config:
        _validate_enum(
            stage_config.get("on_failure"),
            f"{field}.on_failure",
            {
                "error",
                "drop",
                "mark_invalid",
                "use_full_image",
                "skip",
                "noop",
                "no_op",
            },
            issues,
        )

    if stage == "decode":
        _validate_enum(
            stage_config.get("backend", "pillow"),
            f"{field}.backend",
            {"pillow", "pil", "opencv", "cv2"},
            issues,
        )
        if "output_color_order" in stage_config:
            _validate_enum(
                stage_config.get("output_color_order"),
                f"{field}.output_color_order",
                {"rgb"},
                issues,
            )
    elif stage == "face_landmarks":
        _validate_enum(
            stage_config.get("source", "annotation"),
            f"{field}.source",
            {
                "annotation",
                "detector",
                "opencv_haar",
                "haar",
                "opencv-haar",
                "mediapipe_face_mesh",
                "mediapipe_face_landmarker",
            },
            issues,
        )
        fallback = stage_config.get("fallback_detector")
        if fallback is not None:
            _validate_enum(
                fallback,
                f"{field}.fallback_detector",
                {
                    "detector",
                    "opencv_haar",
                    "haar",
                    "opencv-haar",
                    "mediapipe_face_mesh",
                    "mediapipe_face_landmarker",
                },
                issues,
            )
        if "expected_landmarks" in stage_config:
            expected_landmarks = stage_config.get("expected_landmarks")
            if (
                isinstance(expected_landmarks, bool)
                or not isinstance(expected_landmarks, int)
                or expected_landmarks < 4
            ):
                issues.append(f"{field}.expected_landmarks는 4 이상의 정수여야 합니다.")
        for key in (
            "output_uvz",
            "output_visibility_presence",
            "output_face_transform",
            "mirror_retry",
        ):
            if key in stage_config and not isinstance(stage_config.get(key), bool):
                issues.append(f"{field}.{key}는 true/false여야 합니다.")
    elif stage == "eye_selection":
        if "method" in stage_config:
            issues.append(f"{field}.method 대신 {field}.mode를 사용해야 합니다.")
        mode = stage_config.get("mode", "both")
        _validate_enum(
            mode,
            f"{field}.mode",
            {"both", "fixed", "best_visible"},
            issues,
        )
        target_eye = stage_config.get("target_eye")
        if target_eye is not None:
            _validate_enum(
                target_eye,
                f"{field}.target_eye",
                {"left", "right", "auto"},
                issues,
            )
        if (
            isinstance(mode, str)
            and mode.lower() == "fixed"
            and target_eye not in ("left", "right")
        ):
            issues.append(f"{field}.mode='fixed'이면 target_eye는 'left' 또는 'right'여야 합니다.")
        if "min_visibility" in stage_config:
            min_visibility = stage_config.get("min_visibility")
            if not _is_finite_number_in_range(min_visibility, minimum=0.0, maximum=1.0):
                issues.append(f"{field}.min_visibility는 0 이상 1 이하의 숫자여야 합니다.")
        if "canonicalize_to" in stage_config:
            _validate_enum(
                stage_config.get("canonicalize_to"),
                f"{field}.canonicalize_to",
                {"none", "left", "right"},
                issues,
            )
    elif stage == "metric_head_pose":
        _validate_enum(
            stage_config.get("source", "reconstruct"),
            f"{field}.source",
            {
                "reconstruct",
                "precomputed",
                "precomputed_only",
                "precomputed_or_reconstruct",
            },
            issues,
        )
        _validate_enum(
            stage_config.get("precomputed_face_origin_unit", "cm"),
            f"{field}.precomputed_face_origin_unit",
            {"cm", "mm"},
            issues,
        )
        _validate_enum(
            stage_config.get("method", "webeyetrack_radial_procrustes"),
            f"{field}.method",
            {
                "webeyetrack_radial_procrustes",
                "webeyetrack_metric_face",
                "annotation_rt",
            },
            issues,
        )
        _validate_enum(
            stage_config.get("output_unit", "cm"),
            f"{field}.output_unit",
            {"cm", "mm"},
            issues,
        )
        _validate_enum(
            stage_config.get("output_representation", "webeyetrack_components"),
            f"{field}.output_representation",
            {"webeyetrack_components", "rt_matrix", "both"},
            issues,
        )
        _validate_enum(
            stage_config.get("face_origin_source", "annotation_or_reconstruct"),
            f"{field}.face_origin_source",
            {
                "annotation",
                "annotation_only",
                "annotation_or_reconstruct",
                "annotation_first",
                "reconstruct",
                "webeyetrack_reconstruct",
            },
            issues,
        )
        _validate_enum(
            stage_config.get("annotation_unit", "mm"),
            f"{field}.annotation_unit",
            {"mm", "cm"},
            issues,
        )
        _validate_enum(
            stage_config.get("required_irises", "both"),
            f"{field}.required_irises",
            {"both", "selected_eye", "left", "right"},
            issues,
        )
        for key in ("iris_diameter_cm", "initial_depth_cm", "max_depth_step_cm"):
            if key in stage_config and not _is_positive_finite_number(stage_config.get(key)):
                issues.append(f"{field}.{key}는 양의 숫자여야 합니다.")
        if "max_iterations" in stage_config:
            max_iterations = stage_config.get("max_iterations")
            if (
                isinstance(max_iterations, bool)
                or not isinstance(max_iterations, int)
                or max_iterations <= 0
            ):
                issues.append(f"{field}.max_iterations는 양의 정수여야 합니다.")
        if "keep_metric_face" in stage_config and not isinstance(
            stage_config.get("keep_metric_face"), bool
        ):
            issues.append(f"{field}.keep_metric_face는 true/false여야 합니다.")
        intrinsics = stage_config.get("camera_intrinsics")
        if intrinsics is not None:
            if not isinstance(intrinsics, Mapping):
                issues.append(f"{field}.camera_intrinsics는 mapping이어야 합니다.")
            else:
                _validate_enum(
                    intrinsics.get("source", "webeyetrack_estimate"),
                    f"{field}.camera_intrinsics.source",
                    {
                        "webeyetrack_estimate",
                        "annotation",
                        "manifest",
                        "calibration_file",
                    },
                    issues,
                )
        face_scale = stage_config.get("face_scale")
        if face_scale is not None:
            if not isinstance(face_scale, Mapping):
                issues.append(f"{field}.face_scale는 mapping이어야 합니다.")
            else:
                _validate_enum(
                    face_scale.get("method", "iris_diameter"),
                    f"{field}.face_scale.method",
                    {"iris_diameter", "annotation", "fixed"},
                    issues,
                )
                _validate_enum(
                    face_scale.get("required_irises", "both"),
                    f"{field}.face_scale.required_irises",
                    {"both", "selected_eye", "any_visible"},
                    issues,
                )
                if "iris_diameter_cm" in face_scale:
                    iris_diameter = face_scale.get("iris_diameter_cm")
                    if not _is_positive_finite_number(iris_diameter):
                        issues.append(
                            f"{field}.face_scale.iris_diameter_cm는 양의 숫자여야 합니다."
                        )
    elif stage == "eye_region_warp":
        if "method" in stage_config:
            _validate_enum(
                stage_config.get("method"),
                f"{field}.method",
                {
                    "landmark_homography",
                    "webeyetrack_obtain_eyepatch_v1",
                    "webeyetrack_homography",
                    "selected_eye_similarity",
                    "single_eye_similarity",
                    "profile90_annotation",
                    "profile_annotation",
                    "annotation_eye_bbox",
                },
                issues,
            )
        if "size_hw" in stage_config:
            _validate_hw(stage_config.get("size_hw"), f"{field}.size_hw", issues)
        if "face_crop_size" in stage_config:
            face_crop_size = stage_config.get("face_crop_size")
            if (
                isinstance(face_crop_size, bool)
                or not isinstance(face_crop_size, int)
                or face_crop_size <= 0
            ):
                issues.append(f"{field}.face_crop_size는 양의 정수여야 합니다.")
        if "intermediate_size_hw" in stage_config:
            _validate_hw(
                stage_config.get("intermediate_size_hw"),
                f"{field}.intermediate_size_hw",
                issues,
            )
        if "source_quad_indices" in stage_config and not _is_nonnegative_int_list(
            stage_config.get("source_quad_indices"), length=4
        ):
            issues.append(f"{field}.source_quad_indices는 음이 아닌 정수 4개의 list여야 합니다.")
        if "vertical_crop_indices" in stage_config and not _is_nonnegative_int_list(
            stage_config.get("vertical_crop_indices"), length=2
        ):
            issues.append(f"{field}.vertical_crop_indices는 음이 아닌 정수 2개의 list여야 합니다.")
        if "radial_padding_xy" in stage_config:
            radial_padding = stage_config.get("radial_padding_xy")
            if not (
                _is_numeric_pair(radial_padding)
                and all(math.isfinite(float(item)) and float(item) >= 0 for item in radial_padding)
            ):
                issues.append(f"{field}.radial_padding_xy는 0 이상의 숫자 2개여야 합니다.")
        if "center_index" in stage_config:
            center_index = stage_config.get("center_index")
            if (
                isinstance(center_index, bool)
                or not isinstance(center_index, int)
                or center_index < 0
            ):
                issues.append(f"{field}.center_index는 음이 아닌 정수여야 합니다.")
        method = str(stage_config.get("method", "landmark_homography")).lower()
        if method in {
            "profile90_annotation",
            "profile_annotation",
            "annotation_eye_bbox",
        }:
            _validate_enum(
                stage_config.get("crop_mode", "affine"),
                f"{field}.crop_mode",
                {"affine", "letterbox", "stretch"},
                issues,
            )
            for key in ("landmark_crop_scale_xy", "bbox_scale_xy"):
                if key in stage_config:
                    scale = stage_config.get(key)
                    if not (
                        _is_numeric_pair(scale)
                        and all(math.isfinite(float(item)) and float(item) > 0 for item in scale)
                    ):
                        issues.append(f"{field}.{key}는 양의 숫자 2개여야 합니다.")
            if "eyelid_tail_indices" in stage_config:
                tail_indices = stage_config.get("eyelid_tail_indices")
                valid_tail_indices = _is_nonnegative_int_list(tail_indices, length=3) and tuple(
                    tail_indices
                ) in {(0, 1, 5), (3, 2, 4)}
                if not valid_tail_indices:
                    issues.append(
                        f"{field}.eyelid_tail_indices는 corner와 인접 upper/lower index여야 합니다."
                    )
            for key in (
                "bbox_key",
                "eyelid_keypoints_key",
                "iris_center_key",
                "head_origin_key",
                "head_forward_key",
            ):
                value = stage_config.get(key)
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    issues.append(
                        f"{field}.{key}는 비어 있지 않은 metadata field 이름이어야 합니다."
                    )
    elif stage == "face_roi":
        if "source" in stage_config:
            _validate_enum(
                stage_config.get("source"),
                f"{field}.source",
                {"landmarks_bbox"},
                issues,
            )
        if "mode" in stage_config:
            _validate_enum(
                stage_config.get("mode"),
                f"{field}.mode",
                {"crop", "crop_to_roi", "preserve_canvas", "keep_canvas", "mask_only"},
                issues,
            )
    elif stage == "background_mask":
        _validate_enum(
            stage_config.get("method", "face_hull"),
            f"{field}.method",
            {
                "face_hull",
                "bbox",
                "face_roi_bbox",
                "roi_bbox",
                "none",
                "noop",
                "no_op",
                "already_masked",
                "existing_black_canvas",
            },
            issues,
        )
    elif stage == "resize":
        if "size_hw" in stage_config:
            _validate_hw(stage_config.get("size_hw"), f"{field}.size_hw", issues)
        for key in ("interpolation", "interpolation_train", "interpolation_eval"):
            if key in stage_config:
                _validate_enum(
                    stage_config.get(key),
                    f"{field}.{key}",
                    {"nearest", "bilinear", "bicubic", "lanczos"},
                    issues,
                )
    elif stage == "normalize":
        _validate_enum(
            stage_config.get("mode", "zero_one"),
            f"{field}.mode",
            {
                "zero_one",
                "0_1",
                "unit",
                "mean_std",
                "standardize",
                "imagenet",
                "none",
                "identity",
                "raw",
            },
            issues,
        )
        _validate_enum(
            stage_config.get("output_dtype", "float32"),
            f"{field}.output_dtype",
            {"float32", "float", "float16"},
            issues,
        )
        _validate_enum(
            stage_config.get("channel_order", "CHW"),
            f"{field}.channel_order",
            {"chw", "hwc"},
            issues,
        )
    elif stage == "augment" and "apply_to" in stage_config:
        apply_to = stage_config.get("apply_to")
        if not (
            apply_to == "train_only"
            or (
                isinstance(apply_to, list)
                and all(item in {"train", "validation", "test"} for item in apply_to)
            )
        ):
            issues.append(
                f"{field}.apply_to는 train_only 또는 train/validation/test list여야 합니다."
            )


def _validate_enum(value: Any, field: str, allowed: set[str], issues: list[str]) -> None:
    normalized = value.lower() if isinstance(value, str) else None
    if normalized not in allowed:
        issues.append(f"{field}={value!r}은 지원하지 않습니다. 허용값: {sorted(allowed)}")


def _validate_models(
    model: Mapping[str, Any],
    available_branches: set[str],
    *,
    require_model_entrypoints: bool,
    issues: list[str],
) -> set[str]:
    if model.get("backend") != "pytorch":
        issues.append("현재 model.backend는 'pytorch'만 지원합니다.")

    enabled_models: set[str] = set()
    for branch in ("front", "side"):
        branch_config = model.get(branch)
        if not isinstance(branch_config, Mapping):
            issues.append(f"model.{branch}는 mapping이어야 합니다.")
            continue
        enabled = branch_config.get("enabled")
        if not isinstance(enabled, bool):
            issues.append(f"model.{branch}.enabled는 true/false여야 합니다.")
            continue
        if enabled:
            enabled_models.add(branch)
            if branch not in available_branches:
                issues.append(
                    f"model.{branch}.enabled=true이지만 data.views.available에 "
                    f"'{branch}'가 없습니다."
                )

        entrypoint = branch_config.get("entrypoint")
        _validate_entrypoint(
            entrypoint,
            f"model.{branch}.entrypoint",
            required=enabled and require_model_entrypoints,
            issues=issues,
        )
        adapter = branch_config.get("adapter_entrypoint")
        _validate_entrypoint(
            adapter,
            f"model.{branch}.adapter_entrypoint",
            required=False,
            issues=issues,
        )

        input_contract = branch_config.get("input_contract")
        if enabled and not isinstance(input_contract, Mapping):
            issues.append(f"model.{branch}.input_contract mapping이 필요합니다.")
        elif isinstance(input_contract, Mapping):
            _validate_image_shape(
                input_contract.get("shape"),
                f"model.{branch}.input_contract.shape",
                issues,
            )
        output_contract = branch_config.get("output_contract")
        if enabled and not isinstance(output_contract, Mapping):
            issues.append(f"model.{branch}.output_contract mapping이 필요합니다.")
        elif isinstance(output_contract, Mapping):
            gaze_shape = output_contract.get("gaze_shape")
            if branch == "front":
                if gaze_shape != ["B", 2] and gaze_shape != ["B", 2.0]:
                    issues.append("model.front.output_contract.gaze_shape은 [B, 2]여야 합니다.")
            else:
                delta_shape = output_contract.get("delta_y_shape")
                delta_key = output_contract.get("delta_y_key")
                if gaze_shape not in (None, ["B", 2], ["B", 2.0]):
                    issues.append(
                        "model.side.output_contract.gaze_shape은 null 또는 [B, 2]여야 합니다."
                    )
                if gaze_shape is None and delta_shape not in (["B", 1], ["B", 1.0]):
                    issues.append(
                        "Side y-residual 계약은 "
                        "model.side.output_contract.delta_y_shape=[B, 1]이 필요합니다."
                    )
                if gaze_shape is None and (not isinstance(delta_key, str) or not delta_key.strip()):
                    issues.append(
                        "Side y-residual 계약은 "
                        "model.side.output_contract.delta_y_key가 필요합니다."
                    )

        pretrained = branch_config.get("pretrained")
        if isinstance(pretrained, Mapping):
            sha256 = pretrained.get("sha256")
            if sha256 is not None and (
                not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256)
            ):
                issues.append(
                    f"model.{branch}.pretrained.sha256은 64자리 SHA-256 hex 문자열이어야 합니다."
                )
    return enabled_models


def _validate_side_feature_selection(
    preprocessing: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    enabled_models: set[str],
    pairing_enabled: bool,
    issues: list[str],
) -> None:
    """Validate strict-profile auxiliary toggles and cross-view head input."""

    overrides = preprocessing.get("branch_overrides")
    side_override = overrides.get("side") if isinstance(overrides, Mapping) else None
    if not isinstance(side_override, Mapping):
        return
    effective_side = _effective_branch_stages(preprocessing, side_override)
    warp = effective_side.get("eye_region_warp", {})
    features = warp.get("feature_extraction") if isinstance(warp, Mapping) else None
    side_model = model.get("side")
    side_contract = side_model.get("input_contract") if isinstance(side_model, Mapping) else None
    forward_keys = side_contract.get("forward_keys") if isinstance(side_contract, Mapping) else None

    if features is None:
        if forward_keys == "auto":
            issues.append(
                "model.side.input_contract.forward_keys=auto이면 side "
                "eye_region_warp.feature_extraction mapping이 필요합니다."
            )
        return
    if not isinstance(features, Mapping):
        issues.append(
            "preprocessing.branch_overrides.side.eye_region_warp.feature_extraction은 "
            "mapping이어야 합니다."
        )
        return

    field_prefix = "preprocessing.branch_overrides.side.eye_region_warp.feature_extraction"
    parsed: dict[str, Mapping[str, Any]] = {}
    for name in ("side_headpose", "side_eyeangle", "side_eyelidangle"):
        feature = features.get(name)
        if not isinstance(feature, Mapping):
            issues.append(f"{field_prefix}.{name} mapping이 필요합니다.")
            continue
        parsed[name] = feature
        if not isinstance(feature.get("enabled"), bool):
            issues.append(f"{field_prefix}.{name}.enabled는 true/false여야 합니다.")

    if "side" in enabled_models:
        if warp.get("enabled") is not True:
            issues.append(
                "strict-profile side model은 feature toggle과 무관하게 "
                "eye_region_warp.enabled=true여야 합니다."
            )
        if str(warp.get("method", "")).lower() not in {
            "profile90_annotation",
            "profile_annotation",
            "annotation_eye_bbox",
        }:
            issues.append(
                "strict-profile side feature_extraction에는 annotation 기반 eye ROI가 필요합니다."
            )

    head = parsed.get("side_headpose")
    if head is not None:
        source = str(head.get("source", "")).lower()
        if source not in {"side_2d", "front_3d"}:
            issues.append(f"{field_prefix}.side_headpose.source는 side_2d/front_3d여야 합니다.")
        for key_field in ("side_2d_key", "front_3d_key"):
            key = head.get(key_field)
            if not isinstance(key, str) or not key.strip():
                issues.append(f"{field_prefix}.side_headpose.{key_field}가 필요합니다.")
        canonical_head_keys = {
            "side_2d_key": "side_head_pose_2d",
            "front_3d_key": "front_head_vector",
            "front_3d_validity_key": "front_head_orientation_valid",
        }
        for key_field, expected in canonical_head_keys.items():
            if head.get(key_field) != expected:
                issues.append(
                    f"{field_prefix}.side_headpose.{key_field}는 현재 dataset이 생성하는 "
                    f"canonical key '{expected}'여야 합니다."
                )
        _validate_enum(
            head.get("front_3d_invalid_policy", "zero_fill_and_mask"),
            f"{field_prefix}.side_headpose.front_3d_invalid_policy",
            {"zero_fill_and_mask", "error"},
            issues,
        )
        if head.get("front_3d_coordinate_frame", "front_camera") != "front_camera":
            issues.append(
                f"{field_prefix}.side_headpose.front_3d_coordinate_frame는 front_camera여야 합니다."
            )
        if head.get("enabled") is True and source == "front_3d":
            if not pairing_enabled:
                issues.append(
                    "side_headpose.source=front_3d이면 data.pairing.enabled=true여야 합니다."
                )
            front_override = overrides.get("front") if isinstance(overrides, Mapping) else None
            effective_front = (
                _effective_branch_stages(preprocessing, front_override)
                if isinstance(front_override, Mapping)
                else {}
            )
            if not isinstance(front_override, Mapping) or front_override.get("enabled") is not True:
                issues.append(
                    "side_headpose.source=front_3d이면 front preprocessing branch가 켜져야 합니다."
                )
            if effective_front.get("metric_head_pose", {}).get("enabled") is not True:
                issues.append(
                    "side_headpose.source=front_3d이면 front metric_head_pose가 켜져야 합니다."
                )

    for name in ("side_eyeangle", "side_eyelidangle"):
        feature = parsed.get(name)
        if feature is None:
            continue
        key = feature.get("key")
        if not isinstance(key, str) or not key.strip():
            issues.append(f"{field_prefix}.{name}.key가 필요합니다.")

    canonical_feature_keys = {
        "side_eyeangle": "side_eye_angles",
        "side_eyelidangle": "side_iris_pose_2d",
    }
    for name, expected in canonical_feature_keys.items():
        feature = parsed.get(name)
        if feature is not None and feature.get("key") != expected:
            issues.append(
                f"{field_prefix}.{name}.key는 현재 dataset이 생성하는 canonical key "
                f"'{expected}'여야 합니다."
            )

    iris_feature = parsed.get("side_eyelidangle")
    if iris_feature is not None and not isinstance(iris_feature.get("vertical_only"), bool):
        issues.append(f"{field_prefix}.side_eyelidangle.vertical_only는 true/false여야 합니다.")
    if isinstance(warp, Mapping) and "eye_vector_vertical_only" in warp:
        issues.append(
            "preprocessing.branch_overrides.side.eye_region_warp.eye_vector_vertical_only는 "
            "더 이상 지원하지 않습니다. feature_extraction.side_eyelidangle.vertical_only를 "
            "사용하세요."
        )

    if isinstance(side_contract, Mapping) and side_contract.get("image_key") != "side_image":
        issues.append(
            "strict-profile model.side.input_contract.image_key는 'side_image'여야 합니다."
        )

    if forward_keys != "auto" and features is not None and "side" in enabled_models:
        issues.append(
            "selectable side features에는 model.side.input_contract.forward_keys='auto'를 "
            "사용해야 enabled toggle이 자동 반영됩니다."
        )


def _validate_preprocessing_model_contracts(
    preprocessing: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    enabled_models: set[str],
    issues: list[str],
) -> None:
    raw_order = preprocessing.get("stage_order")
    if not isinstance(raw_order, list):
        return
    stage_order = [
        stage
        for stage in raw_order
        if isinstance(stage, str) and stage in SUPPORTED_PREPROCESSING_STAGES
    ]
    overrides = preprocessing.get("branch_overrides", {})
    if not isinstance(overrides, Mapping):
        return

    for branch in sorted(enabled_models):
        branch_model = model.get(branch)
        branch_override = overrides.get(branch)
        if not isinstance(branch_model, Mapping) or not isinstance(branch_override, Mapping):
            continue
        input_contract = branch_model.get("input_contract")
        if not isinstance(input_contract, Mapping):
            continue

        effective = _effective_branch_stages(preprocessing, branch_override)
        fixed_stage: str | None = None
        fixed_size: list[int] | None = None
        for stage in stage_order:
            if stage not in {"eye_region_warp", "resize"}:
                continue
            stage_config = effective.get(stage, {})
            if stage_config.get("enabled") is not True:
                continue
            size_hw = stage_config.get("size_hw")
            if _is_positive_hw(size_hw):
                fixed_stage = stage
                fixed_size = [int(size_hw[0]), int(size_hw[1])]

        shape = input_contract.get("shape")
        if _is_bchw_shape(shape):
            expected_size = [int(shape[2]), int(shape[3])]
            if fixed_size is None:
                issues.append(
                    f"model.{branch}.input_contract.shape={shape}은 고정 H,W를 요구하지만 "
                    f"{branch} preprocessing에 enabled eye_region_warp/resize가 없습니다."
                )
            elif fixed_size != expected_size:
                issues.append(
                    f"{branch} preprocessing의 마지막 fixed-size stage '{fixed_stage}' 출력은 "
                    f"{fixed_size}이지만 model.{branch}.input_contract.shape의 H,W는 "
                    f"{expected_size}입니다."
                )

        normalize = effective.get("normalize", {})
        normalize_runs = "normalize" in stage_order and normalize.get("enabled") is True
        model_dtype = _canonical_float_dtype(input_contract.get("dtype"))
        model_channel_order = "CHW" if _is_bchw_shape(shape) else None
        if not normalize_runs:
            if model_dtype is not None or model_channel_order == "CHW":
                issues.append(
                    f"model.{branch}.input_contract는 BCHW floating tensor를 요구하지만 "
                    f"{branch} preprocessing.normalize가 실행되지 않아 uint8 HWC image가 남습니다."
                )
            continue

        normalize_channel_order = str(normalize.get("channel_order", "CHW")).upper()
        if model_channel_order == "CHW" and normalize_channel_order != "CHW":
            issues.append(
                f"{branch} normalize.channel_order={normalize_channel_order!r}이지만 "
                f"model.{branch}.input_contract.shape은 [B,3,H,W] CHW 형식입니다."
            )

        normalize_dtype = _canonical_float_dtype(normalize.get("output_dtype", "float32"))
        if (
            model_dtype is not None
            and normalize_dtype is not None
            and normalize_dtype != model_dtype
        ):
            issues.append(
                f"{branch} normalize.output_dtype={normalize.get('output_dtype')!r}와 "
                f"model.{branch}.input_contract.dtype={input_contract.get('dtype')!r}가 "
                "일치하지 않습니다."
            )

        color_order = input_contract.get("color_order")
        if isinstance(color_order, str) and color_order.upper() != "RGB":
            issues.append(
                f"preprocessing은 RGB를 출력하지만 model.{branch}.input_contract.color_order="
                f"{color_order!r}입니다. adapter 변환이 config에 명시되지 않았습니다."
            )

        mode = str(normalize.get("mode", "zero_one")).lower()
        model_range = input_contract.get("value_range")
        expected_range: list[float] | None = None
        if mode in {"zero_one", "0_1", "unit"}:
            expected_range = [0.0, 1.0]
        elif mode in {"none", "identity", "raw"}:
            expected_range = [0.0, 255.0]
        if expected_range is not None and _is_numeric_pair(model_range):
            observed = [float(model_range[0]), float(model_range[1])]
            if observed != expected_range:
                issues.append(
                    f"{branch} normalize.mode={mode!r}의 명시적 value range는 "
                    f"{expected_range}이지만 model.{branch}.input_contract.value_range는 "
                    f"{observed}입니다."
                )
        elif mode in {"mean_std", "standardize", "imagenet"} and _is_numeric_pair(model_range):
            issues.append(
                f"{branch} normalize.mode={mode!r}는 고정 [0,1]/[0,255] 범위를 출력하지 "
                f"않으므로 model.{branch}.input_contract.value_range를 숫자 범위로 선언할 "
                "수 없습니다. null 또는 standardized contract를 사용하세요."
            )


def _effective_branch_stages(
    preprocessing: Mapping[str, Any], branch_override: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    effective: dict[str, Mapping[str, Any]] = {}
    for stage in SUPPORTED_PREPROCESSING_STAGES:
        base = preprocessing.get(stage)
        base_mapping = base if isinstance(base, Mapping) else {"enabled": False}
        override = branch_override.get(stage)
        effective[stage] = (
            deep_merge(base_mapping, override) if isinstance(override, Mapping) else base_mapping
        )
    return effective


def _canonical_float_dtype(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.lower()
    if normalized in {"float", "float32"}:
        return "float32"
    if normalized == "float16":
        return "float16"
    return None


def _validate_fusion(
    fusion: Mapping[str, Any],
    *,
    available_branches: set[str],
    enabled_models: set[str],
    pairing_enabled: bool,
    model: Mapping[str, Any],
    require_model_entrypoints: bool,
    issues: list[str],
) -> None:
    enabled = fusion.get("enabled")
    if not isinstance(enabled, bool):
        issues.append("fusion.enabled는 true 또는 false여야 합니다.")
        return
    if not enabled:
        return

    if not pairing_enabled:
        issues.append("fusion.enabled=true이면 data.pairing.enabled도 true여야 합니다.")
    missing_views = {"front", "side"} - available_branches
    if missing_views:
        issues.append(
            f"fusion에는 front/side view가 모두 필요합니다 (누락: {sorted(missing_views)})."
        )
    missing_models = {"front", "side"} - enabled_models
    if missing_models:
        issues.append(
            "fusion에는 front/side model이 모두 enabled여야 합니다 "
            f"(누락: {sorted(missing_models)})."
        )
    if fusion.get("stage") != "late":
        issues.append("현재 fusion.stage는 'late'만 지원합니다.")
    method = fusion.get("method")
    if not isinstance(method, str) or not method.strip():
        issues.append("fusion.method가 필요합니다.")
    elif method != "y_axis_residual":
        issues.append("현재 fusion.method는 y_axis_residual만 지원합니다.")
    if fusion.get("output_shape") != ["B", 2]:
        issues.append("fusion.output_shape은 [B, 2]여야 합니다.")
    if fusion.get("missing_branch_policy") != "use_available_branch":
        issues.append("현재 fusion.missing_branch_policy는 use_available_branch만 지원합니다.")

    if method == "y_axis_residual":
        expected_inputs = {"front.gaze_xy", "side.delta_y_side"}
        input_keys = fusion.get("input_keys")
        if not isinstance(input_keys, list) or set(input_keys) != expected_inputs:
            issues.append(
                "fusion.method=y_axis_residual이면 input_keys는 "
                "front.gaze_xy와 side.delta_y_side여야 합니다."
            )
        side_config = model.get("side", {})
        side_output = (
            side_config.get("output_contract", {}) if isinstance(side_config, Mapping) else {}
        )
        if not isinstance(side_output, Mapping) or side_output.get("delta_y_shape") not in (
            ["B", 1],
            ["B", 1.0],
        ):
            issues.append(
                "y_axis_residual fusion에는 model.side.output_contract.delta_y_shape=[B, 1]이 "
                "필요합니다."
            )
    _validate_entrypoint(
        fusion.get("entrypoint"),
        "fusion.entrypoint",
        required=False,
        issues=issues,
    )
    if fusion.get("entrypoint") not in {None, ""}:
        issues.append("현재 fusion.entrypoint는 지원하지 않습니다.")

    if fusion.get("use_uncertainty") is True:
        for branch in ("front", "side"):
            branch_config = model.get(branch, {})
            output_contract = (
                branch_config.get("output_contract", {})
                if isinstance(branch_config, Mapping)
                else {}
            )
            uncertainty_key = (
                output_contract.get("uncertainty_key")
                if isinstance(output_contract, Mapping)
                else None
            )
            if not uncertainty_key:
                issues.append(
                    f"fusion.use_uncertainty=true이면 "
                    f"model.{branch}.output_contract.uncertainty_key가 필요합니다."
                )

    # This parameter is intentionally unused for prepare; keeping it in this
    # signature makes the phase distinction explicit for future train commands.
    _ = require_model_entrypoints


def _validate_entrypoint(value: Any, path: str, *, required: bool, issues: list[str]) -> None:
    if value is None or value == "":
        if required:
            issues.append(
                f"{path}가 필요합니다. 'package.module:create_model' 형식으로 지정하세요."
            )
        return
    if not isinstance(value, str) or not _ENTRYPOINT_RE.fullmatch(value):
        issues.append(f"{path}는 'package.module:callable' 형식이어야 합니다 (입력값: {value!r}).")


def _validate_execution_config(config: Mapping[str, Any], issues: list[str]) -> None:
    """Reject values that the built-in train/evaluate runner cannot honor.

    These checks intentionally cover only the runner's stable, shared fields.
    Profiles may keep model- or experiment-specific extension namespaces.
    """

    training = _expect_mapping(config, "training", issues)
    for key in (
        "max_epochs",
        "gradient_accumulation_steps",
        "validate_every_n_epochs",
    ):
        value = training.get(key, _MISSING)
        if not _is_positive_int(value):
            issues.append(f"training.{key}는 1 이상의 정수여야 합니다 (입력값: {value!r}).")
    devices = training.get("devices", _MISSING)
    if not (isinstance(devices, int) and not isinstance(devices, bool) and devices == 1):
        issues.append(f"현재 training.devices=1만 지원합니다 (입력값: {devices!r}).")
    precision = training.get("precision", _MISSING)
    if not (isinstance(precision, int) and not isinstance(precision, bool) and precision == 32):
        issues.append(f"현재 training.precision=32만 지원합니다 (입력값: {precision!r}).")

    optimizer = _expect_mapping(config, "optimizer", issues)
    optimizer_name = optimizer.get("name")
    normalized_optimizer = optimizer_name.lower() if isinstance(optimizer_name, str) else None
    if normalized_optimizer not in {"adam", "adamw"}:
        issues.append(f"optimizer.name은 Adam 또는 AdamW여야 합니다 (입력값: {optimizer_name!r}).")
    learning_rate = optimizer.get("learning_rate", _MISSING)
    if not _is_positive_finite_number(learning_rate):
        issues.append(
            f"optimizer.learning_rate는 0보다 큰 유한한 수여야 합니다 (입력값: {learning_rate!r})."
        )

    scheduler = _expect_mapping(config, "scheduler", issues)
    _validate_scheduler_execution_config(scheduler, issues)

    loss = _expect_mapping(config, "loss", issues)
    primary = loss.get("primary")
    if not isinstance(primary, Mapping):
        issues.append("loss.primary는 YAML mapping이어야 합니다.")
    else:
        loss_name = primary.get("name")
        normalized_loss = loss_name.lower() if isinstance(loss_name, str) else None
        supported_losses = {
            "huber",
            "huber_xy",
            "smooth_l1",
            "mse",
            "mse_xy",
            "weighted_l2_xy",
            "l2",
        }
        if normalized_loss not in supported_losses:
            issues.append(
                "loss.primary.name은 Huber, MSE 또는 weighted_l2 계열이어야 합니다 "
                f"(입력값: {loss_name!r})."
            )

    metrics = _expect_mapping(config, "metrics", issues)
    threshold_rates = metrics.get("threshold_rates", [])
    threshold_names: set[str] = set()
    if not isinstance(threshold_rates, list):
        issues.append("metrics.threshold_rates는 mapping list여야 합니다.")
    else:
        for index, specification in enumerate(threshold_rates):
            field = f"metrics.threshold_rates[{index}]"
            if not isinstance(specification, Mapping):
                issues.append(f"{field}는 mapping이어야 합니다.")
                continue
            name = specification.get("name")
            if not isinstance(name, str) or not name.strip():
                issues.append(f"{field}.name은 비어 있지 않은 문자열이어야 합니다.")
            elif name in threshold_names:
                issues.append(f"{field}.name은 중복될 수 없습니다: {name!r}.")
            else:
                threshold_names.add(name)
            _validate_enum(
                specification.get("unit", "normalized"),
                f"{field}.unit",
                {"normalized", "pixel", "cm"},
                issues,
            )
            if not _is_positive_finite_number(specification.get("threshold")):
                issues.append(f"{field}.threshold는 0보다 큰 유한한 수여야 합니다.")
            if "enabled" in specification and not isinstance(specification.get("enabled"), bool):
                issues.append(f"{field}.enabled는 true/false여야 합니다.")
    report = metrics.get("report")
    if not isinstance(report, list) or not all(isinstance(name, str) for name in report):
        issues.append("metrics.report는 metric 이름 문자열 list여야 합니다.")
    else:
        known_metrics = {
            "euclidean_normalized_mean",
            "euclidean_normalized_median",
            "mae_x_normalized",
            "mae_y_normalized",
            "rmse_normalized",
            "out_of_bounds_rate",
            "subject_macro_euclidean_normalized",
            "euclidean_pixel_mean",
            "subject_macro_euclidean_pixel",
            "euclidean_cm_mean",
            "p90_euclidean_cm",
            "p95_euclidean_cm",
            "subject_macro_euclidean_cm",
            *threshold_names,
        }
        unknown_metrics = sorted(set(report) - known_metrics)
        if unknown_metrics:
            issues.append(f"metrics.report에 지원하지 않는 이름이 있습니다: {unknown_metrics}.")
    for key in ("selection_metric", "fallback_selection_metric"):
        value = metrics.get(key)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"metrics.{key}은 비어 있지 않은 metric 이름이어야 합니다.")

    checkpoint = _expect_mapping(config, "checkpoint", issues)
    save_best = checkpoint.get("save_best")
    if not isinstance(save_best, Mapping):
        issues.append("checkpoint.save_best는 YAML mapping이어야 합니다.")
    else:
        mode = save_best.get("mode")
        normalized_mode = mode.lower() if isinstance(mode, str) else None
        if normalized_mode not in {"min", "max"}:
            issues.append(
                f"checkpoint.save_best.mode는 min 또는 max여야 합니다 (입력값: {mode!r})."
            )

    model_export = _expect_mapping(config, "model_export", issues)
    export_format = model_export.get("format")
    if export_format != "pytorch_state_dict":
        issues.append(
            "현재 model_export.format은 'pytorch_state_dict'만 지원합니다 "
            f"(입력값: {export_format!r})."
        )

    mlflow = _expect_mapping(config, "mlflow", issues)
    for key in (
        "enabled",
        "log_system_metrics",
        "log_resolved_config",
        "log_dataset_manifest",
        "log_split_manifest",
        "log_predictions",
        "log_environment",
    ):
        value = mlflow.get(key, _MISSING)
        if value is not _MISSING and not isinstance(value, bool):
            issues.append(f"mlflow.{key}는 true 또는 false여야 합니다 (입력값: {value!r}).")


def _validate_scheduler_execution_config(scheduler: Mapping[str, Any], issues: list[str]) -> None:
    enabled = scheduler.get("enabled", _MISSING)
    if not isinstance(enabled, bool):
        issues.append(f"scheduler.enabled는 true 또는 false여야 합니다 (입력값: {enabled!r}).")
        return
    if not enabled:
        return

    name = scheduler.get("name")
    normalized_name = name.lower() if isinstance(name, str) else None
    if normalized_name not in {"reducelronplateau", "exponentiallr"}:
        issues.append(
            "scheduler.name은 ReduceLROnPlateau 또는 ExponentialLR이어야 합니다 "
            f"(입력값: {name!r})."
        )
        return

    if normalized_name == "exponentiallr":
        gamma = scheduler.get("gamma", _MISSING)
        if not _is_positive_finite_number(gamma):
            issues.append(f"scheduler.gamma는 0보다 큰 유한한 수여야 합니다 (입력값: {gamma!r}).")
        return

    mode = scheduler.get("mode")
    normalized_mode = mode.lower() if isinstance(mode, str) else None
    if normalized_mode not in {"min", "max"}:
        issues.append(f"scheduler.mode는 min 또는 max여야 합니다 (입력값: {mode!r}).")
    factor = scheduler.get("factor", _MISSING)
    if not _is_finite_number_in_range(factor, minimum=0.0, maximum=1.0) or float(factor) in {
        0.0,
        1.0,
    }:
        issues.append(
            "ReduceLROnPlateau scheduler.factor는 0보다 크고 1보다 작아야 합니다 "
            f"(입력값: {factor!r})."
        )
    patience = scheduler.get("patience_epochs", _MISSING)
    if not (isinstance(patience, int) and not isinstance(patience, bool) and patience >= 0):
        issues.append(
            "ReduceLROnPlateau scheduler.patience_epochs는 0 이상의 정수여야 합니다 "
            f"(입력값: {patience!r})."
        )
    minimum_learning_rate = scheduler.get("min_learning_rate", _MISSING)
    if not _is_nonnegative_finite_number(minimum_learning_rate):
        issues.append(
            "ReduceLROnPlateau scheduler.min_learning_rate는 0 이상의 유한한 수여야 합니다 "
            f"(입력값: {minimum_learning_rate!r})."
        )


def _validate_filesystem_paths(
    config: Mapping[str, Any], *, base_dir: Path, issues: list[str]
) -> None:
    data = config.get("data", {})
    dataset_root = data.get("dataset_root") if isinstance(data, Mapping) else None
    if isinstance(dataset_root, str) and dataset_root:
        resolved_data_root = _resolve_local_path(dataset_root, base_dir)
        if not resolved_data_root.exists():
            issues.append(
                "dataset 경로가 없습니다: "
                f"{resolved_data_root}. GAZE_DATA_ROOT를 실제 dataset 폴더로 설정하세요."
            )
        elif not resolved_data_root.is_dir():
            issues.append(f"data.dataset_root가 폴더가 아닙니다: {resolved_data_root}")

    paths = config.get("paths", {})
    output_root = paths.get("output_root") if isinstance(paths, Mapping) else None
    if isinstance(output_root, str) and output_root:
        resolved_output_root = _resolve_local_path(output_root, base_dir)
        if resolved_output_root.exists() and not resolved_output_root.is_dir():
            issues.append(f"paths.output_root가 폴더가 아닙니다: {resolved_output_root}")
        else:
            parent = _nearest_existing_parent(resolved_output_root)
            if parent is None or not os.access(parent, os.W_OK):
                issues.append(
                    f"output 경로를 만들 권한이 없습니다: {resolved_output_root}. "
                    "GAZE_OUTPUT_ROOT를 쓰기 가능한 폴더로 바꾸세요."
                )

    model = config.get("model", {})
    if isinstance(model, Mapping):
        for branch in ("front", "side"):
            branch_config = model.get(branch, {})
            if not isinstance(branch_config, Mapping):
                continue
            source_dir = branch_config.get("source_dir")
            if source_dir is not None:
                _check_existing_path(
                    source_dir,
                    f"model.{branch}.source_dir",
                    base_dir,
                    want_directory=True,
                    issues=issues,
                )
            pretrained = branch_config.get("pretrained", {})
            if isinstance(pretrained, Mapping) and pretrained.get("path") is not None:
                _check_existing_path(
                    pretrained.get("path"),
                    f"model.{branch}.pretrained.path",
                    base_dir,
                    want_directory=False,
                    issues=issues,
                )

    checkpoint = config.get("checkpoint", {})
    if isinstance(checkpoint, Mapping) and checkpoint.get("resume_from") is not None:
        _check_existing_path(
            checkpoint.get("resume_from"),
            "checkpoint.resume_from",
            base_dir,
            want_directory=False,
            issues=issues,
        )


def _check_existing_path(
    value: Any,
    field: str,
    base_dir: Path,
    *,
    want_directory: bool,
    issues: list[str],
) -> None:
    if not isinstance(value, str) or not value.strip():
        issues.append(f"{field}는 비어 있지 않은 경로 문자열이어야 합니다.")
        return
    path = _resolve_local_path(value, base_dir)
    if not path.exists():
        issues.append(f"{field} 경로가 없습니다: {path}")
    elif want_directory and not path.is_dir():
        issues.append(f"{field}는 폴더여야 합니다: {path}")
    elif not want_directory and not path.is_file():
        issues.append(f"{field}는 파일이어야 합니다: {path}")


def _resolve_local_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while not current.exists():
        if current.parent == current:
            return None
        current = current.parent
    return current if current.is_dir() else current.parent


def _validate_hw(value: Any, field: str, issues: list[str]) -> None:
    if not (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value)
    ):
        issues.append(f"{field}는 [height, width] 형식의 양의 정수 2개여야 합니다.")


def _validate_image_shape(value: Any, field: str, issues: list[str]) -> None:
    if not _is_bchw_shape(value):
        issues.append(f"{field}는 [B, 3, H, W] RGB image 형식이어야 합니다.")


def _is_bchw_shape(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and value[0] == "B"
        and value[1] == 3
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value[1:]
        )
    )


def _is_positive_hw(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value)
    )


def _is_numeric_pair(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int | float) and not isinstance(item, bool) for item in value)
    )


def _is_positive_finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _is_nonnegative_finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_finite_number_in_range(value: Any, *, minimum: float, maximum: float) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and minimum <= float(value) <= maximum
    )


def _is_nonnegative_int_list(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value
        )
    )


def _is_three_numbers(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, int | float) and not isinstance(item, bool) for item in value)
    )


def _is_rgb_triplet(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 255
            for item in value
        )
    )


def _find_unresolved_tokens(value: Any, issues: list[str], path: str = "<root>") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = str(key) if path == "<root>" else f"{path}.{key}"
            _find_unresolved_tokens(child, issues, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _find_unresolved_tokens(child, issues, f"{path}[{index}]")
    elif isinstance(value, str) and _INTERPOLATION_RE.search(value):
        issues.append(f"{path}에 resolve되지 않은 interpolation이 있습니다: {value!r}")


__all__ = [
    "ConfigError",
    "ConfigLoadError",
    "ConfigResolutionError",
    "ConfigValidationError",
    "DEFAULT_CONFIG_PATH",
    "SUPPORTED_SCHEMA_VERSION",
    "apply_overrides",
    "deep_merge",
    "load_and_validate_config",
    "load_config",
    "resolve_model_forward_keys",
    "select_model_forward_inputs",
    "resolve_interpolations",
    "validate_config",
]
