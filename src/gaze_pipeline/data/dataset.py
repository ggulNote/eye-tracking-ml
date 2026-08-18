"""PyTorch datasets for canonical static-image gaze manifests."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import multiprocessing as mp
import random
import re
from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from multiprocessing.context import get_spawning_popen
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from .preprocessing_cache import (
    PreprocessingCacheCorruptionError,
    TrustedLocalPreprocessingCache,
)
from .transforms import GazePreprocessor, SampleDroppedError, SampleValidationError


class ManifestError(ValueError):
    """Raised when canonical manifest rows do not satisfy the dataset contract."""


_EMPTY_VALUES = {"", "none", "null", "nan", "na", "n/a"}
_VIEW_ALIASES = {
    "front": "front",
    "frontal": "front",
    "webcam": "front",
    "mpiifacegaze": "front",
    "side": "side",
    "profile": "side",
    "phonecam": "side",
    "phone": "side",
}

_SELECTED_EYE_TO_INDEX = {"left": 0, "right": 1, "both": 2}
_MODEL_AUXILIARY_SUFFIXES = (
    "head_vector",
    "face_origin_3d",
    "head_orientation_valid",
    "face_origin_valid",
    "head_pose_valid",
    "gaze_valid",
    "selected_eye",
    "selected_eye_index",
    "eye_selection_valid",
    "head_pose_2d",
    "head_pose_2d_valid",
    "eye_pose_2d",
    "iris_pose_2d",
    "eye_angles",
)
_MODEL_AUXILIARY_KEYS = frozenset(
    f"{view}_{suffix}" for view in ("front", "side") for suffix in _MODEL_AUXILIARY_SUFFIXES
)


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in _EMPTY_VALUES else text


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float | int | np.number):
        result = float(value)
    else:
        text = _clean_string(value)
        if not text:
            return None
        try:
            result = float(text)
        except ValueError:
            return None
    return result if math.isfinite(result) else None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | np.integer):
        return bool(value)
    text = _clean_string(value).lower()
    if not text:
        return None
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ManifestError(f"expected a boolean value, got {value!r}")


def _first_float(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _parse_numeric_array(value: Any, *, field_name: str) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        try:
            return np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"{field_name} is not a numeric array") from exc

    text = _clean_string(value)
    if not text:
        return None
    parsed: Any = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            break
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    if parsed is None:
        tokens = [token for token in re.split(r"[,;\s]+", text.strip("[]()")) if token]
        try:
            parsed = [float(token) for token in tokens]
        except ValueError as exc:
            raise ManifestError(f"could not parse numeric field {field_name!r}: {text!r}") from exc
    try:
        return np.asarray(parsed, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{field_name} is not a rectangular numeric array") from exc


def _extract_landmarks(row: Mapping[str, Any], prefix: str = "") -> np.ndarray | None:
    fields = (
        f"{prefix}facial_landmarks_xy",
        f"{prefix}landmarks_json",
        f"{prefix}landmarks_xy",
    )
    for field in fields:
        if field in row and _clean_string(row.get(field)):
            values = _parse_numeric_array(row.get(field), field_name=field)
            if values is None:
                return None
            if values.ndim == 1:
                if values.size % 2:
                    raise ManifestError(f"{field} must contain an even number of x/y values")
                values = values.reshape(-1, 2)
            if values.ndim != 2 or values.shape[1] != 2:
                raise ManifestError(f"{field} must have shape [N, 2], got {tuple(values.shape)}")
            return values.astype(np.float32)

    indexed: list[tuple[int, str, float]] = []
    pattern = re.compile(rf"^{re.escape(prefix)}landmark_(\d+)_(x|y)$")
    for key, value in row.items():
        match = pattern.match(str(key))
        numeric = _float_or_none(value)
        if match and numeric is not None:
            indexed.append((int(match.group(1)), match.group(2), numeric))
    if not indexed:
        return None
    by_index: dict[int, dict[str, float]] = defaultdict(dict)
    for index, axis, value in indexed:
        by_index[index][axis] = value
    missing = [index for index, axes in by_index.items() if set(axes) != {"x", "y"}]
    if missing:
        raise ManifestError(f"landmark columns are missing x or y for indices {missing}")
    return np.asarray(
        [[by_index[index]["x"], by_index[index]["y"]] for index in sorted(by_index)],
        dtype=np.float32,
    )


def _coordinate_name(task_config: Mapping[str, Any]) -> str:
    coordinate = task_config.get("coordinate_system", {})
    if isinstance(coordinate, Mapping):
        return str(coordinate.get("name", "centered_normalized_screen")).lower()
    return str(coordinate or "centered_normalized_screen").lower()


def _extract_target(
    row: Mapping[str, Any],
    task_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return normalized target and optional original screen-pixel target."""

    # Canonical records.py names first, then documented compatibility aliases.
    normalized_pairs = (
        ("target_x_normalized", "target_y_normalized"),
        ("target_x_norm", "target_y_norm"),
        ("gaze_x_norm", "gaze_y_norm"),
        ("target_gaze_x", "target_gaze_y"),
    )
    for x_key, y_key in normalized_pairs:
        x = _float_or_none(row.get(x_key))
        y = _float_or_none(row.get(y_key))
        if x is not None or y is not None:
            if x is None or y is None:
                raise ManifestError(f"target requires both {x_key!r} and {y_key!r}")
            return np.asarray([x, y], dtype=np.float32), _extract_pixel_target(row)

    for field in ("target_gaze_xy", "gaze_xy_normalized", "gaze_xy_norm"):
        values = _parse_numeric_array(row.get(field), field_name=field) if field in row else None
        if values is not None:
            values = values.reshape(-1)
            if len(values) != 2:
                raise ManifestError(f"{field} must contain exactly x and y")
            return values.astype(np.float32), _extract_pixel_target(row)

    pixel = _extract_pixel_target(row)
    if pixel is None:
        raise ManifestError(
            "manifest row has no gaze target; expected target_x_normalized/target_y_normalized "
            "or target_x_px/target_y_px with screen dimensions"
        )
    width = _first_float(row, ("screen_width_px", "screen_w_px", "display_width_px"))
    height = _first_float(row, ("screen_height_px", "screen_h_px", "display_height_px"))
    if width is None or height is None or width <= 0 or height <= 0:
        raise ManifestError(
            "pixel gaze targets require positive screen_width_px and screen_height_px"
        )
    normalized = pixel / np.asarray([width, height], dtype=np.float32)
    coordinate_name = _coordinate_name(task_config)
    if "centered" in coordinate_name:
        normalized -= 0.5
    elif "unit" not in coordinate_name and "zero_one" not in coordinate_name:
        raise ManifestError(
            f"cannot normalize pixel target for unsupported coordinate system {coordinate_name!r}"
        )
    return normalized.astype(np.float32), pixel


def _extract_pixel_target(row: Mapping[str, Any]) -> np.ndarray | None:
    pairs = (
        ("target_x_px", "target_y_px"),
        ("gaze_screen_x_px", "gaze_screen_y_px"),
        ("gaze_x_px", "gaze_y_px"),
    )
    for x_key, y_key in pairs:
        x = _float_or_none(row.get(x_key))
        y = _float_or_none(row.get(y_key))
        if x is not None or y is not None:
            if x is None or y is None:
                raise ManifestError(f"pixel target requires both {x_key!r} and {y_key!r}")
            return np.asarray([x, y], dtype=np.float32)
    for field in ("gaze_screen_xy_px", "target_xy_px"):
        values = _parse_numeric_array(row.get(field), field_name=field) if field in row else None
        if values is not None:
            values = values.reshape(-1)
            if len(values) != 2:
                raise ManifestError(f"{field} must contain exactly x and y")
            return values.astype(np.float32)
    return None


def _extract_target_in_screen_bounds(
    row: Mapping[str, Any],
    target: np.ndarray,
    pixel_target: np.ndarray | None,
    task_config: Mapping[str, Any],
) -> bool:
    declared = _bool_or_none(row.get("target_in_screen_bounds"))
    if declared is not None:
        return declared

    width = _first_float(row, ("screen_width_px", "screen_w_px", "display_width_px"))
    height = _first_float(row, ("screen_height_px", "screen_h_px", "display_height_px"))
    if (
        pixel_target is not None
        and width is not None
        and height is not None
        and width > 0
        and height > 0
    ):
        return bool(0 <= float(pixel_target[0]) < width and 0 <= float(pixel_target[1]) < height)

    coordinate_name = _coordinate_name(task_config)
    x, y = (float(target[0]), float(target[1]))
    if "centered" in coordinate_name:
        return -0.5 <= x < 0.5 and -0.5 <= y < 0.5
    if "unit" in coordinate_name or "zero_one" in coordinate_name:
        return 0.0 <= x < 1.0 and 0.0 <= y < 1.0
    return False


def _extract_vector(row: Mapping[str, Any], names: Sequence[str], length: int) -> np.ndarray | None:
    for name in names:
        if name in row and _clean_string(row.get(name)):
            value = _parse_numeric_array(row.get(name), field_name=name)
            if value is None:
                return None
            value = value.reshape(-1)
            if len(value) != length:
                raise ManifestError(f"{name} must contain {length} values")
            return value.astype(np.float32)
    return None


def _extract_fixed_points(
    row: Mapping[str, Any], names: Sequence[str], *, count: int
) -> np.ndarray | None:
    for name in names:
        if name not in row or not _clean_string(row.get(name)):
            continue
        value = _parse_numeric_array(row.get(name), field_name=name)
        if value is None:
            return None
        if value.ndim == 1:
            if value.size != count * 2:
                raise ManifestError(f"{name} must contain exactly {count} x/y points")
            value = value.reshape(count, 2)
        if value.shape != (count, 2):
            raise ManifestError(f"{name} must have shape [{count}, 2], got {tuple(value.shape)}")
        return value.astype(np.float32)
    return None


def _extract_visible_eye(row: Mapping[str, Any]) -> str | None:
    raw_value = _clean_string(row.get("visible_eye")).lower()
    if not raw_value:
        return None
    aliases = {
        "l": "left",
        "left": "left",
        "subject_left": "left",
        "r": "right",
        "right": "right",
        "subject_right": "right",
    }
    visible_eye = aliases.get(raw_value)
    if visible_eye is None:
        raise ManifestError(f"visible_eye must be left/right when present, got {raw_value!r}")
    return visible_eye


def _canonical_view(value: Any) -> str:
    text = _clean_string(value).lower() or "front"
    if text not in _VIEW_ALIASES:
        raise ManifestError(f"unsupported view {text!r}; expected front/webcam or side/phonecam")
    return _VIEW_ALIASES[text]


def _read_manifest(manifest: Any) -> tuple[list[dict[str, Any]], Path | None]:
    if isinstance(manifest, str | Path):
        path = Path(manifest).expanduser()
        if not path.is_file():
            raise ManifestError(f"manifest CSV does not exist: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
        if not rows:
            raise ManifestError(f"manifest contains no data rows: {path}")
        return rows, path

    # pandas.DataFrame without importing pandas eagerly.
    if hasattr(manifest, "to_dict") and not isinstance(manifest, Mapping):
        try:
            rows = manifest.to_dict(orient="records")
            return [dict(row) for row in rows], None
        except TypeError:
            pass
    if isinstance(manifest, Sequence) and not isinstance(manifest, str | bytes):
        rows = []
        for index, row in enumerate(manifest):
            if not isinstance(row, Mapping):
                raise ManifestError(f"manifest row {index} is not a mapping")
            rows.append(dict(row))
        if not rows:
            raise ManifestError("manifest contains no rows")
        return rows, None
    raise TypeError("manifest must be a CSV path, DataFrame, or sequence of row mappings")


def _select_split(rows: list[dict[str, Any]], split: str | None) -> list[dict[str, Any]]:
    if split is None or not any(_clean_string(row.get("split")) for row in rows):
        return rows
    aliases = {"val": "validation", "valid": "validation", "dev": "validation"}
    wanted = aliases.get(str(split).lower(), str(split).lower())
    selected = []
    for row in rows:
        actual = aliases.get(
            _clean_string(row.get("split")).lower(), _clean_string(row.get("split")).lower()
        )
        if actual == wanted:
            selected.append(row)
    if not selected:
        available = sorted({_clean_string(row.get("split")) for row in rows})
        raise ManifestError(
            f"manifest has no rows for split {wanted!r}; available splits: {available}"
        )
    return selected


def _expand_wide_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        has_wide = _clean_string(row.get("front_image_path")) or _clean_string(
            row.get("side_image_path")
        )
        if not has_wide:
            expanded.append(row)
            continue
        for view in ("front", "side"):
            path = _clean_string(row.get(f"{view}_image_path"))
            if not path:
                continue
            branch = dict(row)
            branch["view"] = view
            branch["image_path"] = path
            branch["sample_id"] = _clean_string(row.get(f"{view}_sample_id")) or (
                f"{_clean_string(row.get('pair_id'))}:{view}"
            )
            for field in (
                "facial_landmarks_xy",
                "landmarks_json",
                "landmarks_xy",
                "visible_eye",
                "visible_eye_bbox_xyxy",
                "visible_eye_keypoints_xy",
                "iris_center_xy",
                "profile_head_origin_xy",
                "profile_head_forward_xy",
                "eye_annotation_valid",
            ):
                prefixed = f"{view}_{field}"
                if prefixed in row:
                    branch[field] = row[prefixed]
            expanded.append(branch)
    return expanded


def _resolve_path(raw_path: Any, data_root: Path) -> Path:
    text = _clean_string(raw_path)
    if not text:
        raise ManifestError("manifest row has an empty image_path")
    path = Path(text).expanduser()
    return path if path.is_absolute() else data_root / path


def _side_front_3d_policy(
    preprocessing_config: Mapping[str, Any],
) -> dict[str, str] | None:
    """Return the paired-front head gate requested by the profile90 contract."""

    overrides = preprocessing_config.get("branch_overrides")
    side = overrides.get("side") if isinstance(overrides, Mapping) else None
    if not isinstance(side, Mapping):
        return None
    warp = side.get("eye_region_warp")
    features = warp.get("feature_extraction") if isinstance(warp, Mapping) else None
    head = features.get("side_headpose") if isinstance(features, Mapping) else None
    if not isinstance(head, Mapping):
        return None
    if head.get("enabled") is not True or str(head.get("source", "")).lower() != "front_3d":
        return None
    policy = {
        "vector_key": str(head.get("front_3d_key", "front_head_vector")),
        "validity_key": str(head.get("front_3d_validity_key", "front_head_orientation_valid")),
        "invalid_policy": str(head.get("front_3d_invalid_policy", "zero_fill_and_mask")).lower(),
    }
    if policy["vector_key"] != "front_head_vector":
        raise ManifestError("side front_3d vector key must be canonical 'front_head_vector'")
    if policy["validity_key"] != "front_head_orientation_valid":
        raise ManifestError(
            "side front_3d validity key must be canonical 'front_head_orientation_valid'"
        )
    if policy["invalid_policy"] not in {"zero_fill_and_mask", "error"}:
        raise ManifestError("side front_3d invalid policy must be zero_fill_and_mask or error")
    return policy


class GazeImageDataset(Dataset[dict[str, Any]]):
    """Static-image gaze dataset backed by a canonical CSV manifest.

    ``config`` may be the fully resolved project config or only its
    ``preprocessing`` section.  A long paired manifest has one row per view and
    joins rows solely by explicit ``pair_id``; ordering or filename similarity is
    never used as an implicit pairing signal.
    """

    def __init__(
        self,
        manifest: str | Path | Sequence[Mapping[str, Any]] | Any,
        config: Mapping[str, Any] | None = None,
        *,
        preprocessing_config: Mapping[str, Any] | None = None,
        task_config: Mapping[str, Any] | None = None,
        split: str | None = "train",
        data_root: str | Path | None = None,
        view: str | None = None,
        paired: bool | None = None,
        landmark_detector: Any = None,
    ) -> None:
        rows, manifest_path = _read_manifest(manifest)
        rows = _select_split(rows, split)
        rows = _expand_wide_rows(rows)

        full_config: Mapping[str, Any] = config or {}
        if preprocessing_config is None:
            preprocessing_config = (
                full_config.get("preprocessing", full_config)
                if isinstance(full_config, Mapping)
                else {}
            )
        if task_config is None:
            task_config = full_config.get("task", {}) if isinstance(full_config, Mapping) else {}
        self.task_config = task_config or {}
        self.data_config = full_config.get("data", {}) if isinstance(full_config, Mapping) else {}
        self.side_front_3d_policy = _side_front_3d_policy(preprocessing_config or {})

        self.preprocessing_cache: TrustedLocalPreprocessingCache | None = None
        self.preprocessing_cache_mode = "off"
        self._refreshed_cache_keys: set[str] = set()
        cache_config = (
            preprocessing_config.get("cache", {})
            if isinstance(preprocessing_config, Mapping)
            else {}
        )
        if cache_config is not None and not isinstance(cache_config, Mapping):
            raise ManifestError("preprocessing.cache must be a mapping")
        if isinstance(cache_config, Mapping) and bool(cache_config.get("enabled", False)):
            mode = str(cache_config.get("mode", "read_write")).lower()
            if mode not in {"read_write", "read_only", "refresh"}:
                raise ManifestError(
                    "preprocessing.cache.mode must be read_write, read_only, or refresh"
                )
            cache_dir = str(cache_config.get("dir", "")).strip()
            if not cache_dir or cache_dir.startswith("${"):
                raise ManifestError("preprocessing.cache.dir must resolve to a local path")
            deterministic_config = deepcopy(dict(preprocessing_config or {}))
            deterministic_config.pop("cache", None)
            cache_identity_config = {
                "preprocessing": deterministic_config,
                "task": deepcopy(dict(self.task_config)),
                # Train/evaluation interpolation and the augmentation boundary
                # can differ even when a source manifest row has no split
                # column, so split must be part of the cache namespace.
                "dataset_split": "validation" if split == "val" else str(split or "train"),
            }
            self.preprocessing_cache = TrustedLocalPreprocessingCache(
                cache_dir,
                cache_identity_config,
                trusted_local=bool(cache_config.get("trusted_local", False)),
                schema_version=int(cache_config.get("schema_version", 1)),
                implementation_id=str(
                    cache_config.get("implementation_id", "ordered-gaze-preprocessor-v2")
                ),
                max_entry_bytes=int(cache_config.get("max_entry_bytes", 512 * 1024 * 1024)),
            )
            self.preprocessing_cache_mode = mode

        if data_root is None:
            configured_root = (
                self.data_config.get("dataset_root")
                if isinstance(self.data_config, Mapping)
                else None
            )
            if configured_root and not str(configured_root).startswith("${"):
                data_root = configured_root
            elif manifest_path is not None:
                data_root = manifest_path.parent
            else:
                data_root = Path.cwd()
        self.data_root = Path(data_root).expanduser()
        self.split = "validation" if split == "val" else str(split or "train")

        experiment_config = (
            full_config.get("experiment", {}) if isinstance(full_config, Mapping) else {}
        )
        split_config = (
            self.data_config.get("split", {}) if isinstance(self.data_config, Mapping) else {}
        )
        seed_value = (
            experiment_config.get("seed") if isinstance(experiment_config, Mapping) else None
        )
        if seed_value is None and isinstance(split_config, Mapping):
            seed_value = split_config.get("seed")
        self.augmentation_seed = int(42 if seed_value is None else seed_value)
        # RawValue uses an inherited/spawn-reduced anonymous mapping and avoids
        # Torch's external shm manager. It keeps set_epoch visible to persistent
        # workers while remaining lightweight for num_workers=0.
        self._epoch_state = mp.RawValue("q", 0)

        pairing_config = (
            self.data_config.get("pairing", {}) if isinstance(self.data_config, Mapping) else {}
        )
        if not isinstance(pairing_config, Mapping):
            raise ManifestError("data.pairing must be a mapping")
        self.paired = bool(pairing_config.get("enabled", False)) if paired is None else bool(paired)
        self.pairing_config = pairing_config
        strategy = str(pairing_config.get("strategy", "explicit_pair_id")).lower()
        if self.paired and strategy != "explicit_pair_id":
            raise ManifestError(
                "paired image datasets require data.pairing.strategy=explicit_pair_id"
            )

        for index, row in enumerate(rows):
            if not _clean_string(row.get("sample_id")):
                row["sample_id"] = f"row-{index:08d}"
            row["view"] = _canonical_view(row.get("view") or row.get("modality") or "front")
        sample_ids = [_clean_string(row.get("sample_id")) for row in rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise ManifestError("manifest sample_id values must be unique")
        if view is not None:
            requested_view = _canonical_view(view)
            rows = [row for row in rows if row["view"] == requested_view]
            if not rows:
                raise ManifestError(f"manifest has no rows for requested view {requested_view!r}")

        self.preprocessor = GazePreprocessor(
            preprocessing_config or {},
            split=self.split,
            task_config=self.task_config,
            landmark_detector=landmark_detector,
        )

        if self.paired:
            self.records: list[Any] = self._build_pairs(rows)
        else:
            available_views = sorted({row["view"] for row in rows})
            if len(available_views) > 1 and view is None:
                raise ManifestError(
                    "single-view dataset received multiple views; pass "
                    "view='front'/'side' or paired=True"
                )
            self.records = rows
        if not self.records:
            raise ManifestError("no usable dataset samples remain after filtering")

    def _build_pairs(
        self, rows: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Build complete fusion pairs.

        ``branch_only`` keeps incomplete records in prepared manifests, but a
        paired Dataset intentionally exposes complete front/side units only.
        Train those incomplete records through a separate single-view Dataset
        constructed with ``paired=False`` and an explicit ``view`` filter.
        """

        pair_key = str(self.pairing_config.get("pair_id_key", "pair_id"))
        policy = str(self.pairing_config.get("unpaired_policy", "branch_only")).lower()
        if policy not in {"branch_only", "drop", "error"}:
            raise ManifestError("data.pairing.unpaired_policy must be branch_only, drop, or error")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        unpaired_rows: list[dict[str, Any]] = []
        for row in rows:
            # Source manifests may use a configured custom key, while prepared
            # canonical manifests always write ``pair_id``.
            pair_id = _clean_string(row.get(pair_key)) or _clean_string(row.get("pair_id"))
            if not pair_id:
                unpaired_rows.append(row)
                continue
            row["pair_id"] = pair_id
            grouped[pair_id].append(row)
        if unpaired_rows and policy == "error":
            examples = [_clean_string(row.get("sample_id")) for row in unpaired_rows[:3]]
            raise ManifestError(
                f"{len(unpaired_rows)} rows have no explicit {pair_key}; examples: {examples}"
            )

        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        incomplete: list[str] = []
        tolerance = float(self.pairing_config.get("max_target_distance_normalized", 0.0))
        require_same_subject = bool(self.pairing_config.get("require_same_subject", True))
        require_same_target = bool(self.pairing_config.get("require_same_target", True))
        for pair_id, group in sorted(grouped.items()):
            by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in group:
                by_view[row["view"]].append(row)
            if len(by_view.get("front", [])) != 1 or len(by_view.get("side", [])) != 1:
                if len(by_view.get("front", [])) > 1 or len(by_view.get("side", [])) > 1:
                    raise ManifestError(
                        f"pair_id {pair_id!r} has duplicate front or side rows; "
                        "pair IDs must be explicit and unique"
                    )
                incomplete.append(pair_id)
                continue
            front, side = by_view["front"][0], by_view["side"][0]
            if require_same_subject and _clean_string(front.get("subject_id")) != _clean_string(
                side.get("subject_id")
            ):
                raise ManifestError(f"pair_id {pair_id!r} joins different subjects")
            if require_same_target:
                front_target, _ = _extract_target(front, self.task_config)
                side_target, _ = _extract_target(side, self.task_config)
                distance = float(np.linalg.norm(front_target - side_target))
                if distance > tolerance + 1e-8:
                    raise ManifestError(
                        f"pair_id {pair_id!r} target mismatch {distance:.6g} "
                        f"exceeds tolerance {tolerance}"
                    )
            pairs.append((front, side))
        if incomplete and policy == "error":
            raise ManifestError(
                f"{len(incomplete)} pair IDs do not contain exactly one front and one side row; "
                f"examples: {incomplete[:3]}"
            )
        return pairs

    def __len__(self) -> int:
        return len(self.records)

    @property
    def epoch(self) -> int:
        return int(self._epoch_state.value)

    def set_epoch(self, epoch: int) -> None:
        """Select deterministic per-sample augmentations for a training epoch."""

        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("dataset epoch must be non-negative")
        self._epoch_state.value = epoch

    def __getstate__(self) -> dict[str, Any]:
        """Support ordinary pickle and multiprocessing spawn semantics."""

        state = dict(self.__dict__)
        if get_spawning_popen() is None:
            # RawValue is reducible only during process spawning. For normal
            # serialization, store a scalar and rebuild the state on load.
            state["_pickled_epoch"] = self.epoch
            state["_epoch_state"] = None
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        restored = dict(state)
        pickled_epoch = int(restored.pop("_pickled_epoch", 0))
        if restored.get("_epoch_state") is None:
            restored["_epoch_state"] = mp.RawValue("q", pickled_epoch)
        self.__dict__.update(restored)

    def _stable_augmentation_seed(self, identity: str) -> int:
        payload = (
            f"gaze-augmentation-v1\0{self.augmentation_seed}\0{self.epoch}\0{identity}"
        ).encode()
        # Torch Generator accepts signed 64-bit seeds portably.
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)

    def _augmentation_context(self, *, identity: str, view: str) -> tuple[Mapping[str, Any], int]:
        seed = self._stable_augmentation_seed(identity)
        generator = torch.Generator().manual_seed(seed)
        return (
            self.preprocessor.make_augmentation_context(view, generator=generator),
            seed,
        )

    def _auxiliary_expectations(self, view: str) -> dict[str, bool]:
        """Return which optional WebEyeTrack auxiliary groups should be stable.

        Expectations come from enabled stages, rather than from successful stage
        outputs.  A failed pose or eye-quality stage therefore still yields fixed
        shape tensors plus a false validity flag, while legacy configurations that
        do not run these stages retain their original item keys.
        """

        stage_order = tuple(getattr(self.preprocessor, "stage_order", ()))

        def stage_runs(stage: str) -> bool:
            if stage not in stage_order:
                return False
            stage_config = self.preprocessor._stage_config(stage, view)
            return bool(stage_config.get("enabled", True))

        return {
            "pose": stage_runs("metric_head_pose"),
            "eye_selection": stage_runs("eye_selection"),
        }

    def _row_to_sample(
        self,
        row: Mapping[str, Any],
        *,
        augmentation_context: Mapping[str, Any] | None = None,
        augmentation_seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        view = _canonical_view(row.get("view") or row.get("modality") or "front")
        expected_auxiliary = self._auxiliary_expectations(view)
        target, pixel_target = _extract_target(row, self.task_config)
        target_in_screen_bounds = _extract_target_in_screen_bounds(
            row, target, pixel_target, self.task_config
        )
        landmarks = _extract_landmarks(row)
        image_path = _resolve_path(row.get("image_path"), self.data_root)

        screen_width_px = _first_float(row, ("screen_width_px", "screen_w_px"))
        screen_height_px = _first_float(row, ("screen_height_px", "screen_h_px"))
        screen_width_mm = _first_float(row, ("screen_width_mm", "screen_w_mm"))
        screen_height_mm = _first_float(row, ("screen_height_mm", "screen_h_mm"))
        metadata: dict[str, Any] = {
            "sample_id": _clean_string(row.get("sample_id")),
            "subject_id": _clean_string(row.get("subject_id")),
            "session_id": _clean_string(row.get("session_id")),
            "view": view,
            "pair_id": _clean_string(row.get("pair_id")),
            "image_path": str(image_path),
            "screen_size_px": np.asarray(
                [
                    screen_width_px if screen_width_px is not None else np.nan,
                    screen_height_px if screen_height_px is not None else np.nan,
                ],
                dtype=np.float32,
            ),
            "screen_size_mm": np.asarray(
                [
                    screen_width_mm if screen_width_mm is not None else np.nan,
                    screen_height_mm if screen_height_mm is not None else np.nan,
                ],
                dtype=np.float32,
            ),
            "gaze_screen_xy_px": (
                pixel_target.astype(np.float32)
                if pixel_target is not None
                else np.asarray([np.nan, np.nan], dtype=np.float32)
            ),
            "target_in_screen_bounds": target_in_screen_bounds,
            "augmentation_epoch": self.epoch,
            "augmentation_seed": (int(augmentation_seed) if augmentation_seed is not None else -1),
        }
        for key, names in {
            "head_vector": ("head_vector",),
            "face_origin_3d": ("face_origin_3d", "face_origin_3d_cm"),
            "head_rotation_3d": ("head_rotation_3d", "head_rotation_json"),
            "head_translation_3d": ("head_translation_3d", "head_translation_json"),
            "face_center_3d": ("face_center_3d", "face_center_json"),
            "gaze_target_3d": ("gaze_target_3d", "gaze_target_3d_json"),
        }.items():
            vector = _extract_vector(row, names, 3)
            if vector is not None:
                metadata[key] = vector
        declared_head_pose_valid = _bool_or_none(row.get("head_pose_valid"))
        if declared_head_pose_valid is not None:
            metadata["head_pose_valid"] = declared_head_pose_valid
        evaluation_eye = _clean_string(row.get("evaluation_eye"))
        if evaluation_eye:
            metadata["evaluation_eye"] = evaluation_eye

        visible_eye = _extract_visible_eye(row)
        if visible_eye is not None:
            metadata["visible_eye"] = visible_eye
        for key, length in {
            "visible_eye_bbox_xyxy": 4,
            "iris_center_xy": 2,
            "profile_head_origin_xy": 2,
            "profile_head_forward_xy": 2,
        }.items():
            vector = _extract_vector(row, (key,), length)
            if vector is not None:
                metadata[key] = vector
        eye_keypoints = _extract_fixed_points(
            row,
            ("visible_eye_keypoints_xy",),
            count=6,
        )
        if eye_keypoints is not None:
            metadata["visible_eye_keypoints_xy"] = eye_keypoints
        eye_annotation_valid = _bool_or_none(row.get("eye_annotation_valid"))
        if eye_annotation_valid is not None:
            metadata["eye_annotation_valid"] = eye_annotation_valid

        sample: dict[str, Any] = {
            "image_path": image_path,
            "view": view,
            "target_gaze_xy": torch.as_tensor(target, dtype=torch.float32),
            "metadata": metadata,
        }
        if landmarks is not None:
            sample["landmarks_xy"] = landmarks.copy()
            sample["facial_landmarks_xy"] = landmarks.copy()
            metadata["source_landmarks_xy"] = landmarks.copy()
        try:
            processed: MutableMapping[str, Any]
            if self.preprocessing_cache is None:
                if augmentation_context is not None:
                    sample["_augmentation_context"] = dict(augmentation_context)
                processed = self.preprocessor(sample)
            else:
                processed = self._preprocess_with_cache(
                    sample,
                    row=row,
                    view=view,
                    image_path=image_path,
                    augmentation_context=augmentation_context,
                    augmentation_seed=augmentation_seed,
                )
        except SampleDroppedError as exc:
            raise SampleDroppedError(
                f"sample {_clean_string(row.get('sample_id'))!r} was configured to drop; "
                "filter it during manifest preparation before using DataLoader"
            ) from exc

        image = processed["image"]
        if not torch.is_tensor(image):
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[-1] != 3:
                raise SampleValidationError(
                    "preprocessing output image must be RGB HWC or tensor CHW"
                )
            image = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()
        if image.ndim != 3:
            raise SampleValidationError(
                f"model image must have 3 dimensions, got {tuple(image.shape)}"
            )
        if image.shape[0] != 3:
            if image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
            else:
                raise SampleValidationError(
                    f"model image must have three RGB channels, got {tuple(image.shape)}"
                )

        target_tensor = torch.as_tensor(processed["target_gaze_xy"], dtype=torch.float32).reshape(2)
        processed_metadata = processed.get("metadata", metadata)
        if not isinstance(processed_metadata, Mapping):
            raise SampleValidationError("preprocessing output metadata must be a mapping")
        result_metadata = self._tensorize_metadata(
            processed,
            processed_metadata,
            image,
            expected_auxiliary=expected_auxiliary,
        )
        return image.contiguous(), target_tensor, result_metadata

    def _preprocess_with_cache(
        self,
        sample: MutableMapping[str, Any],
        *,
        row: Mapping[str, Any],
        view: str,
        image_path: Path,
        augmentation_context: Mapping[str, Any] | None,
        augmentation_seed: int | None,
    ) -> MutableMapping[str, Any]:
        assert self.preprocessing_cache is not None
        cache_identity = self.preprocessing_cache.identity(
            row=row,
            view=view,
            image_path=image_path,
        )
        cached_payload: Any | None = None
        should_refresh = (
            self.preprocessing_cache_mode == "refresh"
            and cache_identity.key not in self._refreshed_cache_keys
        )
        if not should_refresh:
            cached_payload = self.preprocessing_cache.load(cache_identity)
        cache_hit = cached_payload is not None
        if cache_hit:
            if not isinstance(cached_payload, MutableMapping):
                raise PreprocessingCacheCorruptionError(
                    "preprocessing cache payload must be a mutable mapping"
                )
            processed = cached_payload
        else:
            processed = self.preprocessor.preprocess_deterministic(sample)
            if self.preprocessing_cache_mode in {"read_write", "refresh"}:
                cache_payload = deepcopy(dict(processed))
                cache_payload.pop("_augmentation_context", None)
                if isinstance(cache_payload.get("image_path"), Path):
                    cache_payload["image_path"] = str(cache_payload["image_path"])
                self.preprocessing_cache.store(cache_identity, cache_payload)
                if self.preprocessing_cache_mode == "refresh":
                    self._refreshed_cache_keys.add(cache_identity.key)
        processed_metadata = processed.get("metadata")
        if isinstance(processed_metadata, MutableMapping):
            # Deterministic cache entries can originate in a previous epoch.
            # Augmentation is re-applied on every access, so its provenance
            # must also describe the current access instead of the cache fill.
            processed_metadata["augmentation_epoch"] = self.epoch
            processed_metadata["augmentation_seed"] = (
                int(augmentation_seed) if augmentation_seed is not None else -1
            )
            processed_metadata["preprocessing_cache_hit"] = cache_hit
            processed_metadata["preprocessing_cache_key"] = cache_identity.key
        if augmentation_context is not None:
            processed["_augmentation_context"] = dict(augmentation_context)
        return self.preprocessor.apply_augmentation(processed)

    @staticmethod
    def _tensorize_metadata(
        sample: Mapping[str, Any],
        metadata: Mapping[str, Any],
        image: torch.Tensor,
        *,
        expected_auxiliary: Mapping[str, bool] | None = None,
    ) -> dict[str, Any]:
        expected_auxiliary = expected_auxiliary or {}

        def auxiliary_value(*names: str, default: Any = None) -> Any:
            for name in names:
                if name in sample and sample[name] is not None:
                    return sample[name]
                if name in metadata and metadata[name] is not None:
                    return metadata[name]
            return default

        result: dict[str, Any] = {
            "sample_id": str(metadata.get("sample_id", "")),
            "subject_id": str(metadata.get("subject_id", "")),
            "session_id": str(metadata.get("session_id", "")),
            "view": str(metadata.get("view", "")),
            "pair_id": str(metadata.get("pair_id", "")),
            "image_path": str(metadata.get("image_path", "")),
            "source_size_hw": torch.as_tensor(
                metadata.get("source_size_hw", [0, 0]), dtype=torch.int64
            ).reshape(2),
            "processed_size_hw": torch.tensor(
                [image.shape[-2], image.shape[-1]], dtype=torch.int64
            ),
            "image_transform": torch.as_tensor(
                metadata.get("image_transform", np.eye(3)), dtype=torch.float32
            ).reshape(3, 3),
            "screen_size_px": torch.as_tensor(
                metadata.get("screen_size_px", [np.nan, np.nan]), dtype=torch.float32
            ).reshape(2),
            "screen_size_mm": torch.as_tensor(
                metadata.get("screen_size_mm", [np.nan, np.nan]), dtype=torch.float32
            ).reshape(2),
            "gaze_screen_xy_px": torch.as_tensor(
                metadata.get("gaze_screen_xy_px", [np.nan, np.nan]), dtype=torch.float32
            ).reshape(2),
            "target_in_screen_bounds": torch.tensor(
                bool(metadata.get("target_in_screen_bounds", False)), dtype=torch.bool
            ),
            "augmentation_epoch": torch.tensor(
                int(metadata.get("augmentation_epoch", 0)), dtype=torch.int64
            ),
            "augmentation_seed": torch.tensor(
                int(metadata.get("augmentation_seed", -1)), dtype=torch.int64
            ),
        }
        landmarks = sample.get("landmarks_xy")
        source_landmarks = metadata.get("source_landmarks_xy")
        if landmarks is not None:
            result["landmarks_xy"] = torch.as_tensor(landmarks, dtype=torch.float32).reshape(-1, 2)
        if source_landmarks is not None:
            result["source_landmarks_xy"] = torch.as_tensor(
                source_landmarks, dtype=torch.float32
            ).reshape(-1, 2)
        for key in ("head_rotation_3d", "head_translation_3d", "face_center_3d", "gaze_target_3d"):
            if key in metadata:
                result[key] = torch.as_tensor(metadata[key], dtype=torch.float32).reshape(3)
        if "evaluation_eye" in metadata:
            result["evaluation_eye"] = str(metadata["evaluation_eye"])
        if "representation" in metadata:
            result["representation"] = str(metadata["representation"])
        if "preprocessing_cache_hit" in metadata:
            result["preprocessing_cache_hit"] = bool(metadata["preprocessing_cache_hit"])
        if "preprocessing_cache_key" in metadata:
            result["preprocessing_cache_key"] = str(metadata["preprocessing_cache_key"])
        if "landmark_kind" in metadata:
            result["landmark_kind"] = str(metadata["landmark_kind"])
        if "landmark_source" in metadata:
            result["landmark_source"] = str(metadata["landmark_source"])
        if "face_bbox_xywh" in metadata:
            result["face_bbox_xywh"] = torch.as_tensor(
                metadata["face_bbox_xywh"], dtype=torch.float32
            ).reshape(4)
        if "detector_mirrored_retry" in metadata:
            result["detector_mirrored_retry"] = torch.tensor(
                bool(metadata["detector_mirrored_retry"]), dtype=torch.bool
            )
        for key, shape in {
            "face_rt": (4, 4),
            "metric_transform": (4, 4),
            "head_euler_degrees": (3,),
            "eye_visibility_score": (2,),
            "visible_eye_bbox_xyxy": (4,),
            "visible_eye_keypoints_xy": (6, 2),
            "iris_center_xy": (2,),
            "profile_head_origin_xy": (2,),
            "profile_head_forward_xy": (2,),
        }.items():
            if key in metadata:
                result[key] = torch.as_tensor(metadata[key], dtype=torch.float32).reshape(shape)
        for key in ("face_origin_unit", "face_origin_source", "visible_eye"):
            if key in metadata:
                result[key] = str(metadata[key])
        if "eye_annotation_valid" in metadata:
            result["eye_annotation_valid"] = torch.tensor(
                bool(metadata["eye_annotation_valid"]), dtype=torch.bool
            )
        if "invalid_reasons" in metadata:
            result["invalid_reasons"] = [str(reason) for reason in metadata["invalid_reasons"]]
        pose_expected = bool(expected_auxiliary.get("pose")) or any(
            key in sample or key in metadata
            for key in (
                "head_vector",
                "face_origin_3d",
                "face_origin_3d_cm",
                "head_orientation_valid",
                "face_origin_valid",
                "head_pose_valid",
            )
        )
        if pose_expected:
            head_vector = torch.as_tensor(
                auxiliary_value("head_vector", default=[np.nan, np.nan, np.nan]),
                dtype=torch.float32,
            ).reshape(3)
            face_origin = torch.as_tensor(
                auxiliary_value(
                    "face_origin_3d",
                    "face_origin_3d_cm",
                    default=[np.nan, np.nan, np.nan],
                ),
                dtype=torch.float32,
            ).reshape(3)
            default_orientation_valid = bool(torch.isfinite(head_vector).all().item())
            default_origin_valid = bool(torch.isfinite(face_origin).all().item())
            orientation_valid = bool(
                auxiliary_value(
                    "head_orientation_valid",
                    default=default_orientation_valid,
                )
            )
            origin_valid = bool(auxiliary_value("face_origin_valid", default=default_origin_valid))
            result["head_vector"] = head_vector
            result["face_origin_3d"] = face_origin
            result["head_orientation_valid"] = torch.tensor(orientation_valid, dtype=torch.bool)
            result["face_origin_valid"] = torch.tensor(origin_valid, dtype=torch.bool)
            result["head_pose_valid"] = torch.tensor(
                bool(
                    auxiliary_value(
                        "head_pose_valid",
                        "pose_valid",
                        default=orientation_valid and origin_valid,
                    )
                ),
                dtype=torch.bool,
            )

        selection_expected = bool(expected_auxiliary.get("eye_selection")) or any(
            key in sample or key in metadata
            for key in ("selected_eye", "eye_selection_valid", "eye_visibility_score")
        )
        if selection_expected:
            selected_eye = str(auxiliary_value("selected_eye", default="unknown")).lower()
            selected_eye_index = _SELECTED_EYE_TO_INDEX.get(selected_eye, -1)
            if selected_eye_index < 0:
                selected_eye = "unknown"
            result["selected_eye"] = selected_eye
            result["selected_eye_index"] = torch.tensor(selected_eye_index, dtype=torch.int64)
            result["eye_selection_valid"] = torch.tensor(
                bool(
                    auxiliary_value(
                        "eye_selection_valid",
                        default=selected_eye_index >= 0,
                    )
                ),
                dtype=torch.bool,
            )

        view = str(metadata.get("view", "")).lower()
        for key in ("head_pose_2d", "eye_pose_2d", "iris_pose_2d", "eye_angles"):
            value = auxiliary_value(key, f"{view}_{key}")
            if value is not None:
                result[key] = torch.as_tensor(value, dtype=torch.float32).reshape(2)
        head_pose_2d_valid = auxiliary_value("head_pose_2d_valid", f"{view}_head_pose_2d_valid")
        if head_pose_2d_valid is not None:
            result["head_pose_2d_valid"] = torch.tensor(bool(head_pose_2d_valid), dtype=torch.bool)

        if pose_expected or selection_expected:
            result["gaze_valid"] = torch.tensor(
                bool(auxiliary_value("gaze_valid", default=True)), dtype=torch.bool
            )
        return result

    @staticmethod
    def _branch_auxiliary_outputs(metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Promote optional metadata tensors to unambiguous model batch keys.

        ``selected_eye_index`` uses ``left=0``, ``right=1``, and ``both=2``;
        ``-1`` represents an unavailable/invalid selection.
        """

        view = str(metadata.get("view", "")).lower()
        if view not in {"front", "side"}:
            return {}
        return {
            f"{view}_{suffix}": metadata[suffix]
            for suffix in _MODEL_AUXILIARY_SUFFIXES
            if suffix in metadata
        }

    def _gate_paired_front_3d_head_input(
        self, result: MutableMapping[str, Any], side_metadata: MutableMapping[str, Any]
    ) -> None:
        """Make a paired front head vector finite and propagate its validity."""

        policy = self.side_front_3d_policy
        if policy is None:
            return
        vector_key = policy["vector_key"]
        validity_key = policy["validity_key"]
        vector_value = result.get(vector_key)
        validity_value = result.get(validity_key, False)
        pair_value = result.get("pair_mask", False)
        try:
            vector = torch.as_tensor(vector_value, dtype=torch.float32).reshape(-1)
            orientation_valid = bool(torch.as_tensor(validity_value).reshape(()).item())
            pair_valid = bool(torch.as_tensor(pair_value).reshape(()).item())
            valid = pair_valid and orientation_valid and vector.numel() == 3
            valid = valid and bool(torch.isfinite(vector).all().item())
        except (TypeError, ValueError, RuntimeError):
            vector = torch.full((3,), torch.nan, dtype=torch.float32)
            valid = False

        side_metadata["front_3d_head_input_valid"] = valid
        if valid:
            result[vector_key] = vector.reshape(3)
            return
        reason = "side_headpose:front_3d_unavailable"
        if policy["invalid_policy"] == "error":
            raise SampleValidationError(reason)

        # A masked NaN can still contaminate a forward pass (0 * NaN is NaN),
        # so use a finite sentinel and explicitly exclude the side loss/metrics.
        result[vector_key] = torch.zeros(3, dtype=torch.float32)
        result["side_gaze_valid"] = torch.tensor(False, dtype=torch.bool)
        side_metadata["gaze_valid"] = False
        reasons = side_metadata.setdefault("invalid_reasons", [])
        if isinstance(reasons, list) and reason not in reasons:
            reasons.append(reason)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if not self.paired:
            row = record
            identity = f"sample:{_clean_string(row.get('sample_id'))}"
            context, augmentation_seed = self._augmentation_context(
                identity=identity, view=str(row["view"])
            )
            image, target, metadata = self._row_to_sample(
                row,
                augmentation_context=context,
                augmentation_seed=augmentation_seed,
            )
            key = "front_image" if metadata["view"] == "front" else "side_image"
            result = {
                key: image,
                "target_gaze_xy": target,
                "target_in_screen_bounds": metadata["target_in_screen_bounds"],
                "metadata": metadata,
            }
            result.update(self._branch_auxiliary_outputs(metadata))
            if metadata["view"] == "side" and self.side_front_3d_policy is not None:
                raise ManifestError(
                    "side_headpose.source=front_3d requires a paired front/side Dataset"
                )
            return result

        front_row, side_row = record
        pair_id = _clean_string(front_row.get("pair_id"))
        context, augmentation_seed = self._augmentation_context(
            identity=f"pair:{pair_id}", view="front"
        )
        front_image, front_target, front_metadata = self._row_to_sample(
            front_row,
            augmentation_context=context,
            augmentation_seed=augmentation_seed,
        )
        side_image, side_target, side_metadata = self._row_to_sample(
            side_row,
            augmentation_context=context,
            augmentation_seed=augmentation_seed,
        )
        if not torch.allclose(front_target, side_target, atol=1e-6, rtol=0):
            raise ManifestError(
                f"pair_id {front_metadata['pair_id']!r} produced inconsistent "
                "targets after preprocessing"
            )
        target_in_screen_bounds = (
            front_metadata["target_in_screen_bounds"] & side_metadata["target_in_screen_bounds"]
        )
        metadata = {
            "pair_id": front_metadata["pair_id"],
            "subject_id": front_metadata["subject_id"],
            "session_id": front_metadata["session_id"],
            "target_in_screen_bounds": target_in_screen_bounds,
            "augmentation_epoch": front_metadata["augmentation_epoch"],
            "augmentation_seed": front_metadata["augmentation_seed"],
            "front": front_metadata,
            "side": side_metadata,
        }
        result = {
            "front_image": front_image,
            "side_image": side_image,
            "pair_mask": torch.tensor(True, dtype=torch.bool),
            "target_gaze_xy": front_target,
            "target_in_screen_bounds": target_in_screen_bounds,
            "metadata": metadata,
        }
        result.update(self._branch_auxiliary_outputs(front_metadata))
        result.update(self._branch_auxiliary_outputs(side_metadata))
        self._gate_paired_front_3d_head_input(result, side_metadata)
        return result


def gaze_collate_fn(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate fixed model tensors while keeping heterogeneous metadata as a list.

    Generic detectors may return a different number of points, and optional 3D
    annotations can vary by row.  Keeping metadata sample-oriented prevents the
    default PyTorch collator from failing or silently imposing a false fixed
    metadata schema.
    """

    if not batch:
        raise ValueError("cannot collate an empty gaze batch")
    fixed_keys = (
        "front_image",
        "side_image",
        "pair_mask",
        "target_gaze_xy",
        "target_in_screen_bounds",
    )
    optional_auxiliary_keys = sorted(
        {key for sample in batch for key in sample if key in _MODEL_AUXILIARY_KEYS}
    )
    fixed_batch: list[dict[str, Any]] = []
    present_keys = {key for key in fixed_keys if key in batch[0]}
    for index, sample in enumerate(batch):
        sample_keys = {key for key in fixed_keys if key in sample}
        if sample_keys != present_keys:
            raise ManifestError(
                f"batch sample {index} has fixed keys {sorted(sample_keys)}, expected "
                f"{sorted(present_keys)}; use a view-filtered or paired Dataset"
            )
        fixed_sample = {key: sample[key] for key in present_keys}
        for key in optional_auxiliary_keys:
            fixed_sample[key] = sample.get(key, _missing_auxiliary_value(key))
        fixed_batch.append(fixed_sample)
    collated = dict(default_collate(fixed_batch))
    collated["metadata"] = [sample.get("metadata", {}) for sample in batch]
    return collated


def _missing_auxiliary_value(key: str) -> Any:
    """Return a shape-stable sentinel for a missing optional model input."""

    suffix = key.removeprefix("front_").removeprefix("side_")
    if suffix in {"head_vector", "face_origin_3d"}:
        return torch.full((3,), torch.nan, dtype=torch.float32)
    if suffix in {"head_pose_2d", "eye_pose_2d", "iris_pose_2d", "eye_angles"}:
        return torch.full((2,), torch.nan, dtype=torch.float32)
    if suffix in {
        "head_pose_valid",
        "head_pose_2d_valid",
        "head_orientation_valid",
        "face_origin_valid",
        "gaze_valid",
        "eye_selection_valid",
    }:
        return torch.tensor(False, dtype=torch.bool)
    if suffix == "selected_eye_index":
        return torch.tensor(-1, dtype=torch.int64)
    if suffix == "selected_eye":
        return "unknown"
    raise KeyError(f"unsupported optional auxiliary key {key!r}")


def _seed_dataloader_worker(_worker_id: int) -> None:
    """Seed non-Torch RNGs from the deterministic worker seed assigned by Torch."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    dataset: Dataset[Any],
    config: Mapping[str, Any],
    *,
    split: str = "train",
) -> DataLoader[Any]:
    """Build a split-aware, deterministically seeded PyTorch DataLoader.

    ``config`` may be the full resolved project config, its ``data`` section,
    or the ``data.dataloader`` section itself.  Worker-only options are omitted
    when ``num_workers=0`` because PyTorch rejects those combinations.
    """

    if not isinstance(config, Mapping):
        raise ManifestError("DataLoader config must be a mapping")
    data_config: Mapping[str, Any]
    if isinstance(config.get("data"), Mapping):
        data_config = config["data"]
    else:
        data_config = config
    if isinstance(data_config.get("dataloader"), Mapping):
        loader_config = data_config["dataloader"]
    else:
        loader_config = data_config

    aliases = {"val": "validation", "valid": "validation", "dev": "validation"}
    split_name = aliases.get(str(split).lower(), str(split).lower())
    if split_name not in {"train", "validation", "test"}:
        raise ManifestError("DataLoader split must be train, validation/val, or test")

    batch_size_key = f"{split_name}_batch_size"
    batch_size = int(loader_config.get(batch_size_key, loader_config.get("batch_size", 1)))
    num_workers = int(loader_config.get("num_workers", 0))
    if batch_size <= 0:
        raise ManifestError("data.dataloader.batch_size must be positive")
    if num_workers < 0:
        raise ManifestError("data.dataloader.num_workers must be non-negative")

    shuffle = bool(loader_config.get(f"{split_name}_shuffle", split_name == "train"))
    drop_last = bool(
        loader_config.get(
            f"drop_last_{split_name}",
            loader_config.get("drop_last", False),
        )
    )

    seed_value: Any = loader_config.get("seed")
    if seed_value is None and isinstance(data_config.get("split"), Mapping):
        seed_value = data_config["split"].get("seed")
    if seed_value is None and isinstance(config.get("experiment"), Mapping):
        seed_value = config["experiment"].get("seed")
    seed = int(42 if seed_value is None else seed_value)
    generator = torch.Generator()
    generator.manual_seed(seed)

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(loader_config.get("pin_memory", False)),
        "drop_last": drop_last,
        "generator": generator,
        "collate_fn": gaze_collate_fn,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(loader_config.get("persistent_workers", False))
        prefetch_factor = loader_config.get("prefetch_factor", 2)
        if prefetch_factor is not None:
            prefetch_factor = int(prefetch_factor)
            if prefetch_factor <= 0:
                raise ManifestError("data.dataloader.prefetch_factor must be positive")
            kwargs["prefetch_factor"] = prefetch_factor
        kwargs["worker_init_fn"] = _seed_dataloader_worker
    return DataLoader(dataset, **kwargs)


# Compatibility names kept intentionally small; all refer to the same contract.
GazeDataset = GazeImageDataset
StaticGazeDataset = GazeImageDataset


__all__ = [
    "GazeDataset",
    "GazeImageDataset",
    "ManifestError",
    "StaticGazeDataset",
    "build_dataloader",
    "gaze_collate_fn",
]
