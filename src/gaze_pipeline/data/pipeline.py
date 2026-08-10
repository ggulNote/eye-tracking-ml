"""Config-driven dataset preparation and reproducible manifest generation.

Preparation is intentionally metadata-only: source images are validated in
place and referenced by path, never copied into the run directory.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import yaml

from .mpiifacegaze import read_image_shape, read_mpiifacegaze_dataset
from .records import (
    MANIFEST_COLUMNS,
    CanonicalRecord,
    DataContractError,
    ScreenCalibration,
    ensure_unique_records,
    record_counts,
)
from .split import (
    SPLIT_NAMES,
    PairValidationResult,
    deterministic_group_split,
    validate_explicit_pairs,
)

GENERIC_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "sample_id",
        "subject_id",
        "view",
        "image_path",
        "target_x_px",
        "target_y_px",
        "screen_width_px",
        "screen_height_px",
    }
)
_GENERIC_READER_TYPES = frozenset({"generic_csv", "dual_view_csv", "csv_manifest"})
_GENERIC_OPTIONAL_METADATA_COLUMNS = frozenset(
    {
        "session_id",
        "facial_landmarks_xy",
        "head_rotation_3d",
        "head_translation_3d",
        "face_center_3d",
        "gaze_target_3d",
        "evaluation_eye",
        "visible_eye",
        "visible_eye_bbox_xyxy",
        "visible_eye_keypoints_xy",
        "iris_center_xy",
        "profile_head_origin_xy",
        "profile_head_forward_xy",
        "eye_annotation_valid",
    }
)


class DataPreparationError(DataContractError):
    """Raised when config-driven data preparation cannot be completed."""


@dataclass(frozen=True, slots=True)
class DataPreparationResult:
    """Prepared records plus paths and hashes needed by a later trainer."""

    records: tuple[CanonicalRecord, ...]
    splits: Mapping[str, tuple[CanonicalRecord, ...]]
    manifest_paths: Mapping[str, Path]
    hash_paths: Mapping[str, Path]
    artifact_hashes: Mapping[str, str]
    summary_path: Path
    dataset_manifest_hash: str
    resolved_config_path: Path
    resolved_config_hash_path: Path
    resolved_config_hash: str
    pair_validation: PairValidationResult
    summary: Mapping[str, Any]

    @property
    def split_counts(self) -> Mapping[str, int]:
        return {name: len(self.splits[name]) for name in SPLIT_NAMES}

    @property
    def dataset_hash(self) -> str:
        """Backward-compatible alias for the dataset manifest byte hash."""

        return self.dataset_manifest_hash


def prepare_data(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> DataPreparationResult:
    """Prepare records and deterministic subject-grouped split manifests.

    Supported readers are the strict ``mpiifacegaze_text`` source reader and a
    generic dual-view CSV contract.  Pairing is accepted only from an explicit
    ``pair_id`` column; this function never guesses pairs from filenames or
    row order.
    """

    if not isinstance(config, Mapping):
        raise DataPreparationError("config must be a mapping")
    base_dir = _config_base_dir(config_path)
    data_config = _required_mapping(config, "data")
    reader_config = _required_mapping(data_config, "reader")
    pairing_config = _mapping(data_config.get("pairing"), "data.pairing")
    split_config = _required_mapping(data_config, "split")
    _validate_supported_prepare_options(reader_config, split_config)
    reader_type = str(reader_config.get("type", "")).strip().lower()
    dataset_name = str(data_config.get("dataset_name", "generic")).strip()
    target_bounds_policy = _target_bounds_policy(
        reader_config,
        default="keep_flagged" if reader_type == "mpiifacegaze_text" else "error",
    )

    root_value = data_config.get("dataset_root")
    if root_value is None:
        root_value = _required_mapping(config, "paths").get("data_root")
    dataset_root = _resolve_config_path(
        root_value,
        base_dir=base_dir,
        field_name="data.dataset_root",
    )

    if reader_type == "mpiifacegaze_text":
        merged_reader_config = dict(reader_config)
        merged_reader_config.setdefault(
            "image_extensions",
            data_config.get("image_extensions", (".jpg", ".jpeg", ".png")),
        )
        source_records = read_mpiifacegaze_dataset(dataset_root, merged_reader_config)
    elif reader_type in _GENERIC_READER_TYPES:
        source_records = read_generic_csv_manifest(
            dataset_root=dataset_root,
            data_config=data_config,
            reader_config=reader_config,
            base_dir=base_dir,
            dataset_name=dataset_name or "generic",
        )
    else:
        supported = ["mpiifacegaze_text", *_GENERIC_READER_TYPES]
        raise DataPreparationError(
            f"unsupported data.reader.type {reader_type!r}; choose one of {sorted(supported)}"
        )

    pair_validation = validate_explicit_pairs(
        source_records,
        enabled=bool(pairing_config.get("enabled", False)),
        require_same_subject=bool(pairing_config.get("require_same_subject", True)),
        require_same_target=bool(pairing_config.get("require_same_target", True)),
        max_target_distance_normalized=float(
            pairing_config.get("max_target_distance_normalized", 0.0)
        ),
        unpaired_policy=str(pairing_config.get("unpaired_policy", "branch_only")),
    )
    records = tuple(sorted(pair_validation.records, key=_record_sort_key))
    if not records:
        raise DataPreparationError("pairing policy removed every record; no data remains to split")
    ensure_unique_records(records)

    split_strategy = str(split_config.get("strategy", "grouped_ratio")).strip().lower()
    if split_strategy != "grouped_ratio":
        raise DataPreparationError(
            "data.split.strategy must be 'grouped_ratio' for leakage-safe "
            f"preparation, got {split_strategy!r}"
        )
    ratios = _mapping(split_config.get("ratios"), "data.split.ratios")
    experiment_config = _mapping(config.get("experiment"), "experiment")
    seed = int(split_config.get("seed", experiment_config.get("seed", 42)))
    group_key = str(split_config.get("group_key", "subject_id")).strip()
    splits_mutable = deterministic_group_split(
        records,
        ratios=ratios or None,
        seed=seed,
        group_key=group_key,
        shuffle_groups=bool(split_config.get("shuffle_groups", True)),
    )
    splits = {name: tuple(splits_mutable[name]) for name in SPLIT_NAMES}

    manifest_dir = _resolve_manifest_dir(
        config,
        split_config=split_config,
        base_dir=base_dir,
    )
    return _write_preparation_artifacts(
        records=records,
        resolved_config=config,
        source_record_count=len(source_records),
        splits=splits,
        pair_validation=pair_validation,
        manifest_dir=manifest_dir,
        dataset_root=dataset_root,
        dataset_name=dataset_name or "generic",
        reader_type=reader_type,
        group_key=group_key,
        ratios=ratios
        or {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        seed=seed,
        pairing_enabled=bool(pairing_config.get("enabled", False)),
        unpaired_policy=str(pairing_config.get("unpaired_policy", "branch_only")),
        target_bounds_policy=target_bounds_policy,
    )


def read_generic_csv_manifest(
    *,
    dataset_root: str | Path,
    data_config: Mapping[str, Any],
    reader_config: Mapping[str, Any],
    base_dir: str | Path,
    dataset_name: str = "generic",
) -> list[CanonicalRecord]:
    """Read the explicit, model-independent front/side CSV contract."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise DataPreparationError(f"generic dataset_root is not a directory: {root}")
    manifest_value = (
        reader_config.get("manifest_path")
        or reader_config.get("csv_path")
        or data_config.get("manifest_path")
    )
    manifest_path = _resolve_config_path(
        manifest_value,
        base_dir=Path(base_dir),
        field_name="data.reader.manifest_path",
    )
    if not manifest_path.is_file():
        raise DataPreparationError(f"generic CSV manifest does not exist: {manifest_path}")

    verify_image_exists = bool(reader_config.get("verify_image_exists", True))
    verify_image_shape = bool(reader_config.get("verify_image_shape", True))
    if verify_image_shape:
        verify_image_exists = True
    source_to_branch = _source_to_branch(data_config)
    directory_to_branch = _directory_to_branch(data_config)
    target_bounds_policy = _target_bounds_policy(reader_config, default="error")
    pairing_config = _mapping(data_config.get("pairing"), "data.pairing")
    pair_id_key = _pair_id_key(pairing_config)
    pairing_enabled = bool(pairing_config.get("enabled", False))

    records: list[CanonicalRecord] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise DataPreparationError(f"generic CSV manifest has no header: {manifest_path}")
        normalized_fieldnames = [name.strip() for name in fieldnames]
        if len(set(normalized_fieldnames)) != len(normalized_fieldnames):
            raise DataPreparationError(
                f"generic CSV manifest has duplicate header names: {manifest_path}"
            )
        required_columns = set(GENERIC_REQUIRED_COLUMNS)
        if pairing_enabled:
            required_columns.add(pair_id_key)
        missing = sorted(required_columns - set(normalized_fieldnames))
        if missing:
            raise DataPreparationError(
                f"generic CSV manifest is missing columns {missing}: {manifest_path}"
            )

        for line_number, raw_row in enumerate(reader, start=2):
            row = {
                str(key).strip(): ("" if value is None else value.strip())
                for key, value in raw_row.items()
                if key is not None
            }
            try:
                record = _generic_row_to_record(
                    row,
                    line_number=line_number,
                    manifest_path=manifest_path,
                    dataset_root=root,
                    dataset_name=dataset_name,
                    source_to_branch=source_to_branch,
                    directory_to_branch=directory_to_branch,
                    pair_id_key=pair_id_key,
                    verify_image_exists=verify_image_exists,
                    verify_image_shape=verify_image_shape,
                    allow_out_of_bounds_targets=(target_bounds_policy == "keep_flagged"),
                )
            except DataContractError as exc:
                raise DataPreparationError(f"{manifest_path}:{line_number}: {exc}") from exc
            records.append(record)

    if not records:
        raise DataPreparationError(f"generic CSV manifest is empty: {manifest_path}")
    ensure_unique_records(records)
    return sorted(records, key=_record_sort_key)


def _generic_row_to_record(
    row: Mapping[str, str],
    *,
    line_number: int,
    manifest_path: Path,
    dataset_root: Path,
    dataset_name: str,
    source_to_branch: Mapping[str, str],
    directory_to_branch: Mapping[str, str],
    pair_id_key: str,
    verify_image_exists: bool,
    verify_image_shape: bool,
    allow_out_of_bounds_targets: bool,
) -> CanonicalRecord:
    sample_id = _required_cell(row, "sample_id")
    subject_id = _required_cell(row, "subject_id")
    raw_image_path = _required_cell(row, "image_path")
    image_path, image_relative_path = _resolve_dataset_image_path(raw_image_path, dataset_root)
    if verify_image_exists and not image_path.is_file():
        raise DataPreparationError(f"referenced image does not exist: {image_path}")
    if verify_image_shape:
        image_width, image_height, image_channels = read_image_shape(image_path)
    else:
        image_width, image_height, image_channels = 1, 1, None

    raw_view = row.get("view", "").strip().lower()
    if raw_view:
        view = _canonical_view(raw_view, source_to_branch=source_to_branch)
    else:
        view = _view_from_directory(image_relative_path, directory_to_branch=directory_to_branch)

    width_mm = _optional_finite_float(row.get("screen_width_mm", ""))
    height_mm = _optional_finite_float(row.get("screen_height_mm", ""))
    if (width_mm is None) != (height_mm is None):
        raise DataPreparationError(
            "screen_width_mm and screen_height_mm must both be filled or blank"
        )
    screen = ScreenCalibration(
        width_px=_positive_integer_cell(row, "screen_width_px"),
        height_px=_positive_integer_cell(row, "screen_height_px"),
        width_mm=width_mm,
        height_mm=height_mm,
    )
    return CanonicalRecord(
        sample_id=sample_id,
        subject_id=subject_id,
        session_id=row.get("session_id", ""),
        view=view,
        pair_id=row.get(pair_id_key, "") or None,
        image_path=image_path,
        image_relative_path=image_relative_path,
        image_width_px=image_width,
        image_height_px=image_height,
        image_channels=image_channels,
        gaze_screen_xy_px=(
            _finite_float_cell(row, "target_x_px"),
            _finite_float_cell(row, "target_y_px"),
        ),
        screen=screen,
        facial_landmarks_xy=_optional_json_landmarks(row, "facial_landmarks_xy"),
        head_rotation_3d=_optional_json_vector(row, "head_rotation_3d"),
        head_translation_3d=_optional_json_vector(row, "head_translation_3d"),
        face_center_3d=_optional_json_vector(row, "face_center_3d"),
        gaze_target_3d=_optional_json_vector(row, "gaze_target_3d"),
        evaluation_eye=row.get("evaluation_eye", "") or None,
        visible_eye=row.get("visible_eye", "") or None,
        visible_eye_bbox_xyxy=cast(
            tuple[float, float, float, float] | None,
            _optional_json_finite_sequence(row, "visible_eye_bbox_xyxy", length=4),
        ),
        visible_eye_keypoints_xy=_optional_json_eye_keypoints(row, "visible_eye_keypoints_xy"),
        iris_center_xy=cast(
            tuple[float, float] | None,
            _optional_json_finite_sequence(row, "iris_center_xy", length=2),
        ),
        profile_head_origin_xy=cast(
            tuple[float, float] | None,
            _optional_json_finite_sequence(row, "profile_head_origin_xy", length=2),
        ),
        profile_head_forward_xy=cast(
            tuple[float, float] | None,
            _optional_json_finite_sequence(row, "profile_head_forward_xy", length=2),
        ),
        eye_annotation_valid=_optional_boolean_cell(
            row,
            "eye_annotation_valid",
            default=False,
        ),
        source_dataset=dataset_name,
        annotation_path=manifest_path,
        annotation_line=line_number,
        allow_out_of_screen_target=allow_out_of_bounds_targets,
    )


def _write_preparation_artifacts(
    *,
    records: Sequence[CanonicalRecord],
    resolved_config: Mapping[str, Any],
    source_record_count: int,
    splits: Mapping[str, Sequence[CanonicalRecord]],
    pair_validation: PairValidationResult,
    manifest_dir: Path,
    dataset_root: Path,
    dataset_name: str,
    reader_type: str,
    group_key: str,
    ratios: Mapping[str, float],
    seed: int,
    pairing_enabled: bool,
    unpaired_policy: str,
    target_bounds_policy: str,
) -> DataPreparationResult:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (
        resolved_config_path,
        resolved_config_hash_path,
        resolved_config_hash,
    ) = _write_resolved_config_snapshot(
        resolved_config,
        path=manifest_dir.parent / "resolved_config.yaml",
    )
    split_by_sample = {
        record.sample_id: split_name for split_name in SPLIT_NAMES for record in splits[split_name]
    }

    all_path = manifest_dir / "dataset_manifest.csv"
    split_path = manifest_dir / "split_manifest.csv"
    split_paths = {split_name: manifest_dir / f"{split_name}.csv" for split_name in SPLIT_NAMES}
    artifact_hashes: dict[str, str] = {}
    artifact_hashes["dataset"] = _write_manifest_csv(
        all_path,
        records,
        pair_validation=pair_validation,
    )
    artifact_hashes["split"] = _write_manifest_csv(
        split_path,
        records,
        pair_validation=pair_validation,
        split_by_sample=split_by_sample,
    )
    for split_name in SPLIT_NAMES:
        artifact_hashes[split_name] = _write_manifest_csv(
            split_paths[split_name],
            splits[split_name],
            pair_validation=pair_validation,
            split_by_sample=split_by_sample,
        )

    manifest_paths: dict[str, Path] = {
        "dataset": all_path,
        "split": split_path,
        **split_paths,
    }
    hash_paths: dict[str, Path] = {}
    for name, path in manifest_paths.items():
        hash_paths[name] = _write_hash_sidecar(path, artifact_hashes[name])

    summary = _build_summary(
        records=records,
        source_record_count=source_record_count,
        splits=splits,
        pair_validation=pair_validation,
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        reader_type=reader_type,
        group_key=group_key,
        ratios=ratios,
        seed=seed,
        pairing_enabled=pairing_enabled,
        unpaired_policy=unpaired_policy,
        target_bounds_policy=target_bounds_policy,
        resolved_config_path=resolved_config_path,
        resolved_config_hash=resolved_config_hash,
        manifest_paths=manifest_paths,
        artifact_hashes=artifact_hashes,
    )
    summary_path = manifest_dir / "summary.json"
    summary_payload = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    _atomic_write(summary_path, summary_payload)
    summary_hash = _sha256(summary_payload)
    artifact_hashes["summary"] = summary_hash
    hash_paths["summary"] = _write_hash_sidecar(summary_path, summary_hash)

    frozen_splits = {name: tuple(splits[name]) for name in SPLIT_NAMES}
    return DataPreparationResult(
        records=tuple(records),
        splits=frozen_splits,
        manifest_paths=manifest_paths,
        hash_paths=hash_paths,
        artifact_hashes=artifact_hashes,
        summary_path=summary_path,
        dataset_manifest_hash=artifact_hashes["dataset"],
        resolved_config_path=resolved_config_path,
        resolved_config_hash_path=resolved_config_hash_path,
        resolved_config_hash=resolved_config_hash,
        pair_validation=pair_validation,
        summary=summary,
    )


def _write_manifest_csv(
    path: Path,
    records: Sequence[CanonicalRecord],
    *,
    pair_validation: PairValidationResult,
    split_by_sample: Mapping[str, str] | None = None,
) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=MANIFEST_COLUMNS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for record in sorted(records, key=_record_sort_key):
        split_name = (
            split_by_sample.get(record.sample_id, "") if split_by_sample is not None else ""
        )
        writer.writerow(
            record.to_manifest_row(
                split=split_name,
                pair_complete=pair_validation.is_complete(record),
            )
        )
    payload = stream.getvalue().encode()
    _atomic_write(path, payload)
    return _sha256(payload)


def _build_summary(
    *,
    records: Sequence[CanonicalRecord],
    source_record_count: int,
    splits: Mapping[str, Sequence[CanonicalRecord]],
    pair_validation: PairValidationResult,
    dataset_root: Path,
    dataset_name: str,
    reader_type: str,
    group_key: str,
    ratios: Mapping[str, float],
    seed: int,
    pairing_enabled: bool,
    unpaired_policy: str,
    target_bounds_policy: str,
    resolved_config_path: Path,
    resolved_config_hash: str,
    manifest_paths: Mapping[str, Path],
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    total = len(records)
    split_summary: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        split_records = splits[split_name]
        groups = sorted({str(getattr(record, group_key)) for record in split_records})
        split_summary[split_name] = {
            "records": len(split_records),
            "groups": len(groups),
            "group_ids": groups,
            "realized_record_ratio": len(split_records) / total,
        }

    return {
        "schema_version": 1,
        "dataset_manifest_hash": artifact_hashes["dataset"],
        "resolved_config": {
            "path": str(resolved_config_path),
            "sha256": resolved_config_hash,
            "sensitive_values_redacted": True,
        },
        "dataset": {
            "name": dataset_name,
            "reader_type": reader_type,
            "root": str(dataset_root),
            "source_records": source_record_count,
            "prepared_records": len(records),
            "counts": record_counts(records),
            "target_bounds": {
                "policy": target_bounds_policy,
                "flag_column": "target_in_screen_bounds",
                "raw_target_preserved": True,
                "flagged_records_remain_in_split": True,
                "metric_filter": "target_in_screen_bounds == true",
            },
        },
        "pairing": {
            "enabled": pairing_enabled,
            "strategy": "explicit_pair_id",
            "unpaired_policy": unpaired_policy,
            "complete_pair_count": len(pair_validation.complete_pair_ids),
            "incomplete_pair_count": len(pair_validation.incomplete_pair_ids),
            "unpaired_sample_count": len(pair_validation.unpaired_sample_ids),
            "dropped_sample_count": len(pair_validation.dropped_sample_ids),
        },
        "split": {
            "strategy": "grouped_ratio",
            "group_key": group_key,
            "seed": seed,
            "configured_ratios": {name: float(ratios[name]) for name in SPLIT_NAMES},
            "partitions": split_summary,
        },
        "artifacts": {
            name: {
                "path": str(manifest_paths[name]),
                "sha256": artifact_hashes[name],
            }
            for name in manifest_paths
        },
        "raw_images_copied": False,
    }


def _source_to_branch(data_config: Mapping[str, Any]) -> Mapping[str, str]:
    return _branch_mapping(data_config, "source_to_branch")


def _directory_to_branch(data_config: Mapping[str, Any]) -> Mapping[str, str]:
    return _branch_mapping(data_config, "directory_to_branch")


def _branch_mapping(data_config: Mapping[str, Any], mapping_name: str) -> Mapping[str, str]:
    views = _mapping(data_config.get("views"), "data.views")
    raw_mapping = _mapping(
        views.get(mapping_name),
        f"data.views.{mapping_name}",
    )
    result: dict[str, str] = {}
    for source, branch in raw_mapping.items():
        source_name = str(source).strip().lower()
        branch_name = str(branch).strip().lower()
        if not source_name:
            raise DataPreparationError(f"data.views.{mapping_name} contains an empty source name")
        if branch_name not in {"front", "side"}:
            raise DataPreparationError(
                f"data.views.{mapping_name}.{source} must map to front/side, got {branch!r}"
            )
        previous = result.setdefault(source_name, branch_name)
        if previous != branch_name:
            raise DataPreparationError(
                f"data.views.{mapping_name} has a case-insensitive collision for {source_name!r}"
            )
    return result


def _canonical_view(raw_view: str, *, source_to_branch: Mapping[str, str]) -> str:
    normalized = raw_view.strip().lower()
    if normalized in {"front", "side"}:
        return normalized
    branch = source_to_branch.get(normalized)
    if branch is None:
        raise DataPreparationError(
            f"view value {raw_view!r} is not front/side and has no "
            "data.views.source_to_branch mapping"
        )
    return branch


def _pair_id_key(pairing_config: Mapping[str, Any]) -> str:
    pair_id_key = str(pairing_config.get("pair_id_key", "pair_id")).strip()
    if not pair_id_key:
        raise DataPreparationError("data.pairing.pair_id_key must not be empty")
    reserved = GENERIC_REQUIRED_COLUMNS | _GENERIC_OPTIONAL_METADATA_COLUMNS
    if pair_id_key in reserved:
        raise DataPreparationError(
            f"data.pairing.pair_id_key {pair_id_key!r} collides with a canonical generic CSV column"
        )
    return pair_id_key


def _validate_supported_prepare_options(
    reader_config: Mapping[str, Any], split_config: Mapping[str, Any]
) -> None:
    if reader_config.get("fail_on_bad_row", True) is not True:
        raise DataPreparationError(
            "data.reader.fail_on_bad_row=false is unsupported; preparation is "
            "strict and never silently skips malformed rows"
        )
    if split_config.get("reuse_existing_manifest", False) is not False:
        raise DataPreparationError(
            "data.split.reuse_existing_manifest=true is unsupported; existing "
            "manifests are not loaded by this implementation"
        )
    if split_config.get("stratify_by") is not None:
        raise DataPreparationError(
            "data.split.stratify_by is unsupported; only null is currently accepted"
        )


def _target_bounds_policy(reader_config: Mapping[str, Any], *, default: str) -> str:
    policy = str(reader_config.get("target_bounds_policy", default)).strip().lower()
    if policy not in {"error", "keep_flagged"}:
        raise DataPreparationError(
            f"data.reader.target_bounds_policy must be 'error' or 'keep_flagged', got {policy!r}"
        )
    return policy


def _view_from_directory(
    image_relative_path: str,
    *,
    directory_to_branch: Mapping[str, str],
) -> str:
    path_parts = {part.lower() for part in PurePosixPath(image_relative_path).parts}
    matches = {
        branch for directory, branch in directory_to_branch.items() if directory in path_parts
    }
    if len(matches) == 1:
        return next(iter(matches))
    if not matches:
        raise DataPreparationError(
            "view is blank and image_path does not match any configured "
            "data.views.directory_to_branch entry"
        )
    raise DataPreparationError(
        f"view is ambiguous for {image_relative_path!r}; matched branches {sorted(matches)}"
    )


def _optional_json_landmarks(row: Mapping[str, str], key: str) -> tuple[tuple[float, float], ...]:
    decoded = _optional_json_array(row, key)
    if decoded is None:
        return ()
    landmarks: list[tuple[float, float]] = []
    for index, point in enumerate(decoded):
        if not isinstance(point, list) or len(point) != 2:
            raise DataPreparationError(f"column {key!r} point {index} must be a JSON [x, y] array")
        landmarks.append(
            (
                _finite_json_number(point[0], field_name=f"{key}[{index}][0]"),
                _finite_json_number(point[1], field_name=f"{key}[{index}][1]"),
            )
        )
    return tuple(landmarks)


def _optional_json_vector(row: Mapping[str, str], key: str) -> tuple[float, float, float] | None:
    return cast(
        tuple[float, float, float] | None,
        _optional_json_finite_sequence(row, key, length=3),
    )


def _optional_json_finite_sequence(
    row: Mapping[str, str], key: str, *, length: int
) -> tuple[float, ...] | None:
    decoded = _optional_json_array(row, key)
    if decoded is None:
        return None
    if len(decoded) != length:
        raise DataPreparationError(
            f"column {key!r} must contain exactly {length} JSON numbers, got {len(decoded)}"
        )
    return tuple(
        _finite_json_number(value, field_name=f"{key}[{index}]")
        for index, value in enumerate(decoded)
    )


def _optional_json_eye_keypoints(
    row: Mapping[str, str], key: str
) -> tuple[tuple[float, float], ...]:
    decoded = _optional_json_array(row, key)
    if decoded is None:
        return ()
    if len(decoded) != 6:
        raise DataPreparationError(
            f"column {key!r} must contain exactly 6 JSON [x, y] points, got {len(decoded)}"
        )
    points: list[tuple[float, float]] = []
    for index, point in enumerate(decoded):
        if not isinstance(point, list) or len(point) != 2:
            raise DataPreparationError(f"column {key!r} point {index} must be a JSON [x, y] array")
        points.append(
            (
                _finite_json_number(point[0], field_name=f"{key}[{index}][0]"),
                _finite_json_number(point[1], field_name=f"{key}[{index}][1]"),
            )
        )
    return tuple(points)


def _optional_json_array(row: Mapping[str, str], key: str) -> list[Any] | None:
    raw_value = row.get(key, "").strip()
    if not raw_value:
        return None
    try:
        decoded = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise DataPreparationError(f"column {key!r} must contain valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, list):
        raise DataPreparationError(f"column {key!r} must contain a JSON array")
    return decoded


def _finite_json_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DataPreparationError(f"{field_name} must be a JSON number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise DataPreparationError(f"{field_name} must be finite, got {value!r}")
    return result


def _resolve_dataset_image_path(raw_path: str, dataset_root: Path) -> tuple[Path, str]:
    normalized = raw_path.strip().replace("\\", "/")
    if not normalized or "${" in normalized:
        raise DataPreparationError(f"image_path must be a resolved path, got {raw_path!r}")
    supplied = Path(normalized).expanduser()
    if supplied.is_absolute():
        resolved = supplied.resolve()
    else:
        relative = PurePosixPath(normalized)
        if relative.is_absolute() or ".." in relative.parts:
            raise DataPreparationError(f"image_path must not escape dataset_root: {raw_path!r}")
        resolved = (dataset_root / Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(dataset_root):
        raise DataPreparationError(f"image_path is outside dataset_root {dataset_root}: {resolved}")
    relative_path = resolved.relative_to(dataset_root).as_posix()
    return resolved, relative_path


def _required_cell(row: Mapping[str, str], key: str) -> str:
    value = row.get(key, "").strip()
    if not value:
        raise DataPreparationError(f"column {key!r} must not be blank")
    return value


def _finite_float_cell(row: Mapping[str, str], key: str) -> float:
    value = _required_cell(row, key)
    try:
        result = float(value)
    except ValueError as exc:
        raise DataPreparationError(f"column {key!r} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise DataPreparationError(f"column {key!r} must be finite, got {value!r}")
    return result


def _optional_finite_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        result = float(stripped)
    except ValueError as exc:
        raise DataPreparationError(f"optional numeric value is invalid: {stripped!r}") from exc
    if not math.isfinite(result):
        raise DataPreparationError(f"optional numeric value must be finite: {stripped!r}")
    return result


def _optional_boolean_cell(row: Mapping[str, str], key: str, *, default: bool) -> bool:
    raw_value = row.get(key, "").strip().lower()
    if not raw_value:
        return bool(default)
    if raw_value in {"true", "1", "yes", "y"}:
        return True
    if raw_value in {"false", "0", "no", "n"}:
        return False
    raise DataPreparationError(
        f"column {key!r} must be a boolean true/false value, got {row.get(key)!r}"
    )


def _positive_integer_cell(row: Mapping[str, str], key: str) -> int:
    numeric = _finite_float_cell(row, key)
    rounded = round(numeric)
    if numeric <= 0 or not math.isclose(numeric, rounded):
        raise DataPreparationError(f"column {key!r} must be a positive integer, got {numeric!r}")
    return int(rounded)


def _resolve_manifest_dir(
    config: Mapping[str, Any],
    *,
    split_config: Mapping[str, Any],
    base_dir: Path,
) -> Path:
    value = split_config.get("manifest_dir")
    if value is None:
        paths = _mapping(config.get("paths"), "paths")
        run_dir = paths.get("run_dir")
        if run_dir is None:
            value = base_dir / "manifests"
        else:
            value = Path(str(run_dir)) / "manifests"
    return _resolve_config_path(
        value,
        base_dir=base_dir,
        field_name="data.split.manifest_dir",
    )


def _resolve_config_path(
    value: Any,
    *,
    base_dir: Path,
    field_name: str,
) -> Path:
    if value is None or not str(value).strip():
        raise DataPreparationError(f"{field_name} must be a non-empty path")
    rendered = str(value).strip()
    if "${" in rendered:
        raise DataPreparationError(f"{field_name} contains unresolved interpolation: {rendered!r}")
    path = Path(rendered).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _config_base_dir(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd().resolve()
    path = Path(config_path).expanduser().resolve()
    if path.parent.name == "configs":
        return path.parent.parent
    return path.parent


def _required_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in mapping:
        raise DataPreparationError(f"config is missing mapping {key!r}")
    return _mapping(mapping[key], key)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DataPreparationError(f"{field_name} must be a mapping")
    return value


def _record_sort_key(record: CanonicalRecord) -> tuple[str, str]:
    return record.subject_id, record.sample_id


def _write_resolved_config_snapshot(
    config: Mapping[str, Any], *, path: Path
) -> tuple[Path, Path, str]:
    redacted = _redact_config_value(config)
    payload = yaml.safe_dump(
        redacted,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).encode()
    _atomic_write(path, payload)
    digest = _sha256(payload)
    hash_path = _write_hash_sidecar(path, digest)
    return path, hash_path, digest


def _redact_config_value(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _is_sensitive_config_key(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(child_key): _redact_config_value(
                child_value,
                key=str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_redact_config_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        if key is not None and _is_tracking_uri_key(key):
            return _redact_uri_userinfo(value)
        return value
    if value is None or isinstance(value, bool | int | float):
        return value
    raise DataPreparationError(
        f"resolved config contains a value that cannot be safely serialized: {type(value).__name__}"
    )


def _is_sensitive_config_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(
        marker in normalized for marker in ("secret", "token", "password", "credential", "api_key")
    )


def _is_tracking_uri_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return normalized == "tracking_uri" or normalized.endswith("_tracking_uri")


def _redact_uri_userinfo(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    safe_netloc = parsed.netloc.rsplit("@", 1)[1] if "@" in parsed.netloc else parsed.netloc
    safe_query = urlencode(
        [
            (
                query_key,
                "<redacted>" if _is_sensitive_config_key(query_key) else query_value,
            )
            for query_key, query_value in parse_qsl(
                parsed.query,
                keep_blank_values=True,
            )
        ],
        doseq=True,
    )
    safe_fragment = "<redacted>" if _is_sensitive_config_key(parsed.fragment) else parsed.fragment
    return urlunsplit((parsed.scheme, safe_netloc, parsed.path, safe_query, safe_fragment))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _write_hash_sidecar(path: Path, digest: str) -> Path:
    sidecar_path = path.with_name(f"{path.name}.sha256")
    _atomic_write(sidecar_path, f"{digest}  {path.name}\n".encode())
    return sidecar_path


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# Short aliases for orchestration code and external adapters.
prepare = prepare_data
load_generic_records = read_generic_csv_manifest
