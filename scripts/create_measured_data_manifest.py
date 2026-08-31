"""Build a dual-view manifest for the measured, named-subject session DB.

The source DB is treated as read-only.  Each selected subject/session must
contain ``feature_maps/training.csv`` and ``feature_maps/evaluation.csv``.
Those tables point at the lens-corrected ``web/frames`` and ``phone/frames``
images. Front always uses the corrected web frame. With ``--side-roi-root``,
Side instead uses ``<subject>/<session>/side_eye_roi`` images joined by the
original phone-frame filename; missing ROI sessions are excluded. Existing
WebEyeTrack eye patches are deliberately never selected as ``image_path``.

Front metric head pose and the upstream sample-validity decision are read only from
``feature_maps/webeyetrack/inputs.csv``.  A row with ``valid=1`` supplies the
MediaPipe-derived ``head_vector`` and the face origin in centimetres.  A row
with ``valid=0`` removes the complete webcam/phonecam pair before pairing and
subject-wise splitting.  The ordinary ``feature_maps/web/features.csv`` table
is kept only as source provenance; producer diagnostic columns are ignored here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SESSIONS = ("head_down", "neutral")
MASTER_TABLES = ("training.csv", "evaluation.csv")
SIDE_ROI_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
FRONT_INPUT_TABLE = "inputs.csv"
FRONT_INPUT_SCHEMA = "webeyetrack_input_v1"
MASTER_REQUIRED_COLUMNS = frozenset(
    {
        "sample_id",
        "subject_id",
        "head_pose",
        "view",
        "image_path",
        "pair_id",
        "target_x_px",
        "target_y_px",
        "screen_width_px",
        "screen_height_px",
        "collection_split",
        "protocol",
        "source_frame",
    }
)
FRONT_INPUT_REQUIRED_COLUMNS = frozenset(
    {
        "schema_version",
        "sample_id",
        "participant",
        "head_pose",
        "pair_id",
        "collection_split",
        "protocol",
        "source_image_path",
        "target_x_centered",
        "target_y_centered",
        "head_vector_x",
        "head_vector_y",
        "head_vector_z",
        "face_origin_x_cm",
        "face_origin_y_cm",
        "face_origin_z_cm",
        "valid",
        "invalid_reason",
    }
)
SIDE_ANNOTATION_FIELDS = (
    "visible_eye",
    "visible_eye_bbox_xyxy",
    "visible_eye_keypoints_xy",
    "iris_center_xy",
    "profile_head_origin_xy",
    "profile_head_forward_xy",
)
SIDE_ROI_FIELDS = (
    "visible_eye",
    "visible_eye_bbox_xyxy",
)
FIELDNAMES = (
    "sample_id",
    "subject_id",
    "session_id",
    "view",
    "image_path",
    "pair_id",
    "target_x_px",
    "target_y_px",
    "screen_width_px",
    "screen_height_px",
    "head_vector",
    "face_origin_3d",
    "head_pose_valid",
    "visible_eye",
    "visible_eye_bbox_xyxy",
    "visible_eye_keypoints_xy",
    "iris_center_xy",
    "profile_head_origin_xy",
    "profile_head_forward_xy",
    "eye_annotation_valid",
    "collection_split",
    "protocol",
    "source_frame",
    "feature_csv_path",
    "feature_csv_line",
    "input_csv_path",
    "input_csv_line",
)


class MeasuredManifestError(ValueError):
    """Raised when the measured DB cannot satisfy the explicit contract."""


@dataclass(frozen=True, slots=True)
class ManifestSummary:
    path: Path
    subjects: tuple[str, ...]
    sessions: tuple[str, ...]
    front_input_rows: int
    source_pairs: int
    output_pairs: int
    dropped_invalid_pairs: int
    dropped_missing_side_roi_pairs: int
    skipped_missing_side_roi_sessions: int
    output_rows: int
    front_pose_valid_rows: int
    front_pose_missing_rows: int
    side_annotation_valid_rows: int
    side_annotation_missing_rows: int


@dataclass(frozen=True, slots=True)
class _IndexedRow:
    row: Mapping[str, str]
    source: Path
    line: int


@dataclass(frozen=True, slots=True)
class _FrontInput:
    usable: bool
    pose: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _SessionResult:
    rows: tuple[dict[str, str | int], ...]
    source_pairs: int
    dropped_invalid_pairs: int
    dropped_missing_side_roi_pairs: int
    front_input_rows: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Common root containing source-root and side-roi-root. Manifest image paths are "
            "written relative to this root. Defaults to source-root."
        ),
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject directory names. NFC and NFD spellings are treated equally.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=list(DEFAULT_SESSIONS),
        help="Session directories to include (default: head_down neutral).",
    )
    parser.add_argument(
        "--require-side-annotations",
        action="store_true",
        help=(
            "Fail unless every Side row has a real visible-eye bbox. Optional eyelid, "
            "iris and head geometry is not required for image-only Side training."
        ),
    )
    parser.add_argument(
        "--side-annotations",
        type=Path,
        default=None,
        help=(
            "Optional generated Side annotation CSV keyed by sample_id. Values override the "
            "header-only feature_maps/phone/features.csv without modifying the source DB."
        ),
    )
    parser.add_argument(
        "--side-roi-root",
        type=Path,
        default=None,
        help=(
            "Optional <subject>/<session>/side_eye_roi image root. Sessions without this "
            "directory or without images are skipped. Images are joined to phone rows by "
            "the original phone frame filename."
        ),
    )
    parser.add_argument(
        "--require-front-pose",
        action="store_true",
        help=(
            "Compatibility safety flag. inputs.csv is mandatory and every emitted Front row "
            "already requires a valid head vector and face origin."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing output manifest.",
    )
    return parser


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", str(value).strip())


def _safe_session(value: str) -> str:
    session = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", session):
        raise MeasuredManifestError(f"unsafe session directory name: {value!r}")
    return session


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _discover_subjects(root: Path, requested: Sequence[str] | None) -> tuple[Path, ...]:
    discovered: dict[str, Path] = {}
    for candidate in root.iterdir():
        if not candidate.is_dir() or candidate.name.startswith("."):
            continue
        normalized = _nfc(candidate.name)
        previous = discovered.setdefault(normalized, candidate)
        if previous != candidate:
            raise MeasuredManifestError(
                "subject directories collide after NFC normalization: "
                f"{previous.name!r}, {candidate.name!r}"
            )
    if not discovered:
        raise MeasuredManifestError(f"no subject directories found under {root}")
    if requested is None:
        names = sorted(discovered)
    else:
        names = [_nfc(name) for name in requested]
        if not all(names) or len(set(names)) != len(names):
            raise MeasuredManifestError("--subjects must contain unique, non-empty names")
        missing = sorted(set(names) - set(discovered))
        if missing:
            raise MeasuredManifestError(f"requested subjects are absent: {missing}")
    return tuple(discovered[name] for name in names)


def _read_table(path: Path, required: frozenset[str]) -> list[_IndexedRow]:
    if not path.is_file():
        raise MeasuredManifestError(f"required CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = tuple(str(name).strip() for name in reader.fieldnames or ())
        if not headers:
            raise MeasuredManifestError(f"CSV has no header: {path}")
        if len(set(headers)) != len(headers):
            raise MeasuredManifestError(f"CSV has duplicate header names: {path}")
        missing = sorted(required - set(headers))
        if missing:
            raise MeasuredManifestError(f"{path}: missing columns {missing}")
        rows: list[_IndexedRow] = []
        for line, raw in enumerate(reader, start=2):
            row = {
                str(key).strip(): ("" if value is None else str(value).strip())
                for key, value in raw.items()
                if key is not None
            }
            rows.append(_IndexedRow(row=row, source=path, line=line))
    return rows


def _index_rows(rows: Sequence[_IndexedRow], *, key: str) -> dict[str, _IndexedRow]:
    result: dict[str, _IndexedRow] = {}
    for item in rows:
        value = _nfc(item.row.get(key, ""))
        if not value:
            raise MeasuredManifestError(f"{item.source}:{item.line}: {key} is empty")
        if value in result:
            previous = result[value]
            raise MeasuredManifestError(
                f"duplicate {key}={value!r}: {previous.source}:{previous.line} and "
                f"{item.source}:{item.line}"
            )
        result[value] = item
    return result


def _resolve_unicode_relative(base: Path, raw_value: str, *, source: Path, line: int) -> Path:
    raw_path = Path(str(raw_value).strip())
    if not raw_path.parts or raw_path.is_absolute() or ".." in raw_path.parts:
        raise MeasuredManifestError(f"{source}:{line}: unsafe relative path {raw_value!r}")
    current = base
    for raw_part in raw_path.parts:
        direct = current / raw_part
        if direct.exists():
            current = direct
            continue
        normalized = _nfc(raw_part)
        matches = [child for child in current.iterdir() if _nfc(child.name) == normalized]
        if len(matches) != 1:
            raise MeasuredManifestError(
                f"{source}:{line}: cannot resolve Unicode path component {raw_part!r} under "
                f"{current}"
            )
        current = matches[0]
    resolved = current.resolve(strict=False)
    base_resolved = base.resolve(strict=True)
    if not _is_within(resolved, base_resolved) or not resolved.is_file():
        raise MeasuredManifestError(f"{source}:{line}: referenced file does not exist: {resolved}")
    return resolved


def _unicode_child_directory(parent: Path, name: str) -> Path | None:
    if not parent.is_dir():
        return None
    normalized = _nfc(name)
    matches = [
        child for child in parent.iterdir() if child.is_dir() and _nfc(child.name) == normalized
    ]
    if len(matches) > 1:
        raise MeasuredManifestError(
            f"directories collide after NFC normalization under {parent}: {name!r}"
        )
    return matches[0] if matches else None


def _index_side_roi_images(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for image in sorted(directory.iterdir(), key=lambda path: _nfc(path.name)):
        if not image.is_file() or image.suffix.lower() not in SIDE_ROI_IMAGE_EXTENSIONS:
            continue
        key = _nfc(image.name)
        if key in result:
            raise MeasuredManifestError(
                f"duplicate Side ROI filename after NFC normalization: {result[key]}, {image}"
            )
        result[key] = image.resolve(strict=True)
    return result


def _manifest_image_path(image: Path, dataset_root: Path) -> str:
    resolved = image.resolve(strict=True)
    if not _is_within(resolved, dataset_root):
        raise MeasuredManifestError(f"image is outside dataset-root {dataset_root}: {resolved}")
    return resolved.relative_to(dataset_root).as_posix()


def _required(row: Mapping[str, str], key: str, *, source: Path, line: int) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise MeasuredManifestError(f"{source}:{line}: {key} is empty")
    return value


def _finite(value: str, *, field: str, source: Path, line: int) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise MeasuredManifestError(f"{source}:{line}: {field} is not numeric") from exc
    if not math.isfinite(number):
        raise MeasuredManifestError(f"{source}:{line}: {field} is not finite")
    return number


def _positive_int(value: str, *, field: str, source: Path, line: int) -> int:
    number = _finite(value, field=field, source=source, line=line)
    if number <= 0 or not number.is_integer():
        raise MeasuredManifestError(f"{source}:{line}: {field} must be a positive integer")
    return int(number)


def _target(value: str, *, limit: int, field: str, source: Path, line: int) -> str:
    number = _finite(value, field=field, source=source, line=line)
    if not 0 <= number < limit:
        raise MeasuredManifestError(f"{source}:{line}: {field}={number} is outside [0, {limit})")
    return str(value).strip()


def _compact_numeric_json(
    raw: str,
    *,
    field: str,
    shape: tuple[int, ...],
    source: Path,
    line: int,
) -> str:
    if not str(raw).strip():
        return ""
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MeasuredManifestError(f"{source}:{line}: {field} must contain valid JSON") from exc

    def validate(value: Any, dimensions: tuple[int, ...], location: str) -> Any:
        if not dimensions:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise MeasuredManifestError(
                    f"{source}:{line}: {location} must contain numeric values"
                )
            number = float(value)
            if not math.isfinite(number):
                raise MeasuredManifestError(
                    f"{source}:{line}: {location} contains a non-finite value"
                )
            return number
        if not isinstance(value, list) or len(value) != dimensions[0]:
            raise MeasuredManifestError(f"{source}:{line}: {location} must have shape {shape}")
        return [
            validate(item, dimensions[1:], f"{location}[{index}]")
            for index, item in enumerate(value)
        ]

    checked = validate(decoded, shape, field)
    return json.dumps(checked, ensure_ascii=False, separators=(",", ":"))


def _side_annotations(feature: _IndexedRow | None) -> dict[str, str]:
    blank = {field: "" for field in SIDE_ANNOTATION_FIELDS}
    blank.update({"eye_annotation_valid": "false"})
    if feature is None:
        return blank
    row, source, line = feature.row, feature.source, feature.line
    visible_eye = str(row.get("visible_eye", "")).strip().lower()
    aliases = {"l": "left", "left": "left", "r": "right", "right": "right"}
    if visible_eye and visible_eye not in aliases:
        raise MeasuredManifestError(
            f"{source}:{line}: visible_eye must be left/right, got {visible_eye!r}"
        )
    values = {
        "visible_eye": aliases.get(visible_eye, ""),
        "visible_eye_bbox_xyxy": _compact_numeric_json(
            row.get("visible_eye_bbox_xyxy", ""),
            field="visible_eye_bbox_xyxy",
            shape=(4,),
            source=source,
            line=line,
        ),
        "visible_eye_keypoints_xy": _compact_numeric_json(
            row.get("visible_eye_keypoints_xy", ""),
            field="visible_eye_keypoints_xy",
            shape=(6, 2),
            source=source,
            line=line,
        ),
        "iris_center_xy": _compact_numeric_json(
            row.get("iris_center_xy", ""),
            field="iris_center_xy",
            shape=(2,),
            source=source,
            line=line,
        ),
        "profile_head_origin_xy": _compact_numeric_json(
            row.get("profile_head_origin_xy", ""),
            field="profile_head_origin_xy",
            shape=(2,),
            source=source,
            line=line,
        ),
        "profile_head_forward_xy": _compact_numeric_json(
            row.get("profile_head_forward_xy", ""),
            field="profile_head_forward_xy",
            shape=(2,),
            source=source,
            line=line,
        ),
    }
    # A Side image-only encoder needs only the visible-eye label and its bbox.
    # The six eyelid points, iris and head anchors are independent optional
    # features and must not invalidate an otherwise usable eye ROI.
    complete = all(values[field] for field in SIDE_ROI_FIELDS)
    declared = str(row.get("eye_annotation_valid", "")).strip().lower()
    if declared and declared not in {"true", "false", "1", "0", "yes", "no"}:
        raise MeasuredManifestError(
            f"{source}:{line}: eye_annotation_valid is not boolean: {declared!r}"
        )
    declared_valid = declared not in {"false", "0", "no"} if declared else True
    return {**values, "eye_annotation_valid": str(complete and declared_valid).lower()}


def _precomputed_side_roi_annotations() -> dict[str, str]:
    """Mark a complete external ROI image as usable without bbox/landmark geometry."""

    return {
        **{field: "" for field in SIDE_ANNOTATION_FIELDS},
        "eye_annotation_valid": "true",
    }


def _front_input(item: _IndexedRow) -> _FrontInput:
    """Validate one authoritative inputs.csv row using only its valid/pose contract."""

    row, source, line = item.row, item.source, item.line
    schema = _required(row, "schema_version", source=source, line=line)
    if schema != FRONT_INPUT_SCHEMA:
        raise MeasuredManifestError(
            f"{source}:{line}: unsupported schema_version {schema!r}; "
            f"expected {FRONT_INPUT_SCHEMA!r}"
        )
    raw_valid = _required(row, "valid", source=source, line=line)
    if raw_valid not in {"0", "1"}:
        raise MeasuredManifestError(f"{source}:{line}: valid must be exactly 0 or 1")

    pose_fields = (
        "head_vector_x",
        "head_vector_y",
        "head_vector_z",
        "face_origin_x_cm",
        "face_origin_y_cm",
        "face_origin_z_cm",
    )
    populated = [bool(str(row.get(key, "")).strip()) for key in pose_fields]
    if raw_valid == "0":
        _required(row, "invalid_reason", source=source, line=line)
        if any(populated) and not all(populated):
            raise MeasuredManifestError(
                f"{source}:{line}: invalid inputs.csv row has a partial 3D pose"
            )
        # A producer may retain a complete pose while rejecting an eye state.
        # Validate such values, but never emit or train on this pair.
        if all(populated):
            for key in pose_fields:
                _finite(row[key], field=key, source=source, line=line)
        return _FrontInput(
            usable=False,
            pose={"head_vector": "", "face_origin_3d": "", "head_pose_valid": "false"},
        )

    if not all(populated):
        raise MeasuredManifestError(f"{source}:{line}: valid=1 requires a complete 3D head pose")
    head = [
        _finite(row[key], field=key, source=source, line=line)
        for key in ("head_vector_x", "head_vector_y", "head_vector_z")
    ]
    if math.sqrt(sum(value * value for value in head)) <= 1e-8:
        raise MeasuredManifestError(f"{source}:{line}: head_vector has zero length")
    origin = [
        _finite(row[key], field=key, source=source, line=line)
        for key in ("face_origin_x_cm", "face_origin_y_cm", "face_origin_z_cm")
    ]
    return _FrontInput(
        usable=True,
        pose={
            "head_vector": json.dumps(head, separators=(",", ":")),
            # The authoritative header explicitly declares centimetres.
            "face_origin_3d": json.dumps(origin, separators=(",", ":")),
            "head_pose_valid": "true",
        },
    )


def _blank_front_pose() -> dict[str, str]:
    return {"head_vector": "", "face_origin_3d": "", "head_pose_valid": "false"}


def _validate_join_identity(
    item: _IndexedRow,
    *,
    subject: str,
    session: str,
    pair_id: str,
    participant_key: str,
) -> None:
    row = item.row
    participant = _nfc(row.get(participant_key, ""))
    if participant and participant != subject:
        raise MeasuredManifestError(
            f"{item.source}:{item.line}: participant {participant!r} does not match {subject!r}"
        )
    head_pose = str(row.get("head_pose", "")).strip()
    if head_pose and head_pose != session:
        raise MeasuredManifestError(
            f"{item.source}:{item.line}: head_pose {head_pose!r} does not match {session!r}"
        )
    joined_pair = _nfc(row.get("pair_id", ""))
    if joined_pair and joined_pair != pair_id:
        raise MeasuredManifestError(
            f"{item.source}:{item.line}: pair_id {joined_pair!r} does not match {pair_id!r}"
        )


def _validate_front_input_contract(
    item: _IndexedRow,
    *,
    image: Path,
    master_row: Mapping[str, str],
    width: int,
    height: int,
) -> None:
    """Cross-check inputs.csv identity, target and corrected source frame."""

    row, source, line = item.row, item.source, item.line
    source_image = _required(row, "source_image_path", source=source, line=line)
    source_name = _nfc(source_image.replace("\\", "/").rsplit("/", maxsplit=1)[-1])
    if source_name != _nfc(image.name):
        raise MeasuredManifestError(
            f"{source}:{line}: source_image_path filename does not match corrected frame "
            f"{image.name!r}"
        )

    for key in ("collection_split", "protocol"):
        actual = _required(row, key, source=source, line=line)
        expected = _required(master_row, key, source=source, line=line)
        if actual != expected:
            raise MeasuredManifestError(
                f"{source}:{line}: {key}={actual!r} does not match master value {expected!r}"
            )

    target_x_px = _finite(master_row["target_x_px"], field="target_x_px", source=source, line=line)
    target_y_px = _finite(master_row["target_y_px"], field="target_y_px", source=source, line=line)
    expected_centered = (target_x_px / width - 0.5, target_y_px / height - 0.5)
    declared_centered = (
        _finite(row["target_x_centered"], field="target_x_centered", source=source, line=line),
        _finite(row["target_y_centered"], field="target_y_centered", source=source, line=line),
    )
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
        for actual, expected in zip(declared_centered, expected_centered, strict=True)
    ):
        raise MeasuredManifestError(
            f"{source}:{line}: centered target does not match the master pixel target"
        )


def _session_rows(
    root: Path,
    subject_dir: Path,
    session: str,
    *,
    dataset_root: Path | None = None,
    side_roi_dir: Path | None = None,
    side_annotation_index: Mapping[str, _IndexedRow] | None = None,
) -> _SessionResult:
    manifest_root = dataset_root or root
    subject = _nfc(subject_dir.name)
    session_dir = subject_dir / session
    if not session_dir.is_dir():
        raise MeasuredManifestError(f"selected session directory is missing: {session_dir}")
    feature_root = session_dir / "feature_maps"

    camera_indexes: dict[str, dict[str, _IndexedRow]] = {}
    for view, directory in (("webcam", "web"), ("phonecam", "phone")):
        feature_csv = feature_root / directory / "features.csv"
        # An empty header-only phone table is a valid statement that no Side
        # annotations have been extracted yet.
        feature_rows = _read_table(feature_csv, frozenset({"sample_id"}))
        camera_indexes[view] = _index_rows(feature_rows, key="sample_id")

    input_csv = feature_root / "webeyetrack" / FRONT_INPUT_TABLE
    input_rows = _read_table(input_csv, FRONT_INPUT_REQUIRED_COLUMNS)
    input_index = _index_rows(input_rows, key="sample_id")
    side_roi_index = _index_side_roi_images(side_roi_dir) if side_roi_dir is not None else None

    output: list[dict[str, str | int]] = []
    seen_samples: set[str] = set()
    seen_front_inputs: set[str] = set()
    invalid_pairs: set[str] = set()
    missing_side_roi_pairs: set[str] = set()
    seen_side_roi_names: set[str] = set()
    for table in MASTER_TABLES:
        source = feature_root / table
        for item in _read_table(source, MASTER_REQUIRED_COLUMNS):
            row, line = item.row, item.line
            raw_subject = _nfc(_required(row, "subject_id", source=source, line=line))
            if raw_subject != subject:
                raise MeasuredManifestError(
                    f"{source}:{line}: subject_id={raw_subject!r} does not match directory "
                    f"{subject!r}"
                )
            raw_session = _required(row, "head_pose", source=source, line=line)
            if raw_session != session:
                raise MeasuredManifestError(
                    f"{source}:{line}: head_pose={raw_session!r} does not match {session!r}"
                )
            view = _required(row, "view", source=source, line=line).lower()
            if view not in {"webcam", "phonecam"}:
                raise MeasuredManifestError(
                    f"{source}:{line}: view must be webcam/phonecam, got {view!r}"
                )
            sample_id = _nfc(_required(row, "sample_id", source=source, line=line))
            pair_id = _nfc(_required(row, "pair_id", source=source, line=line))
            if sample_id in seen_samples:
                raise MeasuredManifestError(f"{source}:{line}: duplicate sample_id {sample_id!r}")
            seen_samples.add(sample_id)

            master_image = _resolve_unicode_relative(
                feature_root,
                _required(row, "image_path", source=source, line=line),
                source=source,
                line=line,
            )
            expected_frames = feature_root / ("web/frames" if view == "webcam" else "phone/frames")
            if not _is_within(master_image, expected_frames.resolve(strict=True)):
                raise MeasuredManifestError(
                    f"{source}:{line}: {view} image must come from {expected_frames}, "
                    f"got {master_image}"
                )
            image = master_image
            if view == "phonecam" and side_roi_index is not None:
                roi_name = _nfc(master_image.name)
                side_roi_image = side_roi_index.get(roi_name)
                if side_roi_image is None:
                    missing_side_roi_pairs.add(pair_id)
                else:
                    image = side_roi_image
                    seen_side_roi_names.add(roi_name)

            width = _positive_int(
                row["screen_width_px"], field="screen_width_px", source=source, line=line
            )
            height = _positive_int(
                row["screen_height_px"], field="screen_height_px", source=source, line=line
            )
            camera_feature = camera_indexes[view].get(sample_id)
            if camera_feature is not None:
                _validate_join_identity(
                    camera_feature,
                    subject=subject,
                    session=session,
                    pair_id=pair_id,
                    participant_key="participant",
                )
                feature_image_raw = str(camera_feature.row.get("image_path", "")).strip()
                if feature_image_raw:
                    feature_image = _resolve_unicode_relative(
                        feature_root,
                        feature_image_raw,
                        source=camera_feature.source,
                        line=camera_feature.line,
                    )
                    if feature_image != master_image:
                        raise MeasuredManifestError(
                            f"{camera_feature.source}:{camera_feature.line}: feature image does "
                            f"not match master row image for {sample_id!r}"
                        )

            input_item = input_index.get(sample_id) if view == "webcam" else None
            if view == "webcam":
                if input_item is None:
                    raise MeasuredManifestError(
                        f"{source}:{line}: authoritative inputs.csv row is missing for "
                        f"{sample_id!r}"
                    )
                _validate_join_identity(
                    input_item,
                    subject=subject,
                    session=session,
                    pair_id=pair_id,
                    participant_key="participant",
                )
                _validate_front_input_contract(
                    input_item,
                    image=master_image,
                    master_row=row,
                    width=width,
                    height=height,
                )
                parsed_input = _front_input(input_item)
                seen_front_inputs.add(sample_id)
                if not parsed_input.usable:
                    invalid_pairs.add(pair_id)
                pose = dict(parsed_input.pose)
            else:
                pose = _blank_front_pose()
            annotation_feature = camera_feature
            if view == "phonecam" and side_roi_index is None and side_annotation_index is not None:
                annotation_feature = side_annotation_index.get(sample_id, camera_feature)
                if annotation_feature is not None and annotation_feature is not camera_feature:
                    _validate_join_identity(
                        annotation_feature,
                        subject=subject,
                        session=session,
                        pair_id=pair_id,
                        participant_key="participant",
                    )
            if (
                view == "phonecam"
                and side_roi_index is not None
                and pair_id not in missing_side_roi_pairs
            ):
                annotations = _precomputed_side_roi_annotations()
            else:
                annotations = (
                    _side_annotations(annotation_feature)
                    if view == "phonecam"
                    else _side_annotations(None)
                )
            output.append(
                {
                    "sample_id": sample_id,
                    "subject_id": subject,
                    "session_id": session,
                    "view": view,
                    "image_path": _manifest_image_path(image, manifest_root),
                    "pair_id": pair_id,
                    "target_x_px": _target(
                        row["target_x_px"],
                        limit=width,
                        field="target_x_px",
                        source=source,
                        line=line,
                    ),
                    "target_y_px": _target(
                        row["target_y_px"],
                        limit=height,
                        field="target_y_px",
                        source=source,
                        line=line,
                    ),
                    "screen_width_px": width,
                    "screen_height_px": height,
                    **pose,
                    **annotations,
                    "collection_split": _required(
                        row, "collection_split", source=source, line=line
                    ),
                    "protocol": _required(row, "protocol", source=source, line=line),
                    "source_frame": _required(row, "source_frame", source=source, line=line),
                    "feature_csv_path": (
                        annotation_feature.source.relative_to(manifest_root).as_posix()
                        if annotation_feature is not None
                        and _is_within(annotation_feature.source, manifest_root)
                        else (
                            feature_root / ("web" if view == "webcam" else "phone") / "features.csv"
                        )
                        .relative_to(manifest_root)
                        .as_posix()
                    ),
                    "feature_csv_line": (
                        annotation_feature.line if annotation_feature is not None else ""
                    ),
                    "input_csv_path": (
                        input_item.source.relative_to(manifest_root).as_posix()
                        if input_item is not None
                        else ""
                    ),
                    "input_csv_line": input_item.line if input_item is not None else "",
                }
            )
    _validate_session_pairs(output, subject=subject, session=session)
    if side_roi_index is not None:
        unused_side_roi = sorted(set(side_roi_index) - seen_side_roi_names)
        if unused_side_roi:
            raise MeasuredManifestError(
                f"{side_roi_dir}: {len(unused_side_roi)} ROI images do not match a phone "
                f"master frame; examples={unused_side_roi[:3]}"
            )
    unexpected_inputs = sorted(set(input_index) - seen_front_inputs)
    if unexpected_inputs:
        raise MeasuredManifestError(
            f"{input_csv}: {len(unexpected_inputs)} inputs.csv rows do not match a Front "
            "master sample"
        )
    excluded_pairs = invalid_pairs | missing_side_roi_pairs
    filtered = tuple(row for row in output if str(row["pair_id"]) not in excluded_pairs)
    _validate_session_pairs(filtered, subject=subject, session=session)
    if any(row["head_pose_valid"] != "true" for row in filtered if row["view"] == "webcam"):
        raise MeasuredManifestError(
            f"{subject}/{session}: an emitted Front row lacks a valid inputs.csv 3D pose"
        )
    return _SessionResult(
        rows=filtered,
        source_pairs=len(output) // 2,
        dropped_invalid_pairs=len(invalid_pairs),
        dropped_missing_side_roi_pairs=len(missing_side_roi_pairs - invalid_pairs),
        front_input_rows=len(input_rows),
    )


def _validate_session_pairs(
    rows: Sequence[Mapping[str, str | int]], *, subject: str, session: str
) -> None:
    pairs: dict[str, list[Mapping[str, str | int]]] = {}
    for row in rows:
        pairs.setdefault(str(row["pair_id"]), []).append(row)
    for pair_id, pair_rows in pairs.items():
        views = {str(row["view"]) for row in pair_rows}
        if len(pair_rows) != 2 or views != {"webcam", "phonecam"}:
            raise MeasuredManifestError(
                f"{subject}/{session}: pair {pair_id!r} must contain one webcam and "
                "one phonecam row"
            )
        targets = {(str(row["target_x_px"]), str(row["target_y_px"])) for row in pair_rows}
        if len(targets) != 1:
            raise MeasuredManifestError(
                f"{subject}/{session}: pair {pair_id!r} has mismatched gaze targets"
            )


def _write_manifest(path: Path, rows: Iterable[Mapping[str, object]], *, force: bool) -> None:
    if path.exists() and not force:
        raise MeasuredManifestError(f"output already exists; pass --force to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=FIELDNAMES,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_manifest(
    source_root: str | Path,
    output_manifest: str | Path,
    *,
    dataset_root: str | Path | None = None,
    subjects: Sequence[str] | None = None,
    sessions: Sequence[str] = DEFAULT_SESSIONS,
    side_annotations: str | Path | None = None,
    side_roi_root: str | Path | None = None,
    require_side_annotations: bool = False,
    require_front_pose: bool = False,
    force: bool = False,
) -> ManifestSummary:
    """Build the measured-data manifest without modifying or copying source files."""

    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise MeasuredManifestError(f"source root is not a directory: {root}")
    output = Path(output_manifest).expanduser().resolve(strict=False)
    manifest_root = (
        Path(dataset_root).expanduser().resolve(strict=True) if dataset_root is not None else root
    )
    if not manifest_root.is_dir():
        raise MeasuredManifestError(f"dataset root is not a directory: {manifest_root}")
    if not _is_within(root, manifest_root):
        raise MeasuredManifestError(
            f"source root must be contained by dataset root {manifest_root}: {root}"
        )
    if _is_within(output, root):
        raise MeasuredManifestError("output manifest must be outside the read-only source root")
    if side_annotations is not None and side_roi_root is not None:
        raise MeasuredManifestError("side_annotations and side_roi_root are mutually exclusive")
    selected_sessions = tuple(_safe_session(session) for session in sessions)
    if not selected_sessions or len(set(selected_sessions)) != len(selected_sessions):
        raise MeasuredManifestError("--sessions must contain unique session names")
    subject_dirs = _discover_subjects(root, subjects)
    side_roi_base: Path | None = None
    side_roi_subjects: dict[str, Path] = {}
    if side_roi_root is not None:
        side_roi_base = Path(side_roi_root).expanduser().resolve(strict=True)
        if not side_roi_base.is_dir():
            raise MeasuredManifestError(f"Side ROI root is not a directory: {side_roi_base}")
        if not _is_within(side_roi_base, manifest_root):
            raise MeasuredManifestError(
                f"Side ROI root must be contained by dataset root {manifest_root}: {side_roi_base}"
            )
        for candidate in side_roi_base.iterdir():
            if not candidate.is_dir() or candidate.name.startswith("."):
                continue
            key = _nfc(candidate.name)
            if key in side_roi_subjects:
                raise MeasuredManifestError(
                    f"Side ROI subject directories collide after NFC normalization: {key!r}"
                )
            side_roi_subjects[key] = candidate
    side_annotation_index: dict[str, _IndexedRow] | None = None
    if side_annotations is not None:
        side_path = Path(side_annotations).expanduser().resolve(strict=True)
        side_rows = _read_table(
            side_path,
            frozenset(
                {
                    "sample_id",
                    "visible_eye",
                    "visible_eye_bbox_xyxy",
                    "eye_annotation_valid",
                }
            ),
        )
        side_annotation_index = _index_rows(side_rows, key="sample_id")

    rows: list[dict[str, str | int]] = []
    source_pairs = 0
    dropped_invalid_pairs = 0
    dropped_missing_side_roi_pairs = 0
    skipped_missing_side_roi_sessions = 0
    front_input_rows = 0
    included_subjects: set[str] = set()
    for subject_dir in subject_dirs:
        for session in selected_sessions:
            side_roi_dir: Path | None = None
            if side_roi_base is not None:
                roi_subject = side_roi_subjects.get(_nfc(subject_dir.name))
                roi_session = (
                    _unicode_child_directory(roi_subject, session)
                    if roi_subject is not None
                    else None
                )
                side_roi_dir = (
                    _unicode_child_directory(roi_session, "side_eye_roi")
                    if roi_session is not None
                    else None
                )
                if side_roi_dir is None or not _index_side_roi_images(side_roi_dir):
                    skipped_missing_side_roi_sessions += 1
                    continue
            session_result = _session_rows(
                root,
                subject_dir,
                session,
                dataset_root=manifest_root,
                side_roi_dir=side_roi_dir,
                side_annotation_index=side_annotation_index,
            )
            rows.extend(session_result.rows)
            included_subjects.add(_nfc(subject_dir.name))
            source_pairs += session_result.source_pairs
            dropped_invalid_pairs += session_result.dropped_invalid_pairs
            dropped_missing_side_roi_pairs += session_result.dropped_missing_side_roi_pairs
            front_input_rows += session_result.front_input_rows

    if not rows:
        raise MeasuredManifestError("no inputs.csv-approved dual-view pairs remain after filtering")

    front_rows = [row for row in rows if row["view"] == "webcam"]
    side_rows = [row for row in rows if row["view"] == "phonecam"]
    front_valid = sum(row["head_pose_valid"] == "true" for row in front_rows)
    side_valid = sum(row["eye_annotation_valid"] == "true" for row in side_rows)
    if require_front_pose and front_valid != len(front_rows):
        raise MeasuredManifestError(
            f"Front pose is missing for {len(front_rows) - front_valid}/{len(front_rows)} rows"
        )
    if require_side_annotations and side_valid != len(side_rows):
        raise MeasuredManifestError(
            f"Side annotations are missing for {len(side_rows) - side_valid}/{len(side_rows)} rows"
        )

    _write_manifest(output, rows, force=force)
    return ManifestSummary(
        path=output,
        subjects=tuple(sorted(included_subjects)),
        sessions=selected_sessions,
        front_input_rows=front_input_rows,
        source_pairs=source_pairs,
        output_pairs=len(rows) // 2,
        dropped_invalid_pairs=dropped_invalid_pairs,
        dropped_missing_side_roi_pairs=dropped_missing_side_roi_pairs,
        skipped_missing_side_roi_sessions=skipped_missing_side_roi_sessions,
        output_rows=len(rows),
        front_pose_valid_rows=front_valid,
        front_pose_missing_rows=len(front_rows) - front_valid,
        side_annotation_valid_rows=side_valid,
        side_annotation_missing_rows=len(side_rows) - side_valid,
    )


def main() -> int:
    args = _parser().parse_args()
    try:
        result = create_manifest(
            args.source_root,
            args.output_manifest,
            dataset_root=args.dataset_root,
            subjects=args.subjects,
            sessions=args.sessions,
            side_annotations=args.side_annotations,
            side_roi_root=args.side_roi_root,
            require_side_annotations=args.require_side_annotations,
            require_front_pose=args.require_front_pose,
            force=args.force,
        )
    except (MeasuredManifestError, OSError) as exc:
        raise SystemExit(f"manifest 생성 실패: {exc}") from exc
    print(f"manifest: {result.path}")
    print(f"subjects: {', '.join(result.subjects)}")
    print(f"sessions: {', '.join(result.sessions)}")
    print(f"inputs.csv rows: {result.front_input_rows}")
    print(f"inputs.csv-approved pairs kept: {result.output_pairs}")
    print(f"inputs.csv-rejected pairs dropped: {result.dropped_invalid_pairs}")
    print(f"missing Side ROI pairs dropped: {result.dropped_missing_side_roi_pairs}")
    print(
        "subject/session groups skipped for missing Side ROI data: "
        f"{result.skipped_missing_side_roi_sessions}"
    )
    print(f"source pairs: {result.source_pairs}, output rows: {result.output_rows}")
    print(
        f"front pose valid/missing: {result.front_pose_valid_rows}/{result.front_pose_missing_rows}"
    )
    print(
        "side annotation valid/missing: "
        f"{result.side_annotation_valid_rows}/{result.side_annotation_missing_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
