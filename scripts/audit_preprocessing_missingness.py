"""Audit preprocessing missingness for every measured-data participant.

The audit is intentionally read-only. It scans the selected participant/session
directories, reports structural and sample-level blockers, and writes CSV/JSON
artifacts that can be visualized without running MediaPipe or model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SESSIONS = ("head_down", "neutral")
MASTER_TABLES = ("training.csv", "evaluation.csv")
POSE_FIELDS = (
    "head_vector_x",
    "head_vector_y",
    "head_vector_z",
    "face_origin_x_cm",
    "face_origin_y_cm",
    "face_origin_z_cm",
)
MASTER_REQUIRED_FIELDS = (
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
)
INPUT_REQUIRED_FIELDS = (
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
    *POSE_FIELDS,
    "valid",
    "invalid_reason",
)
SIDE_ANNOTATION_FIELDS = (
    "sample_id",
    "visible_eye",
    "visible_eye_bbox_xyxy",
    "eye_annotation_valid",
)
DETAIL_FIELDS = (
    "category",
    "subject_id",
    "session_id",
    "sample_id",
    "pair_id",
    "view",
    "field",
    "source",
    "reason",
)
CORE_SUMMARY_CATEGORIES = (
    "missing_session",
    "missing_required_file",
    "unreadable_csv",
    "missing_required_column",
    "missing_master_value",
    "duplicate_sample_id",
    "invalid_pair",
    "missing_image",
    "missing_camera_feature_row",
    "invalid_camera_feature",
    "unusable_camera_feature",
    "missing_webeyetrack_input",
    "closed_eye_pair_excluded",
    "invalid_webeyetrack_input",
    "missing_front_pose",
    "missing_side_annotation",
    "invalid_side_annotation",
)


@dataclass(slots=True)
class FieldCounter:
    total_rows: int = 0
    missing_rows: int = 0
    affected_subjects: set[str] = field(default_factory=set)


@dataclass(slots=True)
class SubjectCounter:
    sessions_present: int = 0
    total_samples: int = 0
    front_samples: int = 0
    side_samples: int = 0
    total_pairs: int = 0
    excluded_closed_pairs: int = 0
    ready_pairs: int = 0


@dataclass(slots=True)
class PairState:
    front_rows: int = 0
    side_rows: int = 0
    front_image_ok: bool = False
    side_image_ok: bool = False
    front_input_ok: bool = False
    side_annotation_ok: bool = False
    excluded_by_input_valid_0: bool = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", default=list(DEFAULT_SESSIONS))
    parser.add_argument("--side-annotations", type=Path, default=None)
    return parser


def _nfc(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "").strip())


def _blank(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"", "na", "n/a", "nan", "none", "null"}


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(str(value).strip()))
    except (TypeError, ValueError):
        return False


def _issue(
    details: list[dict[str, str]],
    category: str,
    subject: str,
    session: str,
    *,
    sample_id: str = "",
    pair_id: str = "",
    view: str = "",
    field_name: str = "",
    source: Path | str = "",
    reason: str,
) -> None:
    details.append(
        {
            "category": category,
            "subject_id": subject,
            "session_id": session,
            "sample_id": sample_id,
            "pair_id": pair_id,
            "view": view,
            "field": field_name,
            "source": str(source),
            "reason": reason,
        }
    )


def _update_field_stats(
    field_stats: dict[tuple[str, str], FieldCounter],
    table_kind: str,
    rows: Sequence[Mapping[str, str]],
    headers: Sequence[str],
    subject: str,
) -> None:
    for header in headers:
        counter = field_stats[(table_kind, header)]
        counter.total_rows += len(rows)
        missing = sum(_blank(row.get(header, "")) for row in rows)
        counter.missing_rows += missing
        if missing:
            counter.affected_subjects.add(subject)


def _read_csv(
    path: Path,
    *,
    table_kind: str,
    required_fields: Iterable[str],
    subject: str,
    session: str,
    details: list[dict[str, str]],
    field_stats: dict[tuple[str, str], FieldCounter],
    required: bool = True,
) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    if not path.is_file():
        if required:
            _issue(
                details,
                "missing_required_file",
                subject,
                session,
                source=path,
                reason=f"required {table_kind} CSV is missing",
            )
        return [], ()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = tuple(str(name).strip() for name in reader.fieldnames or ())
            rows = [
                {
                    str(key).strip(): "" if value is None else str(value).strip()
                    for key, value in raw.items()
                    if key is not None
                }
                for raw in reader
            ]
    except (OSError, UnicodeError, csv.Error) as exc:
        _issue(
            details,
            "unreadable_csv",
            subject,
            session,
            source=path,
            reason=f"{type(exc).__name__}: {exc}",
        )
        return [], ()
    missing_headers = sorted(set(required_fields) - set(headers))
    for name in missing_headers:
        _issue(
            details,
            "missing_required_column",
            subject,
            session,
            field_name=name,
            source=path,
            reason=f"{table_kind} CSV has no {name!r} column",
        )
    _update_field_stats(field_stats, table_kind, rows, headers, subject)
    return rows, headers


def _index_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    subject: str,
    session: str,
    source: Path,
    details: list[dict[str, str]],
) -> dict[str, Mapping[str, str]]:
    result: dict[str, Mapping[str, str]] = {}
    for row in rows:
        sample_id = _nfc(row.get("sample_id", ""))
        if not sample_id:
            continue
        if sample_id in result:
            _issue(
                details,
                "duplicate_sample_id",
                subject,
                session,
                sample_id=sample_id,
                source=source,
                reason="sample_id occurs more than once in the same CSV role",
            )
            continue
        result[sample_id] = row
    return result


def _resolve_image(feature_root: Path, raw_value: str) -> Path | None:
    if _blank(raw_value):
        return None
    raw = Path(str(raw_value).strip())
    if raw.is_absolute() or ".." in raw.parts:
        return None
    candidate = feature_root.joinpath(*raw.parts)
    if candidate.is_file():
        return candidate
    # Handle NFC/NFD filename differences without guessing across directories.
    current = feature_root
    try:
        for part in raw.parts:
            direct = current / part
            if direct.exists():
                current = direct
                continue
            matches = [child for child in current.iterdir() if _nfc(child.name) == _nfc(part)]
            if len(matches) != 1:
                return None
            current = matches[0]
    except OSError:
        return None
    return current if current.is_file() else None


def _load_side_annotations(
    path: Path | None,
    *,
    details: list[dict[str, str]],
    field_stats: dict[tuple[str, str], FieldCounter],
) -> dict[str, Mapping[str, str]] | None:
    if path is None:
        return None
    rows, _ = _read_csv(
        path,
        table_kind="side_annotations",
        required_fields=SIDE_ANNOTATION_FIELDS,
        subject="<all>",
        session="<all>",
        details=details,
        field_stats=field_stats,
    )
    return _index_rows(
        rows,
        subject="<all>",
        session="<all>",
        source=path,
        details=details,
    )


def _side_annotation_ok(row: Mapping[str, str] | None) -> bool:
    if row is None:
        return False
    if _blank(row.get("visible_eye")) or _blank(row.get("visible_eye_bbox_xyxy")):
        return False
    declared = row.get("eye_annotation_valid")
    return True if _blank(declared) else _truthy(declared)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def audit(
    data_root: Path,
    output_dir: Path,
    *,
    sessions: Sequence[str] = DEFAULT_SESSIONS,
    side_annotations: Path | None = None,
) -> dict[str, Any]:
    root = data_root.expanduser().resolve(strict=True)
    output = output_dir.expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    details: list[dict[str, str]] = []
    field_stats: dict[tuple[str, str], FieldCounter] = defaultdict(FieldCounter)
    external_side_index = _load_side_annotations(
        side_annotations.expanduser().resolve(strict=False) if side_annotations else None,
        details=details,
        field_stats=field_stats,
    )
    subjects = sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: _nfc(path.name),
    )
    subject_counts = {_nfc(path.name): SubjectCounter() for path in subjects}
    input_eye_states: Counter[str] = Counter()

    for subject_dir in subjects:
        subject = _nfc(subject_dir.name)
        counter = subject_counts[subject]
        for session in sessions:
            session_dir = subject_dir / session
            if not session_dir.is_dir():
                _issue(
                    details,
                    "missing_session",
                    subject,
                    session,
                    source=session_dir,
                    reason="selected session directory is missing",
                )
                continue
            counter.sessions_present += 1
            feature_root = session_dir / "feature_maps"
            if not feature_root.is_dir():
                _issue(
                    details,
                    "missing_required_file",
                    subject,
                    session,
                    source=feature_root,
                    reason="feature_maps directory is missing",
                )
                continue

            master_rows: list[Mapping[str, str]] = []
            for filename in MASTER_TABLES:
                rows, _ = _read_csv(
                    feature_root / filename,
                    table_kind=filename.removesuffix(".csv"),
                    required_fields=MASTER_REQUIRED_FIELDS,
                    subject=subject,
                    session=session,
                    details=details,
                    field_stats=field_stats,
                )
                master_rows.extend(rows)

            web_path = feature_root / "web" / "features.csv"
            phone_path = feature_root / "phone" / "features.csv"
            input_path = feature_root / "webeyetrack" / "inputs.csv"
            web_rows, _ = _read_csv(
                web_path,
                table_kind="web_features",
                required_fields=("sample_id",),
                subject=subject,
                session=session,
                details=details,
                field_stats=field_stats,
            )
            phone_rows, _ = _read_csv(
                phone_path,
                table_kind="phone_features",
                required_fields=("sample_id",),
                subject=subject,
                session=session,
                details=details,
                field_stats=field_stats,
            )
            input_rows, _ = _read_csv(
                input_path,
                table_kind="webeyetrack_inputs",
                required_fields=INPUT_REQUIRED_FIELDS,
                subject=subject,
                session=session,
                details=details,
                field_stats=field_stats,
            )
            web_index = _index_rows(
                web_rows,
                subject=subject,
                session=session,
                source=web_path,
                details=details,
            )
            phone_index = _index_rows(
                phone_rows,
                subject=subject,
                session=session,
                source=phone_path,
                details=details,
            )
            input_index = _index_rows(
                input_rows,
                subject=subject,
                session=session,
                source=input_path,
                details=details,
            )
            closed_pair_ids = {
                _nfc(row.get("pair_id", ""))
                for row in input_rows
                if str(row.get("valid", "")).strip() == "0" and _nfc(row.get("pair_id", ""))
            }

            seen_samples: set[str] = set()
            pair_states: dict[str, PairState] = defaultdict(PairState)
            for row in master_rows:
                sample_id = _nfc(row.get("sample_id", ""))
                pair_id = _nfc(row.get("pair_id", ""))
                view = str(row.get("view", "")).strip().lower()
                counter.total_samples += 1
                if view == "webcam":
                    counter.front_samples += 1
                elif view == "phonecam":
                    counter.side_samples += 1
                if sample_id and sample_id in seen_samples:
                    _issue(
                        details,
                        "duplicate_sample_id",
                        subject,
                        session,
                        sample_id=sample_id,
                        pair_id=pair_id,
                        view=view,
                        reason="sample_id occurs in more than one master row",
                    )
                seen_samples.add(sample_id)
                for field_name in MASTER_REQUIRED_FIELDS:
                    if _blank(row.get(field_name, "")):
                        _issue(
                            details,
                            "missing_master_value",
                            subject,
                            session,
                            sample_id=sample_id,
                            pair_id=pair_id,
                            view=view,
                            field_name=field_name,
                            reason=f"master row has a blank {field_name!r} value",
                        )

                state = pair_states[pair_id]
                state.excluded_by_input_valid_0 = pair_id in closed_pair_ids
                image = _resolve_image(feature_root, row.get("image_path", ""))
                image_ok = image is not None and image.stat().st_size > 0
                if not image_ok:
                    _issue(
                        details,
                        "missing_image",
                        subject,
                        session,
                        sample_id=sample_id,
                        pair_id=pair_id,
                        view=view,
                        field_name="image_path",
                        source=feature_root / str(row.get("image_path", "")),
                        reason="referenced image is missing, unsafe, or empty",
                    )

                if view == "webcam":
                    state.front_rows += 1
                    state.front_image_ok = state.front_image_ok or image_ok
                    feature_row = web_index.get(sample_id)
                    input_row = input_index.get(sample_id)
                    if input_row is None:
                        _issue(
                            details,
                            "missing_webeyetrack_input",
                            subject,
                            session,
                            sample_id=sample_id,
                            pair_id=pair_id,
                            view=view,
                            source=input_path,
                            reason="Front master sample has no matching inputs.csv row",
                        )
                    else:
                        raw_valid = str(input_row.get("valid", "")).strip()
                        if raw_valid == "1":
                            input_eye_states["open_valid_1"] += 1
                        elif raw_valid == "0":
                            input_eye_states["closed_valid_0"] += 1
                        else:
                            input_eye_states["invalid_or_missing_value"] += 1
                    if input_row is not None and str(input_row.get("valid", "")).strip() == "0":
                        reason = (
                            str(input_row.get("invalid_reason", "")).strip() or "inputs.csv valid=0"
                        )
                        _issue(
                            details,
                            "closed_eye_pair_excluded",
                            subject,
                            session,
                            sample_id=sample_id,
                            pair_id=pair_id,
                            view=view,
                            field_name="valid",
                            source=input_path,
                            reason=reason,
                        )
                    elif input_row is not None and str(input_row.get("valid", "")).strip() != "1":
                        _issue(
                            details,
                            "invalid_webeyetrack_input",
                            subject,
                            session,
                            sample_id=sample_id,
                            pair_id=pair_id,
                            view=view,
                            field_name="valid",
                            source=input_path,
                            reason="valid must be exactly 0 or 1",
                        )
                    elif input_row is not None:
                        missing_pose = [
                            name for name in POSE_FIELDS if not _finite(input_row.get(name))
                        ]
                        if missing_pose:
                            _issue(
                                details,
                                "missing_front_pose",
                                subject,
                                session,
                                sample_id=sample_id,
                                pair_id=pair_id,
                                view=view,
                                field_name="|".join(missing_pose),
                                source=input_path,
                                reason="valid=1 row has missing or non-finite 3D pose values",
                            )
                        else:
                            head = [float(input_row[name]) for name in POSE_FIELDS[:3]]
                            head_norm = math.sqrt(sum(value * value for value in head))
                            state.front_input_ok = head_norm > 1e-8
                elif view == "phonecam":
                    state.side_rows += 1
                    state.side_image_ok = state.side_image_ok or image_ok
                    feature_row = phone_index.get(sample_id)
                    annotation_row = (
                        external_side_index.get(sample_id)
                        if external_side_index is not None
                        else feature_row
                    )
                    if state.excluded_by_input_valid_0:
                        # Closed-eye pairs are intentionally absent from training and Side QC.
                        state.side_annotation_ok = True
                    elif (
                        annotation_row is None
                        or _blank(annotation_row.get("visible_eye"))
                        or _blank(annotation_row.get("visible_eye_bbox_xyxy"))
                    ):
                        _issue(
                            details,
                            "missing_side_annotation",
                            subject,
                            session,
                            sample_id=sample_id,
                            pair_id=pair_id,
                            view=view,
                            field_name="visible_eye|visible_eye_bbox_xyxy",
                            source=side_annotations or phone_path,
                            reason="Side sample has no visible-eye label and bbox",
                        )
                    elif not _side_annotation_ok(annotation_row):
                        _issue(
                            details,
                            "invalid_side_annotation",
                            subject,
                            session,
                            sample_id=sample_id,
                            pair_id=pair_id,
                            view=view,
                            field_name="eye_annotation_valid",
                            source=side_annotations or phone_path,
                            reason="Side eye annotation is explicitly invalid",
                        )
                    else:
                        state.side_annotation_ok = True
                else:
                    feature_row = None

                if view in {"webcam", "phonecam"}:
                    feature_path = web_path if view == "webcam" else phone_path
                    if feature_row is None:
                        _issue(
                            details,
                            "missing_camera_feature_row",
                            subject,
                            session,
                            sample_id=sample_id,
                            pair_id=pair_id,
                            view=view,
                            source=feature_path,
                            reason="master sample has no matching camera features.csv row",
                        )
                    else:
                        if not _blank(feature_row.get("feature_valid")) and not _truthy(
                            feature_row.get("feature_valid")
                        ):
                            _issue(
                                details,
                                "invalid_camera_feature",
                                subject,
                                session,
                                sample_id=sample_id,
                                pair_id=pair_id,
                                view=view,
                                field_name="feature_valid",
                                source=feature_path,
                                reason=str(feature_row.get("invalid_reason", "")).strip()
                                or "feature_valid is false",
                            )
                        if not _blank(feature_row.get("usable")) and not _truthy(
                            feature_row.get("usable")
                        ):
                            _issue(
                                details,
                                "unusable_camera_feature",
                                subject,
                                session,
                                sample_id=sample_id,
                                pair_id=pair_id,
                                view=view,
                                field_name="usable",
                                source=feature_path,
                                reason=str(feature_row.get("invalid_reason", "")).strip()
                                or "usable is false",
                            )

            for pair_id, state in pair_states.items():
                counter.total_pairs += 1
                complete = state.front_rows == 1 and state.side_rows == 1
                if not complete:
                    _issue(
                        details,
                        "invalid_pair",
                        subject,
                        session,
                        pair_id=pair_id,
                        reason=(
                            f"expected one webcam and one phonecam row; got "
                            f"front={state.front_rows}, side={state.side_rows}"
                        ),
                    )
                excluded = complete and state.excluded_by_input_valid_0
                counter.excluded_closed_pairs += int(excluded)
                ready = (
                    complete
                    and not excluded
                    and state.front_image_ok
                    and state.side_image_ok
                    and state.front_input_ok
                    and state.side_annotation_ok
                )
                counter.ready_pairs += int(ready)

    category_cases = Counter(row["category"] for row in details)
    category_subjects: dict[str, set[str]] = defaultdict(set)
    subject_issue_counts: dict[str, Counter[str]] = defaultdict(Counter)
    affected_samples: dict[str, set[str]] = defaultdict(set)
    for row in details:
        subject = row["subject_id"]
        if subject in subject_counts:
            category_subjects[row["category"]].add(subject)
            subject_issue_counts[subject][row["category"]] += 1
            if row["sample_id"]:
                affected_samples[subject].add(row["sample_id"])

    all_categories = sorted(set(CORE_SUMMARY_CATEGORIES) | set(category_cases))
    subject_rows: list[dict[str, Any]] = []
    for subject, counts in subject_counts.items():
        issues = subject_issue_counts[subject]
        row: dict[str, Any] = {
            "subject_id": subject,
            "sessions_present": counts.sessions_present,
            "sessions_expected": len(sessions),
            "total_samples": counts.total_samples,
            "front_samples": counts.front_samples,
            "side_samples": counts.side_samples,
            "total_pairs": counts.total_pairs,
            "excluded_closed_pairs": counts.excluded_closed_pairs,
            "ready_pairs": counts.ready_pairs,
            "affected_samples": len(affected_samples[subject]),
            "issue_rows": sum(issues.values()),
        }
        row.update({category: issues.get(category, 0) for category in all_categories})
        subject_rows.append(row)

    category_rows = [
        {
            "category": category,
            "cases": category_cases[category],
            "affected_subjects": len(category_subjects.get(category, set())),
        }
        for category in all_categories
        if category_cases[category]
    ]
    field_rows = [
        {
            "table": table,
            "column": column,
            "total_rows": counter.total_rows,
            "missing_rows": counter.missing_rows,
            "missing_rate": (
                round(counter.missing_rows / counter.total_rows, 6) if counter.total_rows else 0.0
            ),
            "affected_subjects": len(counter.affected_subjects),
        }
        for (table, column), counter in sorted(field_stats.items())
        if counter.missing_rows
    ]
    reason_counter: Counter[tuple[str, str]] = Counter()
    reason_subjects: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in details:
        if row["category"] in {
            "closed_eye_pair_excluded",
            "invalid_webeyetrack_input",
            "invalid_camera_feature",
            "unusable_camera_feature",
        }:
            key = (row["category"], row["reason"])
            reason_counter[key] += 1
            if row["subject_id"] in subject_counts:
                reason_subjects[key].add(row["subject_id"])
    reason_rows = [
        {
            "category": category,
            "reason": reason,
            "cases": count,
            "affected_subjects": len(reason_subjects[(category, reason)]),
        }
        for (category, reason), count in reason_counter.most_common()
    ]

    _write_csv(output / "missingness_details.csv", details, DETAIL_FIELDS)
    subject_fields = (
        "subject_id",
        "sessions_present",
        "sessions_expected",
        "total_samples",
        "front_samples",
        "side_samples",
        "total_pairs",
        "excluded_closed_pairs",
        "ready_pairs",
        "affected_samples",
        "issue_rows",
        *all_categories,
    )
    _write_csv(output / "missingness_by_subject.csv", subject_rows, subject_fields)
    _write_csv(
        output / "missingness_by_category.csv",
        category_rows,
        ("category", "cases", "affected_subjects"),
    )
    _write_csv(
        output / "field_missingness.csv",
        field_rows,
        ("table", "column", "total_rows", "missing_rows", "missing_rate", "affected_subjects"),
    )
    _write_csv(
        output / "invalid_reason_summary.csv",
        reason_rows,
        ("category", "reason", "cases", "affected_subjects"),
    )

    total_pairs = sum(item.total_pairs for item in subject_counts.values())
    excluded_closed_pairs = sum(item.excluded_closed_pairs for item in subject_counts.values())
    eligible_pairs = total_pairs - excluded_closed_pairs
    ready_pairs = sum(item.ready_pairs for item in subject_counts.values())
    affected_people = {
        subject
        for subjects_for_category in category_subjects.values()
        for subject in subjects_for_category
    }
    report = {
        "data_root": str(root),
        "sessions": list(sessions),
        "side_annotations": str(side_annotations) if side_annotations else None,
        "overview": {
            "participants": len(subjects),
            "affected_participants": len(affected_people),
            "total_samples": sum(item.total_samples for item in subject_counts.values()),
            "total_pairs": total_pairs,
            "eligible_pairs": eligible_pairs,
            "excluded_closed_pairs": excluded_closed_pairs,
            "ready_pairs": ready_pairs,
            "blocked_pairs": eligible_pairs - ready_pairs,
            "ready_pair_rate": round(ready_pairs / eligible_pairs, 6) if eligible_pairs else 0.0,
            "issue_rows": len(details),
        },
        "input_eye_state": {
            "rule": "inputs.csv valid=1 -> open/use; valid=0 -> closed/exclude pair",
            **dict(input_eye_states),
        },
        "categories": category_rows,
        "subjects": subject_rows,
        "top_invalid_reasons": reason_rows[:30],
        "outputs": {
            "details": "missingness_details.csv",
            "by_subject": "missingness_by_subject.csv",
            "by_category": "missingness_by_category.csv",
            "field_missingness": "field_missingness.csv",
            "invalid_reasons": "invalid_reason_summary.csv",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    args = _parser().parse_args()
    report = audit(
        args.data_root,
        args.output_dir,
        sessions=tuple(args.sessions),
        side_annotations=args.side_annotations,
    )
    overview = report["overview"]
    print(json.dumps(overview, ensure_ascii=False, indent=2))
    print(f"report: {Path(args.output_dir).resolve() / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
