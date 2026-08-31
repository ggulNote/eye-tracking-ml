"""Build a strict paired manifest from the self-contained ``process_data`` tree.

Only ``inputs.csv`` rows with ``valid=1`` and complete, readable Front/Side ROI
paths are eligible. ``valid=0`` rows and incomplete sessions are written to a
separate rejection report and never enter training.

The source CSV contains centered normalized targets but no physical display
calibration. The manifest therefore uses a large virtual pixel plane that
round-trips those normalized coordinates exactly enough for training. The
``process_data`` profile deliberately reports normalized metrics only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
VIRTUAL_SCREEN_SIZE = 1_000_000
SIDE_FRAME_RE = re.compile(r"^(?P<pair_id>.+)_phone_frame_\d+$")

INPUT_REQUIRED_COLUMNS = frozenset(
    {
        "sample_id",
        "participant",
        "head_pose",
        "pair_id",
        "eye_patch_path",
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
    "input_csv_path",
    "input_csv_line",
    "target_x_centered",
    "target_y_centered",
    "input_valid",
)

REJECTION_FIELDNAMES = (
    "subject_id",
    "session_id",
    "pair_id",
    "input_csv_path",
    "input_csv_line",
    "input_valid",
    "invalid_reason",
    "exclusion_reason",
    "front_roi_path",
    "side_roi_path",
    "source_image_path",
)


class ProcessDataManifestError(ValueError):
    """Raised for an ambiguous or structurally unsafe ``process_data`` tree."""


@dataclass(frozen=True, slots=True)
class ProcessPair:
    subject: str
    session: str
    pair_id: str
    sample_id: str
    csv_path: Path
    csv_line: int
    valid: bool
    invalid_reason: str
    front_path: Path | None
    side_path: Path | None
    source_image_path: str
    target_xy_centered: tuple[float, float] | None
    head_vector: tuple[float, float, float] | None
    face_origin_cm: tuple[float, float, float] | None
    collection_split: str
    protocol: str
    issues: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return self.valid and not self.issues

    @property
    def exclusion_reason(self) -> str:
        if not self.valid:
            reason = self.invalid_reason or "unspecified"
            return f"inputs_csv_valid_0:{reason}"
        return ";".join(self.issues)


@dataclass(frozen=True, slots=True)
class MissingSession:
    subject: str
    session: str
    reason: str
    session_path: Path


@dataclass(frozen=True, slots=True)
class ProcessInventory:
    root: Path
    subjects: tuple[str, ...]
    pairs: tuple[ProcessPair, ...]
    missing_sessions: tuple[MissingSession, ...]
    orphan_front_images: tuple[Path, ...]
    orphan_side_images: tuple[Path, ...]

    @property
    def usable_pairs(self) -> tuple[ProcessPair, ...]:
        return tuple(pair for pair in self.pairs if pair.usable)


@dataclass(frozen=True, slots=True)
class ManifestSummary:
    path: Path
    rejection_path: Path
    subjects_found: int
    subjects_used: int
    sessions_with_inputs: int
    missing_sessions: int
    input_rows: int
    upstream_invalid_pairs: int
    valid_incomplete_pairs: int
    quality_excluded_pairs: int
    output_pairs: int
    output_rows: int
    orphan_front_images: int
    orphan_side_images: int


def _nfc(value: object) -> str:
    return unicodedata.normalize("NFC", str(value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-data-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--rejection-csv", type=Path)
    parser.add_argument(
        "--quality-exclusions",
        type=Path,
        help="CSV with subject_id, session_id, exclusion_reason from the quality audit",
    )
    parser.add_argument("--sessions", nargs="+", default=["head_down", "neutral"])
    parser.add_argument("--force", action="store_true")
    return parser


def _image_index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: _nfc(item.name)):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = _nfc(path.name)
        if key in result:
            raise ProcessDataManifestError(
                f"Unicode-normalized duplicate image name in {directory}: {key}"
            )
        result[key] = path
    return result


def _side_index(directory: Path) -> tuple[dict[str, Path], tuple[Path, ...]]:
    result: dict[str, Path] = {}
    unkeyed: list[Path] = []
    for path in _image_index(directory).values():
        stem = _nfc(path.stem)
        match = SIDE_FRAME_RE.fullmatch(stem)
        pair_id = match.group("pair_id") if match else stem
        if pair_id in result:
            raise ProcessDataManifestError(
                f"multiple Side ROI images map to pair_id={pair_id!r}: {result[pair_id]} and {path}"
            )
        if not pair_id:
            unkeyed.append(path)
            continue
        result[pair_id] = path
    return result, tuple(unkeyed)


def _required_columns(reader: csv.DictReader[str], path: Path) -> None:
    fields = set(reader.fieldnames or ())
    missing = sorted(INPUT_REQUIRED_COLUMNS - fields)
    if missing:
        raise ProcessDataManifestError(f"{path} is missing required columns: {missing}")


def _finite(value: object, *, field: str, source: Path, line: int) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ProcessDataManifestError(f"{source}:{line}: {field} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ProcessDataManifestError(f"{source}:{line}: {field} must be finite")
    return parsed


def _optional_vector(
    row: Mapping[str, str],
    fields: Sequence[str],
    *,
    source: Path,
    line: int,
) -> tuple[tuple[float, float, float] | None, str | None]:
    raw = [str(row.get(field, "")).strip() for field in fields]
    if not any(raw):
        return None, f"missing_{fields[0].rsplit('_', 1)[0]}"
    if not all(raw):
        return None, f"partial_{fields[0].rsplit('_', 1)[0]}"
    try:
        values = tuple(
            _finite(row[field], field=field, source=source, line=line) for field in fields
        )
    except ProcessDataManifestError:
        return None, f"invalid_{fields[0].rsplit('_', 1)[0]}"
    return (values[0], values[1], values[2]), None


def _optional_target(
    row: Mapping[str, str], *, source: Path, line: int
) -> tuple[tuple[float, float] | None, str | None]:
    try:
        x = _finite(
            row.get("target_x_centered", ""), field="target_x_centered", source=source, line=line
        )
        y = _finite(
            row.get("target_y_centered", ""), field="target_y_centered", source=source, line=line
        )
    except ProcessDataManifestError:
        return None, "invalid_target_xy_centered"
    if not (-0.5 <= x < 0.5 and -0.5 <= y < 0.5):
        return None, "target_xy_centered_out_of_bounds"
    return (x, y), None


def _read_session(
    root: Path,
    subject_dir: Path,
    session: str,
) -> tuple[list[ProcessPair], MissingSession | None, set[Path], set[Path]]:
    subject = _nfc(subject_dir.name)
    session_dir = subject_dir / session
    csv_path = session_dir / "inputs.csv"
    front_index = _image_index(session_dir / "eye_roi")
    side_index, unkeyed_side = _side_index(session_dir / "side_eye_roi")
    if not csv_path.is_file():
        return (
            [],
            MissingSession(subject, session, "missing_inputs_csv", session_dir),
            set(front_index.values()),
            set(side_index.values()) | set(unkeyed_side),
        )

    pairs: list[ProcessPair] = []
    used_front: set[Path] = set()
    used_side: set[Path] = set()
    pair_ids: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _required_columns(reader, csv_path)
        for line, row in enumerate(reader, start=2):
            pair_id = _nfc(row.get("pair_id", "")).strip()
            if not pair_id:
                raise ProcessDataManifestError(f"{csv_path}:{line}: pair_id is blank")
            if pair_id in pair_ids:
                raise ProcessDataManifestError(f"{csv_path}:{line}: duplicate pair_id={pair_id!r}")
            pair_ids.add(pair_id)
            row_subject = _nfc(row.get("participant", "")).strip()
            row_session = str(row.get("head_pose", "")).strip()
            if row_subject != subject or row_session != session:
                raise ProcessDataManifestError(
                    f"{csv_path}:{line}: participant/head_pose does not match its directory"
                )
            valid_text = str(row.get("valid", "")).strip()
            if valid_text not in {"0", "1"}:
                raise ProcessDataManifestError(f"{csv_path}:{line}: valid must be exactly 0 or 1")
            valid = valid_text == "1"
            front_name = _nfc(Path(str(row.get("eye_patch_path", ""))).name)
            front_path = front_index.get(front_name)
            side_path = side_index.get(pair_id)
            if front_path is not None:
                used_front.add(front_path)
            if side_path is not None:
                used_side.add(side_path)

            issues: list[str] = []
            target, target_issue = _optional_target(row, source=csv_path, line=line)
            head, head_issue = _optional_vector(
                row,
                ("head_vector_x", "head_vector_y", "head_vector_z"),
                source=csv_path,
                line=line,
            )
            origin, origin_issue = _optional_vector(
                row,
                ("face_origin_x_cm", "face_origin_y_cm", "face_origin_z_cm"),
                source=csv_path,
                line=line,
            )
            if valid:
                if front_path is None:
                    issues.append("missing_front_eye_roi")
                if side_path is None:
                    issues.append("missing_side_eye_roi")
                if target_issue:
                    issues.append(target_issue)
                if head_issue:
                    issues.append(head_issue)
                elif head is not None and math.sqrt(sum(value * value for value in head)) <= 1e-8:
                    issues.append("zero_length_head_vector")
                if origin_issue:
                    issues.append(origin_issue)

            pairs.append(
                ProcessPair(
                    subject=subject,
                    session=session,
                    pair_id=pair_id,
                    sample_id=_nfc(row.get("sample_id", "")).strip(),
                    csv_path=csv_path,
                    csv_line=line,
                    valid=valid,
                    invalid_reason=str(row.get("invalid_reason", "")).strip(),
                    front_path=front_path,
                    side_path=side_path,
                    source_image_path=str(row.get("source_image_path", "")).strip(),
                    target_xy_centered=target,
                    head_vector=head,
                    face_origin_cm=origin,
                    collection_split=str(row.get("collection_split", "")).strip(),
                    protocol=str(row.get("protocol", "")).strip(),
                    issues=tuple(issues),
                )
            )
    return (
        pairs,
        None,
        set(front_index.values()) - used_front,
        (set(side_index.values()) | set(unkeyed_side)) - used_side,
    )


def scan_process_data(
    process_data_root: Path,
    *,
    sessions: Iterable[str] = ("head_down", "neutral"),
) -> ProcessInventory:
    root = process_data_root.expanduser().resolve(strict=True)
    selected_sessions = tuple(str(session).strip() for session in sessions if str(session).strip())
    if not selected_sessions:
        raise ProcessDataManifestError("at least one session must be selected")
    subjects = sorted(
        (path for path in root.iterdir() if path.is_dir()), key=lambda path: _nfc(path.name)
    )
    pairs: list[ProcessPair] = []
    missing_sessions: list[MissingSession] = []
    orphan_front: set[Path] = set()
    orphan_side: set[Path] = set()
    for subject_dir in subjects:
        for session in selected_sessions:
            session_pairs, missing, front_orphans, side_orphans = _read_session(
                root, subject_dir, session
            )
            pairs.extend(session_pairs)
            if missing is not None:
                missing_sessions.append(missing)
            orphan_front.update(front_orphans)
            orphan_side.update(side_orphans)
    return ProcessInventory(
        root=root,
        subjects=tuple(_nfc(path.name) for path in subjects),
        pairs=tuple(pairs),
        missing_sessions=tuple(missing_sessions),
        orphan_front_images=tuple(sorted(orphan_front, key=str)),
        orphan_side_images=tuple(sorted(orphan_side, key=str)),
    )


def _relative(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    return path.resolve(strict=False).relative_to(root).as_posix()


def _read_quality_exclusions(path: Path | None) -> dict[tuple[str, str], str]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve(strict=True)
    result: dict[tuple[str, str], str] = {}
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "session_id", "exclusion_reason"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ProcessDataManifestError(
                f"{resolved} is missing quality exclusion columns: {sorted(missing)}"
            )
        for line, row in enumerate(reader, start=2):
            subject = _nfc(row.get("subject_id", "")).strip()
            session = str(row.get("session_id", "")).strip()
            reason = str(row.get("exclusion_reason", "")).strip()
            if not subject or not session or not reason:
                raise ProcessDataManifestError(
                    f"{resolved}:{line}: quality exclusion cells must not be blank"
                )
            key = (subject, session)
            if key in result:
                raise ProcessDataManifestError(
                    f"{resolved}:{line}: duplicate quality exclusion {key}"
                )
            result[key] = reason
    return result


def _selected_pairs(
    inventory: ProcessInventory,
    exclusions: Mapping[tuple[str, str], str],
) -> tuple[ProcessPair, ...]:
    return tuple(
        pair for pair in inventory.usable_pairs if (pair.subject, pair.session) not in exclusions
    )


def _manifest_rows(
    inventory: ProcessInventory,
    exclusions: Mapping[tuple[str, str], str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in _selected_pairs(inventory, exclusions):
        assert pair.front_path is not None
        assert pair.side_path is not None
        assert pair.target_xy_centered is not None
        assert pair.head_vector is not None
        assert pair.face_origin_cm is not None
        x, y = pair.target_xy_centered
        common = {
            "subject_id": pair.subject,
            "session_id": pair.session,
            "pair_id": pair.pair_id,
            "target_x_px": f"{(x + 0.5) * VIRTUAL_SCREEN_SIZE:.8f}",
            "target_y_px": f"{(y + 0.5) * VIRTUAL_SCREEN_SIZE:.8f}",
            "screen_width_px": str(VIRTUAL_SCREEN_SIZE),
            "screen_height_px": str(VIRTUAL_SCREEN_SIZE),
            "visible_eye": "",
            "visible_eye_bbox_xyxy": "",
            "visible_eye_keypoints_xy": "",
            "iris_center_xy": "",
            "profile_head_origin_xy": "",
            "profile_head_forward_xy": "",
            "eye_annotation_valid": "true",
            "collection_split": pair.collection_split,
            "protocol": pair.protocol,
            "source_frame": "",
            "input_csv_path": _relative(pair.csv_path, inventory.root),
            "input_csv_line": str(pair.csv_line),
            "target_x_centered": f"{x:.8f}",
            "target_y_centered": f"{y:.8f}",
            "input_valid": "1",
        }
        rows.append(
            {
                **common,
                "sample_id": pair.sample_id or f"{pair.pair_id}_webcam",
                "view": "front",
                "image_path": _relative(pair.front_path, inventory.root),
                "head_vector": json.dumps(pair.head_vector, separators=(",", ":")),
                "face_origin_3d": json.dumps(pair.face_origin_cm, separators=(",", ":")),
                "head_pose_valid": "true",
            }
        )
        rows.append(
            {
                **common,
                "sample_id": f"{pair.pair_id}_phonecam",
                "view": "side",
                "image_path": _relative(pair.side_path, inventory.root),
                "head_vector": "",
                "face_origin_3d": "",
                "head_pose_valid": "false",
            }
        )
    return rows


def _rejection_rows(
    inventory: ProcessInventory,
    exclusions: Mapping[tuple[str, str], str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for missing in inventory.missing_sessions:
        rows.append(
            {
                "subject_id": missing.subject,
                "session_id": missing.session,
                "pair_id": "",
                "input_csv_path": "",
                "input_csv_line": "",
                "input_valid": "",
                "invalid_reason": "",
                "exclusion_reason": missing.reason,
                "front_roi_path": "",
                "side_roi_path": "",
                "source_image_path": "",
            }
        )
    for pair in inventory.pairs:
        quality_reason = exclusions.get((pair.subject, pair.session))
        if pair.usable and quality_reason is None:
            continue
        exclusion_reason = (
            f"quality_excluded_session:{quality_reason}"
            if pair.usable and quality_reason is not None
            else pair.exclusion_reason
        )
        rows.append(
            {
                "subject_id": pair.subject,
                "session_id": pair.session,
                "pair_id": pair.pair_id,
                "input_csv_path": _relative(pair.csv_path, inventory.root),
                "input_csv_line": str(pair.csv_line),
                "input_valid": "1" if pair.valid else "0",
                "invalid_reason": pair.invalid_reason,
                "exclusion_reason": exclusion_reason,
                "front_roi_path": _relative(pair.front_path, inventory.root),
                "side_roi_path": _relative(pair.side_path, inventory.root),
                "source_image_path": pair.source_image_path,
            }
        )
    return rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def create_manifest(
    process_data_root: Path,
    output_manifest: Path,
    *,
    rejection_csv: Path | None = None,
    quality_exclusions: Path | None = None,
    sessions: Iterable[str] = ("head_down", "neutral"),
    force: bool = False,
) -> ManifestSummary:
    output = output_manifest.expanduser().resolve(strict=False)
    rejection = (
        rejection_csv.expanduser().resolve(strict=False)
        if rejection_csv is not None
        else output.with_name(f"{output.stem}_rejections.csv")
    )
    for path in (output, rejection):
        if path.exists() and not force:
            raise ProcessDataManifestError(f"output already exists; pass --force: {path}")
    inventory = scan_process_data(process_data_root, sessions=sessions)
    exclusions = _read_quality_exclusions(quality_exclusions)
    unknown_exclusions = sorted(
        set(exclusions) - {(pair.subject, pair.session) for pair in inventory.pairs}
    )
    if unknown_exclusions:
        raise ProcessDataManifestError(
            f"quality exclusions do not match inputs.csv sessions: {unknown_exclusions}"
        )
    selected_pairs = _selected_pairs(inventory, exclusions)
    manifest_rows = _manifest_rows(inventory, exclusions)
    rejection_rows = _rejection_rows(inventory, exclusions)
    _write_csv(output, FIELDNAMES, manifest_rows)
    _write_csv(rejection, REJECTION_FIELDNAMES, rejection_rows)
    usable_subjects = {pair.subject for pair in selected_pairs}
    summary = ManifestSummary(
        path=output,
        rejection_path=rejection,
        subjects_found=len(inventory.subjects),
        subjects_used=len(usable_subjects),
        sessions_with_inputs=len({(pair.subject, pair.session) for pair in inventory.pairs}),
        missing_sessions=len(inventory.missing_sessions),
        input_rows=len(inventory.pairs),
        upstream_invalid_pairs=sum(not pair.valid for pair in inventory.pairs),
        valid_incomplete_pairs=sum(pair.valid and bool(pair.issues) for pair in inventory.pairs),
        quality_excluded_pairs=sum(
            pair.usable and (pair.subject, pair.session) in exclusions for pair in inventory.pairs
        ),
        output_pairs=len(selected_pairs),
        output_rows=len(manifest_rows),
        orphan_front_images=len(inventory.orphan_front_images),
        orphan_side_images=len(inventory.orphan_side_images),
    )
    return summary


def main() -> None:
    args = _parser().parse_args()
    summary = create_manifest(
        args.process_data_root,
        args.output_manifest,
        rejection_csv=args.rejection_csv,
        quality_exclusions=args.quality_exclusions,
        sessions=args.sessions,
        force=args.force,
    )
    print(f"manifest: {summary.path}")
    print(f"rejections: {summary.rejection_path}")
    print(f"subjects found/used: {summary.subjects_found}/{summary.subjects_used}")
    print(
        f"sessions with inputs/missing: {summary.sessions_with_inputs}/{summary.missing_sessions}"
    )
    print(f"inputs.csv rows: {summary.input_rows}")
    print(f"valid=0 excluded: {summary.upstream_invalid_pairs}")
    print(f"valid=1 incomplete excluded: {summary.valid_incomplete_pairs}")
    print(f"quality-excluded pairs: {summary.quality_excluded_pairs}")
    print(f"usable pairs/output rows: {summary.output_pairs}/{summary.output_rows}")
    print(f"orphan Front/Side images: {summary.orphan_front_images}/{summary.orphan_side_images}")


if __name__ == "__main__":
    main()
