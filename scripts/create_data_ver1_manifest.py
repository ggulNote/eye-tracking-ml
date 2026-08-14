"""Build a generic dual-view CSV manifest from the read-only data(ver1) DB."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

SCREEN_SIZE = (1920, 1080)
DEFAULT_SUBJECTS = ("p00", "p03")
SOURCE_COLUMNS = frozenset(
    {
        "sample",
        "participant",
        "pair",
        "webcam_image",
        "phonecam_image",
        "x_px",
        "y_px",
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
    "visible_eye",
    "visible_eye_bbox_xyxy",
    "visible_eye_keypoints_xy",
    "iris_center_xy",
    "profile_head_origin_xy",
    "profile_head_forward_xy",
    "eye_annotation_valid",
    "annotation_source",
)


class ManifestBuildError(ValueError):
    """Raised when data(ver1) cannot be converted without guessing."""


@dataclass(frozen=True, slots=True)
class ManifestSummary:
    """Small result object used by the CLI and tests."""

    path: Path
    subjects: tuple[str, ...]
    source_pairs: int
    output_rows: int
    dummy_side_annotations: bool


# Coordinates are normalized against the source image, then scaled per JPEG.
# They are intentionally marked as dummy smoke data and must not be used as labels.
_DUMMY_SIDE_TEMPLATES: Mapping[str, Mapping[str, Any]] = {
    "p00": {
        "visible_eye": "left",
        "bbox": (630 / 1280, 270 / 720, 790 / 1280, 350 / 720),
        "keypoints": (
            (650 / 1280, 310 / 720),
            (670 / 1280, 296 / 720),
            (710 / 1280, 294 / 720),
            (770 / 1280, 310 / 720),
            (710 / 1280, 326 / 720),
            (670 / 1280, 324 / 720),
        ),
        "iris": (710 / 1280, 310 / 720),
        "head_origin": (1050 / 1280, 340 / 720),
        "head_forward": (690 / 1280, 395 / 720),
    },
    "p03": {
        "visible_eye": "left",
        "bbox": (400 / 1280, 260 / 720, 560 / 1280, 340 / 720),
        "keypoints": (
            (420 / 1280, 300 / 720),
            (440 / 1280, 286 / 720),
            (480 / 1280, 284 / 720),
            (540 / 1280, 300 / 720),
            (480 / 1280, 316 / 720),
            (440 / 1280, 314 / 720),
        ),
        "iris": (480 / 1280, 300 / 720),
        "head_origin": (760 / 1280, 300 / 720),
        "head_forward": (370 / 1280, 360 / 720),
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument(
        "--dummy-side-annotations",
        action="store_true",
        help="Add fixed normalized Side annotations for wiring smoke tests only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing output manifest.",
    )
    return parser


def _clean_required(row: Mapping[str, str], key: str, *, source: Path, line: int) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ManifestBuildError(f"{source}:{line}: {key} is empty")
    return value


def _safe_subject(value: str) -> str:
    subject = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", subject):
        raise ManifestBuildError(f"unsafe subject directory name: {value!r}")
    return subject


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _screen_size(subject_dir: Path) -> tuple[int, int]:
    participant_path = subject_dir / "participant.json"
    if not participant_path.is_file():
        raise ManifestBuildError(f"participant metadata is missing: {participant_path}")
    try:
        participant = json.loads(participant_path.read_text(encoding="utf-8"))
        screen = participant["screen"]
        size = (int(screen["canvas_width_pixel"]), int(screen["canvas_height_pixel"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ManifestBuildError(f"invalid screen metadata: {participant_path}") from exc
    if size != SCREEN_SIZE:
        raise ManifestBuildError(
            f"data(ver1) smoke expects a {SCREEN_SIZE[0]}x{SCREEN_SIZE[1]} screen, got {size}"
        )
    return size


def _source_image(
    *, source_root: Path, subject_dir: Path, raw_path: str, source: Path, line: int
) -> tuple[Path, str]:
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ManifestBuildError(f"{source}:{line}: unsafe image path {raw_path!r}")
    image_path = (subject_dir / relative).resolve(strict=False)
    if not _is_within(image_path, subject_dir):
        raise ManifestBuildError(f"{source}:{line}: image escapes subject root: {raw_path!r}")
    if not image_path.is_file():
        raise ManifestBuildError(f"{source}:{line}: image does not exist: {image_path}")
    return image_path, image_path.relative_to(source_root).as_posix()


def _target(value: str, *, limit: int, field: str, source: Path, line: int) -> str:
    try:
        number = float(value)
    except ValueError as exc:
        raise ManifestBuildError(f"{source}:{line}: {field} is not numeric") from exc
    if not math.isfinite(number) or not 0 <= number < limit:
        raise ManifestBuildError(f"{source}:{line}: {field}={value!r} is outside [0, {limit})")
    return value.strip()


def _scale_point(point: Sequence[float], width: int, height: int) -> list[int]:
    return [int(round(float(point[0]) * width)), int(round(float(point[1]) * height))]


def _dummy_annotations(subject: str, image_path: Path) -> dict[str, str]:
    template = _DUMMY_SIDE_TEMPLATES.get(subject)
    if template is None:
        raise ManifestBuildError(
            f"--dummy-side-annotations has no reviewed fixed template for {subject!r}"
        )
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except OSError as exc:
        raise ManifestBuildError(f"cannot read Side image dimensions: {image_path}") from exc

    bbox = template["bbox"]
    scaled_bbox = [
        int(round(float(bbox[0]) * width)),
        int(round(float(bbox[1]) * height)),
        int(round(float(bbox[2]) * width)),
        int(round(float(bbox[3]) * height)),
    ]
    keypoints = [_scale_point(point, width, height) for point in template["keypoints"]]
    return {
        "visible_eye": str(template["visible_eye"]),
        "visible_eye_bbox_xyxy": json.dumps(scaled_bbox, separators=(",", ":")),
        "visible_eye_keypoints_xy": json.dumps(keypoints, separators=(",", ":")),
        "iris_center_xy": json.dumps(
            _scale_point(template["iris"], width, height), separators=(",", ":")
        ),
        "profile_head_origin_xy": json.dumps(
            _scale_point(template["head_origin"], width, height), separators=(",", ":")
        ),
        "profile_head_forward_xy": json.dumps(
            _scale_point(template["head_forward"], width, height), separators=(",", ":")
        ),
        "eye_annotation_valid": "true",
        "annotation_source": "dummy_smoke",
    }


def _blank_annotations() -> dict[str, str]:
    return {
        "visible_eye": "",
        "visible_eye_bbox_xyxy": "",
        "visible_eye_keypoints_xy": "",
        "iris_center_xy": "",
        "profile_head_origin_xy": "",
        "profile_head_forward_xy": "",
        "eye_annotation_valid": "false",
        "annotation_source": "",
    }


def _subject_rows(
    source_root: Path, subject: str, *, dummy_side_annotations: bool
) -> list[dict[str, str | int]]:
    subject_dir = source_root / subject
    source = subject_dir / "labels" / "image_samples.csv"
    if not source.is_file():
        raise ManifestBuildError(f"image sample source is missing: {source}")
    screen_width, screen_height = _screen_size(subject_dir)
    rows: list[dict[str, str | int]] = []
    seen_samples: set[str] = set()
    seen_pairs: set[str] = set()

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = {str(name).strip() for name in reader.fieldnames or ()}
        missing = sorted(SOURCE_COLUMNS - headers)
        if missing:
            raise ManifestBuildError(f"{source}: missing source columns {missing}")

        for line, raw_row in enumerate(reader, start=2):
            row = {str(key).strip(): str(value or "").strip() for key, value in raw_row.items()}
            participant = _clean_required(row, "participant", source=source, line=line)
            if participant != subject:
                raise ManifestBuildError(
                    f"{source}:{line}: participant={participant!r} does not match {subject!r}"
                )
            sample = _clean_required(row, "sample", source=source, line=line)
            raw_pair = _clean_required(row, "pair", source=source, line=line)
            if sample in seen_samples:
                raise ManifestBuildError(f"{source}:{line}: duplicate sample {sample!r}")
            pair_id = f"{subject}_pair_{raw_pair}"
            if pair_id in seen_pairs:
                raise ManifestBuildError(f"{source}:{line}: duplicate pair {raw_pair!r}")
            seen_samples.add(sample)
            seen_pairs.add(pair_id)

            target_x = _target(
                row["x_px"],
                limit=screen_width,
                field="x_px",
                source=source,
                line=line,
            )
            target_y = _target(
                row["y_px"],
                limit=screen_height,
                field="y_px",
                source=source,
                line=line,
            )
            common: dict[str, str | int] = {
                "subject_id": subject,
                "session_id": "",
                "pair_id": pair_id,
                "target_x_px": target_x,
                "target_y_px": target_y,
                "screen_width_px": screen_width,
                "screen_height_px": screen_height,
            }
            front_path, front_relative = _source_image(
                source_root=source_root,
                subject_dir=subject_dir,
                raw_path=_clean_required(row, "webcam_image", source=source, line=line),
                source=source,
                line=line,
            )
            del front_path  # Existence was validated; source images are never copied.
            rows.append(
                {
                    **common,
                    **_blank_annotations(),
                    "sample_id": f"{subject}_{sample}_front",
                    "view": "webcam",
                    "image_path": front_relative,
                }
            )

            side_path, side_relative = _source_image(
                source_root=source_root,
                subject_dir=subject_dir,
                raw_path=_clean_required(row, "phonecam_image", source=source, line=line),
                source=source,
                line=line,
            )
            annotations = (
                _dummy_annotations(subject, side_path)
                if dummy_side_annotations
                else _blank_annotations()
            )
            rows.append(
                {
                    **common,
                    **annotations,
                    "sample_id": f"{subject}_{sample}_side",
                    "view": "phonecam",
                    "image_path": side_relative,
                }
            )
    if not rows:
        raise ManifestBuildError(f"image sample source is empty: {source}")
    return rows


def _write_manifest(path: Path, rows: Iterable[Mapping[str, object]], *, force: bool) -> None:
    if path.exists() and not force:
        raise ManifestBuildError(f"output already exists; pass --force to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
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
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def create_manifest(
    source_root: str | Path,
    output_manifest: str | Path,
    *,
    subjects: Sequence[str] = DEFAULT_SUBJECTS,
    dummy_side_annotations: bool = False,
    force: bool = False,
) -> ManifestSummary:
    """Convert selected participant image-sample tables without touching the raw DB."""

    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ManifestBuildError(f"source root is not a directory: {root}")
    output = Path(output_manifest).expanduser().resolve(strict=False)
    if _is_within(output, root):
        raise ManifestBuildError("output manifest must be outside the read-only source root")

    normalized_subjects = tuple(_safe_subject(subject) for subject in subjects)
    if not normalized_subjects or len(set(normalized_subjects)) != len(normalized_subjects):
        raise ManifestBuildError("--subjects must contain unique subject directory names")

    rows: list[dict[str, str | int]] = []
    for subject in normalized_subjects:
        rows.extend(
            _subject_rows(
                root,
                subject,
                dummy_side_annotations=dummy_side_annotations,
            )
        )
    _write_manifest(output, rows, force=force)
    return ManifestSummary(
        path=output,
        subjects=normalized_subjects,
        source_pairs=len(rows) // 2,
        output_rows=len(rows),
        dummy_side_annotations=bool(dummy_side_annotations),
    )


def main() -> int:
    args = _parser().parse_args()
    try:
        result = create_manifest(
            args.source_root,
            args.output_manifest,
            subjects=args.subjects,
            dummy_side_annotations=args.dummy_side_annotations,
            force=args.force,
        )
    except (ManifestBuildError, OSError) as exc:
        raise SystemExit(f"manifest 생성 실패: {exc}") from exc
    print(f"manifest: {result.path}")
    print(f"subjects: {', '.join(result.subjects)}")
    print(f"pairs: {result.source_pairs}, rows: {result.output_rows}")
    print(f"dummy side annotations: {str(result.dummy_side_annotations).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
