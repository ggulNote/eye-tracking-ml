"""Rebuild Side annotation summaries and contact sheets without re-running detectors."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from generate_side_eye_annotations import (
    GeneratedRow,
    SideGenerationError,
    _contact_sheets,
    _decode_bbox,
    _nfc,
    _read_csv,
    _truthy,
    discover_phone_samples,
)

from gaze_pipeline.data.side_annotation import SideEyeAnnotation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", default=["head_down", "neutral"])
    parser.add_argument("--preview-per-subject", type=int, default=4)
    parser.add_argument("--review-preview-limit", type=int, default=400)
    return parser


def _optional_points(raw: str, *, expected_length: int | None = None) -> Any:
    if not str(raw).strip():
        return None
    value = json.loads(str(raw))
    if not isinstance(value, list) or (
        expected_length is not None and len(value) != expected_length
    ):
        raise SideGenerationError(f"invalid point JSON: {raw!r}")
    return tuple(tuple(float(coordinate) for coordinate in point) for point in value)


def _annotation(row: dict[str, str]) -> SideEyeAnnotation | None:
    if not str(row.get("visible_eye_bbox_xyxy", "")).strip():
        return None
    flags_raw = json.loads(row.get("quality_flags", "[]") or "[]")
    scores_raw = json.loads(row.get("visibility_scores", "{}") or "{}")
    iris_points = _optional_points(
        f"[{row['iris_center_xy']}]" if row.get("iris_center_xy") else "",
        expected_length=1,
    )
    return SideEyeAnnotation(
        visible_eye=str(row.get("visible_eye", "")),
        bbox_xyxy=tuple(_decode_bbox(row["visible_eye_bbox_xyxy"])),
        eyelid_keypoints_xy=_optional_points(row.get("visible_eye_keypoints_xy", "")),
        iris_center_xy=iris_points[0] if iris_points else None,
        confidence=float(row.get("detection_confidence", 0.0) or 0.0),
        valid=_truthy(row.get("eye_annotation_valid", "")),
        quality_flags=tuple(str(flag) for flag in flags_raw),
        method=str(row.get("detection_method", "existing_annotation")),
        visibility_scores={str(key): float(value) for key, value in scores_raw.items()},
    )


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = args.data_root.expanduser().resolve(strict=True)
    annotation_csv = args.annotation_csv.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_phone_samples(root, tuple(args.sessions))
    sample_index = {sample.sample_id: sample for sample in samples}
    rows = _read_csv(annotation_csv)
    row_index = {_nfc(row.get("sample_id", "")): row for row in rows}
    if len(row_index) != len(rows):
        raise SideGenerationError("annotation CSV contains blank or duplicate sample_id values")
    missing = sorted(set(sample_index) - set(row_index))
    extra = sorted(set(row_index) - set(sample_index))
    if missing or extra:
        raise SideGenerationError(
            f"annotation/sample mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    generated = [
        GeneratedRow(
            sample=sample,
            annotation=_annotation(row_index[sample.sample_id]),
            reason=row_index[sample.sample_id].get("fallback_reason", ""),
        )
        for sample in samples
    ]
    accepted = [
        item
        for item in generated
        if item.sample.input_valid is True and item.annotation is not None and item.annotation.valid
    ]
    review = [
        item
        for item in generated
        if item.sample.input_valid is True
        and (item.annotation is None or not item.annotation.valid)
    ]
    accepted_by_subject: dict[str, list[GeneratedRow]] = defaultdict(list)
    for item in accepted:
        accepted_by_subject[item.sample.participant].append(item)
    accepted_preview = [
        item
        for participant in sorted(accepted_by_subject)
        for item in accepted_by_subject[participant][: max(0, args.preview_per_subject)]
    ]
    accepted_files = _contact_sheets(
        accepted_preview,
        output_dir=output_dir,
        prefix="accepted_preview",
    )
    review_files = _contact_sheets(
        review,
        output_dir=output_dir,
        prefix="review_queue",
        limit=max(0, args.review_preview_limit),
    )
    methods = Counter(row.get("detection_method", "") for row in rows)
    summary = {
        "data_root": str(root),
        "output_csv": str(annotation_csv),
        "total_samples": len(generated),
        "open_samples": sum(item.sample.input_valid is True for item in generated),
        "closed_samples": sum(item.sample.input_valid is False for item in generated),
        "unknown_eye_state_samples": sum(item.sample.input_valid is None for item in generated),
        "excluded_samples": sum(item.sample.input_valid is not True for item in generated),
        "accepted_samples": len(accepted),
        "review_required_samples": len(review),
        "complete": not review,
        "methods": dict(methods),
        "accepted_preview_files": accepted_files,
        "review_preview_files": review_files,
        "source": "existing annotation CSV; detectors were not re-run",
    }
    _atomic_write_json(output_dir / "side_annotation_summary.json", summary)
    return summary


def main() -> int:
    args = _parser().parse_args()
    try:
        summary = finalize(args)
    except (OSError, ValueError, json.JSONDecodeError, SideGenerationError) as exc:
        print(f"side annotation finalization failed: {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
