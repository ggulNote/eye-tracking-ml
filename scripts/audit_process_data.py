"""Audit ``process_data`` completeness, pixel preservation, and visual quality.

The audit is intentionally read-only for the source datasets. Reports and
contact sheets are written below ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from gaze_pipeline.data.transforms import center_crop_pad_precomputed_roi
from scripts.create_process_data_manifest import ProcessPair, scan_process_data


def _font(size: int = 12) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    windows_font = Path("C:/Windows/Fonts/malgun.ttf")
    if windows_font.is_file():
        return ImageFont.truetype(str(windows_font), size=size)
    return ImageFont.load_default()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--sessions", nargs="+", default=["head_down", "neutral"])
    parser.add_argument("--samples-per-session", type=int, default=2)
    parser.add_argument("--summary-only", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rgb(path: Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as image:
        mode = image.mode
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8), mode


def _laplacian_variance(rgb: np.ndarray) -> float:
    gray = np.asarray(rgb, dtype=np.float32).mean(axis=2)
    if min(gray.shape) < 3:
        return 0.0
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(laplacian))


def _quality(rgb: np.ndarray) -> dict[str, float]:
    gray = np.asarray(rgb, dtype=np.float32).mean(axis=2)
    low, high = np.percentile(gray, [1.0, 99.0])
    return {
        "focus_laplacian_variance": _laplacian_variance(rgb),
        "contrast_std": float(np.std(gray)),
        "brightness_mean": float(np.mean(gray)),
        "dynamic_range_p01_p99": float(high - low),
    }


def _distribution(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not array.size:
        return {}
    return {
        "min": float(np.min(array)),
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _fit(image: Image.Image, size: tuple[int, int], *, nearest: bool = False) -> Image.Image:
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return ImageOps.contain(image.convert("RGB"), size, method=method)


def _center_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = max(0, (width - side) // 2)
    top = max(0, (height - side) // 2)
    return image.crop((left, top, left + side, top + side))


def _local_raw_front(pair: ProcessPair, reference_root: Path | None) -> Path | None:
    if reference_root is None or not pair.source_image_path:
        return None
    filename = Path(pair.source_image_path).name
    candidate = (
        reference_root / pair.subject / pair.session / "feature_maps" / "web" / "frames" / filename
    )
    return candidate if candidate.is_file() else None


def _local_raw_side(pair: ProcessPair, reference_root: Path | None) -> Path | None:
    if reference_root is None or pair.side_path is None:
        return None
    candidate = (
        reference_root
        / pair.subject
        / pair.session
        / "feature_maps"
        / "phone"
        / "frames"
        / pair.side_path.name
    )
    return candidate if candidate.is_file() else None


def _reference_front_roi(pair: ProcessPair, reference_root: Path | None) -> Path | None:
    if reference_root is None or pair.front_path is None:
        return None
    candidate = (
        reference_root
        / pair.subject
        / pair.session
        / "feature_maps"
        / "webeyetrack"
        / "eye_roi"
        / pair.front_path.name
    )
    return candidate if candidate.is_file() else None


def _write_invalid_original_sheets(
    pairs: Sequence[ProcessPair], reference_root: Path | None, output: Path
) -> list[str]:
    selected = [(pair, _local_raw_front(pair, reference_root)) for pair in pairs]
    selected = [(pair, raw) for pair, raw in selected if raw is not None]
    if not selected:
        return []
    columns, rows = 4, 4
    cell_w, cell_h = 320, 270
    page_size = columns * rows
    written: list[str] = []
    for start in range(0, len(selected), page_size):
        sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "black")
        draw = ImageDraw.Draw(sheet)
        font = _font()
        for offset, (pair, raw_path) in enumerate(selected[start : start + page_size]):
            assert raw_path is not None
            x = (offset % columns) * cell_w
            y = (offset // columns) * cell_h
            label = f"{pair.subject}/{pair.session} {pair.pair_id.rsplit('_', 1)[-1]}"
            draw.text((x + 4, y + 4), label, fill="white", font=font)
            draw.text(
                (x + 4, y + 19),
                pair.exclusion_reason,
                fill=(255, 190, 80),
                font=font,
            )
            with Image.open(raw_path) as raw:
                face = _fit(_center_square(raw), (220, 220))
            sheet.paste(face, (x + 4, y + 44))
            if pair.side_path is not None and pair.side_path.is_file():
                with Image.open(pair.side_path) as side:
                    side_thumb = _fit(side, (88, 88), nearest=True)
                sheet.paste(side_thumb, (x + 228, y + 44))
                draw.text((x + 228, y + 136), "Side ROI", fill="white", font=font)
            draw.text((x + 228, y + 160), "raw center", fill="white", font=font)
        filename = f"invalid_originals_{start // page_size + 1:03d}.jpg"
        sheet.save(output / filename, quality=94)
        written.append(filename)
    return written


def _write_side_quality_sheets(
    ranked: Sequence[tuple[ProcessPair, dict[str, float]]],
    output: Path,
    *,
    prefix: str = "side_quality_review",
) -> list[str]:
    if not ranked:
        return []
    columns, rows = 4, 4
    cell_w, cell_h = 256, 292
    page_size = columns * rows
    written: list[str] = []
    for start in range(0, len(ranked), page_size):
        sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "black")
        draw = ImageDraw.Draw(sheet)
        font = _font()
        for offset, (pair, quality) in enumerate(ranked[start : start + page_size]):
            assert pair.side_path is not None
            x = (offset % columns) * cell_w
            y = (offset // columns) * cell_h
            draw.text(
                (x + 3, y + 3),
                f"{pair.subject}/{pair.session} {pair.pair_id.rsplit('_', 1)[-1]}",
                fill="white",
                font=font,
            )
            draw.text(
                (x + 3, y + 18),
                f"focus={quality['focus_laplacian_variance']:.1f} "
                f"contrast={quality['contrast_std']:.1f}",
                fill=(255, 190, 80),
                font=font,
            )
            with Image.open(pair.side_path) as image:
                enlarged = _fit(image, (256, 256), nearest=True)
            sheet.paste(enlarged, (x, y + 36))
        filename = f"{prefix}_{start // page_size + 1:03d}.png"
        sheet.save(output / filename)
        written.append(filename)
    return written


def _write_front_comparison_sheet(
    pairs: Sequence[ProcessPair], reference_root: Path | None, output: Path
) -> list[str]:
    selected: list[tuple[ProcessPair, Path]] = []
    seen_subjects: set[str] = set()
    for pair in pairs:
        reference = _reference_front_roi(pair, reference_root)
        if reference is None or pair.front_path is None or pair.subject in seen_subjects:
            continue
        selected.append((pair, reference))
        seen_subjects.add(pair.subject)
        if len(selected) == 8:
            break
    if not selected:
        return []
    cell_w, cell_h = 512, 280
    sheet = Image.new("RGB", (cell_w * 2, cell_h * 4), "black")
    draw = ImageDraw.Draw(sheet)
    font = _font()
    for index, (pair, reference_path) in enumerate(selected):
        assert pair.front_path is not None
        x = (index % 2) * cell_w
        y = (index // 2) * cell_h
        with Image.open(pair.front_path) as current, Image.open(reference_path) as reference:
            current_rgb = current.convert("RGB")
            reference_rgb = reference.convert("RGB")
            current_array = np.asarray(current_rgb, dtype=np.int16)
            reference_array = np.asarray(reference_rgb, dtype=np.int16)
            difference = np.abs(current_array - reference_array).astype(np.uint8)
            sheet.paste(current_rgb, (x, y + 24))
            sheet.paste(reference_rgb, (x, y + 24 + 128))
            diff_max = int(difference.max()) if difference.size else 0
        draw.text(
            (x + 3, y + 3),
            f"{pair.subject}/{pair.session} process / reference / diff_max={diff_max}",
            fill="white",
            font=font,
        )
    filename = "front_reference_pixel_comparison.png"
    sheet.save(output / filename)
    return [filename]


def _write_session_sample_sheets(
    pairs: Sequence[ProcessPair], count: int, output: Path
) -> list[str]:
    if count <= 0:
        return []
    grouped: dict[tuple[str, str], list[ProcessPair]] = {}
    for pair in pairs:
        if pair.side_path is not None:
            grouped.setdefault((pair.subject, pair.session), []).append(pair)
    selected: list[ProcessPair] = []
    for key in sorted(grouped):
        values = sorted(grouped[key], key=lambda pair: pair.pair_id)
        if count == 1:
            positions = [len(values) // 2]
        else:
            positions = [round(i * (len(values) - 1) / (count - 1)) for i in range(count)]
        selected.extend(values[position] for position in positions)
    ranked: list[tuple[ProcessPair, dict[str, float]]] = []
    for pair in selected:
        assert pair.side_path is not None
        array, _mode = _load_rgb(pair.side_path)
        ranked.append((pair, _quality(array)))
    return _write_side_quality_sheets(ranked, output, prefix="side_session_samples")


def _write_full_session_sheets(
    session_pairs: Mapping[tuple[str, str], Sequence[ProcessPair]], output: Path
) -> list[str]:
    written: list[str] = []
    font = _font()
    columns, rows = 9, 9
    cell_w, cell_h = 128, 146
    for index, (key, pairs) in enumerate(sorted(session_pairs.items()), start=1):
        sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "black")
        draw = ImageDraw.Draw(sheet)
        for offset, pair in enumerate(sorted(pairs, key=lambda item: item.pair_id)[:81]):
            if pair.side_path is None:
                continue
            x = (offset % columns) * cell_w
            y = (offset // columns) * cell_h
            draw.text(
                (x + 2, y + 2),
                pair.pair_id.rsplit("_", 1)[-1],
                fill="white",
                font=font,
            )
            with Image.open(pair.side_path) as image:
                sheet.paste(image.convert("RGB"), (x, y + 18))
        subject, session = key
        filename = f"flagged_session_{index:02d}_{subject}_{session}.jpg"
        sheet.save(output / filename, quality=94)
        written.append(filename)
    return written


def _write_side_raw_comparison_sheets(
    pairs: Sequence[ProcessPair], reference_root: Path | None, output: Path
) -> list[str]:
    selected = [(pair, _local_raw_side(pair, reference_root)) for pair in pairs]
    selected = [(pair, raw) for pair, raw in selected if raw is not None]
    if not selected:
        return []
    columns, rows = 4, 4
    cell_w, cell_h = 400, 260
    page_size = columns * rows
    written: list[str] = []
    font = _font()
    for start in range(0, len(selected), page_size):
        sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "black")
        draw = ImageDraw.Draw(sheet)
        for offset, (pair, raw_path) in enumerate(selected[start : start + page_size]):
            assert raw_path is not None
            assert pair.side_path is not None
            x = (offset % columns) * cell_w
            y = (offset // columns) * cell_h
            draw.text(
                (x + 3, y + 3),
                f"{pair.subject}/{pair.session} {pair.pair_id.rsplit('_', 1)[-1]}",
                fill="white",
                font=font,
            )
            with Image.open(raw_path) as raw:
                raw_thumb = _fit(raw, (280, 224))
            with Image.open(pair.side_path) as roi:
                roi_thumb = _fit(roi, (112, 112), nearest=True)
            sheet.paste(raw_thumb, (x + 3, y + 26))
            sheet.paste(roi_thumb, (x + 285, y + 26))
            draw.text((x + 285, y + 142), "Side ROI", fill="white", font=font)
            draw.text((x + 3, y + 242), "original phone frame", fill="white", font=font)
        filename = f"side_roi_vs_original_{start // page_size + 1:03d}.jpg"
        sheet.save(output / filename, quality=94)
        written.append(filename)
    return written


def _write_detailed_missingness(inventory: Any, reference_root: Path | None, output: Path) -> None:
    fields = (
        "subject_id",
        "session_id",
        "pair_id",
        "input_valid",
        "invalid_reason",
        "exclusion_reason",
        "front_roi_path",
        "side_roi_path",
        "local_original_front_path",
        "input_csv_path",
        "input_csv_line",
    )
    rows: list[dict[str, Any]] = []
    for missing in inventory.missing_sessions:
        rows.append(
            {
                "subject_id": missing.subject,
                "session_id": missing.session,
                "pair_id": "",
                "input_valid": "",
                "invalid_reason": "",
                "exclusion_reason": missing.reason,
                "front_roi_path": "",
                "side_roi_path": "",
                "local_original_front_path": "",
                "input_csv_path": "",
                "input_csv_line": "",
            }
        )
    for pair in inventory.pairs:
        if pair.usable:
            continue
        raw = _local_raw_front(pair, reference_root)
        rows.append(
            {
                "subject_id": pair.subject,
                "session_id": pair.session,
                "pair_id": pair.pair_id,
                "input_valid": int(pair.valid),
                "invalid_reason": pair.invalid_reason,
                "exclusion_reason": pair.exclusion_reason,
                "front_roi_path": str(pair.front_path or ""),
                "side_roi_path": str(pair.side_path or ""),
                "local_original_front_path": str(raw or ""),
                "input_csv_path": str(pair.csv_path),
                "input_csv_line": pair.csv_line,
            }
        )
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    reference_root = (
        args.reference_root.expanduser().resolve(strict=True)
        if args.reference_root is not None and args.reference_root.is_dir()
        else None
    )
    inventory = scan_process_data(args.process_data_root, sessions=args.sessions)

    unique_front = sorted(
        {pair.front_path for pair in inventory.pairs if pair.front_path is not None}, key=str
    )
    unique_side = sorted(
        {pair.side_path for pair in inventory.pairs if pair.side_path is not None}, key=str
    )
    front_sizes: Counter[str] = Counter()
    side_sizes: Counter[str] = Counter()
    front_modes: Counter[str] = Counter()
    side_modes: Counter[str] = Counter()
    corrupt: list[dict[str, str]] = []
    front_arrays: dict[Path, np.ndarray] = {}
    side_arrays: dict[Path, np.ndarray] = {}
    for kind, paths, sizes, modes, arrays in (
        ("front", unique_front, front_sizes, front_modes, front_arrays),
        ("side", unique_side, side_sizes, side_modes, side_arrays),
    ):
        for path in paths:
            try:
                array, mode = _load_rgb(path)
                arrays[path] = array
                sizes[f"{array.shape[1]}x{array.shape[0]}"] += 1
                modes[mode] += 1
            except Exception as exc:  # Pillow exposes several decoder-specific exceptions.
                corrupt.append({"kind": kind, "path": str(path), "error": str(exc)})

    front_passthrough_exact = 0
    for array in front_arrays.values():
        transformed = center_crop_pad_precomputed_roi(
            array, content_size_hw=(128, 512), output_size_hw=(128, 512)
        )
        if np.array_equal(array, transformed):
            front_passthrough_exact += 1

    side_canvas_exact = 0
    for array in side_arrays.values():
        transformed = center_crop_pad_precomputed_roi(
            array, content_size_hw=(128, 128), output_size_hw=(128, 256)
        )
        if (
            np.array_equal(transformed[:, 64:192], array)
            and not np.any(transformed[:, :64])
            and not np.any(transformed[:, 192:])
        ):
            side_canvas_exact += 1

    reference_byte_exact = 0
    reference_pixel_exact = 0
    reference_missing = 0
    reference_different: list[str] = []
    if reference_root is not None:
        for pair in inventory.usable_pairs:
            assert pair.front_path is not None
            reference = _reference_front_roi(pair, reference_root)
            if reference is None:
                reference_missing += 1
                continue
            if _sha256(pair.front_path) == _sha256(reference):
                reference_byte_exact += 1
                reference_pixel_exact += 1
                continue
            current, _ = _load_rgb(pair.front_path)
            expected, _ = _load_rgb(reference)
            if np.array_equal(current, expected):
                reference_pixel_exact += 1
            else:
                reference_different.append(pair.pair_id)

    side_quality: list[tuple[ProcessPair, dict[str, float]]] = []
    for pair in inventory.usable_pairs:
        if pair.side_path is not None and pair.side_path in side_arrays:
            side_quality.append((pair, _quality(side_arrays[pair.side_path])))
    side_quality.sort(key=lambda item: item[1]["focus_laplacian_variance"])
    lowest_focus = side_quality[:32]
    lowest_contrast = sorted(side_quality, key=lambda item: item[1]["contrast_std"])[:16]
    review_by_pair: dict[str, tuple[ProcessPair, dict[str, float]]] = {
        pair.pair_id: (pair, quality) for pair, quality in (*lowest_focus, *lowest_contrast)
    }
    review_items = sorted(
        review_by_pair.values(), key=lambda item: item[1]["focus_laplacian_variance"]
    )
    quality_fields = (
        "subject_id",
        "session_id",
        "pair_id",
        "side_roi_path",
        "focus_laplacian_variance",
        "contrast_std",
        "brightness_mean",
        "dynamic_range_p01_p99",
    )

    def write_quality_csv(
        path: Path, values: Sequence[tuple[ProcessPair, dict[str, float]]]
    ) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=quality_fields, lineterminator="\n")
            writer.writeheader()
            for pair, quality in values:
                writer.writerow(
                    {
                        "subject_id": pair.subject,
                        "session_id": pair.session,
                        "pair_id": pair.pair_id,
                        "side_roi_path": str(pair.side_path),
                        **{key: f"{value:.6f}" for key, value in quality.items()},
                    }
                )

    write_quality_csv(output / "side_quality_all.csv", side_quality)
    write_quality_csv(output / "side_quality_review.csv", review_items)

    global_focus_p01 = float(
        np.percentile([quality["focus_laplacian_variance"] for _, quality in side_quality], 1)
    )
    session_quality: dict[tuple[str, str], list[tuple[ProcessPair, dict[str, float]]]] = {}
    for pair, quality in side_quality:
        session_quality.setdefault((pair.subject, pair.session), []).append((pair, quality))
    session_quality_summary: list[dict[str, Any]] = []
    flagged_session_pairs: dict[tuple[str, str], list[ProcessPair]] = {}
    for key, values in sorted(session_quality.items()):
        focuses = np.asarray(
            [quality["focus_laplacian_variance"] for _, quality in values],
            dtype=np.float64,
        )
        below = int(np.sum(focuses <= global_focus_p01))
        fraction = below / len(values)
        flagged = below >= 5 and fraction >= 0.20
        if flagged:
            flagged_session_pairs[key] = [pair for pair, _ in values]
        session_quality_summary.append(
            {
                "subject": key[0],
                "session": key[1],
                "images": len(values),
                "focus_median": float(np.median(focuses)),
                "focus_min": float(np.min(focuses)),
                "images_at_or_below_global_p01": below,
                "fraction_at_or_below_global_p01": fraction,
                "clustered_low_focus_flag": flagged,
            }
        )
    with (output / "side_session_quality.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = tuple(session_quality_summary[0]) if session_quality_summary else ()
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(session_quality_summary)
    with (output / "quality_exclusions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("subject_id", "session_id", "exclusion_reason")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for subject, session in flagged_session_pairs:
            writer.writerow(
                {
                    "subject_id": subject,
                    "session_id": session,
                    "exclusion_reason": "side_roi_session_quality_outlier",
                }
            )

    invalid_pairs = [pair for pair in inventory.pairs if not pair.valid]
    valid_incomplete = [pair for pair in inventory.pairs if pair.valid and pair.issues]
    _write_detailed_missingness(inventory, reference_root, output / "missingness.csv")
    invalid_sheets = _write_invalid_original_sheets(invalid_pairs, reference_root, output)
    quality_sheets = _write_side_quality_sheets(review_items, output)
    comparison_sheets = _write_front_comparison_sheet(
        inventory.usable_pairs, reference_root, output
    )
    session_sample_sheets = _write_session_sample_sheets(
        inventory.usable_pairs, max(0, int(args.samples_per_session)), output
    )
    flagged_session_sheets = _write_full_session_sheets(flagged_session_pairs, output)
    flagged_side_pairs = [
        pair
        for pair, _quality_values in side_quality
        if (pair.subject, pair.session) in flagged_session_pairs
    ][:16]
    side_raw_comparison_sheets = _write_side_raw_comparison_sheets(
        flagged_side_pairs, reference_root, output
    )

    head_norms = [
        math.sqrt(sum(value * value for value in pair.head_vector))
        for pair in inventory.usable_pairs
        if pair.head_vector is not None
    ]
    invalid_reasons = Counter(pair.invalid_reason or "unspecified" for pair in invalid_pairs)
    result: dict[str, Any] = {
        "process_data_root": str(inventory.root),
        "reference_root": str(reference_root) if reference_root is not None else None,
        "subjects_found": len(inventory.subjects),
        "subjects_used": len({pair.subject for pair in inventory.usable_pairs}),
        "session_slots": len(inventory.subjects) * len(tuple(args.sessions)),
        "sessions_with_inputs": len({(pair.subject, pair.session) for pair in inventory.pairs}),
        "missing_sessions": [
            asdict(item) | {"session_path": str(item.session_path)}
            for item in inventory.missing_sessions
        ],
        "input_rows": len(inventory.pairs),
        "valid_1_rows": sum(pair.valid for pair in inventory.pairs),
        "valid_0_rows": len(invalid_pairs),
        "valid_0_reasons": dict(sorted(invalid_reasons.items())),
        "usable_pairs": len(inventory.usable_pairs),
        "valid_1_incomplete_pairs": len(valid_incomplete),
        "valid_1_incomplete_details": [
            {"pair_id": pair.pair_id, "issues": list(pair.issues)} for pair in valid_incomplete
        ],
        "orphan_front_images": [str(path) for path in inventory.orphan_front_images],
        "orphan_side_images": [str(path) for path in inventory.orphan_side_images],
        "front": {
            "images": len(unique_front),
            "sizes": dict(sorted(front_sizes.items())),
            "source_modes": dict(sorted(front_modes.items())),
            "corrupt_images": sum(item["kind"] == "front" for item in corrupt),
            "expected_size": "512x128",
            "expected_size_images": front_sizes.get("512x128", 0),
            "model_input_pixel_exact_without_resize": front_passthrough_exact,
            "reference_byte_exact": reference_byte_exact,
            "reference_pixel_exact": reference_pixel_exact,
            "reference_missing": reference_missing,
            "reference_pixel_different_pair_ids": reference_different,
        },
        "side": {
            "images": len(unique_side),
            "sizes": dict(sorted(side_sizes.items())),
            "source_modes": dict(sorted(side_modes.items())),
            "corrupt_images": sum(item["kind"] == "side" for item in corrupt),
            "expected_size": "128x128",
            "expected_size_images": side_sizes.get("128x128", 0),
            "model_128x256_canvas_pixel_exact": side_canvas_exact,
            "quality": {
                key: _distribution(quality[key] for _, quality in side_quality)
                for key in (
                    "focus_laplacian_variance",
                    "contrast_std",
                    "brightness_mean",
                    "dynamic_range_p01_p99",
                )
            },
            "quality_review_images": len(review_items),
            "quality_review_is_not_an_eye_detector": True,
            "global_focus_p01": global_focus_p01,
            "clustered_low_focus_sessions": [
                {
                    "subject": subject,
                    "session": session,
                    "reason": "at least 20% of images are at or below global focus p01",
                }
                for subject, session in flagged_session_pairs
            ],
            "session_quality": session_quality_summary,
        },
        "head_vector_norm": _distribution(head_norms),
        "corrupt_details": corrupt,
        "normalization": "uint8 / 255 -> float32 CHW; no interpolation",
        "recommended_quality_excluded_pairs": sum(
            len(pairs) for pairs in flagged_session_pairs.values()
        ),
        "ready_for_training": (
            not corrupt
            and not valid_incomplete
            and front_passthrough_exact == len(unique_front)
            and side_canvas_exact == len(unique_side)
            and len(inventory.usable_pairs) > 0
            and not flagged_session_pairs
        ),
        "artifacts": {
            "missingness_csv": "missingness.csv",
            "side_quality_review_csv": "side_quality_review.csv",
            "side_quality_all_csv": "side_quality_all.csv",
            "side_session_quality_csv": "side_session_quality.csv",
            "quality_exclusions_csv": "quality_exclusions.csv",
            "invalid_original_sheets": invalid_sheets,
            "side_quality_review_sheets": quality_sheets,
            "front_reference_comparison_sheets": comparison_sheets,
            "side_session_sample_sheets": session_sample_sheets,
            "flagged_session_sheets": flagged_session_sheets,
            "side_roi_vs_original_sheets": side_raw_comparison_sheets,
        },
    }
    report = output / "process_data_audit.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = _parser().parse_args()
    result = audit(args)
    if not args.summary_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"subjects found/used: {result['subjects_found']}/{result['subjects_used']}")
    print(
        "sessions with inputs/missing: "
        f"{result['sessions_with_inputs']}/{len(result['missing_sessions'])}"
    )
    print(f"inputs.csv valid=1/valid=0: {result['valid_1_rows']}/{result['valid_0_rows']}")
    print(f"structurally usable pairs: {result['usable_pairs']}")
    print(f"recommended quality-excluded pairs: {result['recommended_quality_excluded_pairs']}")
    print(f"ready without exclusions: {str(result['ready_for_training']).lower()}")


if __name__ == "__main__":
    main()
