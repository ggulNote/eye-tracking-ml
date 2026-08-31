"""Audit external Side ROI image sizes and render the exact 128x256 preprocessing result."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from gaze_pipeline.data.transforms import center_crop_pad_precomputed_roi

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side-roi-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", default=["head_down", "neutral"])
    parser.add_argument("--samples-per-session", type=int, default=1)
    return parser


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", str(value))


def _images(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: _nfc(path.name),
    )


def _selected(images: list[Path], count: int) -> list[Path]:
    if count <= 0 or not images:
        return []
    if len(images) <= count:
        return images
    if count == 1:
        return [images[len(images) // 2]]
    positions = [round(index * (len(images) - 1) / (count - 1)) for index in range(count)]
    return [images[position] for position in positions]


def _write_contact_sheets(
    selected: list[tuple[str, str, Path, tuple[int, int], np.ndarray]], output: Path
) -> list[str]:
    cell_width, cell_height = 256, 154
    columns, rows = 4, 4
    page_size = columns * rows
    written: list[str] = []
    for start in range(0, len(selected), page_size):
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "black")
        draw = ImageDraw.Draw(sheet)
        for offset, (_subject, session, _path, size, transformed) in enumerate(
            selected[start : start + page_size]
        ):
            x = (offset % columns) * cell_width
            y = (offset // columns) * cell_height
            label = f"{start + offset + 1:03d} {session} {size[0]}x{size[1]}"
            draw.text((x + 3, y + 3), label, fill="white")
            sheet.paste(Image.fromarray(transformed), (x, y + 26))
        filename = f"precomputed_side_roi_preview_{start // page_size + 1:03d}.jpg"
        sheet.save(output / filename, quality=92)
        written.append(filename)
    return written


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.side_roi_root.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    session_counts: dict[tuple[str, str], int] = {}
    selected: list[tuple[str, str, Path, tuple[int, int], np.ndarray]] = []
    corrupt: list[dict[str, str]] = []
    missing_sessions: list[dict[str, str]] = []
    subjects = sorted(
        (path for path in root.iterdir() if path.is_dir()), key=lambda p: _nfc(p.name)
    )
    for subject_dir in subjects:
        for session in args.sessions:
            directory = subject_dir / session / "side_eye_roi"
            images = _images(directory) if directory.is_dir() else []
            if not images:
                missing_sessions.append({"subject": _nfc(subject_dir.name), "session": session})
                continue
            session_counts[(_nfc(subject_dir.name), session)] = len(images)
            selected_paths = set(_selected(images, max(0, int(args.samples_per_session))))
            for path in images:
                try:
                    with Image.open(path) as source:
                        rgb_image = source.convert("RGB")
                        rgb_image.load()
                        size = rgb_image.size
                        mode = source.mode
                        rgb = np.asarray(rgb_image)
                except (OSError, ValueError) as exc:
                    corrupt.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                    continue
                sizes[size] += 1
                modes[mode] += 1
                if path in selected_paths:
                    selected.append(
                        (
                            _nfc(subject_dir.name),
                            session,
                            path,
                            size,
                            center_crop_pad_precomputed_roi(rgb),
                        )
                    )
    total = sum(sizes.values())
    preview_files = _write_contact_sheets(selected, output)
    report = {
        "side_roi_root": str(root),
        "total_images": total,
        "subjects_with_images": len({subject for subject, _session in session_counts}),
        "sessions_with_images": len(session_counts),
        "expected_128x128_images": sizes[(128, 128)],
        "non_128x128_images": total - sizes[(128, 128)],
        "corrupt_images": len(corrupt),
        "ready_for_intended_128x128_training": (
            total > 0 and sizes[(128, 128)] == total and not corrupt
        ),
        "source_sizes": {
            f"{width}x{height}": count for (width, height), count in sorted(sizes.items())
        },
        "source_modes": dict(modes),
        "session_counts": [
            {"subject": subject, "session": session, "images": count}
            for (subject, session), count in sorted(session_counts.items())
        ],
        "missing_sessions": missing_sessions,
        "corrupt_details": corrupt,
        "preprocessing_contract": {
            "content_size_hw": [128, 128],
            "output_size_hw": [128, 256],
            "operation": "center_crop_or_black_pad_without_resampling",
            "content_x_range": [64, 192],
            "normalization": "uint8 / 255 -> float32 CHW",
        },
        "preview_files": preview_files,
    }
    (output / "precomputed_side_roi_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    report = audit(_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
