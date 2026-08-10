"""Create two non-biometric PNGs and a canonical generic CSV manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_SIZE = (640, 360)
SCREEN_SIZE = (1920, 1080)
FIELDNAMES = (
    "sample_id",
    "subject_id",
    "view",
    "image_path",
    "target_x_px",
    "target_y_px",
    "screen_width_px",
    "screen_height_px",
    "facial_landmarks_xy",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _draw_demo_face(path: Path, *, shift_x: int, color: tuple[int, int, int]) -> list[list[int]]:
    image = Image.new("RGB", IMAGE_SIZE, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    center_x = IMAGE_SIZE[0] // 2 + shift_x
    center_y = IMAGE_SIZE[1] // 2
    face_box = (center_x - 105, center_y - 135, center_x + 105, center_y + 135)
    draw.ellipse(face_box, fill=color)
    draw.ellipse((center_x - 60, center_y - 40, center_x - 30, center_y - 15), fill=(25, 25, 25))
    draw.ellipse((center_x + 30, center_y - 40, center_x + 60, center_y - 15), fill=(25, 25, 25))
    draw.arc(
        (center_x - 55, center_y + 25, center_x + 55, center_y + 90),
        start=10,
        end=170,
        fill=(90, 30, 30),
        width=5,
    )
    landmarks = [
        [center_x - 65, center_y - 30],
        [center_x - 25, center_y - 30],
        [center_x + 25, center_y - 30],
        [center_x + 65, center_y - 30],
        [center_x - 45, center_y + 65],
        [center_x + 45, center_y + 65],
    ]
    image.save(path)
    return landmarks


def main() -> int:
    output_root = _parser().parse_args().output_root.expanduser().resolve(strict=False)
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    samples = (
        ("demo-001", "demo-subject-01", "demo_front_01.png", -35, (221, 174, 135), 720, 420),
        ("demo-002", "demo-subject-02", "demo_front_02.png", 45, (176, 139, 105), 1220, 650),
    )
    rows: list[dict[str, str | int]] = []
    for sample_id, subject_id, filename, shift_x, color, target_x, target_y in samples:
        landmarks = _draw_demo_face(image_dir / filename, shift_x=shift_x, color=color)
        rows.append(
            {
                "sample_id": sample_id,
                "subject_id": subject_id,
                "view": "front",
                "image_path": f"images/{filename}",
                "target_x_px": target_x,
                "target_y_px": target_y,
                "screen_width_px": SCREEN_SIZE[0],
                "screen_height_px": SCREEN_SIZE[1],
                "facial_landmarks_xy": json.dumps(landmarks, separators=(",", ":")),
            }
        )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"demo root: {output_root}")
    print(f"manifest: {manifest_path}")
    for row in rows:
        print(f"image: {output_root / str(row['image_path'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
