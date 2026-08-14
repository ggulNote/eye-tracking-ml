"""Create a tiny annotated dual-view DB for end-to-end training smoke tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_SIZE = (320, 240)
SCREEN_SIZE = (1920, 1080)
FIELDNAMES = (
    "sample_id",
    "subject_id",
    "view",
    "image_path",
    "pair_id",
    "target_x_px",
    "target_y_px",
    "screen_width_px",
    "screen_height_px",
    "screen_width_mm",
    "screen_height_mm",
    "facial_landmarks_xy",
    "visible_eye",
    "visible_eye_bbox_xyxy",
    "visible_eye_keypoints_xy",
    "iris_center_xy",
    "profile_head_origin_xy",
    "profile_head_forward_xy",
    "eye_annotation_valid",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=4)
    return parser


def _front_image(path: Path, *, shift: int) -> list[list[int]]:
    image = Image.new("RGB", IMAGE_SIZE, (26, 28, 34))
    draw = ImageDraw.Draw(image)
    cx, cy = IMAGE_SIZE[0] // 2 + shift, IMAGE_SIZE[1] // 2
    draw.ellipse((cx - 70, cy - 95, cx + 70, cy + 95), fill=(207, 166, 132))
    draw.ellipse((cx - 40, cy - 25, cx - 17, cy - 8), fill=(20, 20, 20))
    draw.ellipse((cx + 17, cy - 25, cx + 40, cy - 8), fill=(20, 20, 20))
    image.save(path)
    return [
        [cx - 50, cy - 32],
        [cx - 12, cy - 32],
        [cx + 12, cy - 32],
        [cx + 50, cy - 32],
        [cx - 35, cy + 54],
        [cx + 35, cy + 54],
    ]


def _side_image(path: Path, *, shift: int) -> dict[str, object]:
    image = Image.new("RGB", IMAGE_SIZE, (18, 20, 25))
    draw = ImageDraw.Draw(image)
    cx, cy = 175 + shift, 108
    draw.ellipse((70 + shift, 25, 285 + shift, 225), fill=(205, 164, 132))
    points = [
        [cx - 42, cy],
        [cx - 20, cy - 10],
        [cx + 3, cy - 12],
        [cx + 35, cy],
        [cx + 3, cy + 12],
        [cx - 20, cy + 10],
    ]
    draw.line(points + [points[0]], fill=(35, 25, 22), width=3)
    draw.ellipse((cx - 2, cy - 11, cx + 18, cy + 11), fill=(38, 28, 23))
    image.save(path)
    return {
        "visible_eye": "right",
        "visible_eye_bbox_xyxy": [cx - 50, cy - 28, cx + 48, cy + 28],
        "visible_eye_keypoints_xy": points,
        "iris_center_xy": [cx + 8, cy],
        "profile_head_origin_xy": [85 + shift, 145],
        "profile_head_forward_xy": [250 + shift, 122],
        "eye_annotation_valid": "true",
    }


def main() -> int:
    args = _parser().parse_args()
    if args.subjects < 3:
        raise SystemExit("--subjects must be at least 3 so train/validation/test are non-empty")
    root = args.output_root.expanduser().resolve(strict=False)
    rows: list[dict[str, object]] = []
    for index in range(args.subjects):
        subject = f"demo-subject-{index + 1:02d}"
        pair_id = f"{subject}-pair-001"
        subject_dir = root / subject
        front_dir = subject_dir / "webcam"
        side_dir = subject_dir / "phonecam"
        front_dir.mkdir(parents=True, exist_ok=True)
        side_dir.mkdir(parents=True, exist_ok=True)
        shift = (index - args.subjects // 2) * 4
        front_path = front_dir / "001.png"
        side_path = side_dir / "001.png"
        front_landmarks = _front_image(front_path, shift=shift)
        side_annotations = _side_image(side_path, shift=shift)
        target_x = 640 + index * 180
        target_y = 360 + index * 90
        common = {
            "subject_id": subject,
            "pair_id": pair_id,
            "target_x_px": target_x,
            "target_y_px": target_y,
            "screen_width_px": SCREEN_SIZE[0],
            "screen_height_px": SCREEN_SIZE[1],
            "screen_width_mm": 530,
            "screen_height_mm": 300,
        }
        rows.append(
            {
                **common,
                "sample_id": f"{pair_id}-front",
                "view": "webcam",
                "image_path": front_path.relative_to(root).as_posix(),
                "facial_landmarks_xy": json.dumps(front_landmarks, separators=(",", ":")),
            }
        )
        rows.append(
            {
                **common,
                **side_annotations,
                "sample_id": f"{pair_id}-side",
                "view": "phonecam",
                "image_path": side_path.relative_to(root).as_posix(),
            }
        )

    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"dual-view demo root: {root}")
    print(f"manifest: {manifest_path}")
    print(f"subjects: {args.subjects}, images: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
