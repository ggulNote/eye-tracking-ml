"""Plot only the annotated 3x3 strict-profile head/eye pose grid."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plot_front_and_profile90_preprocessing import (
    _focus_crop,
    _load_profile_annotations,
    _profile_result,
    _rgb,
    _style_diagnostic_axis,
)

LEVELS = ("up", "middle", "low")


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__, parents=[])


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = _parser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=project_root.parent / "project_data/example",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=project_root / "configs/examples/profile90_pose_grid_annotations.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs/profile90_pose_grid",
    )
    return parser.parse_args()


def _fmt2(value: list[float] | np.ndarray) -> str:
    x_value, y_value = np.asarray(value, dtype=float)
    return f"[{x_value:+.3f}, {y_value:+.3f}]"


def _validate_grid(
    annotations: list[tuple[str, dict[str, Any]]],
) -> dict[tuple[str, str], tuple[str, dict[str, Any]]]:
    grid: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for sample_id, annotation in annotations:
        head_level = str(annotation.get("head_level"))
        eye_level = str(annotation.get("eye_level"))
        key = (head_level, eye_level)
        if head_level not in LEVELS or eye_level not in LEVELS:
            raise ValueError(f"{sample_id}: head_level and eye_level must be one of {LEVELS}")
        if key in grid:
            raise ValueError(f"duplicate pose cell {key}")
        grid[key] = (sample_id, annotation)
    expected = {(head, eye) for head in LEVELS for eye in LEVELS}
    if set(grid) != expected:
        raise ValueError(f"pose grid mismatch: missing={sorted(expected - set(grid))}")
    return grid


def _save_grid(
    output_dir: Path,
    results: dict[tuple[str, str], tuple[str, dict[str, np.ndarray], dict[str, Any]]],
) -> Path:
    figure = plt.figure(figsize=(22, 23), facecolor="white")
    outer = figure.add_gridspec(
        3,
        3,
        left=0.035,
        right=0.985,
        bottom=0.045,
        top=0.89,
        hspace=0.26,
        wspace=0.12,
    )
    figure.suptitle(
        "Strict-profile 3×3 pose grid — head direction, iris cue, and two eyelid angles",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.93,
        "magenta: H0 ear/tragion anchor → H1 nose tip  |  orange: iris vertical cue  |  "
        "blue: a0→a1  |  green: a0→a2  |  yellow/orange: directions from eye-local +x",
        ha="center",
        va="center",
        fontsize=13,
        color="#35465b",
    )

    for row, head_level in enumerate(LEVELS):
        for column, eye_level in enumerate(LEVELS):
            sample_id, panels, report = results[(head_level, eye_level)]
            nested = outer[row, column].subgridspec(
                3,
                1,
                height_ratios=(1.15, 0.72, 0.42),
                hspace=0.24,
            )
            head_axis = figure.add_subplot(nested[0, 0])
            eye_axis = figure.add_subplot(nested[1, 0])
            crop_axis = figure.add_subplot(nested[2, 0])
            face_bbox = report["face_bbox_xyxy"]
            eye_bbox = report["eye_bbox_xyxy"]
            head_vector = report["head_pose_2d"]
            pitch = float(report["head_pitch_proxy_degrees"])
            eye_pose = report["iris_pose_2d"]
            ear = float(report["ear"])
            tail_geometry = report["eyelid_tail_geometry"]
            direction_degrees = np.asarray(tail_geometry["direction_angles_degrees"])
            direction_normalized = np.asarray(tail_geometry["direction_angles_normalized_pi"])

            head_axis.imshow(_focus_crop(panels["head"], face_bbox, scale_xy=(1.04, 1.04)))
            head_axis.set_title(
                f"head={head_level} · eye={eye_level}\n"
                f"head_2d={_fmt2(head_vector)} · pitch proxy={pitch:+.1f}°",
                fontsize=12,
                fontweight="bold",
                color="#b000b0",
                pad=7,
            )
            _style_diagnostic_axis(head_axis, color="#d000d0")

            eye_axis.imshow(_focus_crop(panels["tail_vectors"], eye_bbox, scale_xy=(1.45, 2.25)))
            eye_axis.set_title(
                f"iris eye_y={float(eye_pose[1]):+.3f} · EAR={ear:.3f}\n"
                f"upper/lower={_fmt2(direction_degrees)}° · "
                f"feature /π={_fmt2(direction_normalized)}",
                fontsize=10,
                color="#7c5d00",
                pad=6,
            )
            _style_diagnostic_axis(eye_axis, color="#d7aa00")

            crop_axis.imshow(_rgb(panels["model_input"]), interpolation="nearest")
            crop_axis.set_title("final side_image [3,128,256] · no overlay", fontsize=9, pad=4)
            _style_diagnostic_axis(crop_axis, color="#2f78bd")
            crop_axis.set_ylabel(sample_id, fontsize=8, color="#526273")

    figure.text(
        0.5,
        0.015,
        "Manual demo annotations. The two eye-local direction angles are model features; "
        "the raw blue/green vectors are diagnostics only. "
        "The magenta arrow is projected 2D head direction, NOT calibrated 3D pose.",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#596779",
    )
    path = output_dir / "side_pose_3x3_vectors.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> int:
    args = _parse_args()
    input_root = args.input_root.expanduser().resolve(strict=True)
    annotation_path = args.annotations.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_grid = _validate_grid(_load_profile_annotations(annotation_path))
    results: dict[tuple[str, str], tuple[str, dict[str, np.ndarray], dict[str, Any]]] = {}
    report: dict[str, Any] = {
        "warning": "manual demo geometry; no calibrated screen gaze or 3D head pose",
        "point_mapping": {"a0": "p4/index3", "a1": "p3/index2", "a2": "p5/index4"},
        "samples": {},
        "plot_paths": {},
    }
    crop_paths: dict[str, str] = {}
    for head_level in LEVELS:
        for eye_level in LEVELS:
            sample_id, annotation = annotation_grid[(head_level, eye_level)]
            print(f"processing {sample_id}", flush=True)
            panels, sample_report = _profile_result(input_root, sample_id, annotation)
            if "tail_vectors" not in panels:
                raise RuntimeError(f"{sample_id}: eyelid-tail angle is unavailable")
            sample_report["head_level"] = head_level
            sample_report["eye_level"] = eye_level
            sample_report["face_bbox_xyxy"] = annotation["face_bbox_xyxy"]
            sample_report["eye_bbox_xyxy"] = annotation["visible_eye_bbox_xyxy"]
            results[(head_level, eye_level)] = (sample_id, panels, sample_report)
            report["samples"][sample_id] = sample_report
            crop_path = output_dir / f"{sample_id}_eye_crop_128x256.png"
            Image.fromarray(_rgb(panels["model_input"])).save(crop_path)
            crop_paths[sample_id] = str(crop_path)

    plot_path = _save_grid(output_dir, results)
    report["plot_paths"] = {
        "pose_grid": str(plot_path),
        "eye_crops": crop_paths,
    }
    report_path = output_dir / "side_pose_3x3_report.json"
    report_path.write_text(
        json.dumps(deepcopy(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"plot: {plot_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
