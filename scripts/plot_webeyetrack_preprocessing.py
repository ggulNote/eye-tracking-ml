"""Plot the actual front and side model-input preprocessing on example images."""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "gaze-pipeline-matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from gaze_pipeline.config import load_and_validate_config
from gaze_pipeline.data.transforms import GazePreprocessor
from gaze_pipeline.data.webeyetrack_compat import (
    LEFT_EAR_INDICES,
    RIGHT_EAR_INDICES,
    WEBEYETRACK_FACE_QUAD_INDICES,
    MediaPipeFaceLandmarkerDetector,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DISPLAY_STAGES = (
    "original",
    "landmarks",
    "ear",
    "head_pose",
    "model_input",
    "normalized",
)
STAGE_TITLES = {
    "original": "1. Original RGB",
    "landmarks": "2. MediaPipe 478 landmarks",
    "ear": "3. Eye selection + EAR",
    "head_pose": "4. 3D head pose",
    "model_input": "5. Model image input",
    "normalized": "6. Tensor after /255",
}


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=project_root / "configs/config.yaml")
    parser.add_argument(
        "--front-profile",
        type=Path,
        default=project_root / "configs/profiles/blazegaze.yaml",
    )
    parser.add_argument(
        "--dual-profile",
        type=Path,
        default=project_root / "configs/profiles/dual_view_images.yaml",
    )
    parser.add_argument(
        "--side-one-eye-profile",
        type=Path,
        default=project_root / "configs/profiles/side_one_eye.yaml",
    )
    parser.add_argument(
        "--side-full-face-profile",
        type=Path,
        default=project_root / "configs/profiles/side_full_face.yaml",
    )
    parser.add_argument("--model-asset", type=Path, required=True)
    parser.add_argument(
        "--only-profile",
        choices=("front_webeyetrack_exact", "side_one_eye", "side_full_face"),
    )
    return parser


def _display_rgb(image: Any) -> np.ndarray:
    if torch.is_tensor(image):
        array = image.detach().cpu().float().numpy()
        if array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))
        minimum = float(np.nanmin(array))
        maximum = float(np.nanmax(array))
        if minimum < 0.0 or maximum > 1.0:
            scale = maximum - minimum
            array = np.zeros_like(array) if scale <= 1e-12 else (array - minimum) / scale
        return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got {array.shape}")
    if array.dtype != np.uint8:
        maximum = float(np.nanmax(array))
        array = array * 255.0 if maximum <= 1.0 else array
        array = np.rint(np.clip(array, 0, 255)).astype(np.uint8)
    return np.ascontiguousarray(array)


def _preview(image: Any, maximum_size: tuple[int, int] = (960, 540)) -> np.ndarray:
    """Bound diagnostic image memory without changing preprocessing itself."""

    pil_image = Image.fromarray(_display_rgb(image))
    pil_image.thumbnail(maximum_size, resample=Image.Resampling.LANCZOS)
    return np.asarray(pil_image)


def _draw_landmarks(image: Any, sample: dict[str, Any]) -> np.ndarray:
    canvas = Image.fromarray(_display_rgb(image))
    draw = ImageDraw.Draw(canvas)
    points = np.asarray(sample.get("landmarks_xy"), dtype=float)
    radius = max(1, int(round(min(canvas.size) * 0.0025)))
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 230, 255))
    quad = points[np.asarray(WEBEYETRACK_FACE_QUAD_INDICES)]
    draw.line([tuple(point) for point in quad] + [tuple(quad[0])], fill=(255, 210, 0), width=4)
    draw.text((12, 12), f"landmarks={len(points)}", fill=(255, 255, 255), stroke_width=2)
    return np.asarray(canvas)


def _draw_failure(image: Any, message: str) -> np.ndarray:
    canvas = Image.fromarray(_display_rgb(image))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 70), fill=(120, 0, 0))
    draw.text(
        (12, 12),
        message,
        fill=(255, 255, 255),
        stroke_width=1,
    )
    return np.asarray(canvas)


def _draw_eye_state(image: Any, sample: dict[str, Any]) -> np.ndarray:
    canvas = Image.fromarray(_display_rgb(image))
    draw = ImageDraw.Draw(canvas)
    points = np.asarray(sample.get("landmarks_xy"), dtype=float)
    metadata = sample["metadata"]
    ears = np.asarray(metadata.get("ear", [np.nan, np.nan]), dtype=float)
    open_mask = np.asarray(metadata.get("eye_open_mask", [False, False]), dtype=bool)
    selected = str(metadata.get("selected_eye", "unknown"))
    for index, (eye, indices) in enumerate(
        (("left", LEFT_EAR_INDICES), ("right", RIGHT_EAR_INDICES))
    ):
        eye_points = points[np.asarray(indices)]
        color = (40, 230, 80) if open_mask[index] else (255, 70, 70)
        width = 6 if selected in {eye, "both"} else 2
        draw.line(
            [tuple(point) for point in eye_points] + [tuple(eye_points[0])],
            fill=color,
            width=width,
        )
    text = (
        f"selected={selected}  EAR L={ears[0]:.3f} R={ears[1]:.3f}  "
        f"valid={metadata.get('gaze_valid', True)}"
    )
    draw.text((12, 12), text, fill=(255, 255, 255), stroke_width=2)
    return np.asarray(canvas)


def _draw_head_pose(image: Any, sample: dict[str, Any]) -> np.ndarray:
    canvas = Image.fromarray(_display_rgb(image))
    draw = ImageDraw.Draw(canvas)
    metadata = sample["metadata"]
    points = np.asarray(sample.get("landmarks_xy"), dtype=float)
    face_rt = np.asarray(metadata.get("face_rt"), dtype=float)
    origin = points[4]
    bbox = np.asarray(metadata.get("face_bbox_xywh", [0, 0, *canvas.size]), dtype=float)
    length = max(35.0, float(bbox[2]) * 0.28)
    if face_rt.shape == (4, 4):
        rotation = face_rt[:3, :3]
        for axis_index, (color, label) in enumerate(
            (((255, 70, 70), "X"), ((70, 255, 70), "Y"), ((70, 150, 255), "Z"))
        ):
            vector = rotation[:, axis_index]
            endpoint = origin + np.asarray([vector[0], -vector[1]]) * length
            draw.line((tuple(origin), tuple(endpoint)), fill=color, width=6)
            draw.text(tuple(endpoint), label, fill=color, stroke_width=1)
    head_vector = np.asarray(metadata.get("head_vector", [np.nan] * 3), dtype=float)
    if np.isfinite(head_vector).all():
        endpoint = origin + np.asarray([head_vector[0], -head_vector[1]]) * length * 1.3
        draw.line((tuple(origin), tuple(endpoint)), fill=(255, 0, 255), width=5)
        draw.text(tuple(endpoint), "head", fill=(255, 0, 255), stroke_width=1)
    euler = np.asarray(metadata.get("head_euler_degrees", [np.nan] * 3), dtype=float)
    face_origin = np.asarray(metadata.get("face_origin_3d", [np.nan] * 3), dtype=float)
    draw.text(
        (12, 12),
        f"mapped pitch/yaw/roll={np.round(euler, 1).tolist()} deg",
        fill=(255, 255, 255),
        stroke_width=2,
    )
    draw.text(
        (12, 35),
        f"face_origin_3d={np.round(face_origin, 1).tolist()} cm",
        fill=(255, 255, 255),
        stroke_width=2,
    )
    return np.asarray(canvas)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str | bool | int | float):
        return value
    return str(value)


def _image_stats(image: Any) -> dict[str, Any]:
    if torch.is_tensor(image):
        array = image.detach().cpu().float()
        return {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
        }
    array = np.asarray(image)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


_REPORT_METADATA_KEYS = (
    "view",
    "gaze_valid",
    "invalid_reasons",
    "landmark_kind",
    "landmark_source",
    "detector_mirrored_retry",
    "face_bbox_xywh",
    "selected_eye",
    "eye_visibility_score",
    "eye_selection_valid",
    "ear",
    "eye_open_mask",
    "ear_threshold",
    "eye_state_valid",
    "gaze_state",
    "head_vector",
    "head_euler_degrees",
    "head_orientation_valid",
    "face_origin_3d",
    "face_origin_unit",
    "face_origin_source",
    "face_origin_valid",
    "head_pose_valid",
    "estimated_face_width_cm",
    "face_roi_xyxy",
    "face_roi_valid",
    "representation",
)


def _report_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep the report readable by omitting the raw 478-point detector payload."""

    return {
        key: _json_value(deepcopy(metadata[key]))
        for key in _REPORT_METADATA_KEYS
        if key in metadata
    }


def _run(
    preprocessor: GazePreprocessor,
    image_path: Path,
    view: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sample: dict[str, Any] = {
        "image_path": str(image_path),
        "view": view,
        "metadata": {"view": view, "gaze_valid": True, "invalid_reasons": []},
    }
    snapshots: dict[str, np.ndarray] = {}
    stage_stats: dict[str, Any] = {}
    for stage in preprocessor.stage_order:
        stage_config = preprocessor._stage_config(stage, view)
        if not bool(stage_config.get("enabled", True)):
            continue
        getattr(preprocessor, f"_stage_{stage}")(sample, stage_config)
        if stage == "decode":
            snapshots["original"] = _preview(sample["image"])
        elif stage == "face_landmarks":
            if sample.get("landmarks_xy") is None:
                reason = "; ".join(sample["metadata"].get("invalid_reasons", []))
                panel = _preview(_draw_failure(sample["image"], f"FaceLandmarker failed: {reason}"))
                for missing_stage in DISPLAY_STAGES[1:]:
                    snapshots[missing_stage] = panel
                return snapshots, {
                    "image_path": str(image_path),
                    "view": view,
                    "status": "invalid_face_landmarks",
                    "stages": stage_stats,
                    "metadata": _report_metadata(sample["metadata"]),
                }
            snapshots["landmarks"] = _preview(_draw_landmarks(sample["image"], sample))
        elif stage == "eye_state":
            snapshots["ear"] = _preview(_draw_eye_state(sample["image"], sample))
        elif stage == "metric_head_pose":
            snapshots["head_pose"] = _preview(_draw_head_pose(sample["image"], sample))
        elif stage == "eye_region_warp":
            snapshots["model_input"] = _preview(sample["image"])
        elif stage == "resize" and "model_input" not in snapshots:
            snapshots["model_input"] = _preview(sample["image"])
        elif stage == "normalize":
            snapshots["normalized"] = _preview(sample["image"])
        stage_stats[stage] = _image_stats(sample["image"])

    missing = [stage for stage in DISPLAY_STAGES if stage not in snapshots]
    if missing:
        raise RuntimeError(f"{image_path.name}: missing plot stages {missing}")
    report = {
        "image_path": str(image_path),
        "view": view,
        "stages": stage_stats,
        "metadata": _report_metadata(sample["metadata"]),
    }
    return snapshots, report


def _save_individual(
    output_dir: Path,
    profile_name: str,
    image_path: Path,
    snapshots: dict[str, np.ndarray],
) -> None:
    target = output_dir / "stages" / profile_name / image_path.stem
    target.mkdir(parents=True, exist_ok=True)
    for index, stage in enumerate(DISPLAY_STAGES, start=1):
        Image.fromarray(snapshots[stage]).save(target / f"{index:02d}_{stage}.png")


def _plot(
    output_dir: Path,
    profile_name: str,
    results: list[tuple[Path, dict[str, np.ndarray]]],
) -> Path:
    figure, axes = plt.subplots(
        len(results),
        len(DISPLAY_STAGES),
        figsize=(20, 4.2 * len(results)),
        squeeze=False,
    )
    figure.suptitle(profile_name.replace("_", " "), fontsize=18, y=0.995)
    for row, (image_path, snapshots) in enumerate(results):
        for column, stage in enumerate(DISPLAY_STAGES):
            axis = axes[row, column]
            axis.imshow(snapshots[stage])
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(STAGE_TITLES[stage], fontsize=10)
            if column == 0:
                axis.set_ylabel(image_path.name, fontsize=10)
    figure.tight_layout(rect=(0.01, 0.01, 1.0, 0.96))
    path = output_dir / f"{profile_name}.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _config(
    config_path: Path,
    profiles: tuple[Path, ...],
    environment: dict[str, str],
) -> dict[str, Any]:
    return load_and_validate_config(
        config_path,
        profiles=profiles,
        environ=environment,
        check_paths=False,
    )


def main() -> int:
    args = _parser().parse_args()
    input_root = args.input_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    model_asset = args.model_asset.expanduser().resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    environment["GAZE_DATA_ROOT"] = str(input_root)
    environment["DUAL_VIEW_MANIFEST"] = str(input_root / "unused_manifest.csv")
    environment["MEDIAPIPE_FACE_MODEL"] = str(model_asset)

    jobs = (
        (
            "front_webeyetrack_exact",
            "front",
            _config(args.config, (args.front_profile,), environment),
        ),
        (
            "side_one_eye",
            "side",
            _config(
                args.config,
                (args.dual_profile, args.side_one_eye_profile),
                environment,
            ),
        ),
        (
            "side_full_face",
            "side",
            _config(
                args.config,
                (args.dual_profile, args.side_full_face_profile),
                environment,
            ),
        ),
    )
    report: dict[str, Any] = {
        "input_root": str(input_root),
        "mediapipe_model": str(model_asset),
        "profiles": {},
    }
    paths: list[Path] = []
    detector = MediaPipeFaceLandmarkerDetector(
        {"model_asset_path": str(model_asset), "mirror_retry": True}
    )
    try:
        for profile_name, view, config in jobs:
            if args.only_profile and args.only_profile != profile_name:
                continue
            preprocessor = GazePreprocessor(
                config,
                split="validation",
                landmark_detector=detector.detect,
            )
            image_paths = sorted(
                path
                for path in (input_root / view).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not image_paths:
                raise FileNotFoundError(f"no images found under {input_root / view}")
            plot_results: list[tuple[Path, dict[str, np.ndarray]]] = []
            sample_reports: list[dict[str, Any]] = []
            for image_path in image_paths:
                print(f"processing: {profile_name}/{image_path.name}", flush=True)
                snapshots, sample_report = _run(preprocessor, image_path, view)
                _save_individual(output_dir, profile_name, image_path, snapshots)
                plot_results.append((image_path, snapshots))
                sample_reports.append(sample_report)
            paths.append(_plot(output_dir, profile_name, plot_results))
            report["profiles"][profile_name] = sample_reports
            del preprocessor, plot_results
            gc.collect()
    finally:
        detector.close()

    report_path = output_dir / "webeyetrack_preprocessing_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for path in paths:
        print(f"plot: {path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
