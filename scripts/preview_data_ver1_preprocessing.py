"""Preview real ``data(ver1)`` pairs without turning dummy geometry into labels.

The script selects one or two paired rows from each participant's
``labels/image_samples.csv`` and writes local-only diagnostic plots below the
project ``outputs/`` directory.  It never edits or copies over a raw image.

Front panels use the configured BlazeGaze/WebEyeTrack ``GazePreprocessor``
when a runtime MediaPipe detection succeeds.  MediaPipe is isolated in a child
process because some macOS/headless builds abort the process while creating a
graphics service; an abort therefore becomes a reportable preview failure
instead of terminating the whole run.

``data(ver1)`` does not contain the six strict-profile eyelid points, iris
center, or 2-D head anchors required by ``side_profile_90.yaml``.  For a visual
wiring check only, the side branch may derive those points from a runtime
MediaPipe result or from an OpenCV eye box plus deterministic synthetic
geometry.  Such rows are always marked ``eye_annotation_valid=false`` and
``training_eligible=false``.  The strict Dataset transform is intentionally
not invoked with them; only its pure ``preprocess_profile_side`` geometry
function is executed to make the preview patch.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "gaze-pipeline-matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from gaze_pipeline.config import load_and_validate_config
from gaze_pipeline.data.profile_side import preprocess_profile_side
from gaze_pipeline.data.transforms import GazePreprocessor
from gaze_pipeline.data.webeyetrack_compat import (
    LEFT_EAR_INDICES,
    LEFT_IRIS_INDICES,
    NOSE_INDEX,
    RIGHT_EAR_INDICES,
    RIGHT_IRIS_INDICES,
    WEBEYETRACK_FACE_QUAD_INDICES,
    MediaPipeFaceLandmarkerDetector,
    select_eye,
)

PANEL_KEYS = (
    "original",
    "annotation",
    "ear",
    "head",
    "eye_cue",
    "model_input",
    "normalized",
)
PANEL_TITLES = {
    "original": "1. Real RGB",
    "annotation": "2. Landmark / annotation",
    "ear": "3. Eye + EAR",
    "head": "4. Head cue",
    "eye_cue": "5. Eye cue",
    "model_input": "6. Model image input",
    "normalized": "7. Normalized tensor",
}
PROFILE_EAR_THRESHOLD = 0.20
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class SelectedPair:
    """One verified webcam/phonecam row selected from a capture manifest."""

    subject_id: str
    sample_id: str
    pair_id: str
    participant_root: Path
    source_manifest: Path
    front_path: Path
    side_path: Path
    target_x_px: float
    target_y_px: float
    target_x_centered: float
    target_y_centered: float
    screen_width_px: int
    screen_height_px: int
    phone_position: str
    row: Mapping[str, str]
    calibration_audit: Mapping[str, Any]


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root.parent / "project_data/data(ver1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs/preprocessing_ver1_preview",
    )
    parser.add_argument("--config", type=Path, default=project_root / "configs/config.yaml")
    parser.add_argument(
        "--front-profile",
        type=Path,
        default=project_root / "configs/profiles/blazegaze.yaml",
    )
    parser.add_argument(
        "--model-asset",
        type=Path,
        default=project_root / "models/face_landmarker_v2_with_blendshapes.task",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="total paired samples to select (one per participant before a second round)",
    )
    parser.add_argument(
        "--participants",
        nargs="*",
        help="optional participant directory names such as p00 p03",
    )
    parser.add_argument(
        "--mediapipe",
        choices=("auto", "required", "off"),
        default="auto",
        help="run isolated MediaPipe, require it, or skip it",
    )
    parser.add_argument(
        "--side-annotation-source",
        choices=("auto", "mediapipe", "dummy"),
        default="auto",
        help="auto prefers MediaPipe and otherwise uses preview-only dummy geometry",
    )
    parser.add_argument("--mediapipe-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--overwrite-generated",
        action="store_true",
        help="replace known files in the output directory; raw data is never touched",
    )
    return parser


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_preview_paths(data_root: Path, output_dir: Path, project_root: Path) -> None:
    """Reject any output capable of overlapping the raw dataset."""

    data_root = data_root.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    project_root = project_root.resolve(strict=True)
    output_root = (project_root / "outputs").resolve(strict=False)
    if not _inside(output_dir, output_root) or output_dir == output_root:
        raise ValueError(f"output_dir must be a child of project outputs/: {output_root}")
    if _inside(output_dir, data_root) or _inside(data_root, output_dir):
        raise ValueError("output_dir and data_root must not overlap")


def _float(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    try:
        return float(raw) if raw != "" else default
    except (TypeError, ValueError):
        return default


def _read_participant_metadata(participant_root: Path) -> tuple[dict[str, Any], str]:
    path = participant_root / "participant.json"
    document = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    screen = document.get("screen", {}) if isinstance(document, Mapping) else {}
    calibration = document.get("calibration_assets", {}) if isinstance(document, Mapping) else {}
    cameras = calibration.get("cameras", {}) if isinstance(calibration, Mapping) else {}
    phone = cameras.get("iphone_left", {}) if isinstance(cameras, Mapping) else {}
    phone_camera = phone.get("camera", {}) if isinstance(phone, Mapping) else {}
    phone_monitor = phone.get("monitor_pose", {}) if isinstance(phone, Mapping) else {}
    stereo = calibration.get("stereo", {}) if isinstance(calibration, Mapping) else {}
    screen_asset = calibration.get("screen_size", {}) if isinstance(calibration, Mapping) else {}
    phone_position = "unknown"
    for camera in document.get("camera_config", []) if isinstance(document, Mapping) else []:
        if isinstance(camera, Mapping) and str(camera.get("role", "")).startswith("iphone"):
            phone_position = str(camera.get("position", "unknown"))
            break
    audit = {
        "capture_geometry_mode": (
            document.get("geometry", {}).get("mode")
            if isinstance(document.get("geometry", {}), Mapping)
            else None
        ),
        "required_assets_valid": bool(calibration.get("required_assets_valid", False)),
        "camera_intrinsics_valid": bool(calibration.get("camera_intrinsics_valid", False)),
        "phone_camera_intrinsics_valid": bool(phone_camera.get("valid", False)),
        "phone_monitor_pose_valid": bool(phone_monitor.get("valid", False)),
        "screen_size_asset_valid": bool(screen_asset.get("valid", False)),
        "stereo_valid": bool(stereo.get("valid", False)),
        "screen_width_px": int(screen.get("canvas_width_pixel", 1920)),
        "screen_height_px": int(screen.get("canvas_height_pixel", 1080)),
        "camera_intrinsics_consumed_by_preview": False,
    }
    return audit, phone_position


def _resolve_raw_path(data_root: Path, participant_root: Path, relative: str) -> Path:
    path = (participant_root / relative).resolve(strict=True)
    if not _inside(path, data_root):
        raise ValueError(f"raw manifest path escapes data_root: {relative!r}")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"unsupported raw image extension: {path.name}")
    return path


def discover_selected_pairs(
    data_root: Path,
    *,
    count: int,
    participant_names: Sequence[str] | None = None,
) -> list[SelectedPair]:
    """Select quality-ranked rows, round-robin across participants."""

    if count < 1 or count > 2:
        raise ValueError("--samples must be 1 or 2 for this privacy-bounded preview")
    data_root = data_root.resolve(strict=True)
    allow = set(participant_names or ())
    candidates: list[list[SelectedPair]] = []
    for participant_root in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if allow and participant_root.name not in allow:
            continue
        manifest = participant_root / "labels/image_samples.csv"
        if not manifest.is_file():
            continue
        audit, phone_position = _read_participant_metadata(participant_root)
        participant_pairs: list[SelectedPair] = []
        with manifest.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                front_relative = str(row.get("webcam_image", "")).strip()
                side_relative = str(row.get("phonecam_image", "")).strip()
                if not front_relative or not side_relative:
                    continue
                try:
                    front_path = _resolve_raw_path(data_root, participant_root, front_relative)
                    side_path = _resolve_raw_path(data_root, participant_root, side_relative)
                except (FileNotFoundError, ValueError):
                    continue
                participant_pairs.append(
                    SelectedPair(
                        subject_id=participant_root.name,
                        sample_id=str(row.get("sample", front_path.stem)),
                        pair_id=f"{participant_root.name}:{row.get('pair', front_path.stem)}",
                        participant_root=participant_root,
                        source_manifest=manifest,
                        front_path=front_path,
                        side_path=side_path,
                        target_x_px=_float(row, "x_px"),
                        target_y_px=_float(row, "y_px"),
                        target_x_centered=_float(row, "x_centered"),
                        target_y_centered=_float(row, "y_centered"),
                        screen_width_px=int(audit["screen_width_px"]),
                        screen_height_px=int(audit["screen_height_px"]),
                        phone_position=phone_position,
                        row=dict(row),
                        calibration_audit=dict(audit),
                    )
                )
        participant_pairs.sort(
            key=lambda item: (
                int(_float(item.row, "webcam_mediapipe_iris_detected")),
                int(_float(item.row, "webcam_mediapipe_face_detected")),
                min(
                    _float(item.row, "webcam_sharpness"),
                    _float(item.row, "phonecam_sharpness"),
                ),
                item.sample_id,
            ),
            reverse=True,
        )
        if participant_pairs:
            candidates.append(participant_pairs)
    if not candidates:
        raise FileNotFoundError(f"no paired image_samples.csv rows found under {data_root}")

    selected: list[SelectedPair] = []
    round_index = 0
    while len(selected) < count:
        added = False
        for participant_pairs in candidates:
            if round_index < len(participant_pairs):
                selected.append(participant_pairs[round_index])
                added = True
                if len(selected) == count:
                    return selected
        if not added:
            break
        round_index += 1
    if len(selected) < count:
        raise ValueError(f"requested {count} pairs but only found {len(selected)}")
    return selected


def _mediapipe_worker(
    image_path: str,
    model_asset: str,
    view: str,
    queue: Any,
) -> None:
    """Child target: a native MediaPipe abort must not kill the preview parent."""

    detector: MediaPipeFaceLandmarkerDetector | None = None
    try:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        detector = MediaPipeFaceLandmarkerDetector(
            {
                "model_asset_path": model_asset,
                "mirror_retry": True,
                "min_face_detection_confidence": 0.35,
                "min_face_presence_confidence": 0.35,
            }
        )
        result = detector.detect(image, view)
        queue.put(("ok", dict(result)))
    except BaseException as exc:  # child boundary must serialize all Python failures
        queue.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if detector is not None:
            detector.close()


def detect_mediapipe_isolated(
    image_path: Path,
    model_asset: Path,
    view: str,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return a detector mapping or a safe failure description."""

    if timeout_seconds <= 0:
        raise ValueError("mediapipe timeout must be positive")
    context = mp.get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_mediapipe_worker,
        args=(str(image_path), str(model_asset), view, queue),
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        return None, f"MediaPipe timed out after {timeout_seconds:g}s"
    try:
        status, payload = queue.get(timeout=0.5)
    except Empty:
        return None, f"MediaPipe child exited with code {process.exitcode} without a result"
    finally:
        queue.close()
        queue.join_thread()
    if status == "error":
        return None, str(payload)
    return dict(payload), None


def _rgb(image: Any) -> np.ndarray:
    if torch.is_tensor(image):
        array = image.detach().cpu().float().numpy()
        if array.ndim == 3 and array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))
        minimum = float(np.nanmin(array))
        maximum = float(np.nanmax(array))
        if minimum < 0 or maximum > 1:
            span = maximum - minimum
            array = np.zeros_like(array) if span <= 1e-12 else (array - minimum) / span
        return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected RGB HWC or CHW tensor, got {array.shape}")
    if array.dtype != np.uint8:
        maximum = float(np.nanmax(array))
        array = array * 255 if maximum <= 1 else array
        array = np.rint(np.clip(array, 0, 255)).astype(np.uint8)
    return np.ascontiguousarray(array)


def _font(canvas: Image.Image, scale: float = 0.018) -> ImageFont.ImageFont:
    size = max(13, int(round(min(canvas.size) * scale)))
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    canvas: Image.Image,
) -> None:
    draw.text(
        xy,
        value,
        font=_font(canvas),
        fill=(255, 255, 255),
        stroke_width=max(1, int(round(min(canvas.size) * 0.0015))),
        stroke_fill=(0, 0, 0),
    )


def _banner(image: Any, text: str, *, color: tuple[int, int, int] = (140, 0, 0)) -> np.ndarray:
    canvas = Image.fromarray(_rgb(image))
    draw = ImageDraw.Draw(canvas)
    height = max(42, int(round(canvas.height * 0.075)))
    draw.rectangle((0, 0, canvas.width, height), fill=color)
    _text(draw, (12, 8), text, canvas)
    return np.asarray(canvas)


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: Sequence[float],
    end: Sequence[float],
    color: tuple[int, int, int],
    width: int,
) -> None:
    start_xy = np.asarray(start, dtype=float)
    end_xy = np.asarray(end, dtype=float)
    draw.line((tuple(start_xy), tuple(end_xy)), fill=color, width=width)
    delta = end_xy - start_xy
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-8:
        return
    unit = delta / norm
    normal = np.asarray([-unit[1], unit[0]])
    size = max(8.0, width * 3.0)
    draw.polygon(
        [
            tuple(end_xy),
            tuple(end_xy - unit * size + normal * size * 0.45),
            tuple(end_xy - unit * size - normal * size * 0.45),
        ],
        fill=color,
    )


def _front_panels(
    preprocessor: GazePreprocessor,
    pair: SelectedPair,
    detection: Mapping[str, Any] | None,
    detection_error: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    original = np.asarray(Image.open(pair.front_path).convert("RGB"))
    if detection is None:
        reason = detection_error or "MediaPipe was disabled"
        unavailable = _banner(original, f"NOT EXECUTED: {reason}")
        return (
            {key: (original if key == "original" else unavailable) for key in PANEL_KEYS},
            {
                "status": "not_executed_no_mediapipe",
                "reason": reason,
                "training_eligible": False,
                "real_stages": ["decode", "target_label"],
                "not_executed_stages": [
                    "face_landmarks",
                    "eye_state",
                    "metric_head_pose",
                    "eye_region_warp",
                    "normalize",
                ],
            },
        )

    sample: dict[str, Any] = {
        "image_path": str(pair.front_path),
        "view": "front",
        "target_gaze_xy": np.asarray([pair.target_x_centered, pair.target_y_centered]),
        "metadata": {"view": "front", "gaze_valid": True, "invalid_reasons": []},
    }
    panels: dict[str, np.ndarray] = {}
    preprocessor.landmark_detector = lambda _image, _view: detection
    for stage in preprocessor.stage_order:
        config = preprocessor._stage_config(stage, "front")
        if not bool(config.get("enabled", True)):
            continue
        getattr(preprocessor, f"_stage_{stage}")(sample, config)
        image = _rgb(sample["image"])
        if stage == "decode":
            panels["original"] = image.copy()
        elif stage == "face_landmarks":
            canvas = Image.fromarray(image)
            draw = ImageDraw.Draw(canvas)
            points = np.asarray(sample["landmarks_xy"], dtype=float)
            radius = max(1, int(round(min(canvas.size) * 0.002)))
            for x, y in points:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 230, 255))
            quad = points[np.asarray(WEBEYETRACK_FACE_QUAD_INDICES)]
            draw.line(
                [tuple(point) for point in quad] + [tuple(quad[0])],
                fill=(255, 210, 0),
                width=4,
            )
            _text(draw, (12, 12), "REAL runtime MediaPipe 478", canvas)
            panels["annotation"] = np.asarray(canvas)
        elif stage == "eye_state":
            canvas = Image.fromarray(image)
            draw = ImageDraw.Draw(canvas)
            points = np.asarray(sample["landmarks_xy"], dtype=float)
            metadata = sample["metadata"]
            ears = np.asarray(metadata.get("ear", [np.nan, np.nan]), dtype=float)
            open_mask = np.asarray(metadata.get("eye_open_mask", [False, False]), dtype=bool)
            for index, indices in enumerate((LEFT_EAR_INDICES, RIGHT_EAR_INDICES)):
                eye = points[np.asarray(indices)]
                color = (40, 230, 80) if open_mask[index] else (255, 70, 70)
                draw.line([tuple(point) for point in eye] + [tuple(eye[0])], fill=color, width=5)
            _text(draw, (12, 12), f"REAL EAR L={ears[0]:.3f} R={ears[1]:.3f}", canvas)
            panels["ear"] = np.asarray(canvas)
            iris_centers = [
                points[np.asarray(LEFT_IRIS_INDICES)].mean(axis=0),
                points[np.asarray(RIGHT_IRIS_INDICES)].mean(axis=0),
            ]
            cue = Image.fromarray(image)
            cue_draw = ImageDraw.Draw(cue)
            for center in iris_centers:
                r = max(3, int(round(min(cue.size) * 0.006)))
                cue_draw.ellipse(
                    (center[0] - r, center[1] - r, center[0] + r, center[1] + r),
                    outline=(255, 165, 0),
                    width=3,
                )
            _text(cue_draw, (12, 12), "REAL iris landmarks; NOT calibrated screen gaze", cue)
            panels["eye_cue"] = np.asarray(cue)
        elif stage == "metric_head_pose":
            canvas = Image.fromarray(image)
            draw = ImageDraw.Draw(canvas)
            points = np.asarray(sample["landmarks_xy"], dtype=float)
            origin = points[NOSE_INDEX]
            vector = np.asarray(sample["metadata"].get("head_vector", [np.nan] * 3), dtype=float)
            bbox = np.asarray(sample["metadata"].get("face_bbox_xywh", [0, 0, 200, 200]))
            if np.isfinite(vector).all():
                endpoint = origin + np.asarray([vector[0], -vector[1]]) * max(120.0, bbox[2] * 0.8)
                _arrow(draw, origin, endpoint, (255, 0, 255), 6)
            euler = np.asarray(sample["metadata"].get("head_euler_degrees", [np.nan] * 3))
            _text(draw, (12, 12), f"runtime pose={np.round(euler, 1).tolist()} deg", canvas)
            _text(
                draw,
                (12, 45),
                "face origin is reconstructed; Camera.mat is not consumed",
                canvas,
            )
            panels["head"] = np.asarray(canvas)
        elif stage == "eye_region_warp":
            panels["model_input"] = image.copy()
        elif stage == "normalize":
            panels["normalized"] = _banner(
                sample["image"],
                "ACTUAL normalize: float32 CHW in [0,1]",
                color=(0, 95, 65),
            )

    missing = [key for key in PANEL_KEYS if key not in panels]
    if missing:
        raise RuntimeError(f"{pair.subject_id}/{pair.sample_id}: missing front panels {missing}")
    metadata = sample["metadata"]
    return panels, {
        "status": "executed_actual_transform",
        "training_eligible": bool(metadata.get("gaze_valid", False)),
        "detector": "runtime_mediapipe_face_landmarker_478",
        "real_stages": [
            "decode",
            "face_landmarks",
            "eye_selection",
            "eye_state",
            "metric_head_pose_reconstruction",
            "webeyetrack_eye_patch",
            "normalize",
            "target_label",
        ],
        "camera_intrinsics_consumed": False,
        "face_origin_source": metadata.get("face_origin_source"),
        "gaze_valid": bool(metadata.get("gaze_valid", False)),
        "invalid_reasons": list(metadata.get("invalid_reasons", [])),
        "model_input_shape_hwc": list(np.asarray(panels["model_input"]).shape),
        "normalized_tensor_shape_chw": list(sample["image"].shape),
        "target_gaze_xy_centered": [pair.target_x_centered, pair.target_y_centered],
    }


def _clip_point(point: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(
        [np.clip(point[0], 0, width - 1), np.clip(point[1], 0, height - 1)],
        dtype=np.float32,
    )


def _annotation_from_mediapipe(
    image: np.ndarray,
    detection: Mapping[str, Any],
) -> dict[str, Any]:
    points = np.asarray(detection["landmarks_xy"], dtype=np.float32)
    presence = detection.get("landmark_presence")
    eye, scores = select_eye(points, mode="best_visible", presence=presence)
    eyelid_indices = LEFT_EAR_INDICES if eye == "left" else RIGHT_EAR_INDICES
    iris_indices = LEFT_IRIS_INDICES if eye == "left" else RIGHT_IRIS_INDICES
    eyelids = points[np.asarray(eyelid_indices)]
    iris_center = points[np.asarray(iris_indices)].mean(axis=0)
    minimum = np.minimum(eyelids.min(axis=0), iris_center)
    maximum = np.maximum(eyelids.max(axis=0), iris_center)
    size = np.maximum(maximum - minimum, [8.0, 8.0])
    center = (minimum + maximum) / 2.0
    bbox_min = center - size * np.asarray([0.85, 1.20])
    bbox_max = center + size * np.asarray([0.85, 1.20])
    height, width = image.shape[:2]
    bbox = np.asarray(
        [
            np.clip(bbox_min[0], 0, width - 1),
            np.clip(bbox_min[1], 0, height - 1),
            np.clip(bbox_max[0], 1, width),
            np.clip(bbox_max[1], 1, height),
        ],
        dtype=np.float32,
    )
    face_bbox = np.asarray(detection.get("face_bbox_xywh"), dtype=np.float32)
    face_center = face_bbox[:2] + face_bbox[2:] / 2.0
    nose = points[NOSE_INDEX]
    if float(np.linalg.norm(nose - face_center)) <= 1e-6:
        nose = face_center + np.asarray([-max(20.0, face_bbox[2] * 0.2), 0.0])
    return {
        "visible_eye": eye,
        "visible_eye_bbox_xyxy": bbox,
        "visible_eye_keypoints_xy": eyelids,
        "iris_center_xy": iris_center,
        "profile_head_origin_xy": _clip_point(face_center, width, height),
        "profile_head_forward_xy": _clip_point(nose, width, height),
        "face_bbox_xyxy": np.asarray(
            [
                face_bbox[0],
                face_bbox[1],
                face_bbox[0] + face_bbox[2],
                face_bbox[1] + face_bbox[3],
            ],
            dtype=np.float32,
        ),
        "eye_annotation_valid": False,
        "training_eligible": False,
        "annotation_source": "runtime_mediapipe_proxy_not_human_verified",
        "visible_eye_scores": scores,
        "dummy_components": ["head_origin", "head_forward_semantics"],
    }


def _side_facing(phone_position: str) -> str:
    value = phone_position.lower()
    if "right" in value:
        return "right"
    return "left"


def _detect_eye_box(image: np.ndarray) -> np.ndarray | None:
    """Find a plausible profile eye; this is a preview aid, not annotation."""

    import cv2  # type: ignore

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    classifier = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml"))
    if classifier.empty():
        return None
    height, width = gray.shape
    detections = classifier.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(max(18, int(min(width, height) * 0.025)),) * 2,
        maxSize=(max(40, int(min(width, height) * 0.18)),) * 2,
    )
    plausible: list[tuple[float, np.ndarray]] = []
    target = np.asarray([width * 0.5, height * 0.34])
    diagonal = float(np.linalg.norm([width, height]))
    for x, y, box_width, box_height in detections:
        center = np.asarray([x + box_width / 2.0, y + box_height / 2.0])
        if not (width * 0.15 <= center[0] <= width * 0.85):
            continue
        if not (height * 0.08 <= center[1] <= height * 0.62):
            continue
        score = float(box_width * box_height) * (1.5 - np.linalg.norm(center - target) / diagonal)
        plausible.append((score, np.asarray([x, y, x + box_width, y + box_height], dtype=float)))
    return max(plausible, key=lambda item: item[0])[1] if plausible else None


def _dummy_side_annotation(
    image: np.ndarray,
    *,
    phone_position: str,
) -> dict[str, Any]:
    """Create conspicuously invalid geometry for a visual contract smoke test."""

    height, width = image.shape[:2]
    detected_bbox = _detect_eye_box(image)
    if detected_bbox is None:
        center = np.asarray([width * 0.5, height * 0.34])
        detector_width = max(30.0, min(width, height) * 0.075)
        provenance = "full_frame_proportional_dummy"
    else:
        center = np.asarray(
            [
                (detected_bbox[0] + detected_bbox[2]) / 2.0,
                (detected_bbox[1] + detected_bbox[3]) / 2.0,
            ]
        )
        detector_width = float(detected_bbox[2] - detected_bbox[0])
        provenance = "opencv_haar_eye_bbox_plus_synthetic_geometry"
    eye_width = max(18.0, detector_width * 0.72)
    opening = max(6.0, eye_width * 0.22)
    eyelids = np.asarray(
        [
            center + [-eye_width / 2, 0],
            center + [-eye_width / 6, -opening / 2],
            center + [eye_width / 6, -opening / 2],
            center + [eye_width / 2, 0],
            center + [eye_width / 6, opening / 2],
            center + [-eye_width / 6, opening / 2],
        ],
        dtype=np.float32,
    )
    facing = _side_facing(phone_position)
    direction = -1.0 if facing == "left" else 1.0
    head_origin = center + np.asarray([-direction * 2.8 * eye_width, 0.6 * eye_width])
    head_forward = center + np.asarray([direction * 2.0 * eye_width, 1.0 * eye_width])
    face_min = center + np.asarray(
        [-6.0 * eye_width if facing == "left" else -5.0 * eye_width, -3.0 * eye_width]
    )
    face_max = center + np.asarray(
        [5.0 * eye_width if facing == "left" else 6.0 * eye_width, 5.0 * eye_width]
    )
    face_bbox = np.asarray(
        [
            np.clip(face_min[0], 0, width - 1),
            np.clip(face_min[1], 0, height - 1),
            np.clip(face_max[0], 1, width),
            np.clip(face_max[1], 1, height),
        ],
        dtype=np.float32,
    )
    eye_bbox = np.asarray(
        [
            np.clip(center[0] - eye_width * 0.75, 0, width - 1),
            np.clip(center[1] - eye_width * 0.55, 0, height - 1),
            np.clip(center[0] + eye_width * 0.75, 1, width),
            np.clip(center[1] + eye_width * 0.55, 1, height),
        ],
        dtype=np.float32,
    )
    return {
        "visible_eye": "unknown",
        "visible_eye_bbox_xyxy": eye_bbox,
        "visible_eye_keypoints_xy": eyelids,
        "iris_center_xy": center.astype(np.float32),
        "profile_head_origin_xy": _clip_point(head_origin, width, height),
        "profile_head_forward_xy": _clip_point(head_forward, width, height),
        "face_bbox_xyxy": face_bbox,
        "eye_annotation_valid": False,
        "training_eligible": False,
        "annotation_source": provenance,
        "side_facing_assumption": facing,
        "dummy_components": [
            "eyelid_keypoints",
            "iris_center",
            "head_origin",
            "head_forward",
            "visible_eye_semantics",
        ],
    }


def _side_panels(
    pair: SelectedPair,
    detection: Mapping[str, Any] | None,
    *,
    annotation_source: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    image = np.asarray(Image.open(pair.side_path).convert("RGB"))
    use_mediapipe = detection is not None and annotation_source != "dummy"
    if annotation_source == "mediapipe" and detection is None:
        raise RuntimeError(f"{pair.subject_id}/{pair.sample_id}: side MediaPipe was required")
    annotation = (
        _annotation_from_mediapipe(image, detection)
        if use_mediapipe
        else _dummy_side_annotation(image, phone_position=pair.phone_position)
    )
    result = preprocess_profile_side(
        image,
        eye_bbox_xyxy=annotation["visible_eye_bbox_xyxy"],
        eyelid_keypoints_xy=annotation["visible_eye_keypoints_xy"],
        iris_center_xy=annotation["iris_center_xy"],
        head_origin_xy=annotation["profile_head_origin_xy"],
        head_forward_point_xy=annotation["profile_head_forward_xy"],
        output_size_hw=(128, 256),
        crop_mode="affine",
        vertical_only=True,
        eyelid_tail_indices=(3, 2, 4),
    )
    watermark = (
        "MEDIAPIPE PROXY / NOT HUMAN VERIFIED / NOT TRAINING DATA"
        if use_mediapipe
        else "DUMMY GEOMETRY / NOT TRAINING DATA"
    )

    annotation_canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(annotation_canvas)
    width = max(3, int(round(min(annotation_canvas.size) * 0.004)))
    draw.rectangle(tuple(annotation["face_bbox_xyxy"]), outline=(255, 210, 0), width=width)
    draw.rectangle(tuple(annotation["visible_eye_bbox_xyxy"]), outline=(0, 255, 255), width=width)
    eyelids = np.asarray(annotation["visible_eye_keypoints_xy"], dtype=float)
    draw.line(
        [tuple(point) for point in eyelids] + [tuple(eyelids[0])],
        fill=(0, 255, 0),
        width=width,
    )
    source_quad = np.asarray(result.source_quad_xy, dtype=float)
    draw.line(
        [tuple(point) for point in source_quad] + [tuple(source_quad[0])],
        fill=(255, 255, 255),
        width=width,
    )
    iris = np.asarray(annotation["iris_center_xy"], dtype=float)
    draw.ellipse(
        (
            iris[0] - width * 2,
            iris[1] - width * 2,
            iris[0] + width * 2,
            iris[1] + width * 2,
        ),
        outline=(255, 60, 255),
        width=width,
    )

    ear_canvas = Image.fromarray(image)
    ear_draw = ImageDraw.Draw(ear_canvas)
    ear_draw.line(
        [tuple(point) for point in eyelids] + [tuple(eyelids[0])],
        fill=(255, 70, 70),
        width=width + 1,
    )
    _text(ear_draw, (12, 55), f"preview EAR={result.ear:.3f}; annotation_valid=False", ear_canvas)

    head_canvas = Image.fromarray(image)
    head_draw = ImageDraw.Draw(head_canvas)
    head_origin = np.asarray(annotation["profile_head_origin_xy"], dtype=float)
    head_forward = np.asarray(annotation["profile_head_forward_xy"], dtype=float)
    _arrow(head_draw, head_origin, head_forward, (255, 0, 255), width + 2)
    _text(head_draw, (12, 55), "2-D proxy only; Camera.mat/monitor pose not used", head_canvas)

    cue_canvas = Image.fromarray(image)
    cue_draw = ImageDraw.Draw(cue_canvas)
    center = np.asarray(result.eyelid_center_xy, dtype=float)
    eye_pose = np.asarray(result.eye_pose_2d, dtype=float)
    endpoint = center + np.asarray([0.0, eye_pose[1] * result.eye_size_xy[1]])
    _arrow(cue_draw, center, endpoint, (255, 165, 0), width + 1)
    _text(
        cue_draw,
        (12, 55),
        f"preview eye cue={np.round(eye_pose, 3).tolist()}; NOT gaze",
        cue_canvas,
    )

    panels = {
        "original": _banner(
            image,
            "REAL phonecam RGB + REAL paired screen target",
            color=(0, 95, 65),
        ),
        "annotation": _banner(np.asarray(annotation_canvas), watermark),
        "ear": _banner(np.asarray(ear_canvas), watermark),
        "head": _banner(np.asarray(head_canvas), watermark),
        "eye_cue": _banner(np.asarray(cue_canvas), watermark),
        "model_input": _banner(np.asarray(result.patch), watermark),
        "normalized": _banner(
            torch.from_numpy(np.ascontiguousarray(result.patch.transpose(2, 0, 1))).float() / 255.0,
            "PREVIEW normalize only / NOT TRAINING DATA",
        ),
    }
    report = {
        "status": "preview_geometry_executed_not_dataset_eligible",
        "annotation_source": annotation["annotation_source"],
        "eye_annotation_valid": False,
        "training_eligible": False,
        "dataset_transform_status": (
            "intentionally_not_executed: strict profile transform rejects "
            "eye_annotation_valid=false"
        ),
        "geometry_function_status": "preprocess_profile_side executed for visualization only",
        "real_stages": ["decode", "paired_target_label"],
        "preview_only_stages": [
            "profile_eye_annotation",
            "EAR",
            "2d_head_proxy",
            "iris_proxy",
            "profile_eye_patch",
            "zero_one_normalization",
        ],
        "dummy_components": list(annotation.get("dummy_components", [])),
        "phone_position_recorded": pair.phone_position,
        "model_input_shape_hwc": list(result.patch.shape),
        "normalized_tensor_shape_chw": [3, result.patch.shape[0], result.patch.shape[1]],
        "target_gaze_xy_centered": [pair.target_x_centered, pair.target_y_centered],
    }
    return panels, report, annotation


def _safe_component(value: str) -> str:
    result = _SAFE_COMPONENT.sub("_", value).strip("._")
    if not result:
        raise ValueError(f"unsafe empty output component derived from {value!r}")
    return result


def _replace_guard(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"generated output already exists: {path}; pass --overwrite-generated to replace it"
        )


def _save_png(path: Path, image: np.ndarray, *, overwrite: bool) -> None:
    _replace_guard(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.png")
    Image.fromarray(_rgb(image)).save(temporary)
    os.replace(temporary, path)


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    _replace_guard(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _preview(image: np.ndarray, maximum_size: tuple[int, int] = (720, 480)) -> np.ndarray:
    result = Image.fromarray(_rgb(image))
    result.thumbnail(maximum_size, Image.Resampling.LANCZOS)
    return np.asarray(result)


def _save_montage(
    output_dir: Path,
    rows: Sequence[tuple[str, Mapping[str, np.ndarray]]],
    *,
    overwrite: bool,
) -> Path:
    path = output_dir / "data_ver1_preprocessing_preview.png"
    _replace_guard(path, overwrite=overwrite)
    figure, axes = plt.subplots(
        len(rows),
        len(PANEL_KEYS),
        figsize=(21, max(5.0, 4.2 * len(rows))),
        squeeze=False,
    )
    figure.suptitle(
        "data(ver1): real front pipeline vs preview-only side geometry",
        fontsize=18,
        y=0.995,
    )
    for row_index, (label, panels) in enumerate(rows):
        for column_index, key in enumerate(PANEL_KEYS):
            axis = axes[row_index, column_index]
            axis.imshow(_preview(panels[key]))
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(PANEL_TITLES[key], fontsize=10)
            if column_index == 0:
                axis.set_ylabel(label, fontsize=9)
    figure.tight_layout(rect=(0.01, 0.01, 1.0, 0.965))
    temporary = path.with_name(f".{path.name}.tmp.png")
    figure.savefig(temporary, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    os.replace(temporary, path)
    return path


def _relative(path: Path, data_root: Path) -> str:
    return path.resolve(strict=True).relative_to(data_root.resolve(strict=True)).as_posix()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_json_value(child) for child in value]
    if value is None or isinstance(value, str | bool | int | float):
        return value
    return str(value)


def _manifest_csv(pairs: Sequence[SelectedPair], data_root: Path) -> str:
    columns = [
        "sample_id",
        "subject_id",
        "pair_id",
        "view",
        "image_path",
        "target_x_px",
        "target_y_px",
        "target_x_centered",
        "target_y_centered",
        "screen_width_px",
        "screen_height_px",
        "capture_face_detected",
        "capture_iris_detected",
        "annotation_status",
        "training_eligible",
    ]
    lines: list[str] = []
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for pair in pairs:
            common = {
                "sample_id": f"{pair.subject_id}:{pair.sample_id}",
                "subject_id": pair.subject_id,
                "pair_id": pair.pair_id,
                "target_x_px": pair.target_x_px,
                "target_y_px": pair.target_y_px,
                "target_x_centered": pair.target_x_centered,
                "target_y_centered": pair.target_y_centered,
                "screen_width_px": pair.screen_width_px,
                "screen_height_px": pair.screen_height_px,
            }
            writer.writerow(
                {
                    **common,
                    "view": "front",
                    "image_path": _relative(pair.front_path, data_root),
                    "capture_face_detected": pair.row.get("webcam_mediapipe_face_detected", ""),
                    "capture_iris_detected": pair.row.get("webcam_mediapipe_iris_detected", ""),
                    "annotation_status": "runtime_required_not_stored",
                    "training_eligible": "runtime_dependent",
                }
            )
            writer.writerow(
                {
                    **common,
                    "view": "side",
                    "image_path": _relative(pair.side_path, data_root),
                    "capture_face_detected": pair.row.get("phonecam_mediapipe_face_detected", ""),
                    "capture_iris_detected": pair.row.get("phonecam_mediapipe_iris_detected", ""),
                    "annotation_status": "missing_strict_profile_annotation",
                    "training_eligible": "false",
                }
            )
        stream.seek(0)
        lines.append(stream.read())
    return "".join(lines)


def _front_config(
    config_path: Path,
    profile_path: Path,
    data_root: Path,
    model_asset: Path,
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["GAZE_DATA_ROOT"] = str(data_root)
    environment["DUAL_VIEW_MANIFEST"] = str(data_root / "preview_manifest_not_used.csv")
    environment["MEDIAPIPE_FACE_MODEL"] = str(model_asset)
    return load_and_validate_config(
        config_path,
        profiles=(profile_path,),
        environ=environment,
        check_paths=False,
    )


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    config_path = args.config.expanduser().resolve(strict=True)
    profile_path = args.front_profile.expanduser().resolve(strict=True)
    model_asset = args.model_asset.expanduser().resolve(strict=True)
    validate_preview_paths(data_root, output_dir, project_root)
    pairs = discover_selected_pairs(
        data_root,
        count=args.samples,
        participant_names=args.participants,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _front_config(config_path, profile_path, data_root, model_asset)
    preprocessor = GazePreprocessor(config, split="validation")

    report: dict[str, Any] = {
        "schema_version": 1,
        "data_root_name": data_root.name,
        "output_scope": "local_project_outputs_only",
        "privacy": {
            "contains_real_faces": True,
            "uploaded": False,
            "raw_images_modified": False,
        },
        "contract": {
            "front": "actual configured GazePreprocessor when runtime MediaPipe succeeds",
            "side": "preview-only preprocess_profile_side with invalid generated annotation",
            "side_generated_annotations_are_training_labels": False,
        },
        "samples": [],
    }
    rows: list[tuple[str, Mapping[str, np.ndarray]]] = []
    side_annotations: dict[str, Any] = {}
    for pair in pairs:
        detections: dict[str, Mapping[str, Any] | None] = {"front": None, "side": None}
        detection_errors: dict[str, str | None] = {"front": None, "side": None}
        if args.mediapipe != "off":
            for view, path in (("front", pair.front_path), ("side", pair.side_path)):
                result, error = detect_mediapipe_isolated(
                    path,
                    model_asset,
                    view,
                    timeout_seconds=args.mediapipe_timeout_seconds,
                )
                detections[view] = result
                detection_errors[view] = error
                if result is None and args.mediapipe == "required":
                    raise RuntimeError(f"{pair.subject_id}/{pair.sample_id}/{view}: {error}")

        front_panels, front_report = _front_panels(
            preprocessor,
            pair,
            detections["front"],
            detection_errors["front"],
        )
        side_panels, side_report, side_annotation = _side_panels(
            pair,
            detections["side"],
            annotation_source=args.side_annotation_source,
        )
        row_id = f"{_safe_component(pair.subject_id)}_{_safe_component(pair.sample_id)}"
        for view, panels in (("front", front_panels), ("side", side_panels)):
            for index, key in enumerate(PANEL_KEYS, start=1):
                _save_png(
                    output_dir / "stages" / row_id / view / f"{index:02d}_{key}.png",
                    panels[key],
                    overwrite=args.overwrite_generated,
                )
        rows.extend(
            [
                (f"{pair.subject_id}/{pair.sample_id} front", front_panels),
                (f"{pair.subject_id}/{pair.sample_id} side", side_panels),
            ]
        )
        sample_key = f"{pair.subject_id}:{pair.sample_id}"
        side_annotations[sample_key] = {
            **_json_value(side_annotation),
            "eye_annotation_valid": False,
            "training_eligible": False,
            "note": "preview only; do not merge into a training manifest",
        }
        report["samples"].append(
            {
                "sample_id": sample_key,
                "pair_id": pair.pair_id,
                "paths": {
                    "front": _relative(pair.front_path, data_root),
                    "side": _relative(pair.side_path, data_root),
                },
                "target": {
                    "pixel_xy": [pair.target_x_px, pair.target_y_px],
                    "centered_xy": [pair.target_x_centered, pair.target_y_centered],
                    "screen_size_px": [pair.screen_width_px, pair.screen_height_px],
                    "source": "labels/image_samples.csv",
                },
                "capture_flags": {
                    "front_face": pair.row.get("webcam_mediapipe_face_detected"),
                    "front_iris": pair.row.get("webcam_mediapipe_iris_detected"),
                    "side_face": pair.row.get("phonecam_mediapipe_face_detected"),
                    "side_iris": pair.row.get("phonecam_mediapipe_iris_detected"),
                },
                "calibration": dict(pair.calibration_audit),
                "front": front_report,
                "side": side_report,
                "runtime_mediapipe_errors": detection_errors,
            }
        )

    montage = _save_montage(output_dir, rows, overwrite=args.overwrite_generated)
    _write_text(
        output_dir / "selected_preview_manifest.csv",
        _manifest_csv(pairs, data_root),
        overwrite=args.overwrite_generated,
    )
    _write_text(
        output_dir / "side_preview_annotations_NOT_FOR_TRAINING.json",
        json.dumps(_json_value(side_annotations), ensure_ascii=False, indent=2) + "\n",
        overwrite=args.overwrite_generated,
    )
    report["montage"] = montage.name
    _write_text(
        output_dir / "preprocessing_ver1_preview_report.json",
        json.dumps(_json_value(report), ensure_ascii=False, indent=2) + "\n",
        overwrite=args.overwrite_generated,
    )
    print(f"preview: {montage}")
    print(f"report: {output_dir / 'preprocessing_ver1_preview_report.json'}")
    print("raw images modified: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
