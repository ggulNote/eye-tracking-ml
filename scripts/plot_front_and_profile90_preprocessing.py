"""Plot exact BlazeGaze front inputs and annotated strict-profile side inputs.

The image-derived iris/eyelid arrows in this diagnostic are *not* screen-gaze
labels. A calibrated screen target is required before they may be interpreted
as gaze direction or used as a regression target.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
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
import yaml
from PIL import Image, ImageDraw, ImageFont

from gaze_pipeline.config import load_and_validate_config
from gaze_pipeline.data.profile_side import preprocess_profile_side
from gaze_pipeline.data.transforms import GazePreprocessor
from gaze_pipeline.data.webeyetrack_compat import (
    LEFT_EAR_INDICES,
    RIGHT_EAR_INDICES,
    WEBEYETRACK_FACE_QUAD_INDICES,
    MediaPipeFaceLandmarkerDetector,
)

FRONT_IRIS_CENTER_INDICES = {"left": 473, "right": 468}
FRONT_EYE_INDICES = {"left": LEFT_EAR_INDICES, "right": RIGHT_EAR_INDICES}
PANEL_KEYS = ("original", "annotation", "ear", "head", "eye_cue", "model_input")
PANEL_TITLES = {
    "original": "1. Original RGB",
    "annotation": "2. Geometry annotation",
    "ear": "3. EAR / eye validity",
    "head": "4. Head-pose cue",
    "eye_cue": "5. Eye vertical cue\n(NOT calibrated gaze)",
    "model_input": "6. Model image input",
}
PROFILE_EAR_THRESHOLD = 0.20


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    default_examples = project_root.parent / "project_data/example"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(os.environ.get("GAZE_EXAMPLE_ROOT", default_examples)),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs/profile90_preprocessing",
    )
    parser.add_argument("--config", type=Path, default=project_root / "configs/config.yaml")
    parser.add_argument(
        "--front-profile",
        type=Path,
        default=project_root / "configs/profiles/blazegaze.yaml",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=project_root / "configs/examples/profile90_annotations.yaml",
    )
    parser.add_argument(
        "--model-asset",
        type=Path,
        default=project_root / "models/face_landmarker_v2_with_blendshapes.task",
    )
    return parser


def _rgb(image: Any) -> np.ndarray:
    if torch.is_tensor(image):
        array = image.detach().cpu().float().numpy()
        if array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))
        if float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        return np.rint(np.clip(array, 0.0, 255.0)).astype(np.uint8)
    array = np.asarray(image)
    if array.dtype != np.uint8:
        maximum = float(np.nanmax(array))
        array = array * 255.0 if maximum <= 1.0 else array
        array = np.rint(np.clip(array, 0.0, 255.0)).astype(np.uint8)
    return np.ascontiguousarray(array)


def _preview(image: Any, maximum_size: tuple[int, int] = (960, 720)) -> np.ndarray:
    result = Image.fromarray(_rgb(image))
    result.thumbnail(maximum_size, Image.Resampling.LANCZOS)
    return np.asarray(result)


def _font(
    canvas: Image.Image, scale: float = 0.018
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(14, int(round(min(canvas.size) * scale)))
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_text(
    draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, canvas: Image.Image
) -> None:
    draw.text(
        xy,
        text,
        font=_font(canvas),
        fill=(255, 255, 255),
        stroke_width=max(1, int(round(min(canvas.size) * 0.0015))),
        stroke_fill=(0, 0, 0),
    )


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    width: int,
) -> None:
    start_xy = np.asarray(start, dtype=float)
    end_xy = np.asarray(end, dtype=float)
    draw.line((tuple(start_xy), tuple(end_xy)), fill=color, width=width)
    direction = end_xy - start_xy
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return
    unit = direction / norm
    normal = np.asarray([-unit[1], unit[0]])
    head = max(8.0, width * 3.0)
    left = end_xy - unit * head + normal * head * 0.45
    right = end_xy - unit * head - normal * head * 0.45
    draw.polygon([tuple(end_xy), tuple(left), tuple(right)], fill=color)


def _front_eye_vertical_cues(landmarks: np.ndarray) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    for eye_name, indices in FRONT_EYE_INDICES.items():
        points = landmarks[np.asarray(indices, dtype=int)]
        corner_vector = points[3] - points[0]
        corner_distance = float(np.linalg.norm(corner_vector))
        if corner_distance <= 1e-6:
            continue
        horizontal = corner_vector / corner_distance
        vertical = np.asarray([-horizontal[1], horizontal[0]], dtype=float)
        if vertical[1] < 0:
            vertical *= -1.0
        eyelid_center = points.mean(axis=0)
        iris_center = landmarks[FRONT_IRIS_CENTER_INDICES[eye_name]]
        signed_pixels = float(np.dot(iris_center - eyelid_center, vertical))
        aperture = 0.5 * (
            float(np.linalg.norm(points[1] - points[5]))
            + float(np.linalg.norm(points[2] - points[4]))
        )
        normalized_vertical = signed_pixels / max(aperture, 1e-6)
        cues.append(
            {
                "eye": eye_name,
                "anchor_xy": eyelid_center,
                "iris_center_xy": iris_center,
                "image_vertical_axis": vertical,
                "eye_vertical_cue_2d": vertical * normalized_vertical,
                "vertical_offset_aperture_ratio": normalized_vertical,
                "corner_distance": corner_distance,
            }
        )
    return cues


def _draw_front_annotation(image: np.ndarray, sample: Mapping[str, Any]) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    landmarks = np.asarray(sample["landmarks_xy"], dtype=float)
    radius = max(2, int(round(min(canvas.size) * 0.002)))
    for x, y in landmarks:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 225, 255))
    quad = landmarks[np.asarray(WEBEYETRACK_FACE_QUAD_INDICES)]
    draw.line(
        [tuple(point) for point in quad] + [tuple(quad[0])],
        fill=(255, 210, 0),
        width=max(3, radius),
    )
    _draw_text(draw, (18, 18), "MediaPipe 478 + WebEyeTrack source quad", canvas)
    return np.asarray(canvas)


def _draw_front_ear(image: np.ndarray, sample: Mapping[str, Any]) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    landmarks = np.asarray(sample["landmarks_xy"], dtype=float)
    metadata = sample["metadata"]
    ears = np.asarray(metadata.get("ear", [np.nan, np.nan]), dtype=float)
    eye_open = np.asarray(metadata.get("eye_open_mask", [False, False]), dtype=bool)
    width = max(3, int(round(min(canvas.size) * 0.004)))
    for index, (eye_name, indices) in enumerate(FRONT_EYE_INDICES.items()):
        points = landmarks[np.asarray(indices)]
        color = (40, 230, 80) if eye_open[index] else (255, 70, 70)
        draw.line([tuple(point) for point in points] + [tuple(points[0])], fill=color, width=width)
        _draw_text(draw, tuple(points.mean(axis=0)), eye_name, canvas)
    _draw_text(
        draw,
        (18, 18),
        f"EAR left={ears[0]:.3f}, right={ears[1]:.3f}; open={eye_open.tolist()}",
        canvas,
    )
    return np.asarray(canvas)


def _draw_front_head(image: np.ndarray, sample: Mapping[str, Any]) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    landmarks = np.asarray(sample["landmarks_xy"], dtype=float)
    metadata = sample["metadata"]
    origin = landmarks[4]
    head_vector = np.asarray(metadata.get("head_vector", [np.nan] * 3), dtype=float)
    bbox = np.asarray(metadata.get("face_bbox_xywh", [0, 0, canvas.width, canvas.height]))
    # The camera-facing z component is near -1 for a frontal head, leaving a
    # small x/y projection. Scale that projection so its direction remains
    # visible in a diagnostic image without pretending it is a gaze ray.
    length = max(160.0, float(bbox[2]) * 0.85)
    endpoint = origin + np.asarray([head_vector[0], -head_vector[1]]) * length
    line_width = max(4, int(round(min(canvas.size) * 0.005)))
    face_rt = np.asarray(metadata.get("face_rt"), dtype=float)
    if face_rt.shape == (4, 4):
        rotation = face_rt[:3, :3]
        for axis_index, (color, label) in enumerate(
            (((255, 70, 70), "X"), ((70, 255, 70), "Y"), ((70, 150, 255), "Z"))
        ):
            axis = rotation[:, axis_index]
            axis_endpoint = origin + np.asarray([axis[0], -axis[1]]) * length * 0.55
            _arrow(draw, origin, axis_endpoint, color, max(2, line_width // 2))
            _draw_text(draw, tuple(axis_endpoint), label, canvas)
    if np.isfinite(head_vector).all():
        _arrow(draw, origin, endpoint, (255, 0, 255), line_width)
    euler = np.asarray(metadata.get("head_euler_degrees", [np.nan] * 3), dtype=float)
    _draw_text(
        draw,
        (18, 18),
        f"head [x,y,z]={np.round(head_vector, 3).tolist()} (x/y projection scaled)",
        canvas,
    )
    _draw_text(draw, (18, 52), f"pitch/yaw/roll={np.round(euler, 1).tolist()} deg", canvas)
    return np.asarray(canvas)


def _draw_front_eye_cue(
    image: np.ndarray,
    cues: list[dict[str, Any]],
) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    width = max(4, int(round(min(canvas.size) * 0.005)))
    values: list[float] = []
    for cue in cues:
        anchor = np.asarray(cue["anchor_xy"])
        vertical = np.asarray(cue["image_vertical_axis"])
        value = float(cue["vertical_offset_aperture_ratio"])
        values.append(value)
        arrow_length = cue["corner_distance"] * max(0.35, min(1.4, abs(value) * 2.0))
        endpoint = anchor + vertical * np.sign(value or 1.0) * arrow_length
        _arrow(draw, anchor, endpoint, (255, 165, 0), width)
        iris = np.asarray(cue["iris_center_xy"])
        radius = width * 2
        draw.ellipse(
            (iris[0] - radius, iris[1] - radius, iris[0] + radius, iris[1] + radius),
            outline=(0, 255, 255),
            width=width,
        )
    mean_value = float(np.mean(values)) if values else float("nan")
    _draw_text(draw, (18, 18), f"vertical eye cue={mean_value:+.3f} aperture", canvas)
    _draw_text(draw, (18, 52), "NOT calibrated screen gaze (target label unavailable)", canvas)
    return np.asarray(canvas)


def _front_result(
    preprocessor: GazePreprocessor,
    image_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sample: dict[str, Any] = {
        "image_path": str(image_path),
        "view": "front",
        "metadata": {"view": "front", "gaze_valid": True, "invalid_reasons": []},
    }
    panels: dict[str, np.ndarray] = {}
    cues: list[dict[str, Any]] = []
    for stage in preprocessor.stage_order:
        stage_config = preprocessor._stage_config(stage, "front")
        if not bool(stage_config.get("enabled", True)):
            continue
        getattr(preprocessor, f"_stage_{stage}")(sample, stage_config)
        if stage == "decode":
            panels["original"] = _rgb(sample["image"]).copy()
        elif stage == "face_landmarks":
            panels["annotation"] = _draw_front_annotation(_rgb(sample["image"]), sample)
        elif stage == "eye_state":
            panels["ear"] = _draw_front_ear(_rgb(sample["image"]), sample)
            cues = _front_eye_vertical_cues(np.asarray(sample["landmarks_xy"], dtype=float))
            panels["eye_cue"] = _draw_front_eye_cue(_rgb(sample["image"]), cues)
        elif stage == "metric_head_pose":
            panels["head"] = _draw_front_head(_rgb(sample["image"]), sample)
        elif stage == "eye_region_warp":
            panels["model_input"] = _rgb(sample["image"]).copy()

    missing = [key for key in PANEL_KEYS if key not in panels]
    if missing:
        raise RuntimeError(f"{image_path.name}: missing front panels {missing}")
    metadata = sample["metadata"]
    face_bbox_xywh = np.asarray(metadata.get("face_bbox_xywh"), dtype=float)
    face_bbox_xyxy = [
        face_bbox_xywh[0],
        face_bbox_xywh[1],
        face_bbox_xywh[0] + face_bbox_xywh[2],
        face_bbox_xywh[1] + face_bbox_xywh[3],
    ]
    head_vector = np.asarray(metadata.get("head_vector"), dtype=float)
    face_origin = np.asarray(metadata.get("face_origin_3d"), dtype=float)
    ears = np.asarray(metadata.get("ear"), dtype=float)
    eye_open_mask = np.asarray(metadata.get("eye_open_mask"), dtype=bool)
    gaze_valid = bool(metadata.get("gaze_valid", False))
    head_pose_valid = bool(metadata.get("head_pose_valid", False))
    report = {
        "image_path": str(image_path),
        "model_input_shape_hwc": list(panels["model_input"].shape),
        "face_bbox_xyxy": np.asarray(face_bbox_xyxy).tolist(),
        "ear": ears.tolist(),
        "eye_open_mask": eye_open_mask.tolist(),
        "head_vector_3d": head_vector.tolist(),
        "face_origin_3d_cm": face_origin.tolist(),
        "head_euler_degrees": np.asarray(metadata.get("head_euler_degrees"), dtype=float).tolist(),
        "eye_vertical_cues": [
            {
                "eye": cue["eye"],
                "eye_vertical_cue_2d": np.asarray(cue["eye_vertical_cue_2d"]).tolist(),
                "vertical_offset_aperture_ratio": cue["vertical_offset_aperture_ratio"],
            }
            for cue in cues
        ],
        "screen_gaze_available": False,
        "screen_gaze_note": "eye cue only; not calibrated gaze",
        "front_model_inputs": {
            "forward_keys": [
                "front_image",
                "front_head_vector",
                "front_face_origin_3d",
            ],
            "front_image": {
                "shape_chw": [3, 128, 512],
                "dtype": "float32",
                "color_order": "RGB",
                "value_range": [0.0, 1.0],
            },
            "front_head_vector": {
                "shape": [3],
                "value": head_vector.tolist(),
                "coordinate_system": "camera",
            },
            "front_face_origin_3d": {
                "shape": [3],
                "value_cm": face_origin.tolist(),
                "coordinate_system": "camera",
            },
            "validity_gate": {
                "ear": ears.tolist(),
                "ear_threshold": float(metadata.get("ear_threshold", PROFILE_EAR_THRESHOLD)),
                "eye_open_mask": eye_open_mask.tolist(),
                "head_pose_valid": head_pose_valid,
                "preprocessing_valid": gaze_valid and head_pose_valid,
                "passed_to_model": False,
            },
        },
    }
    return panels, report


def _load_profile_annotations(path: Path) -> list[tuple[str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    images = document.get("images", {}) if isinstance(document, Mapping) else {}
    if not isinstance(images, Mapping) or not images:
        raise ValueError("profile annotation YAML must contain at least one images entry")
    annotations: list[tuple[str, dict[str, Any]]] = []
    for sample_id, annotation in images.items():
        if not isinstance(annotation, Mapping):
            raise ValueError(f"annotation for {sample_id!r} must be a mapping")
        annotations.append((str(sample_id), dict(annotation)))
    return annotations


def _draw_profile_annotation(
    image: np.ndarray,
    annotation: Mapping[str, Any],
    source_quad_xy: np.ndarray,
) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    width = max(4, int(round(min(canvas.size) * 0.004)))
    for key, color in (
        ("face_bbox_xyxy", (255, 210, 0)),
        ("visible_eye_bbox_xyxy", (0, 255, 255)),
    ):
        draw.rectangle(tuple(annotation[key]), outline=color, width=width)
    eyelids = np.asarray(annotation["visible_eye_keypoints_xy"], dtype=float)
    draw.line(
        [tuple(point) for point in eyelids] + [tuple(eyelids[0])], fill=(0, 255, 0), width=width
    )
    radius = width * 2
    for index, point in enumerate(eyelids, start=1):
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 255, 0))
        _draw_text(draw, (x + radius, y), f"p{index}", canvas)
    iris = np.asarray(annotation["iris_center_xy"], dtype=float)
    draw.ellipse(
        (iris[0] - radius * 2, iris[1] - radius * 2, iris[0] + radius * 2, iris[1] + radius * 2),
        outline=(255, 60, 255),
        width=width,
    )
    source_quad = np.asarray(source_quad_xy, dtype=float)
    draw.line(
        [tuple(point) for point in source_quad] + [tuple(source_quad[0])],
        fill=(255, 255, 255),
        width=width,
    )
    _draw_text(
        draw,
        (18, 18),
        "yellow=face, cyan=eye bbox, white=actual crop, green=eyelid, magenta=iris",
        canvas,
    )
    return np.asarray(canvas)


def _draw_profile_ear(
    image: np.ndarray,
    annotation: Mapping[str, Any],
    ear: float | None,
    *,
    threshold: float,
    annotation_valid: bool,
) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    points = np.asarray(annotation["visible_eye_keypoints_xy"], dtype=float)
    width = max(4, int(round(min(canvas.size) * 0.005)))
    eye_open = ear is not None and ear >= threshold
    preprocessing_valid = annotation_valid and eye_open
    color = (40, 230, 80) if preprocessing_valid else (255, 70, 70)
    draw.line([tuple(point) for point in points] + [tuple(points[0])], fill=color, width=width)
    for a, b in ((1, 5), (2, 4), (0, 3)):
        draw.line(
            (tuple(points[a]), tuple(points[b])), fill=(255, 255, 0), width=max(2, width // 2)
        )
    status = "unavailable" if ear is None else f"{ear:.3f}"
    comparison = "N/A" if ear is None else (">=" if eye_open else "<")
    _draw_text(
        draw,
        (18, 18),
        f"EAR={status} {comparison} {threshold:.2f}; eye_open={eye_open}",
        canvas,
    )
    _draw_text(
        draw,
        (18, 52),
        f"annotation_valid={annotation_valid}; preprocessing_valid={preprocessing_valid}",
        canvas,
    )
    return np.asarray(canvas)


def _draw_profile_head_direction(
    image: np.ndarray,
    origin: np.ndarray,
    forward_point: np.ndarray,
    head_pose_2d: np.ndarray,
) -> np.ndarray:
    """Draw the exact annotated ear-near anchor to nose-tip direction."""

    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    width = max(4, int(round(min(canvas.size) * 0.006)))
    _arrow(draw, origin, forward_point, (255, 0, 255), width)
    radius = width * 2
    for point, label in ((origin, "H0"), (forward_point, "H1")):
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=(255, 255, 0),
        )
        _draw_text(draw, (point[0] + radius, point[1]), label, canvas)
    delta = forward_point - origin
    pitch_proxy = float(np.degrees(np.arctan2(-float(delta[1]), abs(float(delta[0])))))
    _draw_text(draw, (18, 18), f"2D head direction={np.round(head_pose_2d, 4).tolist()}", canvas)
    _draw_text(
        draw,
        (18, 52),
        f"H0=ear/tragion anchor, H1=nose tip; pitch proxy={pitch_proxy:+.1f} deg",
        canvas,
    )
    _draw_text(draw, (18, 86), "projected 2D cue; NOT calibrated 3D head pose", canvas)
    return np.asarray(canvas)


def _draw_profile_tail_vectors(
    image: np.ndarray,
    eyelid_keypoints_xy: np.ndarray,
    eyelid_center_xy: np.ndarray,
    iris_center_xy: np.ndarray,
    eye_pose_2d: np.ndarray,
    eye_size_xy: np.ndarray,
    tail_points_xy: np.ndarray,
    tail_vectors_px: np.ndarray,
    tail_direction_angles_degrees: np.ndarray,
) -> np.ndarray:
    """Overlay the two eye-local direction-angle features."""

    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    width = max(3, int(round(min(canvas.size) * 0.0025)))
    eyelids = np.asarray(eyelid_keypoints_xy, dtype=float)
    corner_axis = eyelids[3] - eyelids[0]
    corner_axis /= max(float(np.linalg.norm(corner_axis)), 1e-8)
    axis_y = np.asarray([-corner_axis[1], corner_axis[0]], dtype=float)
    if axis_y[1] < 0:
        axis_y *= -1.0
    eyelid_center = np.asarray(eyelid_center_xy, dtype=float)
    iris_center = np.asarray(iris_center_xy, dtype=float)
    projected_endpoint = eyelid_center + axis_y * float(eye_pose_2d[1]) * float(
        np.asarray(eye_size_xy)[1]
    )
    _arrow(draw, eyelid_center, projected_endpoint, (255, 150, 0), width)
    iris_radius = width * 2
    draw.ellipse(
        (
            iris_center[0] - iris_radius,
            iris_center[1] - iris_radius,
            iris_center[0] + iris_radius,
            iris_center[1] + iris_radius,
        ),
        outline=(0, 255, 255),
        width=width,
    )
    a0, a1, a2 = np.asarray(tail_points_xy, dtype=float)
    upper_vector, lower_vector = np.asarray(tail_vectors_px, dtype=float)
    _arrow(draw, a0, a1, (50, 170, 255), width)
    _arrow(draw, a0, a2, (80, 235, 90), width)

    local_x_vector = eyelids[0] - a0
    local_x_vector /= max(float(np.linalg.norm(local_x_vector)), 1e-8)
    local_x_angle = float(np.arctan2(local_x_vector[1], local_x_vector[0]))
    arc_radius = max(
        width * 5.0,
        min(float(np.linalg.norm(upper_vector)), float(np.linalg.norm(lower_vector))) * 0.55,
    )
    local_x_endpoint = a0 + local_x_vector * arc_radius * 1.25
    draw.line([tuple(a0), tuple(local_x_endpoint)], fill=(245, 245, 245), width=width)
    direction_degrees = np.asarray(tail_direction_angles_degrees, dtype=float)
    for direction_value, color in zip(
        direction_degrees,
        ((255, 225, 0), (255, 180, 0)),
        strict=True,
    ):
        direction_radians = float(np.radians(direction_value))
        arc_angles = np.linspace(
            local_x_angle,
            local_x_angle + direction_radians,
            num=24,
        )
        arc_points = [
            tuple(a0 + arc_radius * np.asarray([np.cos(angle), np.sin(angle)]))
            for angle in arc_angles
        ]
        draw.line(arc_points, fill=color, width=width)
    radius = width
    for point, label, color in (
        (a0, "a0", (255, 60, 60)),
        (a1, "a1", (50, 170, 255)),
        (a2, "a2", (80, 235, 90)),
    ):
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=color,
        )
        _draw_text(draw, (point[0] + radius, point[1]), label, canvas)
    _draw_text(draw, (18, 120), "blue=a0→a1, green=a0→a2, white=eye-local +x", canvas)
    _draw_text(
        draw,
        (18, 154),
        f"upper={direction_degrees[0]:+.2f} deg, lower={direction_degrees[1]:+.2f} deg",
        canvas,
    )
    _draw_text(draw, (18, 188), "model feature=[upper/pi, lower/pi]", canvas)
    return np.asarray(canvas)


def _draw_profile_eye_cue(
    image: np.ndarray,
    eyelid_keypoints: np.ndarray,
    eyelid_center: np.ndarray,
    iris_center: np.ndarray,
    eye_pose_2d: np.ndarray,
    eye_size_xy: np.ndarray,
) -> np.ndarray:
    """Draw the local-vertical projection represented by ``eye_pose_2d``.

    With ``vertical_only=True``, the model cue is not the raw center-to-iris
    displacement. It is that displacement projected onto the eye-local
    vertical axis formed from the two eye corners. The iris itself is retained
    as a separate marker so the diagnostic remains auditable.
    """

    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    width = max(4, int(round(min(canvas.size) * 0.006)))
    points = np.asarray(eyelid_keypoints, dtype=float)
    corner_axis = points[3] - points[0]
    corner_length = float(np.linalg.norm(corner_axis))
    if corner_length <= 1e-6:
        raise ValueError("profile eye corners must define a non-zero local horizontal axis")
    axis_x = corner_axis / corner_length
    if axis_x[0] < 0 or (abs(float(axis_x[0])) <= 1e-8 and axis_x[1] < 0):
        axis_x *= -1.0
    axis_y = np.asarray([-axis_x[1], axis_x[0]], dtype=float)
    if axis_y[1] < 0:
        axis_y *= -1.0
    projected_endpoint = eyelid_center + axis_y * float(eye_pose_2d[1]) * float(eye_size_xy[1])
    _arrow(draw, eyelid_center, projected_endpoint, (255, 165, 0), width)
    radius = width * 2
    draw.ellipse(
        (
            iris_center[0] - radius,
            iris_center[1] - radius,
            iris_center[0] + radius,
            iris_center[1] + radius,
        ),
        outline=(0, 255, 255),
        width=width,
    )
    projected_radius = max(3, width)
    draw.ellipse(
        (
            projected_endpoint[0] - projected_radius,
            projected_endpoint[1] - projected_radius,
            projected_endpoint[0] + projected_radius,
            projected_endpoint[1] + projected_radius,
        ),
        fill=(255, 165, 0),
    )
    _draw_text(draw, (18, 18), f"eye_pose_2d={np.round(eye_pose_2d, 4).tolist()}", canvas)
    _draw_text(draw, (18, 52), "local-vertical projection; +x right / +y down", canvas)
    _draw_text(draw, (18, 86), "orange=cue, cyan=actual iris; NOT calibrated screen gaze", canvas)
    return np.asarray(canvas)


def _profile_result(
    input_root: Path,
    sample_id: str,
    annotation: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    image_path = (input_root / str(annotation["relative_path"])).resolve(strict=True)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    expected_wh = tuple(int(value) for value in annotation["expected_size_wh"])
    actual_wh = (image.shape[1], image.shape[0])
    if actual_wh != expected_wh:
        raise ValueError(f"{sample_id}: expected image size {expected_wh}, got {actual_wh}")

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
        eyelid_tail_indices=annotation.get("eyelid_tail_indices", [3, 2, 4]),
    )
    head_origin = np.asarray(annotation["profile_head_origin_xy"], dtype=float)
    annotation_valid = bool(annotation.get("eye_annotation_valid", False))
    eye_open = result.ear is not None and result.ear >= PROFILE_EAR_THRESHOLD
    preprocessing_valid = annotation_valid and eye_open
    panels = {
        "original": image,
        "annotation": _draw_profile_annotation(image, annotation, result.source_quad_xy),
        "ear": _draw_profile_ear(
            image,
            annotation,
            result.ear,
            threshold=PROFILE_EAR_THRESHOLD,
            annotation_valid=annotation_valid,
        ),
        "head": _draw_profile_head_direction(
            image,
            head_origin,
            np.asarray(annotation["profile_head_forward_xy"], dtype=float),
            np.asarray(result.head_pose_2d),
        ),
        "eye_cue": _draw_profile_eye_cue(
            image,
            np.asarray(annotation["visible_eye_keypoints_xy"], dtype=float),
            np.asarray(result.eyelid_center_xy),
            np.asarray(annotation["iris_center_xy"], dtype=float),
            np.asarray(result.eye_pose_2d),
            np.asarray(result.eye_size_xy),
        ),
        "model_input": np.asarray(result.patch),
    }
    if (
        result.eyelid_tail_points_xy is not None
        and result.eyelid_tail_vectors_px is not None
        and result.eyelid_direction_angles_degrees is not None
    ):
        panels["tail_vectors"] = _draw_profile_tail_vectors(
            image,
            np.asarray(annotation["visible_eye_keypoints_xy"], dtype=float),
            np.asarray(result.eyelid_center_xy),
            np.asarray(annotation["iris_center_xy"], dtype=float),
            np.asarray(result.eye_pose_2d),
            np.asarray(result.eye_size_xy),
            np.asarray(result.eyelid_tail_points_xy),
            np.asarray(result.eyelid_tail_vectors_px),
            np.asarray(result.eyelid_direction_angles_degrees),
        )
    report = {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "model_input_shape_hwc": list(np.asarray(result.patch).shape),
        "head_pose_2d": np.asarray(result.head_pose_2d).tolist(),
        "head_pose_definition": "unit projected vector from H0 ear/tragion anchor to H1 nose tip",
        "head_points_xy": {
            "H0_ear_tragion_anchor": np.asarray(annotation["profile_head_origin_xy"]).tolist(),
            "H1_nose_tip": np.asarray(annotation["profile_head_forward_xy"]).tolist(),
        },
        "head_pitch_proxy_degrees": result.head_pitch_proxy_degrees,
        "head_pose_is_calibrated_3d": False,
        "iris_pose_2d": np.asarray(result.eye_pose_2d).tolist(),
        "iris_pose_definition": "vertical iris-vs-eyelid cue; not calibrated screen gaze",
        "ear": result.ear,
        "source_quad_xy": np.asarray(result.source_quad_xy).tolist(),
        "source_to_patch": np.asarray(result.source_to_patch).tolist(),
        "screen_gaze_available": False,
        "screen_gaze_note": "target_gaze_xy is null in the example annotation",
        "annotation_status": annotation.get("annotation_status"),
        "eyelid_tail_geometry": {
            "point_mapping": {
                "a0": "p4 temporal eye corner / index 3",
                "a1": "p3 adjacent upper lid / index 2",
                "a2": "p5 adjacent lower lid / index 4",
            },
            "points_xy": (
                np.asarray(result.eyelid_tail_points_xy).tolist()
                if result.eyelid_tail_points_xy is not None
                else None
            ),
            "vectors_px": {
                "a0_to_a1": (
                    np.asarray(result.eyelid_tail_vectors_px[0]).tolist()
                    if result.eyelid_tail_vectors_px is not None
                    else None
                ),
                "a0_to_a2": (
                    np.asarray(result.eyelid_tail_vectors_px[1]).tolist()
                    if result.eyelid_tail_vectors_px is not None
                    else None
                ),
            },
            "direction_angles_degrees": (
                np.asarray(result.eyelid_direction_angles_degrees).tolist()
                if result.eyelid_direction_angles_degrees is not None
                else None
            ),
            "direction_angles_normalized_pi": (
                np.asarray(result.eyelid_direction_angles_normalized).tolist()
                if result.eyelid_direction_angles_normalized is not None
                else None
            ),
            "included_angle_degrees_diagnostic": result.eyelid_tail_angle_degrees,
            "included_angle_normalized_pi_diagnostic": result.eyelid_tail_angle_normalized,
            "model_feature_key": "side_eye_angles",
            "feature_shape": [2],
            "vectors_passed_to_model": False,
            "meaning": "upper/lower vector directions in semantic eye-local axes",
        },
        "side_model_inputs": {
            "forward_keys": [
                "side_image",
                "side_head_pose_2d",
                "side_eye_angles",
                "side_iris_pose_2d",
            ],
            "side_image": {
                "shape_chw": [3, 128, 256],
                "dtype": "float32",
                "color_order": "RGB",
                "value_range": [0.0, 1.0],
                "visualized_patch_shape_hwc": list(np.asarray(result.patch).shape),
            },
            "side_iris_pose_2d": {
                "shape": [2],
                "value": np.asarray(result.eye_pose_2d).tolist(),
                "coordinate_system": "eye_local_image_plane_x_right_y_down",
                "vertical_only": True,
                "meaning": "iris offset projected onto the eye-local vertical axis",
                "is_calibrated_screen_gaze": False,
            },
            "side_head_pose_2d": {
                "shape": [2],
                "value": np.asarray(result.head_pose_2d).tolist(),
                "coordinate_system": "source_image_plane_x_right_y_down",
                "meaning": "unit vector from the ear-near anchor to the nose tip",
                "is_3d_rotation": False,
            },
            "side_eye_angles": {
                "shape": [2],
                "value": np.asarray(result.eyelid_direction_angles_normalized).tolist(),
                "value_range": [-1.0, 1.0],
                "unit": "radians_divided_by_pi",
                "order": ["a0_to_a1_upper", "a0_to_a2_lower"],
                "meaning": "two eye-local eyelid direction angles",
            },
            "validity_gate": {
                "selected_eye_ear": result.ear,
                "ear_threshold": PROFILE_EAR_THRESHOLD,
                "eye_open": eye_open,
                "annotation_valid": annotation_valid,
                "preprocessing_valid": preprocessing_valid,
                "passed_to_model": False,
            },
        },
    }
    return panels, report


def _save_plot(
    output_dir: Path,
    rows: list[tuple[str, dict[str, np.ndarray]]],
) -> Path:
    figure, axes = plt.subplots(
        len(rows),
        len(PANEL_KEYS),
        figsize=(21, 4.8 * len(rows)),
        squeeze=False,
    )
    figure.suptitle(
        "Front exact BlazeGaze preprocessing + annotation-first strict 90-degree side",
        fontsize=18,
        y=0.995,
    )
    for row_index, (row_name, panels) in enumerate(rows):
        for column_index, key in enumerate(PANEL_KEYS):
            axis = axes[row_index, column_index]
            axis.imshow(_preview(panels[key]))
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(PANEL_TITLES[key], fontsize=10)
            if column_index == 0:
                axis.set_ylabel(row_name, fontsize=10)
    figure.tight_layout(rect=(0.01, 0.01, 1.0, 0.965))
    path = output_dir / "front_two_and_side90_preprocessing.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _focus_crop(
    image: np.ndarray,
    bbox_xyxy: list[float] | np.ndarray,
    *,
    scale_xy: tuple[float, float],
) -> np.ndarray:
    """Crop a diagnostic panel while keeping the requested ROI centered."""

    array = _rgb(image)
    x0, y0, x1, y1 = np.asarray(bbox_xyxy, dtype=float)
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    width = (x1 - x0) * scale_xy[0]
    height = (y1 - y0) * scale_xy[1]
    left = max(0, int(np.floor(center_x - width / 2.0)))
    top = max(0, int(np.floor(center_y - height / 2.0)))
    right = min(array.shape[1], int(np.ceil(center_x + width / 2.0)))
    bottom = min(array.shape[0], int(np.ceil(center_y + height / 2.0)))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid focus crop [{left}, {top}, {right}, {bottom}]")
    return np.ascontiguousarray(array[top:bottom, left:right])


def _format_vector(vector: list[float] | np.ndarray) -> str:
    x_value, y_value = np.asarray(vector, dtype=float)
    return f"[{x_value:+.4f}, {y_value:+.4f}]"


def _format_vector3(vector: list[float] | np.ndarray, *, decimals: int = 3) -> str:
    values = np.asarray(vector, dtype=float)
    return "[" + ", ".join(f"{value:+.{decimals}f}" for value in values) + "]"


def _style_diagnostic_axis(
    axis: plt.Axes,
    *,
    color: str,
) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(color)
        spine.set_linewidth(3.0)


def _save_all_front_model_inputs_plot(
    output_dir: Path,
    results: list[tuple[str, Mapping[str, np.ndarray], Mapping[str, Any]]],
) -> Path:
    """Compare both front examples and the exact BlazeGaze input contract."""

    figure, axes = plt.subplots(
        len(results),
        4,
        figsize=(20, 4.8 * len(results)),
        squeeze=False,
        facecolor="white",
    )
    figure.suptitle(
        "Front examples — extracted BlazeGaze model inputs",
        fontsize=22,
        fontweight="bold",
        y=0.995,
    )
    column_titles = (
        "Source + WebEyeTrack warp quad",
        "#2 head_vector + #3 face_origin_3d",
        "EAR gate + eye vertical cue",
        "#1 front_image [3,128,512]",
    )
    for row_index, (sample_id, panels, front_report) in enumerate(results):
        inputs = front_report["front_model_inputs"]
        head_vector = inputs["front_head_vector"]["value"]
        face_origin = inputs["front_face_origin_3d"]["value_cm"]
        validity = inputs["validity_gate"]
        preprocessing_valid = bool(validity["preprocessing_valid"])
        ears = np.asarray(validity["ear"], dtype=float)
        face_bbox = front_report["face_bbox_xyxy"]
        cue_values = [
            float(cue["vertical_offset_aperture_ratio"])
            for cue in front_report["eye_vertical_cues"]
        ]

        displayed = (
            _focus_crop(panels["annotation"], face_bbox, scale_xy=(1.28, 1.28)),
            _focus_crop(panels["head"], face_bbox, scale_xy=(1.28, 1.28)),
            _focus_crop(panels["eye_cue"], face_bbox, scale_xy=(1.05, 0.68)),
            _rgb(panels["model_input"]),
        )
        subtitles = (
            "yellow polygon = 128×512 homography source",
            f"head={_format_vector3(head_vector)}\n"
            f"origin_cm={_format_vector3(face_origin, decimals=2)}",
            f"EAR=[{ears[0]:.3f}, {ears[1]:.3f}] · "
            f"eye cue={[round(value, 3) for value in cue_values]} · "
            f"{'PASS' if preprocessing_valid else 'BLOCK'}",
            "raw RGB eye strip · no overlay",
        )
        colors = (
            "#5478a8",
            "#d000d0",
            "#27a657" if preprocessing_valid else "#cf3030",
            "#2f78bd",
        )
        for column_index, (image, subtitle, color) in enumerate(
            zip(displayed, subtitles, colors, strict=True)
        ):
            axis = axes[row_index, column_index]
            axis.imshow(image, interpolation="nearest" if column_index == 3 else None)
            axis.set_title(
                (f"{column_titles[column_index]}\n" if row_index == 0 else "") + subtitle,
                fontsize=11,
                color=color,
                pad=8,
            )
            _style_diagnostic_axis(axis, color=color)
            if column_index == 0:
                axis.set_ylabel(sample_id, fontsize=12, fontweight="bold")

    figure.text(
        0.5,
        0.01,
        "BlazeGaze forward inputs: front_image + front_head_vector + front_face_origin_3d · "
        "EAR and eye cue are diagnostics, not model inputs · eye cue is not calibrated gaze",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#536174",
    )
    figure.tight_layout(rect=(0.015, 0.045, 0.995, 0.955), h_pad=2.0, w_pad=1.0)
    path = output_dir / "front_all_model_inputs.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _save_side_model_inputs_plot(
    output_dir: Path,
    sample_id: str,
    panels: Mapping[str, np.ndarray],
    side_report: Mapping[str, Any],
    annotation: Mapping[str, Any],
    *,
    filename: str,
) -> Path:
    """Show the strict-profile extraction and all four SideModel inputs."""

    inputs = side_report["side_model_inputs"]
    head_pose = np.asarray(inputs["side_head_pose_2d"]["value"], dtype=float)
    iris_pose = np.asarray(inputs["side_iris_pose_2d"]["value"], dtype=float)
    eye_angles = np.asarray(inputs["side_eye_angles"]["value"], dtype=float)
    validity = inputs["validity_gate"]
    eye_open = bool(validity["eye_open"])
    preprocessing_valid = bool(validity["preprocessing_valid"])
    ear = validity["selected_eye_ear"]
    ear_text = "unavailable" if ear is None else f"{float(ear):.4f}"

    figure = plt.figure(figsize=(18, 19), facecolor="white")
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=(6.0, 5.0, 7.0),
        hspace=0.36,
        wspace=0.10,
        left=0.045,
        right=0.955,
        bottom=0.045,
        top=0.855,
    )
    source_axis = figure.add_subplot(grid[0, 0])
    head_axis = figure.add_subplot(grid[0, 1])
    eye_axis = figure.add_subplot(grid[1, 0])
    validity_axis = figure.add_subplot(grid[1, 1])
    image_axis = figure.add_subplot(grid[2, :])

    figure.suptitle(
        "Strict 90° Side preprocessing — the four SideModel inputs",
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.925,
        "STRICT 90° SOURCE  →  annotation-based extraction  →  "
        "#1 eye image  +  #2 head pose  +  #3 eyelid direction angles  +  #4 iris pose",
        ha="center",
        va="center",
        fontsize=15,
        color="#27364b",
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#edf4ff",
            "edgecolor": "#7da6d9",
            "linewidth": 1.5,
        },
    )

    source_axis.imshow(_preview(panels["annotation"], maximum_size=(1100, 850)))
    source_axis.set_title(
        "STEP 1 — source + annotation / ROI\n"
        "white polygon = exact affine crop sent to the eye-image branch",
        fontsize=14,
        pad=12,
    )
    _style_diagnostic_axis(source_axis, color="#5478a8")

    face_bbox = np.asarray(annotation["face_bbox_xyxy"], dtype=float)
    head_axis.imshow(_focus_crop(panels["head"], face_bbox, scale_xy=(1.05, 1.05)))
    head_axis.set_title(
        "MODEL INPUT #2 — side_head_pose_2d [2]\n"
        f"{_format_vector(head_pose)}  ·  magenta: ear-near anchor → nose tip",
        fontsize=14,
        color="#a000a0",
        pad=12,
    )
    head_axis.text(
        0.5,
        -0.06,
        "+x right / +y down · unit image-plane vector · NOT 3D rotation",
        transform=head_axis.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        color="#6b315f",
    )
    _style_diagnostic_axis(head_axis, color="#d000d0")

    eye_bbox = np.asarray(annotation["visible_eye_bbox_xyxy"], dtype=float)
    eye_axis.imshow(_focus_crop(panels["tail_vectors"], eye_bbox, scale_xy=(1.45, 2.20)))
    eye_axis.set_title(
        "MODEL INPUT #3 + #4 — side_eye_angles [2] + side_iris_pose_2d [2]\n"
        f"angles/π={_format_vector(eye_angles)} · iris={_format_vector(iris_pose)}",
        fontsize=14,
        color="#b15c00",
        pad=12,
    )
    eye_axis.text(
        0.5,
        -0.07,
        "yellow/orange arcs=upper/lower direction angles · cyan/orange=iris cue",
        transform=eye_axis.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        color="#7a4a13",
    )
    _style_diagnostic_axis(eye_axis, color="#f08a00")

    validity_axis.imshow(_focus_crop(panels["ear"], eye_bbox, scale_xy=(1.45, 2.20)))
    validity_axis.set_title(
        "QUALITY GATE — selected-eye EAR (not a model input)\n"
        f"EAR={ear_text} ≥ {PROFILE_EAR_THRESHOLD:.2f} → "
        f"eye_open={eye_open}, preprocessing_valid={preprocessing_valid}",
        fontsize=14,
        color="#157538" if preprocessing_valid else "#a52323",
        pad=12,
    )
    validity_axis.text(
        0.5,
        -0.07,
        "Closed/invalid samples are masked from loss, metrics, and fusion.",
        transform=validity_axis.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        color="#31573f",
    )
    _style_diagnostic_axis(
        validity_axis,
        color="#27a657" if preprocessing_valid else "#cf3030",
    )

    # This panel intentionally receives the raw patch only: no arrows,
    # landmarks, labels, or matplotlib overlay may alter the model pixels.
    image_axis.imshow(_rgb(panels["model_input"]), interpolation="nearest")
    image_axis.set_title(
        "MODEL INPUT #1 — side_image [3, 128, 256]",
        fontsize=20,
        fontweight="bold",
        color="#174f8a",
        pad=14,
    )
    image_axis.text(
        0.5,
        -0.055,
        "Final 2:1 RGB eye crop — shown before CHW conversion and [0,1] normalization; "
        "no diagnostic overlay",
        transform=image_axis.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        color="#35526f",
    )
    _style_diagnostic_axis(image_axis, color="#2f78bd")

    figure.text(
        0.955,
        0.018,
        f"sample: {sample_id}",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#657080",
    )
    path = output_dir / filename
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _save_all_side_model_inputs_plot(
    output_dir: Path,
    results: list[tuple[str, Mapping[str, np.ndarray], Mapping[str, Any], Mapping[str, Any]]],
) -> Path:
    """Compare all strict-profile examples and their four SideModel inputs."""

    figure, axes = plt.subplots(
        len(results),
        4,
        figsize=(20, 3.8 * len(results)),
        squeeze=False,
        facecolor="white",
    )
    figure.suptitle(
        "All strict 90° side examples — extracted SideModel inputs",
        fontsize=22,
        fontweight="bold",
        y=0.995,
    )
    column_titles = (
        "Source + exact eye crop",
        "#2 head_pose_2d",
        "#3 eye_angles + #4 iris_pose_2d",
        "#1 side_image [3,128,256]",
    )
    for row_index, (sample_id, panels, side_report, annotation) in enumerate(results):
        inputs = side_report["side_model_inputs"]
        head_pose = inputs["side_head_pose_2d"]["value"]
        iris_pose = inputs["side_iris_pose_2d"]["value"]
        eye_angles = inputs["side_eye_angles"]["value"]
        validity = inputs["validity_gate"]
        preprocessing_valid = bool(validity["preprocessing_valid"])
        ear = validity["selected_eye_ear"]
        ear_text = "N/A" if ear is None else f"{float(ear):.3f}"
        face_bbox = annotation["face_bbox_xyxy"]
        eye_bbox = annotation["visible_eye_bbox_xyxy"]

        displayed = (
            _focus_crop(panels["annotation"], face_bbox, scale_xy=(1.05, 1.05)),
            _focus_crop(panels["head"], face_bbox, scale_xy=(1.05, 1.05)),
            _focus_crop(panels["tail_vectors"], eye_bbox, scale_xy=(1.5, 2.2)),
            _rgb(panels["model_input"]),
        )
        subtitles = (
            "white polygon = affine eye crop",
            _format_vector(head_pose),
            f"angles/π={_format_vector(eye_angles)} · iris={_format_vector(iris_pose)} · "
            f"EAR={ear_text} · "
            f"{'PASS' if preprocessing_valid else 'BLOCK'}",
            "raw RGB crop · no overlay",
        )
        colors = (
            "#5478a8",
            "#d000d0",
            "#27a657" if preprocessing_valid else "#cf3030",
            "#2f78bd",
        )
        for column_index, (image, subtitle, color) in enumerate(
            zip(displayed, subtitles, colors, strict=True)
        ):
            axis = axes[row_index, column_index]
            axis.imshow(image, interpolation="nearest" if column_index == 3 else None)
            axis.set_title(
                (f"{column_titles[column_index]}\n" if row_index == 0 else "") + subtitle,
                fontsize=11,
                color=color,
                pad=8,
            )
            _style_diagnostic_axis(axis, color=color)
            if column_index == 0:
                axis.set_ylabel(sample_id, fontsize=12, fontweight="bold")

    figure.text(
        0.5,
        0.006,
        "Manual demo annotations · eye pose is an image-derived vertical cue, not calibrated "
        "screen gaze · a0/a1/a2 contribute two direction angles only · "
        "EAR is a validity gate and is not passed into the model",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#536174",
    )
    figure.tight_layout(rect=(0.015, 0.025, 0.995, 0.972), h_pad=2.0, w_pad=1.0)
    path = output_dir / "side90_all_model_inputs.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> int:
    args = _parser().parse_args()
    input_root = args.input_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    annotation_path = args.annotations.expanduser().resolve(strict=True)
    model_asset = args.model_asset.expanduser().resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    environment["GAZE_DATA_ROOT"] = str(input_root)
    environment["MEDIAPIPE_FACE_MODEL"] = str(model_asset)
    config = load_and_validate_config(
        args.config,
        profiles=(args.front_profile,),
        environ=environment,
        check_paths=False,
    )

    detector = MediaPipeFaceLandmarkerDetector(
        {"model_asset_path": str(model_asset), "mirror_retry": False}
    )
    rows: list[tuple[str, dict[str, np.ndarray]]] = []
    report: dict[str, Any] = {
        "warning": "All eye arrows are image-derived cues, not calibrated screen gaze labels.",
        "front": [],
        "front_model_inputs": {},
        "side_profile_90": [],
        "side_model_inputs": {},
        "plot_paths": {},
        "excluded_inputs": [],
    }
    front_results: list[tuple[str, Mapping[str, np.ndarray], Mapping[str, Any]]] = []
    try:
        preprocessor = GazePreprocessor(
            config,
            split="validation",
            landmark_detector=detector.detect,
        )
        front_paths = sorted((input_root / "front").glob("*"))
        front_paths = [
            path for path in front_paths if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if len(front_paths) != 2:
            raise ValueError(f"expected exactly two front images, found {len(front_paths)}")
        for image_path in front_paths:
            print(f"processing front: {image_path.name}", flush=True)
            panels, sample_report = _front_result(preprocessor, image_path)
            rows.append((f"front / {image_path.name}", panels))
            front_results.append((image_path.stem, panels, sample_report))
            report["front"].append(sample_report)
            report["front_model_inputs"][image_path.stem] = deepcopy(
                sample_report["front_model_inputs"]
            )
    finally:
        detector.close()

    side_results: list[
        tuple[str, Mapping[str, np.ndarray], Mapping[str, Any], Mapping[str, Any]]
    ] = []
    for sample_id, annotation in _load_profile_annotations(annotation_path):
        print(f"processing strict side: {sample_id}", flush=True)
        side_panels, side_report = _profile_result(input_root, sample_id, annotation)
        rows.append((f"side 90° / {sample_id}", side_panels))
        side_results.append((sample_id, side_panels, side_report, annotation))
        report["side_profile_90"].append(side_report)
        report["side_model_inputs"][sample_id] = deepcopy(side_report["side_model_inputs"])

    plot_path = _save_plot(output_dir, rows)
    all_front_inputs_plot_path = _save_all_front_model_inputs_plot(output_dir, front_results)
    front_crop_paths: dict[str, str] = {}
    for sample_id, front_panels, _front_report in front_results:
        crop_path = output_dir / f"{sample_id}_eye_strip_128x512.png"
        Image.fromarray(_rgb(front_panels["model_input"])).save(crop_path)
        front_crop_paths[sample_id] = str(crop_path)
    all_side_inputs_plot_path = _save_all_side_model_inputs_plot(output_dir, side_results)
    side_inputs_plot_paths: dict[str, str] = {}
    crop_paths: dict[str, str] = {}
    for sample_id, side_panels, side_report, annotation in side_results:
        side_inputs_plot_path = _save_side_model_inputs_plot(
            output_dir,
            sample_id,
            side_panels,
            side_report,
            annotation,
            filename=f"side90_model_inputs_{sample_id}.png",
        )
        crop_path = output_dir / f"{sample_id}_eye_crop_128x256.png"
        Image.fromarray(_rgb(side_panels["model_input"])).save(crop_path)
        side_inputs_plot_paths[sample_id] = str(side_inputs_plot_path)
        crop_paths[sample_id] = str(crop_path)
    report["plot_paths"] = {
        "combined_preprocessing": str(plot_path),
        "all_front_model_inputs": str(all_front_inputs_plot_path),
        "front_eye_strips": front_crop_paths,
        "all_side_model_inputs": str(all_side_inputs_plot_path),
        "side_model_inputs": side_inputs_plot_paths,
        "side_eye_crops": crop_paths,
    }
    report_path = output_dir / "front_and_profile90_report.json"
    report_path.write_text(
        json.dumps(deepcopy(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"plot: {plot_path}")
    print(f"all front model inputs plot: {all_front_inputs_plot_path}")
    for sample_id, crop_path in front_crop_paths.items():
        print(f"front eye strip ({sample_id}): {crop_path}")
    print(f"all side model inputs plot: {all_side_inputs_plot_path}")
    for sample_id in side_inputs_plot_paths:
        print(f"side model inputs plot ({sample_id}): {side_inputs_plot_paths[sample_id]}")
        print(f"side eye crop ({sample_id}): {crop_paths[sample_id]}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
