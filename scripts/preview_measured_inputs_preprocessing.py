"""Visualize measured dual-view samples using the ``inputs.csv`` contract.

Only ``head_down`` and ``neutral`` sessions are inspected.  Front samples are
accepted only when ``feature_maps/webeyetrack/inputs.csv`` declares
``valid=1``; in this dataset that flag means the upstream quality check
accepted the sample.

The corrected Front/Side frames are always read from ``feature_maps/*/frames``.
The Front model image is regenerated from the corrected Front frame by the
current MediaPipe + WebEyeTrack eye-region warp.  ``eye_patch_path`` is never
read.  A Side model patch needs a real visible-eye bbox; eyelid, iris and head
geometry are optional diagnostics. Missing Side ROI annotations produce an
explicit non-trainable panel; coordinates are never guessed or synthesized.

Raw data is read-only.  All generated PNG and JSON files go to a caller-owned
output directory that must not overlap the source database.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gaze_pipeline.config import load_and_validate_config
from gaze_pipeline.data.profile_side import preprocess_profile_side
from gaze_pipeline.data.transforms import GazePreprocessor
from gaze_pipeline.data.webeyetrack_compat import MediaPipeFaceLandmarkerDetector

DEFAULT_SESSIONS = ("head_down", "neutral")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
STRICT_SIDE_FIELDS = (
    "visible_eye_bbox_xyxy",
    "visible_eye_keypoints_xy",
    "iris_center_xy",
    "profile_head_origin_xy",
    "profile_head_forward_xy",
)
PANEL_WIDTH = 360
PANEL_HEIGHT = 235


class PreviewError(ValueError):
    """Raised when the preview contract cannot be satisfied safely."""


@dataclass(frozen=True, slots=True)
class PreviewSample:
    subject: str
    session: str
    pair_id: str
    sample_id: str
    inputs_csv: Path
    front_path: Path
    side_path: Path
    head_vector: tuple[float, float, float]
    face_origin_cm: tuple[float, float, float]
    valid: bool
    quality_score: float
    side_annotation: Mapping[str, Any] | None


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs/measured_preprocessing_preview",
    )
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--sessions", nargs="+", default=list(DEFAULT_SESSIONS))
    parser.add_argument(
        "--side-annotations",
        type=Path,
        default=None,
        help="Optional generated bbox annotation CSV keyed by pair_id.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs/config.yaml",
    )
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
    parser.add_argument("--mediapipe-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", str(value).strip())


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_paths(data_root: Path, output_dir: Path) -> tuple[Path, Path]:
    """Resolve paths and reject any output that can overwrite raw data."""

    data_root = data_root.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    if output_dir == data_root or _inside(output_dir, data_root) or _inside(data_root, output_dir):
        raise PreviewError("output directory and raw data root must not overlap")
    return data_root, output_dir


def _boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(str(row.get(field, "")).strip())
    except ValueError as exc:
        raise PreviewError(f"{field} is not numeric for {row.get('sample_id', '?')}") from exc
    if not math.isfinite(value):
        raise PreviewError(f"{field} is not finite for {row.get('sample_id', '?')}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return [dict(row) for row in reader]


def _normalized_child(parent: Path, name: str, *, directory: bool = False) -> Path:
    wanted = _nfc(name)
    matches = [
        child
        for child in parent.iterdir()
        if _nfc(child.name) == wanted and (child.is_dir() if directory else child.is_file())
    ]
    if len(matches) != 1:
        raise PreviewError(
            f"expected one local match for {name!r} below {parent}, got {len(matches)}"
        )
    return matches[0]


def _resolve_frame(session_root: Path, view: str, source_name: str) -> Path:
    frame_root = (session_root / f"feature_maps/{view}/frames").resolve(strict=True)
    basename = Path(str(source_name)).name
    path = _normalized_child(frame_root, basename)
    if path.suffix.lower() not in IMAGE_SUFFIXES or not _inside(path.resolve(), frame_root):
        raise PreviewError(f"invalid corrected {view} frame: {path}")
    return path


def _side_frame(session_root: Path, pair_id: str) -> Path:
    rows: list[dict[str, str]] = []
    for filename in ("training.csv", "evaluation.csv"):
        rows.extend(_read_csv(session_root / "feature_maps" / filename))
    matches = [
        row
        for row in rows
        if _nfc(row.get("pair_id", "")) == _nfc(pair_id)
        and row.get("view", "").strip().lower() == "phonecam"
    ]
    if len(matches) != 1:
        raise PreviewError(f"pair {pair_id!r} has {len(matches)} corrected Side rows")
    return _resolve_frame(session_root, "phone", matches[0].get("image_path", ""))


def _json_array(row: Mapping[str, str], field: str, shape: tuple[int, ...]) -> Any:
    raw = str(row.get(field, "")).strip()
    if not raw:
        raise PreviewError(f"{field} is empty")
    try:
        array = np.asarray(json.loads(raw), dtype=np.float32)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PreviewError(f"{field} is not a numeric JSON array") from exc
    if tuple(array.shape) != shape or not np.isfinite(array).all():
        raise PreviewError(f"{field} must have finite shape {shape}, got {array.shape}")
    return array.tolist()


def _optional_json_array(row: Mapping[str, str], field: str, shape: tuple[int, ...]) -> Any | None:
    if not str(row.get(field, "")).strip():
        return None
    return _json_array(row, field, shape)


def _side_annotation_from_row(row: Mapping[str, str]) -> Mapping[str, Any] | None:
    if not _boolean(row.get("eye_annotation_valid", "")):
        return None
    if any(
        not str(row.get(field, "")).strip() for field in ("visible_eye", "visible_eye_bbox_xyxy")
    ):
        return None
    try:
        return {
            "visible_eye": str(row.get("visible_eye", "unknown")),
            "visible_eye_bbox_xyxy": _json_array(row, "visible_eye_bbox_xyxy", (4,)),
            "visible_eye_keypoints_xy": _optional_json_array(
                row, "visible_eye_keypoints_xy", (6, 2)
            ),
            "iris_center_xy": _optional_json_array(row, "iris_center_xy", (2,)),
            "profile_head_origin_xy": _optional_json_array(row, "profile_head_origin_xy", (2,)),
            "profile_head_forward_xy": _optional_json_array(row, "profile_head_forward_xy", (2,)),
        }
    except PreviewError:
        return None


def _side_annotation(session_root: Path, pair_id: str) -> Mapping[str, Any] | None:
    """Return a real bbox annotation plus any available optional geometry."""

    rows = _read_csv(session_root / "feature_maps/phone/features.csv")
    matches = [row for row in rows if _nfc(row.get("pair_id", "")) == _nfc(pair_id)]
    if len(matches) != 1:
        return None
    return _side_annotation_from_row(matches[0])


def load_side_annotations(path: Path) -> dict[str, Mapping[str, Any]]:
    """Load a generated annotation table without writing to the raw database."""

    annotations: dict[str, Mapping[str, Any]] = {}
    for line, row in enumerate(_read_csv(path.resolve(strict=True)), start=2):
        pair_id = _nfc(row.get("pair_id", ""))
        if not pair_id:
            raise PreviewError(f"{path}:{line}: pair_id is empty")
        if pair_id in annotations:
            raise PreviewError(f"{path}:{line}: duplicate pair_id {pair_id!r}")
        annotation = _side_annotation_from_row(row)
        if annotation is None:
            raise PreviewError(f"{path}:{line}: valid visible-eye bbox is required")
        annotations[pair_id] = annotation
    return annotations


def discover_samples(
    data_root: Path,
    *,
    count: int = 2,
    sessions: Sequence[str] = DEFAULT_SESSIONS,
    side_annotations_by_pair: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[PreviewSample]:
    """Select inputs.csv-approved Front+Side pairs round-robin by subject."""

    if count < 2:
        raise PreviewError("at least two samples are required for a comparison preview")
    data_root = data_root.resolve(strict=True)
    selected_sessions = {_nfc(session) for session in sessions}
    candidates_by_subject: list[list[PreviewSample]] = []
    subjects = sorted(
        (path for path in data_root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: _nfc(path.name),
    )
    for subject_root in subjects:
        subject_candidates: list[PreviewSample] = []
        for session_root in sorted(
            (path for path in subject_root.iterdir() if path.is_dir()), key=lambda path: path.name
        ):
            if _nfc(session_root.name) not in selected_sessions:
                continue
            inputs_csv = session_root / "feature_maps/webeyetrack/inputs.csv"
            for row in _read_csv(inputs_csv):
                if not _boolean(row.get("valid", "")):
                    continue
                try:
                    head = tuple(
                        _finite_float(row, field)
                        for field in ("head_vector_x", "head_vector_y", "head_vector_z")
                    )
                    origin = tuple(
                        _finite_float(row, field)
                        for field in (
                            "face_origin_x_cm",
                            "face_origin_y_cm",
                            "face_origin_z_cm",
                        )
                    )
                    pair_id = str(row.get("pair_id", "")).strip()
                    sample_id = str(row.get("sample_id", "")).strip()
                    front_path = _resolve_frame(
                        session_root, "web", row.get("source_image_path", "")
                    )
                    side_path = _side_frame(session_root, pair_id)
                except (FileNotFoundError, PreviewError):
                    continue
                subject_candidates.append(
                    PreviewSample(
                        subject=_nfc(subject_root.name),
                        session=session_root.name,
                        pair_id=pair_id,
                        sample_id=sample_id,
                        inputs_csv=inputs_csv,
                        front_path=front_path,
                        side_path=side_path,
                        head_vector=head,  # type: ignore[arg-type]
                        face_origin_cm=origin,  # type: ignore[arg-type]
                        valid=True,
                        quality_score=_finite_float(row, "quality_score"),
                        side_annotation=(
                            side_annotations_by_pair.get(_nfc(pair_id))
                            if side_annotations_by_pair is not None
                            else _side_annotation(session_root, pair_id)
                        ),
                    )
                )
        subject_candidates.sort(key=lambda sample: (sample.session, sample.pair_id))
        if subject_candidates:
            candidates_by_subject.append(subject_candidates)

    selected: list[PreviewSample] = []
    selected_keys: set[tuple[str, str, str]] = set()
    selected_subjects: set[str] = set()

    # When both configured sessions exist, show both before adding more rows.
    # Prefer a different anonymous participant for each session.
    for requested_session in sessions:
        if len(selected) == count:
            return selected
        normalized_session = _nfc(requested_session)
        session_candidates = [
            sample
            for subject_candidates in candidates_by_subject
            for sample in subject_candidates
            if _nfc(sample.session) == normalized_session
        ]
        candidate = next(
            (sample for sample in session_candidates if sample.subject not in selected_subjects),
            session_candidates[0] if session_candidates else None,
        )
        if candidate is None:
            continue
        key = (candidate.subject, candidate.session, candidate.pair_id)
        selected.append(candidate)
        selected_keys.add(key)
        selected_subjects.add(candidate.subject)
    if len(selected) == count:
        return selected

    round_index = 0
    maximum_rounds = max((len(candidates) for candidates in candidates_by_subject), default=0)
    while len(selected) < count and round_index < maximum_rounds:
        for candidates in candidates_by_subject:
            if round_index < len(candidates):
                candidate = candidates[round_index]
                key = (candidate.subject, candidate.session, candidate.pair_id)
                if key in selected_keys:
                    continue
                selected.append(candidate)
                selected_keys.add(key)
                if len(selected) == count:
                    return selected
        round_index += 1
    raise PreviewError(f"requested {count} approved pairs, found only {len(selected)}")


def _mediapipe_worker(image_path: str, model_asset: str, queue: Any) -> None:
    """Run native MediaPipe in a child so a native abort cannot kill the preview."""

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
        queue.put(("ok", dict(detector.detect(image, "front"))))
    except BaseException as exc:  # native-process boundary must report every Python failure
        queue.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if detector is not None:
            detector.close()


def detect_mediapipe_isolated(
    image_path: Path, model_asset: Path, *, timeout_seconds: float
) -> dict[str, Any]:
    """Return real MediaPipe landmarks or fail clearly; no stored-ROI fallback."""

    if timeout_seconds <= 0:
        raise PreviewError("MediaPipe timeout must be positive")
    model_asset = model_asset.resolve(strict=True)
    context = mp.get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_mediapipe_worker,
        args=(str(image_path), str(model_asset), queue),
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise PreviewError(f"MediaPipe timed out after {timeout_seconds:g}s for a Front frame")
    try:
        status, payload = queue.get(timeout=0.5)
    except Empty as exc:
        raise PreviewError(
            f"MediaPipe child exited with code {process.exitcode} without a result"
        ) from exc
    finally:
        queue.close()
        queue.join_thread()
    if status != "ok":
        raise PreviewError(f"MediaPipe Front detection failed: {payload}")
    return dict(payload)


def load_front_pipeline_config(
    config_path: Path,
    front_profile: Path,
    *,
    data_root: Path,
    output_dir: Path,
    model_asset: Path,
) -> dict[str, Any]:
    """Resolve the same Front preprocessing profile used by training."""

    environment = dict(os.environ)
    environment.update(
        {
            "GAZE_DATA_ROOT": str(data_root),
            "GAZE_OUTPUT_ROOT": str(output_dir),
            "DUAL_VIEW_MANIFEST": str(output_dir / "preview-only-unused-manifest.csv"),
            "MEDIAPIPE_FACE_MODEL": str(model_asset),
        }
    )
    return load_and_validate_config(
        config_path,
        profiles=(front_profile,),
        environ=environment,
        check_paths=False,
    )


def regenerate_front_roi(
    sample: PreviewSample,
    *,
    config: Mapping[str, Any],
    model_asset: Path,
    timeout_seconds: float,
) -> Image.Image:
    """Run current Front preprocessing through WebEyeTrack eye-region warp."""

    detection = detect_mediapipe_isolated(
        sample.front_path, model_asset, timeout_seconds=timeout_seconds
    )

    def detected(_image: np.ndarray, _view: str) -> Mapping[str, Any]:
        return detection

    preprocessor = GazePreprocessor(
        config,
        split="validation",
        landmark_detector=detected,
    )
    pipeline_sample: dict[str, Any] = {
        "image_path": str(sample.front_path),
        "view": "front",
        "metadata": {
            "view": "front",
            "gaze_valid": True,
            "invalid_reasons": [],
            "head_vector": np.asarray(sample.head_vector, dtype=np.float32),
            "face_origin_3d": np.asarray(sample.face_origin_cm, dtype=np.float32),
            "head_orientation_valid": True,
            "face_origin_valid": True,
            "head_pose_valid": True,
            "face_origin_unit": "cm",
            "face_origin_source": "inputs.csv",
        },
    }
    reached_warp = False
    for stage in preprocessor.stage_order:
        stage_config = preprocessor._stage_config(stage, "front")
        if not bool(stage_config.get("enabled", True)):
            continue
        getattr(preprocessor, f"_stage_{stage}")(pipeline_sample, stage_config)
        if stage == "eye_region_warp":
            reached_warp = True
            break
    if not reached_warp:
        raise PreviewError("Front profile did not execute eye_region_warp")
    metadata = pipeline_sample["metadata"]
    if not bool(metadata.get("eye_region_warp_valid", False)):
        reasons = "; ".join(str(reason) for reason in metadata.get("invalid_reasons", []))
        raise PreviewError(f"fresh Front WebEyeTrack ROI is invalid: {reasons or 'unknown'}")
    patch = np.asarray(pipeline_sample["image"])
    if patch.shape != (128, 512, 3) or patch.dtype != np.uint8:
        raise PreviewError(
            f"fresh Front ROI must be uint8 [128,512,3], got {patch.shape}/{patch.dtype}"
        )
    return Image.fromarray(patch)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "#ececec")
    resized = image.convert("RGB")
    resized.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - resized.width) // 2
    y = (size[1] - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def _text_panel(
    lines: Sequence[str], size: tuple[int, int], *, warning: bool = False
) -> Image.Image:
    canvas = Image.new("RGB", size, "#fff4f4" if warning else "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(19)
    y = 30
    for line in lines:
        draw.text((22, y), line, font=font, fill="#9b1c1c" if warning else "#222222")
        y += 30
    return canvas


def _pose_overlay(sample: PreviewSample) -> Image.Image:
    canvas = Image.open(sample.front_path).convert("RGB")
    canvas.thumbnail((960, 600), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    font = _font(22)
    lines = (
        f"head_vector={tuple(round(value, 4) for value in sample.head_vector)}",
        f"face_origin_cm={tuple(round(value, 3) for value in sample.face_origin_cm)}",
        "numeric 3D values only (camera projection is not defined)",
        f"inputs.csv valid=1 / quality={sample.quality_score:.3f}",
    )
    height = 30 * len(lines) + 16
    draw.rectangle((0, 0, canvas.width, height), fill=(0, 0, 0, 190))
    for index, line in enumerate(lines):
        draw.text((14, 8 + index * 30), line, font=font, fill="white")
    return canvas


def _side_roi(sample: PreviewSample) -> tuple[Image.Image, bool]:
    annotation = sample.side_annotation
    if annotation is None:
        return (
            _text_panel(
                (
                    "SIDE ROI NOT GENERATED",
                    "real profile annotation missing",
                    "no coordinates fabricated",
                    "Side branch: not trainable",
                ),
                (PANEL_WIDTH, PANEL_HEIGHT),
                warning=True,
            ),
            False,
        )
    image = np.asarray(Image.open(sample.side_path).convert("RGB"))
    eyelids = annotation.get("visible_eye_keypoints_xy")
    iris = annotation.get("iris_center_xy")
    head_origin = annotation.get("profile_head_origin_xy")
    head_forward = annotation.get("profile_head_forward_xy")
    has_head = head_origin is not None and head_forward is not None
    result = preprocess_profile_side(
        image,
        eye_bbox_xyxy=annotation["visible_eye_bbox_xyxy"],
        eyelid_keypoints_xy=eyelids,
        iris_center_xy=iris,
        head_origin_xy=head_origin,
        head_forward_point_xy=head_forward,
        output_size_hw=(128, 256),
        crop_mode="affine" if eyelids is not None else "stretch",
        vertical_only=True,
        extract_head_pose=has_head,
        extract_iris_pose=iris is not None,
        extract_eye_angles=eyelids is not None,
    )
    return Image.fromarray(result.patch), True


def _save_stage(image: Image.Image, path: Path, *, preserve_model_size: bool = False) -> None:
    saved = image if preserve_model_size else _fit(image, (960, 600))
    saved.save(path, format="PNG")


def render_preview(
    samples: Sequence[PreviewSample],
    output_dir: Path,
    *,
    front_roi_factory: Callable[[PreviewSample], Image.Image],
    overwrite: bool = False,
) -> Path:
    """Write per-stage images, a comparison panel, and an auditable JSON summary."""

    if len(samples) < 2:
        raise PreviewError("render_preview requires at least two samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / "measured_preprocessing_comparison.png"
    summary_path = output_dir / "preview_summary.json"
    known = [panel_path, summary_path]
    if not overwrite and any(path.exists() for path in known):
        raise FileExistsError("preview output already exists; pass --overwrite to replace it")

    titles = (
        "Corrected Front source",
        "Corrected Side source",
        "Front pose from inputs.csv",
        "Front WebEyeTrack ROI",
        "Side strict-profile ROI",
    )
    margin = 24
    header_height = 98
    row_label_width = 220
    title_height = 42
    canvas_width = margin * 2 + row_label_width + len(titles) * PANEL_WIDTH
    canvas_height = header_height + title_height + len(samples) * PANEL_HEIGHT + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 14),
        "Measured preprocessing: inputs.csv-approved pairs only",
        font=_font(27),
        fill="#111111",
    )
    draw.text(
        (margin, 52),
        "Corrected frames -> Front WebEyeTrack ROI + 3D pose | Side ROI requires real annotation",
        font=_font(20),
        fill="#444444",
    )
    for column, title in enumerate(titles):
        x = margin + row_label_width + column * PANEL_WIDTH
        draw.rectangle(
            (x, header_height, x + PANEL_WIDTH, header_height + title_height), fill="#eeeeee"
        )
        draw.text((x + 12, header_height + 9), title, font=_font(18), fill="#111111")

    summaries: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        prefix = f"sample_{index + 1:02d}"
        front = Image.open(sample.front_path).convert("RGB")
        side = Image.open(sample.side_path).convert("RGB")
        pose = _pose_overlay(sample)
        front_roi = front_roi_factory(sample).convert("RGB")
        if front_roi.size != (512, 128):
            raise PreviewError(f"fresh Front ROI must be 512x128 pixels, got {front_roi.size}")
        side_roi, side_available = _side_roi(sample)
        stages = {
            "front_corrected": front,
            "side_corrected": side,
            "front_pose": pose,
            "front_webeyetrack_roi": front_roi,
            "side_profile_roi": side_roi,
        }
        for name, image in stages.items():
            preserve_model_size = name == "front_webeyetrack_roi" or (
                name == "side_profile_roi" and side_available
            )
            _save_stage(
                image,
                output_dir / f"{prefix}_{name}.png",
                preserve_model_size=preserve_model_size,
            )

        y = header_height + title_height + index * PANEL_HEIGHT
        draw.rectangle((margin, y, margin + row_label_width, y + PANEL_HEIGHT), fill="#fafafa")
        label_lines = (
            f"Sample {index + 1}",
            f"participant: anonymous-{index + 1}",
            f"session: {sample.session}",
            "inputs.csv valid: YES",
            f"Side trainable: {'YES' if side_available else 'NO'}",
        )
        for line_index, line in enumerate(label_lines):
            draw.text(
                (margin + 12, y + 18 + line_index * 31),
                line,
                font=_font(18),
                fill="#117733" if line.startswith("inputs.csv valid") else "#222222",
            )
        for column, image in enumerate(stages.values()):
            x = margin + row_label_width + column * PANEL_WIDTH
            canvas.paste(_fit(image, (PANEL_WIDTH, PANEL_HEIGHT)), (x, y))

        summaries.append(
            {
                "participant": f"anonymous-{index + 1}",
                "session": sample.session,
                "valid": sample.valid,
                "inputs_csv_accepted": True,
                "quality_score": sample.quality_score,
                "head_vector": list(sample.head_vector),
                "face_origin_cm": list(sample.face_origin_cm),
                "front_source": "corrected feature_maps/web/frames image",
                "side_source": "corrected feature_maps/phone/frames image",
                "front_roi_source": (
                    "fresh corrected Front frame -> MediaPipe -> WebEyeTrack eye_region_warp"
                ),
                "front_model_image_shape_hwc": [128, 512, 3],
                "side_annotation_available": side_available,
                "side_training_eligible": side_available,
                "side_model_image_shape_hwc": [128, 256, 3] if side_available else None,
            }
        )

    canvas.save(panel_path, format="PNG")
    summary_path.write_text(
        json.dumps(
            {
                "contract": {
                    "source": "feature_maps/webeyetrack/inputs.csv",
                    "authoritative_fields": ["head_vector", "face_origin", "valid"],
                    "ignored_field": "eye_patch_path (stored ROI is never read)",
                    "accepted_valid_values": ["1", "true", "yes", "y"],
                    "sample_filter": "only rows accepted by inputs.csv valid=1",
                    "sessions": list(DEFAULT_SESSIONS),
                    "side_policy": "real visible-eye bbox; optional geometry; never fabricated",
                },
                "samples": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return panel_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    data_root, output_dir = validate_paths(args.data_root, args.output_dir)
    model_asset = args.model_asset.resolve(strict=True)
    config = load_front_pipeline_config(
        args.config,
        args.front_profile,
        data_root=data_root,
        output_dir=output_dir,
        model_asset=model_asset,
    )
    side_annotations = (
        load_side_annotations(args.side_annotations) if args.side_annotations is not None else None
    )
    samples = discover_samples(
        data_root,
        count=args.samples,
        sessions=args.sessions,
        side_annotations_by_pair=side_annotations,
    )
    panel = render_preview(
        samples,
        output_dir,
        front_roi_factory=lambda sample: regenerate_front_roi(
            sample,
            config=config,
            model_asset=model_asset,
            timeout_seconds=args.mediapipe_timeout_seconds,
        ),
        overwrite=args.overwrite,
    )
    print(f"selected inputs.csv-approved pairs: {len(samples)}")
    for index, sample in enumerate(samples, start=1):
        side_status = "real annotation" if sample.side_annotation is not None else "missing"
        print(f"- anonymous-{index}/{sample.session}: Side annotation={side_status}")
    print(f"comparison panel: {panel}")
    print(f"summary: {output_dir / 'preview_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
