"""Generate and quality-check Side eye bounding boxes from phone frames.

The measured source database is read-only. The script writes one annotation row
for every discovered phonecam master sample, plus a review queue and contact
sheets. ``webeyetrack/inputs.csv`` is authoritative for eye state: ``valid=1``
means open and eligible, while ``valid=0`` means closed and excludes the whole
pair from detection and review. High-confidence MediaPipe/YuNet annotations can
be accepted automatically; lower-confidence candidates remain invalid until
reviewed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from gaze_pipeline.data.side_annotation import (
    SideAnnotationError,
    SideEyeAnnotation,
    annotation_from_bbox_candidate,
    annotation_from_landmarks,
    annotation_from_temporal_neighbors,
    bbox_as_ints,
)
from gaze_pipeline.data.webeyetrack_compat import (
    MediaPipeFaceLandmarkerDetector,
    WebEyeTrackGeometryError,
)

DEFAULT_SESSIONS = ("head_down", "neutral")
MASTER_FILES = ("training.csv", "evaluation.csv")
OUTPUT_FIELDS = (
    "sample_id",
    "participant",
    "head_pose",
    "pair_id",
    "input_valid",
    "eye_state",
    "input_invalid_reason",
    "visible_eye",
    "visible_eye_bbox_xyxy",
    "visible_eye_keypoints_xy",
    "iris_center_xy",
    "eye_annotation_valid",
    "detection_method",
    "detection_confidence",
    "visibility_scores",
    "quality_flags",
    "fallback_reason",
    "source_image_path",
    "review_status",
)


class SideGenerationError(RuntimeError):
    """Raised when Side annotations cannot be generated safely."""


@dataclass(frozen=True, slots=True)
class PhoneSample:
    sample_id: str
    participant: str
    session: str
    pair_id: str
    source_frame: int
    image_path: Path
    input_valid: bool | None
    input_invalid_reason: str


@dataclass(frozen=True, slots=True)
class GeneratedRow:
    sample: PhoneSample
    annotation: SideEyeAnnotation | None
    reason: str


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/side_annotations.csv.",
    )
    parser.add_argument("--sessions", nargs="+", default=list(DEFAULT_SESSIONS))
    parser.add_argument(
        "--model-asset",
        type=Path,
        default=project_root / "models" / "face_landmarker_v2_with_blendshapes.task",
    )
    parser.add_argument(
        "--yunet-model",
        type=Path,
        default=project_root / "models" / "face_detection_yunet_2023mar.onnx",
    )
    parser.add_argument(
        "--detector",
        choices=("hybrid", "mediapipe", "yunet", "haar"),
        default="hybrid",
    )
    parser.add_argument("--review-overrides", type=Path, default=None)
    parser.add_argument("--accept-haar", action="store_true")
    parser.add_argument("--accept-yunet", action="store_true")
    parser.add_argument(
        "--haar-visible-eye",
        choices=("left", "right"),
        default="right",
        help="Semantic eye label for Haar fallback when landmarks are unavailable.",
    )
    parser.add_argument("--min-eye-width-px", type=float, default=12.0)
    parser.add_argument("--min-eye-opening-px", type=float, default=1.5)
    parser.add_argument("--min-visibility-dominance", type=float, default=1.05)
    parser.add_argument("--min-auto-confidence", type=float, default=0.80)
    parser.add_argument("--min-sharpness", type=float, default=18.0)
    parser.add_argument("--yunet-score-threshold", type=float, default=0.60)
    parser.add_argument("--yunet-auto-accept-score", type=float, default=0.84)
    parser.add_argument("--yunet-bbox-face-width-ratio", type=float, default=0.24)
    parser.add_argument(
        "--profile-facing-direction",
        choices=("left", "right", "auto"),
        default="left",
        help="Direction the photographed profile faces in the phone image.",
    )
    parser.add_argument("--temporal-min-auto-confidence", type=float, default=0.60)
    parser.add_argument("--temporal-max-auto-gap", type=int, default=20)
    parser.add_argument("--no-temporal-fill", action="store_true")
    parser.add_argument("--preview-per-subject", type=int, default=4)
    parser.add_argument("--review-preview-limit", type=int, default=400)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--sample-per-session",
        type=int,
        default=None,
        help="Evenly sample this many frames per participant/session for a smoke run.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return exit code 3 when any row still requires manual review.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def _nfc(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "").strip())


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                str(key).strip(): "" if value is None else str(value).strip()
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"))


def _resolve_unicode_path(base: Path, raw_value: str) -> Path:
    raw = Path(str(raw_value).strip())
    if raw.is_absolute() or ".." in raw.parts:
        raise SideGenerationError(f"unsafe image path: {raw_value!r}")
    current = base
    for part in raw.parts:
        direct = current / part
        if direct.exists():
            current = direct
            continue
        matches = [child for child in current.iterdir() if _nfc(child.name) == _nfc(part)]
        if len(matches) != 1:
            raise SideGenerationError(f"cannot resolve {part!r} under {current}")
        current = matches[0]
    if not current.is_file():
        raise SideGenerationError(f"phone frame does not exist: {current}")
    return current


def discover_phone_samples(data_root: Path, sessions: Sequence[str]) -> list[PhoneSample]:
    root = data_root.expanduser().resolve(strict=True)
    samples: list[PhoneSample] = []
    seen: set[str] = set()
    for subject_dir in sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: _nfc(path.name),
    ):
        participant = _nfc(subject_dir.name)
        for session in sessions:
            feature_root = subject_dir / session / "feature_maps"
            if not feature_root.is_dir():
                continue
            inputs_path = feature_root / "webeyetrack" / "inputs.csv"
            input_rows = _read_csv(inputs_path)
            inputs_by_pair: dict[str, tuple[bool, str]] = {}
            for input_row in input_rows:
                pair_id = _nfc(input_row.get("pair_id", ""))
                if not pair_id:
                    raise SideGenerationError(f"{inputs_path}: row has no pair_id")
                if pair_id in inputs_by_pair:
                    raise SideGenerationError(f"{inputs_path}: duplicate pair_id: {pair_id}")
                raw_valid = str(input_row.get("valid", "")).strip()
                if raw_valid not in {"0", "1"}:
                    raise SideGenerationError(
                        f"{inputs_path}: valid must be exactly 0 or 1 for {pair_id}; "
                        f"got {raw_valid!r}"
                    )
                inputs_by_pair[pair_id] = (
                    raw_valid == "1",
                    str(input_row.get("invalid_reason", "")).strip(),
                )
            for master_file in MASTER_FILES:
                for row in _read_csv(feature_root / master_file):
                    if str(row.get("view", "")).strip().lower() != "phonecam":
                        continue
                    sample_id = _nfc(row.get("sample_id", ""))
                    if not sample_id:
                        raise SideGenerationError(
                            f"{feature_root / master_file}: phonecam row has no sample_id"
                        )
                    if sample_id in seen:
                        raise SideGenerationError(f"duplicate phonecam sample_id: {sample_id}")
                    seen.add(sample_id)
                    try:
                        source_frame = int(float(str(row.get("source_frame", ""))))
                    except ValueError as exc:
                        raise SideGenerationError(
                            f"{feature_root / master_file}: invalid source_frame for {sample_id}"
                        ) from exc
                    pair_id = _nfc(row.get("pair_id", ""))
                    if not pair_id:
                        raise SideGenerationError(
                            f"{feature_root / master_file}: phonecam row has no pair_id"
                        )
                    input_state = inputs_by_pair.get(pair_id)
                    if input_state is None:
                        input_state = (
                            None,
                            (
                                f"missing inputs.csv: {inputs_path}"
                                if not inputs_path.is_file()
                                else f"no matching inputs.csv row for pair_id={pair_id}"
                            ),
                        )
                    samples.append(
                        PhoneSample(
                            sample_id=sample_id,
                            participant=participant,
                            session=session,
                            pair_id=pair_id,
                            source_frame=source_frame,
                            image_path=_resolve_unicode_path(
                                feature_root,
                                str(row.get("image_path", "")),
                            ),
                            input_valid=input_state[0],
                            input_invalid_reason=input_state[1],
                        )
                    )
    return samples


def _sample_per_session(samples: Sequence[PhoneSample], count: int) -> list[PhoneSample]:
    if count <= 0:
        return []
    groups: dict[tuple[str, str], list[PhoneSample]] = defaultdict(list)
    for sample in samples:
        groups[(sample.participant, sample.session)].append(sample)
    selected: list[PhoneSample] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda sample: sample.source_frame)
        if len(group) <= count:
            selected.extend(group)
            continue
        if count == 1:
            positions = [len(group) // 2]
        else:
            positions = [round(index * (len(group) - 1) / (count - 1)) for index in range(count)]
        selected.extend(group[position] for position in positions)
    return selected


def _load_overrides(path: Path | None) -> dict[str, Mapping[str, str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise SideGenerationError(f"review override CSV does not exist: {path}")
    rows = _read_csv(path)
    result: dict[str, Mapping[str, str]] = {}
    for row in rows:
        sample_id = _nfc(row.get("sample_id", ""))
        if not sample_id:
            raise SideGenerationError(f"{path}: override row has no sample_id")
        if sample_id in result:
            raise SideGenerationError(f"{path}: duplicate override for {sample_id}")
        result[sample_id] = row
    return result


def _decode_bbox(raw: Any) -> list[float]:
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise SideGenerationError("visible_eye_bbox_xyxy must be valid JSON") from exc
    if not isinstance(value, list) or len(value) != 4:
        raise SideGenerationError("visible_eye_bbox_xyxy must have four values")
    return [float(item) for item in value]


class HaarProfileEyeDetector:
    """Profile-face plus eye-cascade fallback that always requires QC review."""

    def __init__(self) -> None:
        root = Path(cv2.data.haarcascades)
        self.profile = cv2.CascadeClassifier(str(root / "haarcascade_profileface.xml"))
        self.eye = cv2.CascadeClassifier(str(root / "haarcascade_eye_tree_eyeglasses.xml"))
        if self.profile.empty() or self.eye.empty():
            raise SideGenerationError("OpenCV profile-face/eye Haar cascades are unavailable")

    def detect(
        self,
        rgb: np.ndarray,
        *,
        visible_eye: str,
        accept: bool,
    ) -> SideEyeAnnotation:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.profile.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(120, 120),
        )
        mirrored = False
        if len(faces) == 0:
            mirrored = True
            faces = self.profile.detectMultiScale(
                cv2.flip(gray, 1),
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(120, 120),
            )
        if len(faces) == 0:
            raise SideGenerationError("Haar profile-face detector found no face")
        x, y, width, height = max(
            (tuple(int(value) for value in face) for face in faces),
            key=lambda box: box[2] * box[3],
        )
        if mirrored:
            x = gray.shape[1] - x - width
        face_gray = gray[y : y + max(2, int(height * 0.68)), x : x + width]
        eyes = self.eye.detectMultiScale(
            face_gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(12, 8),
        )
        if len(eyes) == 0:
            raise SideGenerationError("Haar eye detector found no eye inside profile face")
        eye_x, eye_y, eye_w, eye_h = max(
            (tuple(int(value) for value in eye) for eye in eyes),
            key=lambda box: box[2] * box[3],
        )
        center_x = x + eye_x + eye_w / 2.0
        center_y = y + eye_y + eye_h / 2.0
        crop_width = max(float(eye_w) * 1.8, float(eye_h) * 2.4)
        crop_height = crop_width / 2.0
        bbox = [
            center_x - crop_width / 2.0,
            center_y - crop_height / 2.0,
            center_x + crop_width / 2.0,
            center_y + crop_height / 2.0,
        ]
        return annotation_from_bbox_candidate(
            bbox,
            image_size_hw=rgb.shape[:2],
            visible_eye=visible_eye,
            method="opencv_haar_profile_eye",
            accept=accept,
        )


class YuNetProfileEyeDetector:
    """OpenCV Zoo YuNet face detector with five-point eye candidates."""

    def __init__(self, model_path: Path, *, score_threshold: float) -> None:
        if not model_path.is_file():
            raise SideGenerationError(f"YuNet model asset is missing: {model_path}")
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            float(score_threshold),
            0.3,
            5000,
        )

    def detect(
        self,
        rgb: np.ndarray,
        *,
        bbox_face_width_ratio: float,
        facing_direction: str,
        accept: bool,
        auto_accept_score: float,
    ) -> SideEyeAnnotation:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image_height, image_width = bgr.shape[:2]
        self.detector.setInputSize((image_width, image_height))
        _, faces = self.detector.detect(bgr)
        if faces is None or len(faces) == 0:
            raise SideGenerationError("YuNet face detector found no face")
        face = max(
            faces,
            key=lambda row: float(row[2] * row[3]) * max(float(row[14]), 0.0),
        )
        x, y, face_width, face_height = (float(value) for value in face[:4])
        if face_width <= 0 or face_height <= 0:
            raise SideGenerationError("YuNet returned a non-positive face bbox")
        first_eye = np.asarray(face[4:6], dtype=np.float64)
        second_eye = np.asarray(face[6:8], dtype=np.float64)
        normalized_direction = facing_direction.strip().lower()
        if normalized_direction == "auto":
            nose_x = float(face[8])
            normalized_direction = (
                "left" if nose_x < float((first_eye[0] + second_eye[0]) / 2.0) else "right"
            )
        looking_left = normalized_direction == "left"
        eye_point = min((first_eye, second_eye), key=lambda point: point[0])
        normalized_eye = "right"
        if not looking_left:
            eye_point = max((first_eye, second_eye), key=lambda point: point[0])
            normalized_eye = "left"
        eye_x, eye_y = float(eye_point[0]), float(eye_point[1])
        if not (x <= eye_x <= x + face_width and y <= eye_y <= y + face_height):
            raise SideGenerationError("YuNet eye landmark lies outside the face bbox")
        crop_width = max(12.0, face_width * float(bbox_face_width_ratio))
        crop_height = crop_width / 2.0
        face_score = float(np.clip(face[14], 0.0, 1.0))
        candidate = annotation_from_bbox_candidate(
            [
                eye_x - crop_width / 2.0,
                eye_y - crop_height / 2.0,
                eye_x + crop_width / 2.0,
                eye_y + crop_height / 2.0,
            ],
            image_size_hw=(image_height, image_width),
            visible_eye=normalized_eye,
            method="opencv_yunet_profile_eye",
            accept=accept or face_score >= auto_accept_score,
        )
        return replace(candidate, confidence=face_score)


def _crop_quality(
    rgb: np.ndarray,
    annotation: SideEyeAnnotation,
    *,
    min_sharpness: float,
) -> SideEyeAnnotation:
    x0, y0, x1, y1 = bbox_as_ints(annotation)
    crop = rgb[max(0, y0) : min(rgb.shape[0], y1), max(0, x0) : min(rgb.shape[1], x1)]
    flags = list(annotation.quality_flags)
    if crop.size == 0:
        flags.append("empty_eye_crop")
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        if sharpness < min_sharpness:
            flags.append("low_sharpness")
        if brightness < 20.0:
            flags.append("too_dark")
        if brightness > 245.0:
            flags.append("too_bright")
    unique_flags = tuple(dict.fromkeys(flags))
    return replace(annotation, quality_flags=unique_flags, valid=not unique_flags)


def _manual_annotation(
    row: Mapping[str, str],
    *,
    image_size_hw: tuple[int, int],
) -> SideEyeAnnotation:
    return annotation_from_bbox_candidate(
        _decode_bbox(row.get("visible_eye_bbox_xyxy", "")),
        image_size_hw=image_size_hw,
        visible_eye=str(row.get("visible_eye", "")),
        method="manual_override",
        accept=_truthy(row.get("eye_annotation_valid", "")),
    )


def _annotation_row(item: GeneratedRow, root: Path) -> dict[str, str]:
    annotation = item.annotation
    base = {
        "sample_id": item.sample.sample_id,
        "participant": item.sample.participant,
        "head_pose": item.sample.session,
        "pair_id": item.sample.pair_id,
        "input_valid": (
            "1"
            if item.sample.input_valid is True
            else "0"
            if item.sample.input_valid is False
            else ""
        ),
        "eye_state": (
            "open"
            if item.sample.input_valid is True
            else "closed"
            if item.sample.input_valid is False
            else "unknown"
        ),
        "input_invalid_reason": item.sample.input_invalid_reason,
        "source_image_path": item.sample.image_path.relative_to(root).as_posix(),
    }
    if item.sample.input_valid is not True:
        known_closed = item.sample.input_valid is False
        return {
            **base,
            "visible_eye": "",
            "visible_eye_bbox_xyxy": "",
            "visible_eye_keypoints_xy": "",
            "iris_center_xy": "",
            "eye_annotation_valid": "false",
            "detection_method": (
                "excluded_inputs_valid_0" if known_closed else "excluded_missing_input_state"
            ),
            "detection_confidence": "0.0",
            "visibility_scores": "{}",
            "quality_flags": _compact_json(
                ["closed_eye_pair_excluded" if known_closed else "missing_input_eye_state"]
            ),
            "fallback_reason": item.sample.input_invalid_reason or "inputs.csv valid=0",
            "review_status": "excluded",
        }
    if annotation is None:
        return {
            **base,
            "visible_eye": "",
            "visible_eye_bbox_xyxy": "",
            "visible_eye_keypoints_xy": "",
            "iris_center_xy": "",
            "eye_annotation_valid": "false",
            "detection_method": "failed",
            "detection_confidence": "0.0",
            "visibility_scores": "{}",
            "quality_flags": _compact_json([item.reason]),
            "fallback_reason": item.reason,
            "review_status": "required",
        }
    bbox = [float(value) for value in bbox_as_ints(annotation)]
    return {
        **base,
        "visible_eye": annotation.visible_eye,
        "visible_eye_bbox_xyxy": _compact_json(bbox),
        "visible_eye_keypoints_xy": (
            _compact_json(annotation.eyelid_keypoints_xy)
            if annotation.eyelid_keypoints_xy is not None
            else ""
        ),
        "iris_center_xy": (
            _compact_json(annotation.iris_center_xy)
            if annotation.iris_center_xy is not None
            else ""
        ),
        "eye_annotation_valid": str(annotation.valid).lower(),
        "detection_method": annotation.method,
        "detection_confidence": f"{annotation.confidence:.6f}",
        "visibility_scores": _compact_json(annotation.visibility_scores),
        "quality_flags": _compact_json(annotation.quality_flags),
        "fallback_reason": item.reason if annotation.method.startswith("temporal_") else "",
        "review_status": "accepted" if annotation.valid else "required",
    }


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _contact_sheets(
    items: Sequence[GeneratedRow],
    *,
    output_dir: Path,
    prefix: str,
    limit: int | None = None,
) -> list[str]:
    selected = list(items[:limit] if limit is not None else items)
    if not selected:
        return []
    cell_width, cell_height = 320, 215
    columns, rows_per_page = 4, 4
    page_size = columns * rows_per_page
    written: list[str] = []
    for page_index in range(0, len(selected), page_size):
        page_items = selected[page_index : page_index + page_size]
        sheet = Image.new("RGB", (cell_width * columns, cell_height * rows_per_page), "black")
        draw = ImageDraw.Draw(sheet)
        for cell_index, item in enumerate(page_items):
            offset_x = (cell_index % columns) * cell_width
            offset_y = (cell_index // columns) * cell_height + 22
            try:
                with Image.open(item.sample.image_path) as source:
                    source_width, source_height = source.size
                    image = source.convert("RGB")
            except OSError as exc:
                draw.text(
                    (offset_x + 4, offset_y + 4),
                    f"SOURCE-UNAVAILABLE: {type(exc).__name__}",
                    fill="orange",
                )
                continue
            image = ImageOps.contain(image, (cell_width, cell_height - 22))
            sheet.paste(image, (offset_x, offset_y))
            scale_x = image.width / source_width
            scale_y = image.height / source_height
            if item.annotation is not None:
                x0, y0, x1, y1 = bbox_as_ints(item.annotation)
                draw.rectangle(
                    [
                        offset_x + x0 * scale_x,
                        offset_y + y0 * scale_y,
                        offset_x + x1 * scale_x,
                        offset_y + y1 * scale_y,
                    ],
                    outline="lime" if item.annotation.valid else "red",
                    width=3,
                )
                status = "ACCEPT" if item.annotation.valid else "REVIEW-NOT-USED"
                label = (
                    f"{page_index + cell_index + 1} {status} "
                    f"{item.annotation.method} {item.annotation.confidence:.2f}"
                )
            else:
                label = f"{page_index + cell_index + 1} detection_failed"
            draw.text((offset_x + 4, offset_y - 19), label, fill="white")
        filename = f"{prefix}_{page_index // page_size + 1:04d}.jpg"
        sheet.save(output_dir / filename, quality=90)
        written.append(filename)
    return written


def _fill_temporal_candidates(
    generated: Sequence[GeneratedRow],
    *,
    min_sharpness: float,
    min_auto_confidence: float,
    max_auto_gap: int,
) -> list[GeneratedRow]:
    groups: dict[tuple[str, str], list[tuple[int, GeneratedRow]]] = defaultdict(list)
    filled: dict[str, GeneratedRow] = {}
    for item in generated:
        if item.sample.input_valid is not True:
            filled[item.sample.sample_id] = item
            continue
        groups[(item.sample.participant, item.sample.session)].append(
            (item.sample.source_frame, item)
        )

    for group_items in groups.values():
        ordered = [item for _, item in sorted(group_items, key=lambda pair: pair[0])]
        anchors: list[tuple[int, SideEyeAnnotation]] = []
        for position, item in enumerate(ordered):
            annotation = item.annotation
            if annotation is None:
                continue
            photometric_only = set(annotation.quality_flags) <= {
                "low_sharpness",
                "too_dark",
                "too_bright",
            }
            if annotation.valid or (
                annotation.method == "mediapipe_face_landmarker" and photometric_only
            ):
                anchors.append((position, annotation))
        for position, item in enumerate(ordered):
            if (item.annotation is not None and item.annotation.valid) or not anchors:
                filled[item.sample.sample_id] = item
                continue
            previous = next(
                (anchor for anchor in reversed(anchors) if anchor[0] < position),
                None,
            )
            following = next((anchor for anchor in anchors if anchor[0] > position), None)
            try:
                rgb = _load_rgb(item.sample.image_path)
                annotation = annotation_from_temporal_neighbors(
                    target_position=position,
                    image_size_hw=rgb.shape[:2],
                    previous=previous,
                    following=following,
                    min_auto_confidence=min_auto_confidence,
                    max_auto_gap=max_auto_gap,
                )
                annotation = _crop_quality(
                    rgb,
                    annotation,
                    min_sharpness=min_sharpness,
                )
                if (
                    item.annotation is None
                    or annotation.valid
                    or annotation.method == "temporal_interpolation"
                ):
                    filled[item.sample.sample_id] = replace(item, annotation=annotation)
                else:
                    filled[item.sample.sample_id] = item
            except (OSError, SideAnnotationError, ValueError) as exc:
                filled[item.sample.sample_id] = replace(
                    item,
                    reason=f"{item.reason} | temporal:{exc}",
                )
    return [filled.get(item.sample.sample_id, item) for item in generated]


def generate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.data_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_csv = (
        args.output_csv.expanduser().resolve(strict=False)
        if args.output_csv is not None
        else output_dir / "side_annotations.csv"
    )
    if output_csv.exists() and not args.force:
        raise SideGenerationError(f"output exists; pass --force to replace it: {output_csv}")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_phone_samples(root, tuple(args.sessions))
    if args.sample_per_session is not None:
        samples = _sample_per_session(samples, max(0, int(args.sample_per_session)))
    if args.max_samples is not None:
        samples = samples[: max(0, int(args.max_samples))]
    if not samples:
        raise SideGenerationError("no phonecam samples were discovered")
    overrides = _load_overrides(args.review_overrides)

    mediapipe_detector: MediaPipeFaceLandmarkerDetector | None = None
    if args.detector in {"hybrid", "mediapipe"}:
        model_asset = args.model_asset.expanduser().resolve(strict=False)
        if not model_asset.is_file():
            raise SideGenerationError(
                f"MediaPipe model asset is missing: {model_asset}; "
                "run 'make webeyetrack-assets' first"
            )
        mediapipe_detector = MediaPipeFaceLandmarkerDetector(
            {
                "model_asset_path": str(model_asset),
                "num_faces": 1,
                "mirror_retry": True,
                "min_face_detection_confidence": 0.35,
                "min_face_presence_confidence": 0.35,
                "min_tracking_confidence": 0.35,
            }
        )
    yunet_detector: YuNetProfileEyeDetector | None = None
    if args.detector in {"hybrid", "yunet"}:
        yunet_detector = YuNetProfileEyeDetector(
            args.yunet_model.expanduser().resolve(strict=False),
            score_threshold=args.yunet_score_threshold,
        )
    haar_detector = HaarProfileEyeDetector() if args.detector in {"hybrid", "haar"} else None
    generated: list[GeneratedRow] = []
    try:
        for index, sample in enumerate(samples, start=1):
            if sample.input_valid is not True:
                generated.append(
                    GeneratedRow(
                        sample=sample,
                        annotation=None,
                        reason=sample.input_invalid_reason or "inputs.csv valid=0",
                    )
                )
                continue
            rgb = _load_rgb(sample.image_path)
            annotation: SideEyeAnnotation | None = None
            failure_reasons: list[str] = []
            override = overrides.get(sample.sample_id)
            if override is not None:
                try:
                    annotation = _manual_annotation(override, image_size_hw=rgb.shape[:2])
                except (SideAnnotationError, SideGenerationError, ValueError) as exc:
                    failure_reasons.append(f"manual_override:{exc}")
            elif mediapipe_detector is not None:
                try:
                    detected = mediapipe_detector.detect(rgb, "side")
                    annotation = annotation_from_landmarks(
                        detected["landmarks_xy"],
                        image_size_hw=rgb.shape[:2],
                        presence=detected.get("landmark_presence"),
                        visibility=detected.get("landmark_visibility"),
                        min_eye_width_px=args.min_eye_width_px,
                        min_eye_opening_px=args.min_eye_opening_px,
                        min_visibility_dominance=args.min_visibility_dominance,
                        min_auto_confidence=args.min_auto_confidence,
                    )
                except (
                    FileNotFoundError,
                    ImportError,
                    SideAnnotationError,
                    WebEyeTrackGeometryError,
                    ValueError,
                ) as exc:
                    failure_reasons.append(f"mediapipe:{exc}")
            if annotation is None and yunet_detector is not None:
                try:
                    annotation = yunet_detector.detect(
                        rgb,
                        bbox_face_width_ratio=args.yunet_bbox_face_width_ratio,
                        facing_direction=args.profile_facing_direction,
                        accept=args.accept_yunet,
                        auto_accept_score=args.yunet_auto_accept_score,
                    )
                except (cv2.error, SideAnnotationError, SideGenerationError, ValueError) as exc:
                    failure_reasons.append(f"yunet:{exc}")
            if annotation is None and haar_detector is not None:
                try:
                    annotation = haar_detector.detect(
                        rgb,
                        visible_eye=args.haar_visible_eye,
                        accept=args.accept_haar,
                    )
                except (SideAnnotationError, SideGenerationError, ValueError) as exc:
                    failure_reasons.append(f"haar:{exc}")
            if annotation is not None:
                annotation = _crop_quality(
                    rgb,
                    annotation,
                    min_sharpness=args.min_sharpness,
                )
            generated.append(
                GeneratedRow(
                    sample=sample,
                    annotation=annotation,
                    reason=" | ".join(failure_reasons) or "detection failed",
                )
            )
            if index % 100 == 0 or index == len(samples):
                accepted = sum(
                    item.annotation is not None and item.annotation.valid for item in generated
                )
                print(f"processed {index}/{len(samples)}; accepted={accepted}")
    finally:
        if mediapipe_detector is not None:
            mediapipe_detector.close()

    if not args.no_temporal_fill:
        generated = _fill_temporal_candidates(
            generated,
            min_sharpness=args.min_sharpness,
            min_auto_confidence=args.temporal_min_auto_confidence,
            max_auto_gap=args.temporal_max_auto_gap,
        )

    rows = [_annotation_row(item, root) for item in generated]
    _atomic_write_csv(output_csv, rows)
    review = [
        item
        for item in generated
        if item.sample.input_valid is True
        and (item.annotation is None or not item.annotation.valid)
    ]
    review_rows = [row for row in rows if row["review_status"] == "required"]
    _atomic_write_csv(output_dir / "side_annotations_review_queue.csv", review_rows)

    accepted_by_subject: dict[str, list[GeneratedRow]] = defaultdict(list)
    for item in generated:
        if (
            item.sample.input_valid is True
            and item.annotation is not None
            and item.annotation.valid
        ):
            accepted_by_subject[item.sample.participant].append(item)
    accepted_preview: list[GeneratedRow] = []
    for participant in sorted(accepted_by_subject):
        accepted_preview.extend(
            accepted_by_subject[participant][: max(0, args.preview_per_subject)]
        )
    preview_files = _contact_sheets(
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

    per_subject: dict[str, Counter[str]] = defaultdict(Counter)
    methods: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    for item in generated:
        if item.sample.input_valid is not True:
            status = "excluded"
        elif item.annotation is not None and item.annotation.valid:
            status = "accepted"
        else:
            status = "review"
        per_subject[item.sample.participant][status] += 1
        if item.sample.input_valid is not True:
            if item.sample.input_valid is False:
                methods["excluded_inputs_valid_0"] += 1
                flags["closed_eye_pair_excluded"] += 1
            else:
                methods["excluded_missing_input_state"] += 1
                flags["missing_input_eye_state"] += 1
        elif item.annotation is None:
            methods["failed"] += 1
            flags[item.reason] += 1
        else:
            methods[item.annotation.method] += 1
            flags.update(item.annotation.quality_flags)
    summary = {
        "data_root": str(root),
        "output_csv": str(output_csv),
        "total_samples": len(generated),
        "open_samples": sum(item.sample.input_valid is True for item in generated),
        "closed_samples": sum(item.sample.input_valid is False for item in generated),
        "unknown_eye_state_samples": sum(item.sample.input_valid is None for item in generated),
        "excluded_samples": sum(item.sample.input_valid is not True for item in generated),
        "accepted_samples": sum(
            item.sample.input_valid is True
            and item.annotation is not None
            and item.annotation.valid
            for item in generated
        ),
        "review_required_samples": len(review),
        "complete": not review,
        "methods": dict(methods),
        "quality_flags": dict(flags.most_common()),
        "subjects": [
            {
                "participant": participant,
                "accepted": counts["accepted"],
                "review_required": counts["review"],
                "excluded": counts["excluded"],
                "total": counts["accepted"] + counts["review"] + counts["excluded"],
            }
            for participant, counts in sorted(per_subject.items())
        ],
        "accepted_preview_files": preview_files,
        "review_preview_files": review_files,
        "review_workflow": {
            "edit": str(output_dir / "side_annotations_review_queue.csv"),
            "rerun_argument": "--review-overrides <edited-review-queue.csv>",
            "training_gate": "complete must be true before measured-manifest",
        },
        "config": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, str | int | float | bool | type(None))
        },
    }
    (output_dir / "side_annotation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = _parser().parse_args()
    try:
        summary = generate(args)
    except (OSError, SideGenerationError) as exc:
        print(f"side annotation generation failed: {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 3 if args.require_complete and not summary["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
