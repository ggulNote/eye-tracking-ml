"""Canonical, model-independent records used by the data pipeline.

The classes in this module deliberately contain paths and metadata only.  Raw
face images are never embedded in, copied into, or serialized with a manifest.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DataContractError(ValueError):
    """Raised when source data cannot satisfy the canonical data contract."""


def _finite_tuple(values: Sequence[float], *, length: int, field_name: str) -> tuple[float, ...]:
    if len(values) != length:
        raise DataContractError(f"{field_name} must contain {length} values, got {len(values)}")
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise DataContractError(f"{field_name} contains a non-finite value")
    return converted


def _optional_finite_tuple(
    values: Sequence[float] | None, *, length: int, field_name: str
) -> tuple[float, ...] | None:
    if values is None:
        return None
    return _finite_tuple(values, length=length, field_name=field_name)


@dataclass(frozen=True, slots=True)
class ScreenCalibration:
    """Participant/device screen dimensions used by gaze labels and metrics.

    Pixel sizes are always required.  Physical sizes are optional for generic
    manifests, but MPIIFaceGaze supplies both and therefore supports physical
    millimetre/centimetre metrics.
    """

    width_px: int
    height_px: int
    width_mm: float | None = None
    height_mm: float | None = None
    source_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "width_px", int(self.width_px))
        object.__setattr__(self, "height_px", int(self.height_px))
        if self.width_px <= 0 or self.height_px <= 0:
            raise DataContractError("screen width_px and height_px must both be positive")

        if (self.width_mm is None) != (self.height_mm is None):
            raise DataContractError(
                "screen width_mm and height_mm must either both be present or both be absent"
            )
        if self.width_mm is not None and self.height_mm is not None:
            object.__setattr__(self, "width_mm", float(self.width_mm))
            object.__setattr__(self, "height_mm", float(self.height_mm))
            if (
                not math.isfinite(self.width_mm)
                or not math.isfinite(self.height_mm)
                or self.width_mm <= 0
                or self.height_mm <= 0
            ):
                raise DataContractError(
                    "screen width_mm and height_mm must be finite positive values"
                )

        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def has_physical_size(self) -> bool:
        return self.width_mm is not None and self.height_mm is not None

    @property
    def mm_per_pixel_xy(self) -> tuple[float, float] | None:
        if not self.has_physical_size:
            return None
        assert self.width_mm is not None and self.height_mm is not None
        return self.width_mm / self.width_px, self.height_mm / self.height_px


@dataclass(frozen=True, slots=True)
class CanonicalRecord:
    """One still-image gaze sample independent of a specific model backend."""

    sample_id: str
    subject_id: str
    session_id: str
    view: str
    image_path: Path
    image_relative_path: str
    image_width_px: int
    image_height_px: int
    image_channels: int | None
    gaze_screen_xy_px: tuple[float, float]
    screen: ScreenCalibration
    pair_id: str | None = None
    facial_landmarks_xy: tuple[tuple[float, float], ...] = ()
    # WebEyeTrack's metric head-pose contract. ``head_vector`` is a unit
    # direction in the front-camera frame and ``face_origin_3d`` is expressed
    # in centimetres.  They intentionally remain separate from the generic
    # rotation/translation fields below because neither value is an Euler
    # rotation nor a camera translation vector.
    head_vector: tuple[float, float, float] | None = None
    face_origin_3d: tuple[float, float, float] | None = None
    head_pose_valid: bool = False
    head_rotation_3d: tuple[float, float, float] | None = None
    head_translation_3d: tuple[float, float, float] | None = None
    face_center_3d: tuple[float, float, float] | None = None
    gaze_target_3d: tuple[float, float, float] | None = None
    evaluation_eye: str | None = None
    visible_eye: str | None = None
    visible_eye_bbox_xyxy: tuple[float, float, float, float] | None = None
    visible_eye_keypoints_xy: tuple[tuple[float, float], ...] = ()
    iris_center_xy: tuple[float, float] | None = None
    profile_head_origin_xy: tuple[float, float] | None = None
    profile_head_forward_xy: tuple[float, float] | None = None
    eye_annotation_valid: bool = False
    source_dataset: str = "generic"
    annotation_path: Path | None = None
    annotation_line: int | None = None
    allow_out_of_screen_target: bool = False

    def __post_init__(self) -> None:
        sample_id = str(self.sample_id).strip()
        subject_id = str(self.subject_id).strip()
        session_id = str(self.session_id).strip()
        view = str(self.view).strip().lower()
        if not sample_id:
            raise DataContractError("sample_id must not be empty")
        if not subject_id:
            raise DataContractError("subject_id must not be empty")
        if view not in {"front", "side"}:
            raise DataContractError(f"view must be 'front' or 'side', got {view!r}")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "view", view)

        image_path = Path(self.image_path)
        object.__setattr__(self, "image_path", image_path)
        relative_path = str(self.image_relative_path).strip().replace("\\", "/")
        if not relative_path:
            raise DataContractError("image_relative_path must not be empty")
        object.__setattr__(self, "image_relative_path", relative_path)

        object.__setattr__(self, "image_width_px", int(self.image_width_px))
        object.__setattr__(self, "image_height_px", int(self.image_height_px))
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise DataContractError("image width and height must both be positive")
        if self.image_channels is not None:
            object.__setattr__(self, "image_channels", int(self.image_channels))
            if self.image_channels <= 0:
                raise DataContractError("image_channels must be positive when present")

        gaze = _finite_tuple(
            self.gaze_screen_xy_px,
            length=2,
            field_name="gaze_screen_xy_px",
        )
        object.__setattr__(self, "gaze_screen_xy_px", gaze)
        object.__setattr__(
            self,
            "allow_out_of_screen_target",
            bool(self.allow_out_of_screen_target),
        )
        if not self.allow_out_of_screen_target and not 0 <= gaze[0] < self.screen.width_px:
            raise DataContractError(
                f"target x={gaze[0]} lies outside screen [0, {self.screen.width_px})"
            )
        if not self.allow_out_of_screen_target and not 0 <= gaze[1] < self.screen.height_px:
            raise DataContractError(
                f"target y={gaze[1]} lies outside screen [0, {self.screen.height_px})"
            )

        landmarks = tuple(
            _finite_tuple(point, length=2, field_name="facial_landmarks_xy point")
            for point in self.facial_landmarks_xy
        )
        if landmarks and len(landmarks) < 4:
            raise DataContractError(
                "facial_landmarks_xy must contain at least 4 points when present, "
                f"got {len(landmarks)}"
            )
        object.__setattr__(self, "facial_landmarks_xy", landmarks)

        for name in (
            "head_vector",
            "face_origin_3d",
            "head_rotation_3d",
            "head_translation_3d",
            "face_center_3d",
            "gaze_target_3d",
        ):
            object.__setattr__(
                self,
                name,
                _optional_finite_tuple(getattr(self, name), length=3, field_name=name),
            )

        object.__setattr__(self, "head_pose_valid", bool(self.head_pose_valid))
        has_head_vector = self.head_vector is not None
        has_face_origin = self.face_origin_3d is not None
        if has_head_vector != has_face_origin:
            raise DataContractError(
                "head_vector and face_origin_3d must either both be present or both be absent"
            )
        if self.head_pose_valid and not has_head_vector:
            raise DataContractError("head_pose_valid=true requires head_vector and face_origin_3d")

        if (self.face_center_3d is None) != (self.gaze_target_3d is None):
            raise DataContractError("face_center_3d and gaze_target_3d must be present together")

        if self.pair_id is not None:
            pair_id = str(self.pair_id).strip()
            object.__setattr__(self, "pair_id", pair_id or None)
        if self.evaluation_eye is not None:
            eye = str(self.evaluation_eye).strip().lower()
            if eye not in {"left", "right"}:
                raise DataContractError(
                    f"evaluation_eye must be left/right when present, got {eye!r}"
                )
            object.__setattr__(self, "evaluation_eye", eye)

        if self.visible_eye is not None:
            visible_eye_aliases = {
                "l": "left",
                "left": "left",
                "subject_left": "left",
                "r": "right",
                "right": "right",
                "subject_right": "right",
            }
            raw_visible_eye = str(self.visible_eye).strip().lower()
            visible_eye = visible_eye_aliases.get(raw_visible_eye)
            if visible_eye is None:
                raise DataContractError(
                    f"visible_eye must be left/right when present, got {self.visible_eye!r}"
                )
            object.__setattr__(self, "visible_eye", visible_eye)

        bbox = _optional_finite_tuple(
            self.visible_eye_bbox_xyxy,
            length=4,
            field_name="visible_eye_bbox_xyxy",
        )
        if bbox is not None and (bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
            raise DataContractError("visible_eye_bbox_xyxy must satisfy x1 > x0 and y1 > y0")
        object.__setattr__(self, "visible_eye_bbox_xyxy", bbox)

        eye_keypoints = tuple(
            _finite_tuple(point, length=2, field_name="visible_eye_keypoints_xy point")
            for point in self.visible_eye_keypoints_xy
        )
        if eye_keypoints and len(eye_keypoints) != 6:
            raise DataContractError(
                "visible_eye_keypoints_xy must contain exactly 6 points when present, "
                f"got {len(eye_keypoints)}"
            )
        object.__setattr__(self, "visible_eye_keypoints_xy", eye_keypoints)

        for name in (
            "iris_center_xy",
            "profile_head_origin_xy",
            "profile_head_forward_xy",
        ):
            object.__setattr__(
                self,
                name,
                _optional_finite_tuple(getattr(self, name), length=2, field_name=name),
            )
        object.__setattr__(self, "eye_annotation_valid", bool(self.eye_annotation_valid))
        if self.annotation_path is not None:
            object.__setattr__(self, "annotation_path", Path(self.annotation_path))
        if self.annotation_line is not None:
            line_number = int(self.annotation_line)
            if line_number <= 0:
                raise DataContractError("annotation_line must be a positive 1-based number")
            object.__setattr__(self, "annotation_line", line_number)

    @property
    def gaze_screen_xy_normalized(self) -> tuple[float, float]:
        """Return centered coordinates using screen width/height denominators."""

        x_px, y_px = self.gaze_screen_xy_px
        return (
            x_px / self.screen.width_px - 0.5,
            y_px / self.screen.height_px - 0.5,
        )

    @property
    def target_in_screen_bounds(self) -> bool:
        x_px, y_px = self.gaze_screen_xy_px
        return 0 <= x_px < self.screen.width_px and 0 <= y_px < self.screen.height_px

    @property
    def gaze_vector_3d(self) -> tuple[float, float, float] | None:
        if self.face_center_3d is None or self.gaze_target_3d is None:
            return None
        return tuple(
            target - origin
            for target, origin in zip(self.gaze_target_3d, self.face_center_3d, strict=True)
        )  # type: ignore[return-value]

    @property
    def gaze_unit_3d(self) -> tuple[float, float, float] | None:
        vector = self.gaze_vector_3d
        if vector is None:
            return None
        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0:
            raise DataContractError(f"record {self.sample_id!r} has a zero-length 3D gaze vector")
        return tuple(component / norm for component in vector)  # type: ignore[return-value]

    def to_manifest_row(
        self, *, split: str = "", pair_complete: bool | None = None
    ) -> dict[str, str | int | float]:
        """Flatten the record into a stable CSV-compatible mapping."""

        target_norm = self.gaze_screen_xy_normalized
        return {
            "split": split,
            "sample_id": self.sample_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "view": self.view,
            "pair_id": self.pair_id or "",
            "pair_complete": ("" if pair_complete is None else str(bool(pair_complete)).lower()),
            "source_dataset": self.source_dataset,
            "image_path": str(self.image_path),
            "image_relative_path": self.image_relative_path,
            "image_width_px": self.image_width_px,
            "image_height_px": self.image_height_px,
            "image_channels": self.image_channels or "",
            "target_x_px": self.gaze_screen_xy_px[0],
            "target_y_px": self.gaze_screen_xy_px[1],
            "target_x_normalized": target_norm[0],
            "target_y_normalized": target_norm[1],
            "target_in_screen_bounds": str(self.target_in_screen_bounds).lower(),
            "screen_width_px": self.screen.width_px,
            "screen_height_px": self.screen.height_px,
            "screen_width_mm": self.screen.width_mm or "",
            "screen_height_mm": self.screen.height_mm or "",
            "facial_landmarks_xy": _json_sequence(self.facial_landmarks_xy),
            "head_vector": _json_sequence(self.head_vector),
            "face_origin_3d": _json_sequence(self.face_origin_3d),
            "head_pose_valid": str(self.head_pose_valid).lower(),
            "head_rotation_3d": _json_sequence(self.head_rotation_3d),
            "head_translation_3d": _json_sequence(self.head_translation_3d),
            "face_center_3d": _json_sequence(self.face_center_3d),
            "gaze_target_3d": _json_sequence(self.gaze_target_3d),
            "evaluation_eye": self.evaluation_eye or "",
            "visible_eye": self.visible_eye or "",
            "visible_eye_bbox_xyxy": _json_sequence(self.visible_eye_bbox_xyxy),
            "visible_eye_keypoints_xy": _json_sequence(self.visible_eye_keypoints_xy),
            "iris_center_xy": _json_sequence(self.iris_center_xy),
            "profile_head_origin_xy": _json_sequence(self.profile_head_origin_xy),
            "profile_head_forward_xy": _json_sequence(self.profile_head_forward_xy),
            "eye_annotation_valid": str(self.eye_annotation_valid).lower(),
            "screen_calibration_path": (
                str(self.screen.source_path) if self.screen.source_path else ""
            ),
            "annotation_path": (str(self.annotation_path) if self.annotation_path else ""),
            "annotation_line": self.annotation_line or "",
        }


def _json_sequence(values: Sequence[Any] | None) -> str:
    if values is None or len(values) == 0:
        return ""
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


MANIFEST_COLUMNS: tuple[str, ...] = (
    "split",
    "sample_id",
    "subject_id",
    "session_id",
    "view",
    "pair_id",
    "pair_complete",
    "source_dataset",
    "image_path",
    "image_relative_path",
    "image_width_px",
    "image_height_px",
    "image_channels",
    "target_x_px",
    "target_y_px",
    "target_x_normalized",
    "target_y_normalized",
    "target_in_screen_bounds",
    "screen_width_px",
    "screen_height_px",
    "screen_width_mm",
    "screen_height_mm",
    "facial_landmarks_xy",
    "head_vector",
    "face_origin_3d",
    "head_pose_valid",
    "head_rotation_3d",
    "head_translation_3d",
    "face_center_3d",
    "gaze_target_3d",
    "evaluation_eye",
    "visible_eye",
    "visible_eye_bbox_xyxy",
    "visible_eye_keypoints_xy",
    "iris_center_xy",
    "profile_head_origin_xy",
    "profile_head_forward_xy",
    "eye_annotation_valid",
    "screen_calibration_path",
    "annotation_path",
    "annotation_line",
)


def ensure_unique_records(records: Iterable[CanonicalRecord]) -> None:
    """Reject duplicate sample IDs and duplicate resolved image paths."""

    sample_ids: dict[str, CanonicalRecord] = {}
    image_paths: dict[Path, CanonicalRecord] = {}
    for record in records:
        previous = sample_ids.get(record.sample_id)
        if previous is not None:
            raise DataContractError(
                f"duplicate sample_id {record.sample_id!r}: "
                f"{previous.image_path} and {record.image_path}"
            )
        sample_ids[record.sample_id] = record

        resolved = record.image_path.resolve()
        previous = image_paths.get(resolved)
        if previous is not None:
            raise DataContractError(
                f"image {resolved} is referenced by both "
                f"{previous.sample_id!r} and {record.sample_id!r}"
            )
        image_paths[resolved] = record


def record_counts(records: Iterable[CanonicalRecord]) -> Mapping[str, Any]:
    """Return small, JSON-serializable counts for data summaries."""

    records = tuple(records)
    by_subject: dict[str, int] = {}
    by_view: dict[str, int] = {}
    by_resolution: dict[str, int] = {}
    out_of_screen_by_subject: dict[str, int] = {}
    for record in records:
        by_subject[record.subject_id] = by_subject.get(record.subject_id, 0) + 1
        out_of_screen_by_subject.setdefault(record.subject_id, 0)
        if not record.target_in_screen_bounds:
            out_of_screen_by_subject[record.subject_id] += 1
        by_view[record.view] = by_view.get(record.view, 0) + 1
        resolution = f"{record.image_width_px}x{record.image_height_px}"
        by_resolution[resolution] = by_resolution.get(resolution, 0) + 1
    return {
        "records": len(records),
        "subjects": len(by_subject),
        "by_subject": dict(sorted(by_subject.items())),
        "by_view": dict(sorted(by_view.items())),
        "by_image_resolution": dict(sorted(by_resolution.items())),
        "physical_screen_calibration_records": sum(
            record.screen.has_physical_size for record in records
        ),
        "out_of_screen_target_records": sum(
            not record.target_in_screen_bounds for record in records
        ),
        "out_of_screen_target_records_by_subject": dict(sorted(out_of_screen_by_subject.items())),
    }
