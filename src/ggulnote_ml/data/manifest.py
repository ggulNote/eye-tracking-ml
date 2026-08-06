from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

import numpy as np

from ggulnote_ml.config import DataConfig
from ggulnote_ml.coordinates import to_top_left_normalized
from ggulnote_ml.contracts import RawDataset, RawVideoSample
from ggulnote_ml.data.base import DataSource
from ggulnote_ml.exceptions import ContractError, OptionalDependencyError


REQUIRED_COLUMNS = {
    "video_path",
    "source",
    "participant_id",
    "session_id",
    "target_x",
    "target_y",
}


class ManifestVideoDataSource(DataSource):
    """Load labeled instants/windows from webcam and phone video.

    A video may appear in many rows. Each row can identify its label time with
    `label_frame_index` or `label_timestamp_ms`, which keeps gaze labels aligned
    with the decoded frames instead of assigning one target to a whole video.
    """

    def __init__(self, config: DataConfig, manifest_path: Path) -> None:
        self.config = config
        self.manifest_path = manifest_path

    def load(self) -> RawDataset:
        try:
            import cv2
        except ImportError as exc:
            raise OptionalDependencyError(
                "Manifest video loading requires the 'video' extra: pip install -e '.[video]'."
            ) from exc

        if not self.manifest_path.exists():
            raise ContractError("Manifest does not exist: %s" % self.manifest_path)

        samples: List[RawVideoSample] = []
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
            if missing:
                raise ContractError(
                    "Manifest is missing required columns: %s" % ", ".join(sorted(missing))
                )
            for row_number, row in enumerate(reader, start=2):
                if not _parse_bool(row.get("valid"), default=True):
                    continue
                try:
                    samples.append(self._load_row(cv2, row, row_number))
                except (TypeError, ValueError) as exc:
                    raise ContractError(
                        "Manifest row %d contains an invalid numeric value: %s"
                        % (row_number, exc)
                    ) from exc

        sample_ids = [sample.sample_id for sample in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ContractError("Manifest sample_id values must be unique.")

        dataset = RawDataset(samples=tuple(samples))
        dataset.validate(self.config.sequence_length)
        return dataset

    def _load_row(self, cv2: object, row: Mapping[str, str], row_number: int) -> RawVideoSample:
        video_path = Path(row["video_path"])
        if not video_path.is_absolute():
            video_path = (self.manifest_path.parent / video_path).resolve()
        if not video_path.exists():
            raise ContractError("Manifest row %d video does not exist: %s" % (row_number, video_path))

        source = row["source"].strip().lower()
        if source not in {"webcam", "phone"}:
            raise ContractError("Manifest row %d source must be webcam or phone." % row_number)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ContractError("Could not open video at manifest row %d: %s" % (row_number, video_path))
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise ContractError("Video has no readable frames: %s" % video_path)
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            indices = _window_indices(
                frame_count=frame_count,
                sequence_length=self.config.sequence_length,
                frame_stride=self.config.frame_stride,
                label_frame_index=_optional_int(row.get("label_frame_index")),
                label_timestamp_ms=_optional_float(row.get("label_timestamp_ms")),
                fps=fps,
            )
            frames = []
            rotation = int(row.get("rotation_degrees") or 0)
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, bgr = capture.read()
                if not ok:
                    raise ContractError(
                        "Could not decode frame %d from %s" % (frame_index, video_path)
                    )
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = _rotate(rgb, rotation)
                frames.append(rgb.astype(np.uint8, copy=False))
        finally:
            capture.release()

        screen_size_px = _screen_size_px(row)
        screen_size_cm = _screen_size_cm(row)
        target = to_top_left_normalized(
            [float(row["target_x"]), float(row["target_y"])],
            coordinate_space=(row.get("target_coordinate_space") or "top_left_normalized").strip(),
            screen_size_px=screen_size_px,
        )
        timestamps_ms = tuple(
            float(index) * 1000.0 / fps if fps > 0 else float("nan") for index in indices
        )
        if fps <= 0 and _optional_float(row.get("label_timestamp_ms")) is not None:
            raise ContractError(
                "Manifest row %d uses label_timestamp_ms but the video FPS is unavailable."
                % row_number
            )
        if fps <= 0:
            timestamps_ms = ()
        sample = RawVideoSample(
            frames=np.stack(frames, axis=0),
            target=target,
            participant_id=row["participant_id"].strip(),
            session_id=row["session_id"].strip(),
            source=source,
            sample_id=(row.get("sample_id") or "%s-row-%d" % (video_path.stem, row_number)).strip(),
            frame_indices=tuple(int(index) for index in indices),
            timestamps_ms=timestamps_ms,
            screen_size_px=screen_size_px,
            screen_size_cm=screen_size_cm,
            device_id=(row.get("device_id") or "").strip(),
            camera_intrinsics_path=_resolve_optional_path(
                row.get("camera_intrinsics_path"), self.manifest_path.parent
            ),
            calibration_point_id=(row.get("calibration_point_id") or "").strip() or None,
            sample_weight=float(row.get("sample_weight") or 1.0),
        )
        sample.validate(self.config.sequence_length)
        return sample


def _rotate(frame: np.ndarray, rotation_degrees: int) -> np.ndarray:
    rotations = {0: 0, 90: 3, 180: 2, 270: 1}
    if rotation_degrees not in rotations:
        raise ContractError("rotation_degrees must be one of 0, 90, 180, 270.")
    turns = rotations[rotation_degrees]
    return np.rot90(frame, k=turns).copy() if turns else frame


def _window_indices(
    frame_count: int,
    sequence_length: int,
    frame_stride: int,
    label_frame_index: Optional[int],
    label_timestamp_ms: Optional[float],
    fps: float,
) -> np.ndarray:
    if label_frame_index is not None and label_timestamp_ms is not None:
        raise ContractError("Use only one of label_frame_index or label_timestamp_ms per row.")
    if label_timestamp_ms is not None:
        if label_timestamp_ms < 0:
            raise ContractError("label_timestamp_ms must be zero or greater.")
        if fps <= 0:
            raise ContractError("label_timestamp_ms requires a video with readable FPS metadata.")
        center = int(round(label_timestamp_ms * fps / 1000.0))
    elif label_frame_index is not None:
        center = label_frame_index
    else:
        # Backward compatible clip-level row. New datasets should always use a
        # frame or timestamp locator so the label is synchronized explicitly.
        return np.linspace(0, max(0, frame_count - 1), sequence_length, dtype=int)

    if center < 0 or center >= frame_count:
        raise ContractError(
            "Label frame %d is outside the video frame range [0,%d]."
            % (center, frame_count - 1)
        )
    offsets = (np.arange(sequence_length, dtype=int) - sequence_length // 2) * frame_stride
    return np.clip(center + offsets, 0, frame_count - 1).astype(int)


def _screen_size_px(row: Mapping[str, str]) -> Optional[Tuple[int, int]]:
    width = _optional_int(row.get("screen_width_px"))
    height = _optional_int(row.get("screen_height_px"))
    if width is None and height is None:
        return None
    if width is None or height is None or width <= 0 or height <= 0:
        raise ContractError("screen_width_px and screen_height_px must both be positive.")
    return (width, height)


def _screen_size_cm(row: Mapping[str, str]) -> Optional[Tuple[float, float]]:
    width = _optional_float(row.get("screen_width_cm"))
    height = _optional_float(row.get("screen_height_cm"))
    if width is None and height is None:
        return None
    if width is None or height is None or width <= 0 or height <= 0:
        raise ContractError("screen_width_cm and screen_height_cm must both be positive.")
    return (width, height)


def _resolve_optional_path(value: Optional[str], base_dir: Path) -> Optional[str]:
    if value is None or not value.strip():
        return None
    path = Path(value.strip())
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.is_file():
        raise ContractError("camera_intrinsics_path does not exist: %s" % path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            intrinsics = json.load(handle)
        matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float32)
        distortion = np.asarray(
            intrinsics.get("distortion_coefficients", []), dtype=np.float32
        )
        image_size = (
            int(intrinsics["image_width_px"]),
            int(intrinsics["image_height_px"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError("Invalid camera intrinsics JSON at %s: %s" % (path, exc)) from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ContractError("camera_matrix must be a finite 3x3 array.")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ContractError("camera_matrix focal lengths fx and fy must be positive.")
    if distortion.ndim != 1 or not np.isfinite(distortion).all():
        raise ContractError("distortion_coefficients must be a finite 1D array.")
    if any(value <= 0 for value in image_size):
        raise ContractError("Camera calibration image dimensions must be positive.")
    return str(path)


def _optional_int(value: Optional[str]) -> Optional[int]:
    return None if value is None or not value.strip() else int(value)


def _optional_float(value: Optional[str]) -> Optional[float]:
    return None if value is None or not value.strip() else float(value)


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ContractError("valid must be a boolean value.")
