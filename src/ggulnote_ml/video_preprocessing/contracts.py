from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


SCHEMA_VERSION = "dual_view_video_v2"
VIDEO_FEATURE_SCHEMA_VERSION = "video_mediapipe_undistorted_2d_v2"

DUAL_VIEW_MANIFEST_COLUMNS = (
    "sample_id",
    "subject_id",
    "view",
    "image_path",
    "pair_id",
    "target_x_px",
    "target_y_px",
    "screen_width_px",
    "screen_height_px",
    "collection_split",
    "protocol",
    "source_frame",
    "source_timestamp",
    "corrected_timestamp",
    "reference_timestamp",
    "target_timestamp",
)

IMAGE_PIPELINE_REQUIRED_COLUMNS = frozenset(
    {
        "sample_id",
        "subject_id",
        "view",
        "image_path",
        "pair_id",
        "target_x_px",
        "target_y_px",
        "screen_width_px",
        "screen_height_px",
    }
)

VIDEO_FEATURE_NAMES = (
    "left_eye_center_x",
    "left_eye_center_y",
    "left_iris_center_x",
    "left_iris_center_y",
    "right_eye_center_x",
    "right_eye_center_y",
    "right_iris_center_x",
    "right_iris_center_y",
)

PROCESSED_FEATURE_COLUMNS = (
    "schema_version",
    "sample_id",
    "participant",
    "camera",
    "pair_id",
    "pair",
    "source_frame",
    "source_timestamp",
    "corrected_timestamp",
    "reference_timestamp",
    "target_timestamp",
    "protocol",
    "collection_split",
    "x_norm",
    "y_norm",
    "sync_valid",
    "usable",
    "intrinsics_applied",
    "intrinsics_rms_px",
    "camera_matrix_path",
    "face_detected",
    "iris_detected",
    "landmark_count",
    "left_ear",
    "right_ear",
    "left_eye_closed",
    "right_eye_closed",
    "eye_closed",
    "feature_valid",
    "invalid_reason",
    *VIDEO_FEATURE_NAMES,
    "image_path",
)


@dataclass(frozen=True)
class SelectedSample:
    """One quality-selected capture sample produced for a confirmed dot."""

    sample: str
    participant: str
    protocol: str
    split: str
    source_pair: int
    webcam_frame: int
    phonecam_frame: int
    webcam_timestamp: int
    phonecam_timestamp: int
    segment: str
    target: str
    direction: str

    @property
    def export_partition(self) -> Optional[str]:
        split = self.split.strip().lower()
        if split == "evaluation":
            return "evaluation"
        if split in {"train", "training"}:
            return "training"
        return None


@dataclass(frozen=True)
class SynchronizedPair:
    participant: str
    pair: int
    webcam_frame: Optional[int]
    phonecam_frame: Optional[int]
    webcam_timestamp: Optional[int]
    phonecam_timestamp: Optional[int]
    webcam_corrected_timestamp: Optional[int]
    phonecam_corrected_timestamp: Optional[int]
    reference_timestamp: int
    target_timestamp: Optional[int]
    x_norm: Optional[float]
    y_norm: Optional[float]
    protocol: str
    split: str
    segment: str
    target: str
    direction: str
    usable: bool
    valid: bool
    invalid_reason: str

    @property
    def cameras_available(self) -> bool:
        return all(
            value is not None
            for value in (
                self.webcam_frame,
                self.phonecam_frame,
                self.webcam_timestamp,
                self.phonecam_timestamp,
                self.webcam_corrected_timestamp,
                self.phonecam_corrected_timestamp,
            )
        )

    @property
    def target_available(self) -> bool:
        return (
            self.target_timestamp is not None
            and self.x_norm is not None
            and self.y_norm is not None
        )

    @property
    def export_partition(self) -> Optional[str]:
        if not self.valid or not self.usable or not self.cameras_available:
            return None
        if not self.target_available:
            return None
        split = self.split.strip().lower()
        if split == "evaluation":
            return "evaluation"
        if split in {"train", "training"}:
            return "training"
        return None


@dataclass(frozen=True)
class SelectedSynchronizedPair:
    """A capture sample mapped to one valid latency-synchronized frame pair."""

    sample: SelectedSample
    synchronized: SynchronizedPair
