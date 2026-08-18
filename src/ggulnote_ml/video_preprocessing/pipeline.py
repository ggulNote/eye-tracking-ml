from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ggulnote_ml.capture.calibration_assets import (
    CameraIntrinsics,
    load_camera_intrinsics,
)
from ggulnote_ml.capture.dataset import (
    normalize_head_pose,
    normalize_participant_id,
    participant_recording_directory,
)
from ggulnote_ml.exceptions import ContractError, OptionalDependencyError
from ggulnote_ml.video_preprocessing.contracts import (
    DUAL_VIEW_MANIFEST_COLUMNS,
    IMAGE_PIPELINE_REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    SynchronizedPair,
    VIDEO_FEATURE_NAMES,
    VIDEO_FEATURE_SCHEMA_VERSION,
)
from ggulnote_ml.video_preprocessing.camera_geometry import FrameUndistorter
from ggulnote_ml.video_preprocessing.mediapipe_features import (
    VideoFrameFeatures,
    extract_video_frame_features,
)
from ggulnote_ml.video_preprocessing.mediapipe_landmarks import (
    MediaPipeFaceIrisExtractor,
)
from ggulnote_ml.video_preprocessing.loader import (
    load_screen_size,
    load_selected_samples,
    load_synchronized_pairs,
    select_synchronized_samples,
)
from ggulnote_ml.video_preprocessing.processed_writer import (
    write_processed_feature_csv,
)


@dataclass(frozen=True)
class VideoPreprocessingResult:
    output_root: Path
    participant_directory: Path
    training_manifest: Path
    evaluation_manifest: Path
    webcam_features: Path
    phonecam_features: Path
    video_training_features: Path
    video_evaluation_features: Path
    summary_json: Path
    training_pairs: int
    evaluation_pairs: int
    valid_feature_rows: int


class SequentialBgrFrameDecoder:
    """Decode monotonically increasing frame indices without timestamp guessing."""

    def __init__(self, video_path: Path, cv2_module: Any = None) -> None:
        self.video_path = video_path.expanduser().resolve()
        self._cv2 = cv2_module
        self._capture: Any = None
        self._next_frame = 0
        self._cached_index: Optional[int] = None
        self._cached_frame: Optional[np.ndarray] = None

    def __enter__(self) -> "SequentialBgrFrameDecoder":
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise OptionalDependencyError(
                    "Video preprocessing requires: pip install -e '.[video]'."
                ) from exc
            self._cv2 = cv2
        if not self.video_path.is_file():
            raise FileNotFoundError("Camera video does not exist: %s" % self.video_path)
        self._capture = self._cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise ContractError("Could not open camera video: %s" % self.video_path)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read(self, frame_index: int) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("SequentialBgrFrameDecoder must be opened first.")
        if frame_index == self._cached_index and self._cached_frame is not None:
            return self._cached_frame.copy()
        if frame_index < self._next_frame:
            raise ContractError(
                "Requested video frames must be non-decreasing; got %d after %d."
                % (frame_index, self._next_frame - 1)
            )
        while self._next_frame <= frame_index:
            ok, frame = self._capture.read()
            current = self._next_frame
            self._next_frame += 1
            if not ok or frame is None:
                raise ContractError(
                    "Could not decode frame %d from %s."
                    % (frame_index, self.video_path)
                )
            if current == frame_index:
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ContractError("Decoded frame must be uint8 BGR [H,W,3].")
                contiguous = np.ascontiguousarray(frame)
                self._cached_index = frame_index
                self._cached_frame = contiguous
                return contiguous.copy()
        raise AssertionError("Frame decoder did not reach the requested index.")


def run_video_preprocessing(
    participant_id: str,
    dataset_root: Path,
    output_root: Optional[Path] = None,
    head_pose: Optional[str] = None,
    cv2_module: Any = None,
    landmark_extractor_factory: Any = None,
    ear_threshold: float = 0.20,
    intrinsics_mode: str = "required",
    feature_cameras: Sequence[str] = ("webcam",),
) -> VideoPreprocessingResult:
    """Export paired frames and configured MediaPipe features from synchronized video."""

    if not 0.0 < ear_threshold < 1.0:
        raise ValueError("ear_threshold must be in (0,1).")
    configured_feature_cameras = tuple(
        str(camera).strip().lower() for camera in feature_cameras
    )
    if not configured_feature_cameras:
        raise ValueError("feature_cameras must contain at least one camera.")
    if len(set(configured_feature_cameras)) != len(configured_feature_cameras):
        raise ValueError("feature_cameras must not contain duplicates.")
    unsupported_feature_cameras = set(configured_feature_cameras) - {
        "webcam",
        "phonecam",
    }
    if unsupported_feature_cameras:
        raise ValueError(
            "feature_cameras contains unsupported cameras: %s"
            % ", ".join(sorted(unsupported_feature_cameras))
        )
    intrinsics_mode = intrinsics_mode.strip().lower()
    if intrinsics_mode not in {"required", "off"}:
        raise ValueError("intrinsics_mode must be required or off.")

    participant = normalize_participant_id(participant_id)
    normalized_head_pose = (
        ""
        if head_pose is None or str(head_pose).strip() == ""
        else normalize_head_pose(head_pose)
    )
    raw_root = dataset_root.expanduser().resolve()
    participant_source = participant_recording_directory(
        raw_root,
        participant,
        normalized_head_pose or None,
    )
    if not participant_source.is_dir():
        raise FileNotFoundError(
            "Participant directory does not exist: %s" % participant_source
        )
    if output_root is None:
        participant_output = participant_source / "feature_maps"
    else:
        destination_root = output_root.expanduser().resolve()
        if destination_root == raw_root:
            raise ContractError(
                "External video preprocessing output_root must differ from raw dataset_root."
            )
        try:
            destination_root.relative_to(raw_root)
        except ValueError:
            pass
        else:
            raise ContractError(
                "External output_root must be outside raw dataset_root; omit it to use "
                "<participant_name>/<head_pose>/feature_maps/."
            )
        participant_output = destination_root / participant
        if normalized_head_pose:
            participant_output = participant_output / normalized_head_pose
    training_manifest = participant_output / "training.csv"
    evaluation_manifest = participant_output / "evaluation.csv"
    video_training_features = participant_output / "training_features.csv"
    video_evaluation_features = participant_output / "evaluation_features.csv"
    summary_json = participant_output / "summary.json"
    web_frames_directory = participant_output / "web" / "frames"
    phone_frames_directory = participant_output / "phone" / "frames"
    protected_outputs = (
        training_manifest,
        evaluation_manifest,
        video_training_features,
        video_evaluation_features,
        summary_json,
        web_frames_directory,
        phone_frames_directory,
        participant_output / "web" / "features.csv",
        participant_output / "phone" / "features.csv",
    )
    existing = [path for path in protected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Video preprocessing outputs already exist and will not be overwritten: %s"
            % ", ".join(str(path) for path in existing)
        )

    synchronized_path = participant_source / "feature_maps" / "synchronized.csv"
    selected_samples_path = participant_source / "labels" / "labels.csv"
    pairs = load_synchronized_pairs(synchronized_path)
    selected_samples = load_selected_samples(selected_samples_path)
    if any(pair.participant != participant for pair in pairs):
        raise ContractError("Synchronized participant does not match the requested participant.")
    if any(pair.head_pose != normalized_head_pose for pair in pairs):
        raise ContractError("Synchronized head_pose does not match the requested head pose.")
    if any(sample.participant != participant for sample in selected_samples):
        raise ContractError("Selected sample participant does not match the requested participant.")
    if any(sample.head_pose != normalized_head_pose for sample in selected_samples):
        raise ContractError("Selected sample head_pose does not match the requested head pose.")
    screen_width, screen_height = load_screen_size(
        participant_source / "participant.json"
    )
    selected_pairs = select_synchronized_samples(pairs, selected_samples)
    if not selected_pairs:
        raise ContractError("No synchronized training or evaluation pairs are exportable.")

    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise OptionalDependencyError(
                "Video preprocessing requires: pip install -e '.[video]'."
            ) from exc
    camera_intrinsics: Dict[str, Optional[CameraIntrinsics]] = {
        "webcam": None,
        "phonecam": None,
    }
    if intrinsics_mode == "required":
        for camera in camera_intrinsics:
            camera_intrinsics[camera] = load_camera_intrinsics(
                participant_source
                / "calibration"
                / ("web" if camera == "webcam" else "phone")
                / "Camera.mat"
            )
    undistorters = {
        camera: (
            FrameUndistorter(intrinsics, cv2_module)
            if intrinsics is not None
            else None
        )
        for camera, intrinsics in camera_intrinsics.items()
    }
    if landmark_extractor_factory is None:
        landmark_extractor_factory = _default_landmark_extractor_factory
    web_frames_directory.mkdir(parents=True, exist_ok=False)
    phone_frames_directory.mkdir(parents=True, exist_ok=False)

    rows = {"training": [], "evaluation": []}
    feature_rows = {"webcam": [], "phonecam": []}
    webcam_video = participant_source / "video" / "web" / "capture.mp4"
    phonecam_video = participant_source / "video" / "phone" / "capture.mp4"
    with ExitStack() as stack:
        webcam_decoder = stack.enter_context(
            SequentialBgrFrameDecoder(webcam_video, cv2_module=cv2_module)
        )
        phonecam_decoder = stack.enter_context(
            SequentialBgrFrameDecoder(phonecam_video, cv2_module=cv2_module)
        )
        landmark_extractors = {
            camera: stack.enter_context(landmark_extractor_factory(camera))
            for camera in configured_feature_cameras
        }
        for selection in selected_pairs:
            pair = selection.synchronized
            assert pair.webcam_frame is not None
            assert pair.phonecam_frame is not None
            partition = pair.export_partition
            assert partition is not None
            webcam_frame = webcam_decoder.read(pair.webcam_frame)
            phonecam_frame = phonecam_decoder.read(pair.phonecam_frame)
            if undistorters["webcam"] is not None:
                webcam_frame = undistorters["webcam"].apply(webcam_frame)
            if undistorters["phonecam"] is not None:
                phonecam_frame = undistorters["phonecam"].apply(phonecam_frame)
            pair_id = "_".join(
                value
                for value in (
                    participant,
                    normalized_head_pose,
                    selection.sample.sample,
                )
                if value
            )
            webcam_image_row = _write_view_image_and_row(
                cv2_module,
                participant_output,
                participant_output,
                pair,
                pair_id,
                "webcam",
                webcam_frame,
                screen_width,
                screen_height,
            )
            phonecam_image_row = _write_view_image_and_row(
                cv2_module,
                participant_output,
                participant_output,
                pair,
                pair_id,
                "phonecam",
                phonecam_frame,
                screen_width,
                screen_height,
            )
            rows[partition].extend((webcam_image_row, phonecam_image_row))
            frames = {"webcam": webcam_frame, "phonecam": phonecam_frame}
            image_rows = {
                "webcam": webcam_image_row,
                "phonecam": phonecam_image_row,
            }
            for camera in configured_feature_cameras:
                features = extract_video_frame_features(
                    landmark_extractors[camera].extract(frames[camera]),
                    ear_threshold,
                )
                feature_rows[camera].append(
                    _processed_feature_row(
                        pair,
                        pair_id,
                        camera,
                        features,
                        str(image_rows[camera]["image_path"]),
                        camera_intrinsics[camera],
                        participant_source,
                    )
                )

    _write_manifest(training_manifest, rows["training"])
    _write_manifest(evaluation_manifest, rows["evaluation"])
    webcam_features = participant_output / "web" / "features.csv"
    phonecam_features = participant_output / "phone" / "features.csv"
    write_processed_feature_csv(webcam_features, feature_rows["webcam"])
    write_processed_feature_csv(phonecam_features, feature_rows["phonecam"])
    all_feature_rows = feature_rows["webcam"] + feature_rows["phonecam"]
    training_ready_pairs = _training_ready_pair_ids(
        all_feature_rows, configured_feature_cameras
    )
    training_feature_rows = [
        row
        for row in all_feature_rows
        if row["pair_id"] in training_ready_pairs
    ]
    evaluation_feature_rows = [
        row for row in all_feature_rows if row["collection_split"] == "evaluation"
    ]
    write_processed_feature_csv(video_training_features, training_feature_rows)
    write_processed_feature_csv(video_evaluation_features, evaluation_feature_rows)
    valid_feature_rows = sum(int(row["feature_valid"]) for row in all_feature_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "participant": participant,
        "head_pose": normalized_head_pose,
        "source_synchronized_csv": str(synchronized_path),
        "source_labels_csv": str(selected_samples_path),
        "output_root": str(participant_output),
        "training_pairs": len(rows["training"]) // 2,
        "evaluation_pairs": len(rows["evaluation"]) // 2,
        "selected_pairs": len(selected_pairs),
        "unselected_synchronized_pairs": len(pairs) - len(selected_pairs),
        "ear_threshold": ear_threshold,
        "intrinsics_mode": intrinsics_mode,
        "mediapipe_feature_cameras": list(configured_feature_cameras),
        "intrinsics": {
            camera: (
                {
                    "applied": True,
                    "camera_matrix_path": intrinsics.path.relative_to(
                        participant_source
                    ).as_posix(),
                    "rms_error_px": intrinsics.rms_error_px,
                    "image_width": intrinsics.image_width,
                    "image_height": intrinsics.image_height,
                }
                if intrinsics is not None
                else {"applied": False}
            )
            for camera, intrinsics in camera_intrinsics.items()
        },
        "video_feature_schema_version": VIDEO_FEATURE_SCHEMA_VERSION,
        "video_feature_order": list(VIDEO_FEATURE_NAMES),
        "processed_feature_rows": len(all_feature_rows),
        "valid_feature_rows": valid_feature_rows,
        "training_feature_pairs": len(training_ready_pairs),
        "face_not_detected_rows": sum(
            row["invalid_reason"] == "face_not_detected" for row in all_feature_rows
        ),
        "iris_not_detected_rows": sum(
            row["invalid_reason"] == "iris_not_detected" for row in all_feature_rows
        ),
        "eye_closed_rows": sum(
            row["invalid_reason"] == "eye_closed" for row in all_feature_rows
        ),
        "training_manifest": str(training_manifest),
        "evaluation_manifest": str(evaluation_manifest),
        "webcam_features": str(webcam_features),
        "phonecam_features": str(phonecam_features),
        "video_training_features": str(video_training_features),
        "video_evaluation_features": str(video_evaluation_features),
        "image_pipeline_required_columns": sorted(IMAGE_PIPELINE_REQUIRED_COLUMNS),
    }
    with summary_json.open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return VideoPreprocessingResult(
        output_root=participant_output,
        participant_directory=participant_output,
        training_manifest=training_manifest,
        evaluation_manifest=evaluation_manifest,
        webcam_features=webcam_features,
        phonecam_features=phonecam_features,
        video_training_features=video_training_features,
        video_evaluation_features=video_evaluation_features,
        summary_json=summary_json,
        training_pairs=len(rows["training"]) // 2,
        evaluation_pairs=len(rows["evaluation"]) // 2,
        valid_feature_rows=valid_feature_rows,
    )


def _default_landmark_extractor_factory(camera: str) -> MediaPipeFaceIrisExtractor:
    del camera
    return MediaPipeFaceIrisExtractor()


def _write_view_image_and_row(
    cv2_module: Any,
    output_root: Path,
    participant_output: Path,
    pair: SynchronizedPair,
    pair_id: str,
    view: str,
    frame: np.ndarray,
    screen_width: int,
    screen_height: int,
) -> Dict[str, object]:
    if view == "webcam":
        source_frame = pair.webcam_frame
        source_timestamp = pair.webcam_timestamp
        corrected_timestamp = pair.webcam_corrected_timestamp
    elif view == "phonecam":
        source_frame = pair.phonecam_frame
        source_timestamp = pair.phonecam_timestamp
        corrected_timestamp = pair.phonecam_corrected_timestamp
    else:
        raise ContractError("view must be webcam or phonecam.")
    assert source_frame is not None
    assert source_timestamp is not None
    assert corrected_timestamp is not None
    assert pair.x_norm is not None and pair.y_norm is not None
    folder_name = "web" if view == "webcam" else "phone"
    filename = "%s_%s_frame_%08d.png" % (pair_id, folder_name, source_frame)
    image_path = participant_output / folder_name / "frames" / filename
    if image_path.exists():
        raise FileExistsError("Frame image already exists: %s" % image_path)
    if not cv2_module.imwrite(str(image_path), frame):
        raise ContractError("Could not write frame image: %s" % image_path)
    relative_path = image_path.relative_to(output_root).as_posix()
    return {
        "sample_id": "%s_%s" % (pair_id, view),
        "subject_id": pair.participant,
        "head_pose": pair.head_pose,
        "view": view,
        "image_path": relative_path,
        "pair_id": pair_id,
        "target_x_px": "%.6f" % (pair.x_norm * screen_width),
        "target_y_px": "%.6f" % (pair.y_norm * screen_height),
        "screen_width_px": screen_width,
        "screen_height_px": screen_height,
        "collection_split": pair.export_partition,
        "protocol": pair.protocol,
        "source_frame": source_frame,
        "source_timestamp": source_timestamp,
        "corrected_timestamp": corrected_timestamp,
        "reference_timestamp": pair.reference_timestamp,
        "target_timestamp": pair.target_timestamp,
    }


def _processed_feature_row(
    pair: SynchronizedPair,
    pair_id: str,
    camera: str,
    features: VideoFrameFeatures,
    image_path: str,
    intrinsics: Optional[CameraIntrinsics],
    participant_source: Path,
) -> Dict[str, object]:
    if camera == "webcam":
        source_frame = pair.webcam_frame
        source_timestamp = pair.webcam_timestamp
        corrected_timestamp = pair.webcam_corrected_timestamp
    elif camera == "phonecam":
        source_frame = pair.phonecam_frame
        source_timestamp = pair.phonecam_timestamp
        corrected_timestamp = pair.phonecam_corrected_timestamp
    else:
        raise ContractError("camera must be webcam or phonecam.")
    assert source_frame is not None
    assert source_timestamp is not None
    assert corrected_timestamp is not None
    assert pair.x_norm is not None and pair.y_norm is not None
    partition = pair.export_partition
    assert partition is not None
    eye_state = features.eye_state
    row: Dict[str, object] = {
        "schema_version": VIDEO_FEATURE_SCHEMA_VERSION,
        "sample_id": "%s_%s" % (pair_id, camera),
        "participant": pair.participant,
        "head_pose": pair.head_pose,
        "camera": camera,
        "pair_id": pair_id,
        "pair": pair.pair,
        "source_frame": source_frame,
        "source_timestamp": source_timestamp,
        "corrected_timestamp": corrected_timestamp,
        "reference_timestamp": pair.reference_timestamp,
        "target_timestamp": pair.target_timestamp,
        "protocol": pair.protocol,
        "collection_split": partition,
        "x_norm": "%.8f" % pair.x_norm,
        "y_norm": "%.8f" % pair.y_norm,
        "sync_valid": int(pair.valid),
        "usable": int(pair.usable),
        "intrinsics_applied": int(intrinsics is not None),
        "intrinsics_rms_px": (
            "" if intrinsics is None else "%.8f" % intrinsics.rms_error_px
        ),
        "camera_matrix_path": (
            ""
            if intrinsics is None
            else intrinsics.path.relative_to(participant_source).as_posix()
        ),
        "face_detected": int(features.face_detected),
        "iris_detected": int(features.iris_detected),
        "landmark_count": features.landmark_count,
        "left_ear": "" if eye_state is None else "%.8f" % eye_state.left_ear,
        "right_ear": "" if eye_state is None else "%.8f" % eye_state.right_ear,
        "left_eye_closed": (
            "" if eye_state is None else int(eye_state.left_eye_closed)
        ),
        "right_eye_closed": (
            "" if eye_state is None else int(eye_state.right_eye_closed)
        ),
        "eye_closed": "" if eye_state is None else int(eye_state.eye_closed),
        "feature_valid": int(features.feature_valid),
        "invalid_reason": features.invalid_reason,
        "image_path": image_path,
    }
    if features.values is None:
        for name in VIDEO_FEATURE_NAMES:
            row[name] = ""
    else:
        for name, value in zip(VIDEO_FEATURE_NAMES, features.values):
            row[name] = "%.8f" % value
    return row


def _training_ready_pair_ids(
    rows: Sequence[Dict[str, object]],
    required_cameras: Sequence[str],
) -> set:
    by_pair: Dict[object, list] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    ready = set()
    for pair_id, pair_rows in by_pair.items():
        cameras = {row["camera"] for row in pair_rows}
        if (
            cameras == set(required_cameras)
            and all(row["collection_split"] == "training" for row in pair_rows)
            and all(row["feature_valid"] == 1 for row in pair_rows)
        ):
            ready.add(pair_id)
    return ready


def _write_manifest(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DUAL_VIEW_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
