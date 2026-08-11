from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ggulnote_ml.capture.dataset import normalize_participant_id
from ggulnote_ml.exceptions import ContractError, OptionalDependencyError
from ggulnote_ml.video_preprocessing.contracts import (
    DUAL_VIEW_MANIFEST_COLUMNS,
    IMAGE_PIPELINE_REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    SynchronizedPair,
    VIDEO_FEATURE_NAMES,
    VIDEO_FEATURE_SCHEMA_VERSION,
)
from ggulnote_ml.video_preprocessing.mediapipe_features import (
    VideoFrameFeatures,
    extract_video_frame_features,
)
from ggulnote_ml.video_preprocessing.mediapipe_landmarks import (
    MediaPipeFaceIrisExtractor,
)
from ggulnote_ml.video_preprocessing.loader import (
    load_screen_size,
    load_synchronized_pairs,
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
    output_root: Path,
    cv2_module: Any = None,
    landmark_extractor_factory: Any = None,
    ear_threshold: float = 0.20,
) -> VideoPreprocessingResult:
    """Export frames and MediaPipe features from A's synchronized video pairs."""

    if not 0.0 < ear_threshold < 1.0:
        raise ValueError("ear_threshold must be in (0,1).")

    participant = normalize_participant_id(participant_id)
    raw_root = dataset_root.expanduser().resolve()
    participant_source = raw_root / participant
    if not participant_source.is_dir():
        raise FileNotFoundError(
            "Participant directory does not exist: %s" % participant_source
        )
    destination_root = output_root.expanduser().resolve()
    if destination_root == raw_root:
        raise ContractError("Video preprocessing output_root must differ from raw dataset_root.")
    try:
        destination_root.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ContractError(
            "Video preprocessing output_root must not be inside raw dataset_root."
        )
    participant_output = destination_root / participant
    manifest_directory = destination_root / "manifests"
    training_manifest = manifest_directory / (participant + "_training.csv")
    evaluation_manifest = manifest_directory / (participant + "_evaluation.csv")
    video_training_features = manifest_directory / (
        participant + "_video_training.csv"
    )
    video_evaluation_features = manifest_directory / (
        participant + "_video_evaluation.csv"
    )
    summary_json = manifest_directory / (participant + "_summary.json")
    protected_outputs = (
        participant_output,
        training_manifest,
        evaluation_manifest,
        video_training_features,
        video_evaluation_features,
        summary_json,
    )
    existing = [path for path in protected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Video preprocessing outputs already exist and will not be overwritten: %s"
            % ", ".join(str(path) for path in existing)
        )

    synchronized_path = (
        participant_source / "synchronized" / "synchronized_frames.csv"
    )
    pairs = load_synchronized_pairs(synchronized_path)
    if any(pair.participant != participant for pair in pairs):
        raise ContractError("Synchronized participant does not match the requested participant.")
    screen_width, screen_height = load_screen_size(
        participant_source / "participant.json"
    )
    export_pairs = tuple(pair for pair in pairs if pair.export_partition is not None)
    if not export_pairs:
        raise ContractError("No synchronized training or evaluation pairs are exportable.")

    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise OptionalDependencyError(
                "Video preprocessing requires: pip install -e '.[video]'."
            ) from exc
    if landmark_extractor_factory is None:
        landmark_extractor_factory = _default_landmark_extractor_factory
    (participant_output / "webcam").mkdir(parents=True, exist_ok=False)
    (participant_output / "phonecam").mkdir(parents=True, exist_ok=False)
    manifest_directory.mkdir(parents=True, exist_ok=True)

    rows = {"training": [], "evaluation": []}
    feature_rows = {"webcam": [], "phonecam": []}
    webcam_video = participant_source / "webcam" / "capture.mp4"
    phonecam_video = participant_source / "phonecam" / "capture.mp4"
    with ExitStack() as stack:
        webcam_decoder = stack.enter_context(
            SequentialBgrFrameDecoder(webcam_video, cv2_module=cv2_module)
        )
        phonecam_decoder = stack.enter_context(
            SequentialBgrFrameDecoder(phonecam_video, cv2_module=cv2_module)
        )
        webcam_landmarks = stack.enter_context(
            landmark_extractor_factory("webcam")
        )
        phonecam_landmarks = stack.enter_context(
            landmark_extractor_factory("phonecam")
        )
        for pair in export_pairs:
            assert pair.webcam_frame is not None
            assert pair.phonecam_frame is not None
            partition = pair.export_partition
            assert partition is not None
            webcam_frame = webcam_decoder.read(pair.webcam_frame)
            phonecam_frame = phonecam_decoder.read(pair.phonecam_frame)
            pair_id = "%s_pair_%08d" % (participant, pair.pair)
            webcam_image_row = _write_view_image_and_row(
                cv2_module,
                destination_root,
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
                destination_root,
                participant_output,
                pair,
                pair_id,
                "phonecam",
                phonecam_frame,
                screen_width,
                screen_height,
            )
            rows[partition].extend((webcam_image_row, phonecam_image_row))
            webcam_result = extract_video_frame_features(
                webcam_landmarks.extract(webcam_frame), ear_threshold
            )
            phonecam_result = extract_video_frame_features(
                phonecam_landmarks.extract(phonecam_frame), ear_threshold
            )
            feature_rows["webcam"].append(
                _processed_feature_row(
                    pair,
                    pair_id,
                    "webcam",
                    webcam_result,
                    str(webcam_image_row["image_path"]),
                )
            )
            feature_rows["phonecam"].append(
                _processed_feature_row(
                    pair,
                    pair_id,
                    "phonecam",
                    phonecam_result,
                    str(phonecam_image_row["image_path"]),
                )
            )

    _write_manifest(training_manifest, rows["training"])
    _write_manifest(evaluation_manifest, rows["evaluation"])
    webcam_features = participant_output / "webcam" / "processed_features.csv"
    phonecam_features = participant_output / "phonecam" / "processed_features.csv"
    write_processed_feature_csv(webcam_features, feature_rows["webcam"])
    write_processed_feature_csv(phonecam_features, feature_rows["phonecam"])
    all_feature_rows = feature_rows["webcam"] + feature_rows["phonecam"]
    training_ready_pairs = _training_ready_pair_ids(all_feature_rows)
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
        "source_synchronized_csv": str(synchronized_path),
        "output_root": str(destination_root),
        "training_pairs": len(rows["training"]) // 2,
        "evaluation_pairs": len(rows["evaluation"]) // 2,
        "excluded_pairs": len(pairs) - len(export_pairs),
        "ear_threshold": ear_threshold,
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
        output_root=destination_root,
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
    filename = "%s_%s_frame_%08d.png" % (pair_id, view, source_frame)
    image_path = participant_output / view / filename
    if image_path.exists():
        raise FileExistsError("Frame image already exists: %s" % image_path)
    if not cv2_module.imwrite(str(image_path), frame):
        raise ContractError("Could not write frame image: %s" % image_path)
    relative_path = image_path.relative_to(output_root).as_posix()
    return {
        "sample_id": "%s_%s" % (pair_id, view),
        "subject_id": pair.participant,
        "session_id": "capture",
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
        "training": int(pair.training),
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
) -> set:
    by_pair: Dict[object, list] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    ready = set()
    for pair_id, pair_rows in by_pair.items():
        cameras = {row["camera"] for row in pair_rows}
        if (
            cameras == {"webcam", "phonecam"}
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
