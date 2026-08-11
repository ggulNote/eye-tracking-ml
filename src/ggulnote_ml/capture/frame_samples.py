from __future__ import annotations

import csv
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import DisplayConfig, FrameCaptureConfig
from .contracts import FramePacket
from .dataset import ParticipantPaths
from .protocols import ProtocolFrameState, normalized_coordinates
from ggulnote_ml.video_preprocessing.mediapipe_landmarks import (
    MediaPipeFaceIrisExtractor,
)


IMAGE_SAMPLE_COLUMNS = (
    "sample",
    "participant",
    "protocol",
    "split",
    "pair",
    "display_timestamp",
    "confirmation_timestamp",
    "confirmation_offset_ms",
    "candidate_count",
    "pair_quality_score",
    "webcam_image",
    "webcam_frame",
    "webcam_timestamp",
    "webcam_face_detected",
    "webcam_eyes_detected",
    "webcam_mediapipe_face_detected",
    "webcam_mediapipe_iris_detected",
    "webcam_sharpness",
    "webcam_brightness",
    "phonecam_image",
    "phonecam_frame",
    "phonecam_timestamp",
    "phonecam_face_detected",
    "phonecam_eyes_detected",
    "phonecam_mediapipe_face_detected",
    "phonecam_mediapipe_iris_detected",
    "phonecam_sharpness",
    "phonecam_brightness",
    "x_px",
    "y_px",
    "x_norm",
    "y_norm",
    "x_centered",
    "y_centered",
    "segment",
    "target",
    "direction",
)


@dataclass(frozen=True)
class FrameQuality:
    face_detected: bool
    eyes_detected: int
    eye_open_detected: bool
    sharpness: float
    brightness: float
    score: float
    mediapipe_face_detected: bool = False
    mediapipe_iris_detected: bool = False

    @property
    def mediapipe_ready(self) -> bool:
        return self.mediapipe_face_detected and self.mediapipe_iris_detected


@dataclass(frozen=True)
class _CandidatePair:
    pair_index: int
    display_timestamp_ns: int
    webcam: FramePacket
    phonecam: FramePacket
    state: ProtocolFrameState
    webcam_quality: FrameQuality
    phonecam_quality: FrameQuality
    combined_score: float


class FrameQualityScorer:
    """Score one full camera frame with dependency-free OpenCV heuristics.

    Eye evidence from the bundled Haar cascade has the highest configured weight,
    followed by face detection, Laplacian sharpness, and exposure. Haar detection
    is intentionally only a collection-time heuristic; MediaPipe preprocessing
    remains the authoritative blink/face validation stage.
    """

    def __init__(self, config: FrameCaptureConfig) -> None:
        self.config = config
        cascade_root = Path(cv2.data.haarcascades)
        self._face = cv2.CascadeClassifier(
            str(cascade_root / "haarcascade_frontalface_default.xml")
        )
        self._eyes = cv2.CascadeClassifier(
            str(cascade_root / "haarcascade_eye_tree_eyeglasses.xml")
        )
        if self._face.empty() or self._eyes.empty():
            raise RuntimeError("OpenCV face/eye quality cascades could not be loaded.")

    def score(self, frame: np.ndarray) -> FrameQuality:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Quality input must be a uint8 BGR image with shape (H, W, 3).")
        height, width = frame.shape[:2]
        scale = min(1.0, self.config.scoring_width_px / float(width))
        resized = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        faces = self._face.detectMultiScale(
            gray,
            scaleFactor=self.config.face_scale_factor,
            minNeighbors=self.config.face_min_neighbors,
        )
        face_detected = len(faces) > 0
        roi = gray
        eyes_detected = 0
        if face_detected:
            x, y, face_width, face_height = max(
                faces, key=lambda item: int(item[2]) * int(item[3])
            )
            roi = gray[y : y + face_height, x : x + face_width]
            upper_face = roi[: max(1, round(face_height * 0.65)), :]
            eyes = self._eyes.detectMultiScale(
                upper_face,
                scaleFactor=self.config.eye_scale_factor,
                minNeighbors=self.config.eye_min_neighbors,
            )
            eyes_detected = min(2, len(eyes))

        sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        brightness = float(roi.mean())
        eye_open_detected = eyes_detected > 0
        sharpness_component = min(1.0, sharpness / self.config.sharpness_reference)
        exposure_span = max(
            self.config.ideal_brightness,
            255.0 - self.config.ideal_brightness,
            1.0,
        )
        brightness_component = max(
            0.0,
            1.0 - abs(brightness - self.config.ideal_brightness) / exposure_span,
        )
        score = (
            self.config.eye_open_weight * int(eye_open_detected)
            + self.config.face_weight * int(face_detected)
            + self.config.sharpness_weight * sharpness_component
            + self.config.brightness_weight * brightness_component
        )
        return FrameQuality(
            face_detected=face_detected,
            eyes_detected=eyes_detected,
            eye_open_detected=eye_open_detected,
            sharpness=sharpness,
            brightness=brightness,
            score=score,
        )


class FrameSampleWriter:
    """Select and persist exactly one paired image sample per confirmed target.

    Each confirmed capture window can contain a variable number of frames because
    real camera FPS varies. ``observe`` scores every paired candidate but retains
    only the best pair. When the window ends, the two frames are written with the
    same sample ID and one manifest row. Original MP4 frames remain untouched.
    """

    def __init__(
        self,
        paths: ParticipantPaths,
        config: FrameCaptureConfig,
        display: DisplayConfig,
        enable_mediapipe_quality: bool = True,
        landmark_extractor_factory=None,
    ) -> None:
        self.paths = paths
        self.config = config
        self.display = display
        self.manifest_path = paths.labels_directory / "image_samples.csv"
        self._sample_index = 0
        self._active_key: Optional[Tuple[str, int, int]] = None
        self._candidate_count = 0
        self._best: Optional[_CandidatePair] = None
        self._file = None
        self._writer = None
        self._scorer = None
        self._landmark_stack = ExitStack()
        self._landmark_extractors = None
        if config.enabled:
            self._scorer = FrameQualityScorer(config)
            if enable_mediapipe_quality:
                factory = landmark_extractor_factory or (
                    lambda _camera: MediaPipeFaceIrisExtractor()
                )
                self._landmark_extractors = {
                    camera: self._landmark_stack.enter_context(factory(camera))
                    for camera in ("webcam", "phonecam")
                }
            self._file = self.manifest_path.open("x", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=IMAGE_SAMPLE_COLUMNS)
            self._writer.writeheader()

    def observe(
        self,
        pair_index: int,
        display_timestamp_ns: int,
        webcam: FramePacket,
        phonecam: FramePacket,
        state: ProtocolFrameState,
    ) -> None:
        is_candidate = (
            self.config.enabled
            and state.confirmation_required
            and state.is_confirmed
            and state.is_usable_window
        )
        if not is_candidate:
            self.flush()
            return
        if (
            state.segment_index is None
            or state.confirmation_timestamp_ns is None
            or state.confirmation_offset_ms is None
            or state.target_x is None
            or state.target_y is None
        ):
            raise ValueError("Confirmed frame candidate is missing target or timing fields.")

        key = (state.protocol_id, state.segment_index, state.confirmation_timestamp_ns)
        if self._active_key is not None and key != self._active_key:
            self.flush()
        self._active_key = key
        webcam_quality = self._scorer.score(webcam.frame)
        phonecam_quality = self._scorer.score(phonecam.frame)
        if self._landmark_extractors is not None:
            webcam_landmarks = self._landmark_extractors["webcam"].extract(
                webcam.frame
            )
            phonecam_landmarks = self._landmark_extractors["phonecam"].extract(
                phonecam.frame
            )
            webcam_quality = replace(
                webcam_quality,
                mediapipe_face_detected=webcam_landmarks.face_detected,
                mediapipe_iris_detected=webcam_landmarks.iris_detected,
            )
            phonecam_quality = replace(
                phonecam_quality,
                mediapipe_face_detected=phonecam_landmarks.face_detected,
                mediapipe_iris_detected=phonecam_landmarks.iris_detected,
            )
        combined_score = (
            webcam_quality.score
            + phonecam_quality.score
            + self.config.mediapipe_ready_weight
            * (
                int(webcam_quality.mediapipe_ready)
                + int(phonecam_quality.mediapipe_ready)
            )
        )
        self._candidate_count += 1
        if self._best is None or combined_score > self._best.combined_score:
            self._best = _CandidatePair(
                pair_index=pair_index,
                display_timestamp_ns=display_timestamp_ns,
                webcam=self._copy_packet(webcam),
                phonecam=self._copy_packet(phonecam),
                state=state,
                webcam_quality=webcam_quality,
                phonecam_quality=phonecam_quality,
                combined_score=combined_score,
            )

    def flush(self) -> None:
        if self._best is None:
            self._active_key = None
            self._candidate_count = 0
            return
        candidate = self._best
        sample_id = "s%06d" % self._sample_index
        suffix = "." + self.config.image_format
        webcam_path = self.paths.webcam_images_directory / (sample_id + suffix)
        phonecam_path = self.paths.phone_images_directory / (sample_id + suffix)
        self._write_image(webcam_path, candidate.webcam.frame)
        try:
            self._write_image(phonecam_path, candidate.phonecam.frame)
        except Exception:
            webcam_path.unlink(missing_ok=True)
            raise
        self._write_manifest_row(sample_id, webcam_path, phonecam_path, candidate)
        self._sample_index += 1
        self._active_key = None
        self._candidate_count = 0
        self._best = None

    @staticmethod
    def _copy_packet(packet: FramePacket) -> FramePacket:
        return FramePacket(
            frame_index=packet.frame_index,
            captured_at_ms=packet.captured_at_ms,
            unix_timestamp_ns=packet.unix_timestamp_ns,
            frame=packet.frame.copy(),
        )

    def _write_manifest_row(
        self,
        sample_id: str,
        webcam_path: Path,
        phonecam_path: Path,
        candidate: _CandidatePair,
    ) -> None:
        state = candidate.state
        x_px, y_px, centered_x, centered_y = normalized_coordinates(
            float(state.target_x),
            float(state.target_y),
            self.display.canvas_width,
            self.display.canvas_height,
        )
        webcam_quality = candidate.webcam_quality
        phonecam_quality = candidate.phonecam_quality
        self._writer.writerow(
            {
                "sample": sample_id,
                "participant": self.paths.participant_id,
                "protocol": state.protocol_id,
                "split": state.split,
                "pair": candidate.pair_index,
                "display_timestamp": candidate.display_timestamp_ns,
                "confirmation_timestamp": state.confirmation_timestamp_ns,
                "confirmation_offset_ms": "%.3f" % float(state.confirmation_offset_ms),
                "candidate_count": self._candidate_count,
                "pair_quality_score": "%.6f" % candidate.combined_score,
                "webcam_image": str(webcam_path.relative_to(self.paths.participant_directory)),
                "webcam_frame": candidate.webcam.frame_index,
                "webcam_timestamp": candidate.webcam.unix_timestamp_ns,
                "webcam_face_detected": int(webcam_quality.face_detected),
                "webcam_eyes_detected": webcam_quality.eyes_detected,
                "webcam_mediapipe_face_detected": int(
                    webcam_quality.mediapipe_face_detected
                ),
                "webcam_mediapipe_iris_detected": int(
                    webcam_quality.mediapipe_iris_detected
                ),
                "webcam_sharpness": "%.6f" % webcam_quality.sharpness,
                "webcam_brightness": "%.6f" % webcam_quality.brightness,
                "phonecam_image": str(phonecam_path.relative_to(self.paths.participant_directory)),
                "phonecam_frame": candidate.phonecam.frame_index,
                "phonecam_timestamp": candidate.phonecam.unix_timestamp_ns,
                "phonecam_face_detected": int(phonecam_quality.face_detected),
                "phonecam_eyes_detected": phonecam_quality.eyes_detected,
                "phonecam_mediapipe_face_detected": int(
                    phonecam_quality.mediapipe_face_detected
                ),
                "phonecam_mediapipe_iris_detected": int(
                    phonecam_quality.mediapipe_iris_detected
                ),
                "phonecam_sharpness": "%.6f" % phonecam_quality.sharpness,
                "phonecam_brightness": "%.6f" % phonecam_quality.brightness,
                "x_px": x_px,
                "y_px": y_px,
                "x_norm": "%.6f" % float(state.target_x),
                "y_norm": "%.6f" % float(state.target_y),
                "x_centered": "%.6f" % centered_x,
                "y_centered": "%.6f" % centered_y,
                "segment": state.segment_index,
                "target": state.target_index,
                "direction": state.direction,
            }
        )

    def _write_image(self, path: Path, frame: np.ndarray) -> None:
        if path.exists():
            raise FileExistsError("Image sample already exists: %s" % path)
        parameters = (
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
            if self.config.image_format == "jpg"
            else []
        )
        if not cv2.imwrite(str(path), frame, parameters):
            raise RuntimeError("Could not write image sample: %s" % path)

    def close(self) -> None:
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None
        self._landmark_stack.close()
