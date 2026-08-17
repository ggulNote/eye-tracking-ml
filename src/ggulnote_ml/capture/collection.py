from __future__ import annotations

import csv
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .camera import CameraSource
from .config import CaptureConfig, DisplayConfig
from .contracts import FramePacket
from .dataset import ParticipantPaths
from .frame_samples import FrameSampleWriter
from .protocols import (
    ProtocolFrameState,
    ProtocolPlan,
    ProtocolRunner,
    normalized_coordinates,
    write_protocol_manifest,
)
from .recorder import SessionRecorder
from ggulnote_ml.video_preprocessing.mediapipe_landmarks import (
    FaceIrisLandmarks,
    MediaPipeFaceIrisExtractor,
)


LABEL_COLUMNS = (
    "participant",
    "head_pose",
    "protocol",
    "split",
    "pair",
    "display_timestamp",
    "webcam_frame",
    "webcam_timestamp",
    "phonecam_frame",
    "phonecam_timestamp",
    "x_px",
    "y_px",
    "x_norm",
    "y_norm",
    "x_centered",
    "y_centered",
    "segment",
    "target",
    "direction",
    "confirmation_timestamp",
    "confirmation_offset_ms",
    "usable",
)


@dataclass(frozen=True)
class ProtocolResult:
    protocol_id: str
    status: str
    frame_pairs: int
    frame_log_path: Path


class PairedLabelWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("x", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=LABEL_COLUMNS)
        self._writer.writeheader()
        self._last_display_timestamp_ns: Optional[int] = None

    def write(
        self,
        paths: ParticipantPaths,
        pair_index: int,
        display_timestamp_ns: int,
        webcam: FramePacket,
        iphone: FramePacket,
        state: ProtocolFrameState,
        display: DisplayConfig,
    ) -> None:
        if display_timestamp_ns <= 0:
            raise ValueError("display_timestamp must be a positive Unix nanosecond value.")
        if (
            self._last_display_timestamp_ns is not None
            and display_timestamp_ns < self._last_display_timestamp_ns
        ):
            raise ValueError("display_timestamp values must be non-decreasing.")
        self._last_display_timestamp_ns = display_timestamp_ns
        if state.target_x is None or state.target_y is None:
            x_px = y_px = ""
            centered_x = centered_y = ""
            x_norm = y_norm = ""
        else:
            x_px, y_px, centered_x, centered_y = normalized_coordinates(
                state.target_x, state.target_y, display.canvas_width, display.canvas_height
            )
            x_norm = "%.6f" % state.target_x
            y_norm = "%.6f" % state.target_y
            centered_x = "%.6f" % centered_x
            centered_y = "%.6f" % centered_y
        self._writer.writerow(
            {
                "participant": paths.participant_id,
                "head_pose": paths.head_pose,
                "protocol": state.protocol_id,
                "split": state.split,
                "pair": pair_index,
                "display_timestamp": display_timestamp_ns,
                "webcam_frame": webcam.frame_index,
                "webcam_timestamp": webcam.unix_timestamp_ns,
                "phonecam_frame": iphone.frame_index,
                "phonecam_timestamp": iphone.unix_timestamp_ns,
                "x_px": x_px,
                "y_px": y_px,
                "x_norm": x_norm,
                "y_norm": y_norm,
                "x_centered": centered_x,
                "y_centered": centered_y,
                "segment": state.segment_index if state.segment_index is not None else "",
                "target": state.target_index if state.target_index is not None else "",
                "direction": state.direction,
                "confirmation_timestamp": (
                    state.confirmation_timestamp_ns
                    if state.confirmation_timestamp_ns is not None
                    else ""
                ),
                "confirmation_offset_ms": (
                    "%.3f" % state.confirmation_offset_ms
                    if state.confirmation_offset_ms is not None
                    else ""
                ),
                "usable": int(state.is_usable_window),
            }
        )

    def close(self) -> None:
        self._file.close()


def render_protocol(
    display: DisplayConfig,
    plan: ProtocolPlan,
    state: ProtocolFrameState,
    head_pose: str = "",
) -> np.ndarray:
    canvas = np.full(
        (display.canvas_height, display.canvas_width, 3),
        display.background_bgr,
        dtype=np.uint8,
    )
    if head_pose and state.phase != "target":
        cv2.putText(
            canvas,
            "HEAD POSE: %s" % head_pose,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if state.phase == "ready":
        _center_text(canvas, "%s starts soon" % plan.protocol_id, display.canvas_height // 2, 1.0)
        _center_text(canvas, "Keep your head still and follow the dot", display.canvas_height // 2 + 50, 0.7)
    elif state.phase == "target" and state.target_x is not None and state.target_y is not None:
        x = round(state.target_x * (display.canvas_width - 1))
        y = round(state.target_y * (display.canvas_height - 1))
        if state.show_guide_line:
            y0 = round(0.1 * (display.canvas_height - 1))
            y1 = round(0.9 * (display.canvas_height - 1))
            cv2.line(canvas, (x, y0), (x, y1), display.guide_bgr, 1, cv2.LINE_AA)
        if state.confirmation_required:
            if state.is_confirmed:
                ring_radius = display.point_radius_px + 5
                ring_color = display.confirmed_ring_bgr
            else:
                ring_radius = round(
                    display.point_radius_px
                    + (display.confirmation_ring_radius_px - display.point_radius_px)
                    * (1.0 - state.fixation_progress)
                )
                ring_color = display.confirmation_ring_bgr
            cv2.circle(canvas, (x, y), ring_radius, ring_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), display.point_radius_px, display.point_bgr, -1, cv2.LINE_AA)
        if state.confirmation_required:
            if state.is_confirmed:
                instruction = "Keep looking..."
            elif state.fixation_progress < 1.0:
                instruction = "Hold your gaze on the dot..."
            else:
                instruction = "Click or press Space"
            _center_text(canvas, instruction, display.canvas_height - 36, 0.65)
    elif state.phase == "complete_message":
        _center_text(canvas, "%s complete" % plan.protocol_id, display.canvas_height // 2, 1.0)
    return canvas


def render_display_check(display: DisplayConfig) -> np.ndarray:
    """Render an edge-to-edge test pattern for the configured collection display."""

    canvas = np.full(
        (display.canvas_height, display.canvas_width, 3),
        display.background_bgr,
        dtype=np.uint8,
    )
    border_width = max(4, round(min(display.canvas_width, display.canvas_height) * 0.008))
    cv2.rectangle(
        canvas,
        (0, 0),
        (display.canvas_width - 1, display.canvas_height - 1),
        (0, 255, 0),
        border_width,
    )
    _center_text(
        canvas,
        "GREEN BORDER MUST TOUCH ALL 4 SCREEN EDGES",
        display.canvas_height // 2,
        0.8,
    )
    _center_text(
        canvas,
        "Press Q or Esc to close",
        display.canvas_height // 2 + 48,
        0.65,
    )
    return canvas


def _open_protocol_window(display: DisplayConfig) -> Tuple[int, int, int, int]:
    """Open the target window without OpenCV letterboxing the dot-test canvas."""

    flags = cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO
    cv2.namedWindow(display.window_name, flags)
    blank_canvas = np.full(
        (display.canvas_height, display.canvas_width, 3),
        display.background_bgr,
        dtype=np.uint8,
    )
    # HighGUI on macOS needs a mapped normal window before switching to
    # fullscreen. Setting fullscreen first can place the window in an invisible
    # Space (reported as a negative y coordinate).
    cv2.imshow(display.window_name, blank_canvas)
    cv2.waitKey(100)
    cv2.moveWindow(display.window_name, 0, 0)
    cv2.setWindowProperty(
        display.window_name,
        cv2.WND_PROP_ASPECT_RATIO,
        cv2.WINDOW_FREERATIO,
    )
    if display.fullscreen:
        cv2.setWindowProperty(
            display.window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
        cv2.waitKey(300)
    cv2.setWindowProperty(
        display.window_name,
        cv2.WND_PROP_ASPECT_RATIO,
        cv2.WINDOW_FREERATIO,
    )
    cv2.imshow(display.window_name, blank_canvas)
    cv2.waitKey(100)
    return tuple(int(value) for value in cv2.getWindowImageRect(display.window_name))


def run_display_check(config: CaptureConfig) -> Tuple[int, int, int, int]:
    """Show a no-data display check and return HighGUI's rendered image rectangle."""

    try:
        image_rect = _open_protocol_window(config.display)
        while True:
            cv2.imshow(config.display.window_name, render_display_check(config.display))
            key = cv2.waitKey(20) & 0xFF
            if key in {ord("q"), 27}:
                return image_rect
    finally:
        cv2.destroyAllWindows()


class _ConfirmationInput:
    """Collect a mouse click without coupling OpenCV callbacks to protocol state."""

    def __init__(self) -> None:
        self._mouse_requested = False

    def on_mouse(self, event, _x, _y, _flags, _parameter) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_requested = True

    def consume(self, key: int) -> bool:
        requested = self._mouse_requested or key in {ord(" "), 13}
        self._mouse_requested = False
        return requested


def _center_text(canvas: np.ndarray, text: str, y: int, scale: float) -> None:
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = max(0, (canvas.shape[1] - size[0]) // 2)
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _collection_recorders(
    config: CaptureConfig, paths: ParticipantPaths, fps: float
) -> Tuple[Optional[SessionRecorder], Optional[SessionRecorder]]:
    if not config.recording.enabled:
        return None, None
    webcam = SessionRecorder(
        paths.webcam_directory / "capture.mp4",
        paths.webcam_directory / "timestamps.csv",
        config.recording.video_codec,
        fps,
        config.recording.timing_mode,
    )
    phone = SessionRecorder(
        paths.phone_directory / "capture.mp4",
        paths.phone_directory / "timestamps.csv",
        config.recording.video_codec,
        fps,
        config.recording.timing_mode,
    )
    return webcam, phone


def _preflight_canvas(
    webcam: FramePacket,
    phonecam: FramePacket,
    camera_width_px: int,
    remaining_seconds: float,
    landmark_results: Optional[Dict[str, FaceIrisLandmarks]] = None,
    successful_detections: int = 0,
    required_detections: int = 0,
    landmark_required_cameras: Tuple[str, ...] = (),
) -> np.ndarray:
    """Build a labeled side-by-side preview without modifying source frames."""

    panels = []
    for label, packet in (("webcam", webcam), ("phonecam", phonecam)):
        height, width = packet.frame.shape[:2]
        scale = camera_width_px / float(width)
        panel = cv2.resize(
            packet.frame,
            (camera_width_px, max(1, round(height * scale))),
        )
        result = landmark_results.get(label) if landmark_results is not None else None
        status = ""
        color = (0, 255, 0)
        if result is not None:
            valid = result.face_detected and result.iris_detected
            status = " - MediaPipe OK" if valid else " - ADJUST CAMERA"
            color = (0, 255, 0) if valid else (0, 0, 255)
        elif landmark_results is not None and label not in landmark_required_cameras:
            status = " - IMAGE ONLY"
            color = (0, 255, 255)
        cv2.putText(
            panel,
            label + status,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    canvas = cv2.hconcat(panels)
    if landmark_results is not None:
        cv2.putText(
            canvas,
            "MediaPipe stability %d/%d (q/Esc: abort)"
            % (successful_detections, required_detections),
            (16, canvas.shape[0] - 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        footer = "phonecam: side image only; check focus, exposure, and one visible eye"
    else:
        footer = "Camera check %.1fs - keep both feeds moving" % max(
            0.0, remaining_seconds
        )
    cv2.putText(
        canvas,
        footer,
        (16, canvas.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _run_camera_preflight(
    config: CaptureConfig,
    webcam_source: CameraSource,
    phone_source: CameraSource,
    head_pose: str = "",
) -> None:
    """Verify both live feeds before recording, without reopening either camera."""

    if not config.preview.enabled or config.preview.preflight_duration_ms == 0:
        return
    started_ns = time.monotonic_ns()
    window_name = "%s - preflight%s" % (
        config.preview.window_name_prefix,
        " - " + head_pose if head_pose else "",
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    successful_detections = 0
    with ExitStack() as stack:
        extractors = None
        if config.preview.landmark_required_cameras:
            extractors = {
                camera: stack.enter_context(
                    MediaPipeFaceIrisExtractor(
                        min_detection_confidence=(
                            config.preview.landmark_detection_confidence
                        )
                    )
                )
                for camera in config.preview.landmark_required_cameras
            }

        try:
            while True:
                elapsed_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
                webcam = webcam_source.read(started_ns)
                phonecam = phone_source.read(started_ns)
                results = None
                if extractors is not None:
                    frames = {"webcam": webcam.frame, "phonecam": phonecam.frame}
                    results = {
                        camera: extractor.extract(frames[camera])
                        for camera, extractor in extractors.items()
                    }
                    required_valid = all(
                        result.face_detected and result.iris_detected
                        for result in results.values()
                    )
                    successful_detections = (
                        successful_detections + 1 if required_valid else 0
                    )
                remaining_seconds = (
                    config.preview.preflight_duration_ms - elapsed_ms
                ) / 1000.0
                cv2.imshow(
                    window_name,
                    _preflight_canvas(
                        webcam,
                        phonecam,
                        config.preview.camera_width_px,
                        remaining_seconds,
                        results,
                        successful_detections,
                        config.preview.required_consecutive_detections,
                        config.preview.landmark_required_cameras,
                    ),
                )
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    raise RuntimeError("Camera preflight was aborted.")
                duration_complete = elapsed_ms >= config.preview.preflight_duration_ms
                landmarks_complete = (
                    extractors is None
                    or successful_detections
                    >= config.preview.required_consecutive_detections
                )
                if duration_complete and landmarks_complete:
                    break
        finally:
            cv2.destroyWindow(window_name)
    webcam_source.reset_session()
    phone_source.reset_session()


def _run_real_protocols_inner(
    config: CaptureConfig,
    paths: ParticipantPaths,
    plans: Tuple[ProtocolPlan, ...],
) -> Tuple[ProtocolResult, ...]:
    by_role = {camera.role: camera for camera in config.cameras}
    results = []
    with ExitStack() as stack:
        webcam_source = stack.enter_context(CameraSource(by_role["webcam_front"]))
        phone_source = stack.enter_context(CameraSource(by_role["iphone_left"]))
        _run_camera_preflight(config, webcam_source, phone_source, paths.head_pose)
        collection_started_ns = time.monotonic_ns()
        webcam_recorder, phone_recorder = _collection_recorders(
            config, paths, by_role["webcam_front"].fps
        )
        frame_log_path = paths.metadata_directory / "frame_log.csv"
        labels = PairedLabelWriter(frame_log_path)
        frame_samples = FrameSampleWriter(paths, config.frame_capture, config.display)
        write_protocol_manifest(
            paths.metadata_directory / "protocol.csv",
            plans,
            participant=paths.participant_id,
            head_pose=paths.head_pose,
        )
        pair_index = 0
        _open_protocol_window(config.display)
        confirmation_input = _ConfirmationInput()
        cv2.setMouseCallback(config.display.window_name, confirmation_input.on_mouse)

        try:
            for plan in plans:
                runner = ProtocolRunner(plan)
                protocol_started_ns = time.monotonic_ns()
                protocol_pairs = 0
                status = "completed"
                while True:
                    elapsed_ms = (time.monotonic_ns() - protocol_started_ns) / 1_000_000.0
                    display_state = runner.state_at(elapsed_ms)
                    if display_state.phase == "finished":
                        break
                    cv2.imshow(
                        config.display.window_name,
                        render_protocol(
                            config.display,
                            plan,
                            display_state,
                            paths.head_pose,
                        ),
                    )
                    key = cv2.waitKey(1) & 0xFF
                    display_timestamp_ns = time.time_ns()
                    if key in {ord("q"), 27}:
                        status = "aborted"
                        break
                    if confirmation_input.consume(key):
                        runner.confirm(elapsed_ms, display_timestamp_ns)
                        display_state = runner.state_at(elapsed_ms)

                    webcam = webcam_source.read(collection_started_ns)
                    iphone = phone_source.read(collection_started_ns)
                    frame_samples.observe(
                        pair_index,
                        display_timestamp_ns,
                        webcam,
                        iphone,
                        display_state,
                    )
                    labels.write(
                        paths,
                        pair_index,
                        display_timestamp_ns,
                        webcam,
                        iphone,
                        display_state,
                        config.display,
                    )
                    if webcam_recorder:
                        webcam_recorder.write(webcam)
                    if phone_recorder:
                        phone_recorder.write(iphone)
                    pair_index += 1
                    protocol_pairs += 1
                results.append(
                    ProtocolResult(plan.protocol_id, status, protocol_pairs, frame_log_path)
                )
                frame_samples.flush()
                if status == "aborted":
                    break
        finally:
            labels.close()
            frame_samples.close()
            if webcam_recorder:
                webcam_recorder.close()
            if phone_recorder:
                phone_recorder.close()
    return tuple(results)


def run_real_protocols(
    config: CaptureConfig,
    paths: ParticipantPaths,
    plans: Tuple[ProtocolPlan, ...],
) -> Tuple[ProtocolResult, ...]:
    """Run hardware collection and always release GUI resources on errors or aborts."""
    try:
        return _run_real_protocols_inner(config, paths, plans)
    finally:
        cv2.destroyAllWindows()


def run_camera_preflight(config: CaptureConfig) -> None:
    """Open both configured cameras and run the A+B landmark readiness gate."""

    by_role = {camera.role: camera for camera in config.cameras}
    try:
        with ExitStack() as stack:
            webcam_source = stack.enter_context(
                CameraSource(by_role["webcam_front"])
            )
            phone_source = stack.enter_context(CameraSource(by_role["iphone_left"]))
            _run_camera_preflight(config, webcam_source, phone_source)
    finally:
        cv2.destroyAllWindows()


def _simulation_frame(width: int, height: int, role: str, pair_index: int) -> np.ndarray:
    color = (40, 80, 140) if role == "webcam_front" else (120, 70, 35)
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.putText(
        frame,
        "%s frame %d" % (role, pair_index),
        (10, max(25, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def run_simulation_protocols(
    config: CaptureConfig,
    paths: ParticipantPaths,
    plans: Tuple[ProtocolPlan, ...],
) -> Tuple[ProtocolResult, ...]:
    simulation = config.simulation
    interval_ms = 1000.0 / simulation.fps
    interval_ns = round(1_000_000_000 / simulation.fps)
    phone_delay_ns = round(simulation.phone_delay_ms * 1_000_000)
    results = []
    pair_index = 0
    webcam_recorder, phone_recorder = _collection_recorders(config, paths, simulation.fps)
    frame_log_path = paths.metadata_directory / "frame_log.csv"
    labels = PairedLabelWriter(frame_log_path)
    frame_samples = FrameSampleWriter(
        paths,
        config.frame_capture,
        config.display,
        enable_mediapipe_quality=False,
    )
    write_protocol_manifest(
        paths.metadata_directory / "protocol.csv",
        plans,
        participant=paths.participant_id,
        head_pose=paths.head_pose,
    )
    try:
        for plan in plans:
            runner = ProtocolRunner(plan)
            protocol_pairs = 0
            while True:
                elapsed_ms = protocol_pairs * interval_ms
                display_state = runner.state_at(elapsed_ms)
                if display_state.phase == "finished":
                    break
                display_timestamp_ns = (
                    simulation.base_unix_timestamp_ns + pair_index * interval_ns
                )
                if (
                    display_state.confirmation_required
                    and not display_state.is_confirmed
                    and display_state.target_elapsed_ms is not None
                    and display_state.target_elapsed_ms >= plan.simulation_confirm_after_ms
                ):
                    runner.confirm(elapsed_ms, display_timestamp_ns)
                    display_state = runner.state_at(elapsed_ms)
                webcam = FramePacket(
                    pair_index,
                    pair_index * interval_ms,
                    simulation.base_unix_timestamp_ns + pair_index * interval_ns,
                    _simulation_frame(simulation.width, simulation.height, "webcam_front", pair_index),
                )
                iphone = FramePacket(
                    pair_index,
                    pair_index * interval_ms + simulation.phone_delay_ms,
                    simulation.base_unix_timestamp_ns + pair_index * interval_ns + phone_delay_ns,
                    _simulation_frame(simulation.width, simulation.height, "iphone_left", pair_index),
                )
                frame_samples.observe(
                    pair_index,
                    display_timestamp_ns,
                    webcam,
                    iphone,
                    display_state,
                )
                labels.write(
                    paths,
                    pair_index,
                    display_timestamp_ns,
                    webcam,
                    iphone,
                    display_state,
                    config.display,
                )
                if webcam_recorder:
                    webcam_recorder.write(webcam)
                if phone_recorder:
                    phone_recorder.write(iphone)
                pair_index += 1
                protocol_pairs += 1
            results.append(
                ProtocolResult(
                    plan.protocol_id,
                    "completed",
                    protocol_pairs,
                    frame_log_path,
                )
            )
            frame_samples.flush()
    finally:
        labels.close()
        frame_samples.close()
        if webcam_recorder:
            webcam_recorder.close()
        if phone_recorder:
            phone_recorder.close()
    return tuple(results)
