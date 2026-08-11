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
from .protocols import ProtocolFrameState, ProtocolPlan, normalized_coordinates, write_protocol_events
from .recorder import SessionRecorder


LABEL_COLUMNS = (
    "participant",
    "protocol",
    "split",
    "frame",
    "display_timestamp",
    "webcam_frame",
    "webcam_timestamp",
    "phonecam_frame",
    "phonecam_timestamp",
    "time_diff_ms",
    "x_px",
    "y_px",
    "x_norm",
    "y_norm",
    "x_centered",
    "y_centered",
    "segment",
    "repeat",
    "target",
    "direction",
    "settling",
    "usable",
    "training",
)


@dataclass(frozen=True)
class ProtocolResult:
    protocol_id: str
    status: str
    frame_pairs: int
    label_path: Path


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
                "protocol": state.protocol_id,
                "split": state.split,
                "frame": pair_index,
                "display_timestamp": display_timestamp_ns,
                "webcam_frame": webcam.frame_index,
                "webcam_timestamp": webcam.unix_timestamp_ns,
                "phonecam_frame": iphone.frame_index,
                "phonecam_timestamp": iphone.unix_timestamp_ns,
                "time_diff_ms": "%.6f"
                % ((iphone.unix_timestamp_ns - webcam.unix_timestamp_ns) / 1_000_000.0),
                "x_px": x_px,
                "y_px": y_px,
                "x_norm": x_norm,
                "y_norm": y_norm,
                "x_centered": centered_x,
                "y_centered": centered_y,
                "segment": state.segment_index if state.segment_index is not None else "",
                "repeat": state.repeat_index if state.repeat_index is not None else "",
                "target": state.target_index if state.target_index is not None else "",
                "direction": state.direction,
                "settling": int(state.is_settling),
                "usable": int(state.is_usable_window),
                "training": int(state.is_training_sample),
            }
        )

    def close(self) -> None:
        self._file.close()


def render_protocol(display: DisplayConfig, plan: ProtocolPlan, state: ProtocolFrameState) -> np.ndarray:
    canvas = np.full(
        (display.canvas_height, display.canvas_width, 3),
        display.background_bgr,
        dtype=np.uint8,
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
        cv2.circle(canvas, (x, y), display.point_radius_px, display.point_bgr, -1, cv2.LINE_AA)
    elif state.phase == "complete_message":
        _center_text(canvas, "%s complete" % plan.protocol_id, display.canvas_height // 2, 1.0)
    return canvas


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
    )
    phone = SessionRecorder(
        paths.phone_directory / "capture.mp4",
        paths.phone_directory / "timestamps.csv",
        config.recording.video_codec,
        fps,
    )
    return webcam, phone


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
        collection_started_ns = time.monotonic_ns()
        webcam_recorder, phone_recorder = _collection_recorders(
            config, paths, by_role["webcam_front"].fps
        )
        label_path = paths.labels_directory / "labels.csv"
        labels = PairedLabelWriter(label_path)
        pair_index = 0
        cv2.namedWindow(config.display.window_name, cv2.WINDOW_NORMAL)
        if config.display.fullscreen:
            cv2.setWindowProperty(
                config.display.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

        try:
            for plan in plans:
                write_protocol_events(paths.events_directory / (plan.protocol_id + ".csv"), plan)
                protocol_started_ns = time.monotonic_ns()
                protocol_pairs = 0
                status = "completed"
                while True:
                    elapsed_ms = (time.monotonic_ns() - protocol_started_ns) / 1_000_000.0
                    display_state = plan.state_at(elapsed_ms)
                    if display_state.phase == "finished":
                        break
                    cv2.imshow(
                        config.display.window_name,
                        render_protocol(config.display, plan, display_state),
                    )
                    key = cv2.waitKey(1) & 0xFF
                    display_timestamp_ns = time.time_ns()
                    if key in {ord("q"), 27}:
                        status = "aborted"
                        break

                    webcam = webcam_source.read(collection_started_ns)
                    iphone = phone_source.read(collection_started_ns)
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
                results.append(ProtocolResult(plan.protocol_id, status, protocol_pairs, label_path))
                if status == "aborted":
                    break
        finally:
            labels.close()
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
    label_path = paths.labels_directory / "labels.csv"
    labels = PairedLabelWriter(label_path)
    try:
        for plan in plans:
            write_protocol_events(paths.events_directory / (plan.protocol_id + ".csv"), plan)
            protocol_pairs = 0
            while True:
                elapsed_ms = protocol_pairs * interval_ms
                display_state = plan.state_at(elapsed_ms)
                if display_state.phase == "finished":
                    break
                display_timestamp_ns = (
                    simulation.base_unix_timestamp_ns + pair_index * interval_ns
                )
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
            results.append(ProtocolResult(plan.protocol_id, "completed", protocol_pairs, label_path))
    finally:
        labels.close()
        if webcam_recorder:
            webcam_recorder.close()
        if phone_recorder:
            phone_recorder.close()
    return tuple(results)
