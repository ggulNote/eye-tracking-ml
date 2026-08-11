from __future__ import annotations

import csv
import json
import shutil
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ggulnote_ml.capture.camera import CameraSource
from ggulnote_ml.capture.config import CaptureConfig
from ggulnote_ml.capture.dataset import normalize_participant_id
from ggulnote_ml.capture.recorder import SessionRecorder

from .config import LatencyConfig
from .latency import (
    BrightnessSample,
    CameraLatencyEstimate,
    DisplayEvent,
    brightness_from_frame,
    build_display_events,
    ensure_estimates_valid,
    estimate_camera_latency,
    simulate_brightness_samples,
)


@dataclass(frozen=True)
class LatencyPaths:
    participant_id: str
    calibration_directory: Path
    run_directory: Path
    final_json: Path
    run_json: Path
    events_csv: Path
    detections_csv: Path
    webcam_brightness_csv: Path
    phonecam_brightness_csv: Path
    webcam_video: Path
    phonecam_video: Path


def _default_measurement_id() -> str:
    return datetime.now(timezone.utc).strftime("latency_%Y%m%dT%H%M%S_%fZ")


def create_latency_paths(
    dataset_root: Path, participant_id: str, measurement_id: Optional[str] = None
) -> LatencyPaths:
    participant_id = normalize_participant_id(participant_id)
    calibration = dataset_root.expanduser().resolve() / participant_id / "Calibration"
    final_json = calibration / "latency.json"
    if final_json.exists():
        raise FileExistsError("Latency calibration already exists and will not be overwritten: %s" % final_json)
    run_directory = calibration / "latency_runs" / (measurement_id or _default_measurement_id())
    run_directory.mkdir(parents=True, exist_ok=False)
    return LatencyPaths(
        participant_id=participant_id,
        calibration_directory=calibration,
        run_directory=run_directory,
        final_json=final_json,
        run_json=run_directory / "result.json",
        events_csv=run_directory / "display_events.csv",
        detections_csv=run_directory / "detections.csv",
        webcam_brightness_csv=run_directory / "webcam_brightness.csv",
        phonecam_brightness_csv=run_directory / "phonecam_brightness.csv",
        webcam_video=run_directory / "webcam.mp4",
        phonecam_video=run_directory / "phonecam.mp4",
    )


def _atomic_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _write_events(path: Path, events: Sequence[DisplayEvent]) -> None:
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["event", "display_timestamp", "light_state"])
        for event in events:
            writer.writerow([event.event_index, event.display_timestamp_ns, event.light_state])


def _write_brightness(path: Path, samples: Sequence[BrightnessSample]) -> None:
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "timestamp", "brightness"])
        for sample in samples:
            writer.writerow([sample.frame, sample.timestamp_ns, "%.6f" % sample.brightness])


def _write_detections(path: Path, estimates: Iterable[CameraLatencyEstimate]) -> None:
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "camera",
                "event",
                "light_state",
                "display_timestamp",
                "detected_frame",
                "detected_timestamp",
                "latency_ms",
                "brightness_change",
                "valid",
                "invalid_reason",
            ]
        )
        for estimate in estimates:
            for detection in estimate.detections:
                writer.writerow(
                    [
                        estimate.camera,
                        detection.event_index,
                        detection.light_state,
                        detection.display_timestamp_ns,
                        detection.detected_frame if detection.detected_frame is not None else "",
                        detection.detected_timestamp_ns
                        if detection.detected_timestamp_ns is not None
                        else "",
                        "%.6f" % detection.latency_ms
                        if detection.latency_ms is not None
                        else "",
                        "%.6f" % detection.brightness_change
                        if detection.brightness_change is not None
                        else "",
                        int(detection.valid),
                        detection.invalid_reason,
                    ]
                )


def _estimate_summary(estimate: CameraLatencyEstimate) -> Dict[str, object]:
    return {
        "status": estimate.status,
        "median_ms": estimate.median_ms,
        "mad_ms": estimate.mad_ms,
        "p95_ms": estimate.p95_ms,
        "valid_events": estimate.valid_events,
        "total_events": estimate.total_events,
    }


def _finish_measurement(
    paths: LatencyPaths,
    latency_config: LatencyConfig,
    events: Sequence[DisplayEvent],
    webcam_samples: Sequence[BrightnessSample],
    phonecam_samples: Sequence[BrightnessSample],
    mode: str,
) -> Dict[str, object]:
    _write_events(paths.events_csv, events)
    _write_brightness(paths.webcam_brightness_csv, webcam_samples)
    _write_brightness(paths.phonecam_brightness_csv, phonecam_samples)
    estimates = (
        estimate_camera_latency("webcam", webcam_samples, events, latency_config.detection),
        estimate_camera_latency("phonecam", phonecam_samples, events, latency_config.detection),
    )
    _write_detections(paths.detections_csv, estimates)
    status = "valid" if all(item.status == "valid" for item in estimates) else "invalid"
    result: Dict[str, object] = {
        "schema_version": 1,
        "participant": paths.participant_id,
        "status": status,
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_run": str(paths.run_directory.relative_to(paths.calibration_directory)),
        "timestamp_unit": "unix_ns",
        "latency_unit": "ms",
        "protocol": asdict(latency_config.protocol),
        "detection": asdict(latency_config.detection),
        "cameras": {estimate.camera: _estimate_summary(estimate) for estimate in estimates},
    }
    _atomic_json(paths.run_json, result)
    if status == "valid":
        _atomic_json(paths.final_json, result)
    ensure_estimates_valid(estimates)
    return result


def _flash_canvas(capture_config: CaptureConfig, latency_config: LatencyConfig, state: int) -> np.ndarray:
    color = latency_config.protocol.light_bgr if state else latency_config.protocol.dark_bgr
    canvas = np.full(
        (capture_config.display.canvas_height, capture_config.display.canvas_width, 3),
        color,
        dtype=np.uint8,
    )
    fixation_color = (0, 0, 0) if state else (255, 255, 255)
    center = (capture_config.display.canvas_width // 2, capture_config.display.canvas_height // 2)
    cv2.circle(
        canvas,
        center,
        latency_config.protocol.fixation_radius_px,
        fixation_color,
        -1,
        cv2.LINE_AA,
    )
    return canvas


def run_real_latency_measurement(
    capture_config: CaptureConfig,
    latency_config: LatencyConfig,
    paths: LatencyPaths,
) -> Dict[str, object]:
    by_role = {camera.role: camera for camera in capture_config.cameras}
    webcam_samples: List[BrightnessSample] = []
    phonecam_samples: List[BrightnessSample] = []
    events: List[DisplayEvent] = []
    status = "running"
    with ExitStack() as stack:
        webcam_source = stack.enter_context(CameraSource(by_role["webcam_front"]))
        phonecam_source = stack.enter_context(CameraSource(by_role["iphone_left"]))
        webcam_recorder = SessionRecorder(
            paths.webcam_video,
            paths.run_directory / "webcam_timestamps.csv",
            capture_config.recording.video_codec,
            by_role["webcam_front"].fps,
        )
        phonecam_recorder = SessionRecorder(
            paths.phonecam_video,
            paths.run_directory / "phonecam_timestamps.csv",
            capture_config.recording.video_codec,
            by_role["iphone_left"].fps,
        )
        collection_started_ns = time.monotonic_ns()
        next_transition_ns = collection_started_ns + round(
            latency_config.protocol.initial_dark_ms * 1_000_000
        )
        interval_ns = round(latency_config.protocol.transition_interval_ms * 1_000_000)
        tail_ns = round(latency_config.protocol.tail_duration_ms * 1_000_000)
        current_state = 0
        cv2.namedWindow(capture_config.display.window_name, cv2.WINDOW_NORMAL)
        if capture_config.display.fullscreen:
            cv2.setWindowProperty(
                capture_config.display.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        try:
            while True:
                now_ns = time.monotonic_ns()
                transitioned = (
                    len(events) < latency_config.protocol.transition_count
                    and now_ns >= next_transition_ns
                )
                if transitioned:
                    current_state = 1 - current_state
                cv2.imshow(
                    capture_config.display.window_name,
                    _flash_canvas(capture_config, latency_config, current_state),
                )
                key = cv2.waitKey(1) & 0xFF
                if transitioned:
                    events.append(DisplayEvent(len(events), time.time_ns(), current_state))
                    next_transition_ns += interval_ns
                if key in {ord("q"), 27}:
                    status = "aborted"
                    break

                webcam = webcam_source.read(collection_started_ns)
                phonecam = phonecam_source.read(collection_started_ns)
                webcam_recorder.write(webcam)
                phonecam_recorder.write(phonecam)
                webcam_samples.append(
                    BrightnessSample(
                        webcam.frame_index,
                        webcam.unix_timestamp_ns,
                        brightness_from_frame(webcam.frame, latency_config.detection.roi_norm),
                    )
                )
                phonecam_samples.append(
                    BrightnessSample(
                        phonecam.frame_index,
                        phonecam.unix_timestamp_ns,
                        brightness_from_frame(phonecam.frame, latency_config.detection.roi_norm),
                    )
                )
                if len(events) == latency_config.protocol.transition_count:
                    last_event_age_ns = time.time_ns() - events[-1].display_timestamp_ns
                    if last_event_age_ns >= tail_ns:
                        status = "completed"
                        break
        finally:
            webcam_recorder.close()
            phonecam_recorder.close()
            cv2.destroyAllWindows()

    if status != "completed":
        _atomic_json(
            paths.run_json,
            {
                "schema_version": 1,
                "participant": paths.participant_id,
                "status": status,
                "mode": "camera",
            },
        )
        raise RuntimeError("Latency measurement was aborted; no final latency.json was written.")
    return _finish_measurement(
        paths,
        latency_config,
        tuple(events),
        tuple(webcam_samples),
        tuple(phonecam_samples),
        "camera",
    )


def run_latency_simulation(
    latency_config: LatencyConfig,
    paths: LatencyPaths,
) -> Dict[str, object]:
    events = build_display_events(
        latency_config.protocol, latency_config.simulation.base_unix_timestamp_ns
    )
    webcam_samples = simulate_brightness_samples(
        events,
        latency_config.protocol,
        latency_config.simulation,
        latency_config.simulation.webcam_latency_ms,
        seed_offset=0,
    )
    phonecam_samples = simulate_brightness_samples(
        events,
        latency_config.protocol,
        latency_config.simulation,
        latency_config.simulation.phonecam_latency_ms,
        seed_offset=1,
    )
    return _finish_measurement(
        paths,
        latency_config,
        events,
        webcam_samples,
        phonecam_samples,
        "simulation",
    )


def copy_config_snapshots(
    paths: LatencyPaths, capture_config_path: Path, latency_config_path: Path
) -> None:
    shutil.copyfile(capture_config_path, paths.run_directory / "capture_config.yaml")
    shutil.copyfile(latency_config_path, paths.run_directory / "latency_config.yaml")
