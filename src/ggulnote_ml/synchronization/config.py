from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass(frozen=True)
class FlashProtocolConfig:
    transition_count: int
    initial_dark_ms: float
    transition_interval_ms: float
    tail_duration_ms: float
    dark_bgr: Tuple[int, int, int]
    light_bgr: Tuple[int, int, int]
    fixation_radius_px: int


@dataclass(frozen=True)
class CameraDetectionConfig:
    roi_norm: Tuple[float, float, float, float]
    roi_candidates: Tuple[Tuple[float, float, float, float], ...]
    min_brightness_change: float


@dataclass(frozen=True)
class DetectionConfig:
    webcam: CameraDetectionConfig
    phonecam: CameraDetectionConfig
    comparison_window_frames: int
    max_latency_ms: float
    mad_multiplier: float
    min_valid_events: int

    def camera(self, name: str) -> CameraDetectionConfig:
        if name == "webcam":
            return self.webcam
        if name == "phonecam":
            return self.phonecam
        raise ValueError("Unknown latency camera: %s" % name)


@dataclass(frozen=True)
class MatchingConfig:
    max_pair_diff_ms: float
    max_target_gap_ms: float
    enforce_one_to_one: bool
    interpolate_dynamic_targets: bool


@dataclass(frozen=True)
class LatencySimulationConfig:
    fps: float
    webcam_latency_ms: float
    phonecam_latency_ms: float
    dark_brightness: float
    light_brightness: float
    noise_std: float
    random_seed: int
    base_unix_timestamp_ns: int


@dataclass(frozen=True)
class LatencyConfig:
    protocol: FlashProtocolConfig
    detection: DetectionConfig
    matching: MatchingConfig
    simulation: LatencySimulationConfig


def _mapping(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError("%s must be a mapping." % key)
    return value


def _required(raw: Dict[str, Any], key: str, section: str) -> Any:
    if key not in raw:
        raise ValueError("Missing latency config value: %s.%s" % (section, key))
    return raw[key]


def _bgr(value: Any, name: str) -> Tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("%s must be a three-item BGR list." % name)
    result = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in result):
        raise ValueError("%s channels must be within [0, 255]." % name)
    return result


def _camera_detection(raw: Dict[str, Any], name: str) -> CameraDetectionConfig:
    section = "detection.cameras.%s" % name
    roi = _roi(_required(raw, "roi_norm", section), "%s.roi_norm" % section)
    candidates_raw = raw.get("roi_candidates", [])
    if not isinstance(candidates_raw, list):
        raise ValueError("%s.roi_candidates must be a list." % section)
    candidates = [roi]
    for index, value in enumerate(candidates_raw):
        candidate = _roi(
            value,
            "%s.roi_candidates[%d]" % (section, index),
        )
        if candidate not in candidates:
            candidates.append(candidate)
    threshold = float(_required(raw, "min_brightness_change", section))
    if threshold <= 0:
        raise ValueError("%s.min_brightness_change must be positive." % section)
    return CameraDetectionConfig(
        roi_norm=roi,
        roi_candidates=tuple(candidates),
        min_brightness_change=threshold,
    )


def _roi(value: Any, name: str) -> Tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("%s must contain x0, y0, x1, y1." % name)
    roi = tuple(float(item) for item in value)
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("%s must be ordered within [0, 1]." % name)
    return roi


def load_latency_config(path: Path) -> LatencyConfig:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Latency config does not exist: %s" % resolved)
    with resolved.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("Latency config root must be a mapping.")

    protocol_raw = _mapping(raw, "protocol")
    detection_raw = _mapping(raw, "detection")
    matching_raw = _mapping(raw, "matching")
    simulation_raw = _mapping(raw, "simulation")

    protocol = FlashProtocolConfig(
        transition_count=int(_required(protocol_raw, "transition_count", "protocol")),
        initial_dark_ms=float(_required(protocol_raw, "initial_dark_ms", "protocol")),
        transition_interval_ms=float(
            _required(protocol_raw, "transition_interval_ms", "protocol")
        ),
        tail_duration_ms=float(_required(protocol_raw, "tail_duration_ms", "protocol")),
        dark_bgr=_bgr(_required(protocol_raw, "dark_bgr", "protocol"), "protocol.dark_bgr"),
        light_bgr=_bgr(
            _required(protocol_raw, "light_bgr", "protocol"), "protocol.light_bgr"
        ),
        fixation_radius_px=int(_required(protocol_raw, "fixation_radius_px", "protocol")),
    )
    if protocol.transition_count < 2:
        raise ValueError("protocol.transition_count must be at least 2.")
    if min(
        protocol.initial_dark_ms,
        protocol.transition_interval_ms,
        protocol.tail_duration_ms,
    ) <= 0:
        raise ValueError("Protocol durations must be positive.")
    if protocol.fixation_radius_px <= 0:
        raise ValueError("protocol.fixation_radius_px must be positive.")

    camera_detection_raw = _mapping(detection_raw, "cameras")
    detection = DetectionConfig(
        webcam=_camera_detection(_mapping(camera_detection_raw, "webcam"), "webcam"),
        phonecam=_camera_detection(_mapping(camera_detection_raw, "phonecam"), "phonecam"),
        comparison_window_frames=int(
            _required(detection_raw, "comparison_window_frames", "detection")
        ),
        max_latency_ms=float(_required(detection_raw, "max_latency_ms", "detection")),
        mad_multiplier=float(_required(detection_raw, "mad_multiplier", "detection")),
        min_valid_events=int(_required(detection_raw, "min_valid_events", "detection")),
    )
    if detection.comparison_window_frames <= 0:
        raise ValueError("detection.comparison_window_frames must be positive.")
    if detection.max_latency_ms <= 0:
        raise ValueError("Detection maximum latency must be positive.")
    if detection.mad_multiplier <= 0 or detection.min_valid_events <= 0:
        raise ValueError("Detection robust-statistic parameters must be positive.")
    if detection.min_valid_events > protocol.transition_count:
        raise ValueError("detection.min_valid_events cannot exceed transition_count.")

    matching = MatchingConfig(
        max_pair_diff_ms=float(
            _required(matching_raw, "max_pair_diff_ms", "matching")
        ),
        max_target_gap_ms=float(
            _required(matching_raw, "max_target_gap_ms", "matching")
        ),
        enforce_one_to_one=bool(
            _required(matching_raw, "enforce_one_to_one", "matching")
        ),
        interpolate_dynamic_targets=bool(
            _required(matching_raw, "interpolate_dynamic_targets", "matching")
        ),
    )
    if matching.max_pair_diff_ms <= 0 or matching.max_target_gap_ms <= 0:
        raise ValueError("Matching time thresholds must be positive.")

    simulation = LatencySimulationConfig(
        fps=float(_required(simulation_raw, "fps", "simulation")),
        webcam_latency_ms=float(
            _required(simulation_raw, "webcam_latency_ms", "simulation")
        ),
        phonecam_latency_ms=float(
            _required(simulation_raw, "phonecam_latency_ms", "simulation")
        ),
        dark_brightness=float(_required(simulation_raw, "dark_brightness", "simulation")),
        light_brightness=float(_required(simulation_raw, "light_brightness", "simulation")),
        noise_std=float(_required(simulation_raw, "noise_std", "simulation")),
        random_seed=int(_required(simulation_raw, "random_seed", "simulation")),
        base_unix_timestamp_ns=int(
            _required(simulation_raw, "base_unix_timestamp_ns", "simulation")
        ),
    )
    if simulation.fps <= 0 or min(simulation.webcam_latency_ms, simulation.phonecam_latency_ms) < 0:
        raise ValueError("Simulation fps must be positive and latencies non-negative.")
    if not 0 <= simulation.dark_brightness < simulation.light_brightness <= 255:
        raise ValueError("Simulation brightness values must be ordered within [0, 255].")
    if simulation.noise_std < 0 or simulation.base_unix_timestamp_ns <= 0:
        raise ValueError("Simulation noise and base timestamp are invalid.")

    return LatencyConfig(protocol, detection, matching, simulation)
