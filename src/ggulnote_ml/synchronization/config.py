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
class DetectionConfig:
    roi_norm: Tuple[float, float, float, float]
    comparison_window_frames: int
    min_brightness_change: float
    max_latency_ms: float
    mad_multiplier: float
    min_valid_events: int


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

    roi_raw = _required(detection_raw, "roi_norm", "detection")
    if not isinstance(roi_raw, list) or len(roi_raw) != 4:
        raise ValueError("detection.roi_norm must contain x0, y0, x1, y1.")
    roi = tuple(float(value) for value in roi_raw)
    detection = DetectionConfig(
        roi_norm=roi,
        comparison_window_frames=int(
            _required(detection_raw, "comparison_window_frames", "detection")
        ),
        min_brightness_change=float(
            _required(detection_raw, "min_brightness_change", "detection")
        ),
        max_latency_ms=float(_required(detection_raw, "max_latency_ms", "detection")),
        mad_multiplier=float(_required(detection_raw, "mad_multiplier", "detection")),
        min_valid_events=int(_required(detection_raw, "min_valid_events", "detection")),
    )
    x0, y0, x1, y1 = detection.roi_norm
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("detection.roi_norm must be ordered within [0, 1].")
    if detection.comparison_window_frames <= 0:
        raise ValueError("detection.comparison_window_frames must be positive.")
    if detection.min_brightness_change <= 0 or detection.max_latency_ms <= 0:
        raise ValueError("Detection thresholds must be positive.")
    if detection.mad_multiplier <= 0 or detection.min_valid_events <= 0:
        raise ValueError("Detection robust-statistic parameters must be positive.")
    if detection.min_valid_events > protocol.transition_count:
        raise ValueError("detection.min_valid_events cannot exceed transition_count.")

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

    return LatencyConfig(protocol, detection, simulation)
