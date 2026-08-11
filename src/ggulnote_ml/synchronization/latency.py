from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .config import DetectionConfig, FlashProtocolConfig, LatencySimulationConfig


@dataclass(frozen=True)
class DisplayEvent:
    event_index: int
    display_timestamp_ns: int
    light_state: int

    def validate(self) -> None:
        if self.event_index < 0 or self.display_timestamp_ns <= 0:
            raise ValueError("Display event index and timestamp must be valid.")
        if self.light_state not in {0, 1}:
            raise ValueError("Display event light_state must be 0 or 1.")


@dataclass(frozen=True)
class BrightnessSample:
    frame: int
    timestamp_ns: int
    brightness: float

    def validate(self) -> None:
        if self.frame < 0 or self.timestamp_ns <= 0:
            raise ValueError("Brightness sample frame and timestamp must be valid.")
        if not np.isfinite(self.brightness):
            raise ValueError("Brightness sample must be finite.")


@dataclass(frozen=True)
class TransitionDetection:
    event_index: int
    light_state: int
    display_timestamp_ns: int
    detected_frame: Optional[int]
    detected_timestamp_ns: Optional[int]
    latency_ms: Optional[float]
    brightness_change: Optional[float]
    valid: bool
    invalid_reason: str


@dataclass(frozen=True)
class CameraLatencyEstimate:
    camera: str
    status: str
    median_ms: Optional[float]
    mad_ms: Optional[float]
    p95_ms: Optional[float]
    valid_events: int
    total_events: int
    detections: Tuple[TransitionDetection, ...]


def build_display_events(
    protocol: FlashProtocolConfig, base_timestamp_ns: int
) -> Tuple[DisplayEvent, ...]:
    if base_timestamp_ns <= 0:
        raise ValueError("base_timestamp_ns must be positive.")
    initial_ns = round(protocol.initial_dark_ms * 1_000_000)
    interval_ns = round(protocol.transition_interval_ms * 1_000_000)
    return tuple(
        DisplayEvent(
            event_index=index,
            display_timestamp_ns=base_timestamp_ns + initial_ns + index * interval_ns,
            light_state=1 if index % 2 == 0 else 0,
        )
        for index in range(protocol.transition_count)
    )


def _validate_ordered_inputs(
    samples: Sequence[BrightnessSample], events: Sequence[DisplayEvent]
) -> None:
    if not samples:
        raise ValueError("At least one brightness sample is required.")
    if not events:
        raise ValueError("At least one display event is required.")
    for sample in samples:
        sample.validate()
    for event in events:
        event.validate()
    if any(right.timestamp_ns <= left.timestamp_ns for left, right in zip(samples, samples[1:])):
        raise ValueError("Brightness sample timestamps must be strictly increasing.")
    if any(
        right.display_timestamp_ns <= left.display_timestamp_ns
        for left, right in zip(events, events[1:])
    ):
        raise ValueError("Display event timestamps must be strictly increasing.")


def estimate_camera_latency(
    camera: str,
    samples: Sequence[BrightnessSample],
    events: Sequence[DisplayEvent],
    config: DetectionConfig,
) -> CameraLatencyEstimate:
    """Detect brightness transitions and summarize camera pipeline latency.

    Input brightness samples must be ordered and use Unix nanoseconds. A positive
    signed before/after brightness difference is selected for light transitions,
    and the sign is reversed for dark transitions.
    """

    _validate_ordered_inputs(samples, events)
    window = config.comparison_window_frames
    if len(samples) < window * 2:
        raise ValueError("Not enough brightness samples for the comparison window.")

    timestamps = np.asarray([sample.timestamp_ns for sample in samples], dtype=np.int64)
    brightness = np.asarray([sample.brightness for sample in samples], dtype=np.float64)
    maximum_delay_ns = round(config.max_latency_ms * 1_000_000)
    used_frames = set()
    detections = []

    for event in events:
        direction = 1.0 if event.light_state == 1 else -1.0
        start = int(np.searchsorted(timestamps, event.display_timestamp_ns, side="left"))
        stop = int(
            np.searchsorted(
                timestamps,
                event.display_timestamp_ns + maximum_delay_ns,
                side="right",
            )
        )
        best_index = None
        best_change = float("-inf")
        for index in range(max(window, start), min(stop, len(samples) - window + 1)):
            if samples[index].frame in used_frames:
                continue
            before = float(np.mean(brightness[index - window : index]))
            after = float(np.mean(brightness[index : index + window]))
            signed_change = direction * (after - before)
            if signed_change > best_change:
                best_change = signed_change
                best_index = index

        if best_index is None:
            detections.append(
                TransitionDetection(
                    event.event_index,
                    event.light_state,
                    event.display_timestamp_ns,
                    None,
                    None,
                    None,
                    None,
                    False,
                    "no_frame_in_search_window",
                )
            )
            continue
        sample = samples[best_index]
        latency_ms = (sample.timestamp_ns - event.display_timestamp_ns) / 1_000_000.0
        if best_change < config.min_brightness_change:
            detections.append(
                TransitionDetection(
                    event.event_index,
                    event.light_state,
                    event.display_timestamp_ns,
                    sample.frame,
                    sample.timestamp_ns,
                    latency_ms,
                    best_change,
                    False,
                    "brightness_change_below_threshold",
                )
            )
            continue
        used_frames.add(sample.frame)
        detections.append(
            TransitionDetection(
                event.event_index,
                event.light_state,
                event.display_timestamp_ns,
                sample.frame,
                sample.timestamp_ns,
                latency_ms,
                best_change,
                True,
                "",
            )
        )

    preliminary = [item.latency_ms for item in detections if item.valid]
    if preliminary:
        center = float(np.median(preliminary))
        mad = float(np.median(np.abs(np.asarray(preliminary) - center)))
        if mad > 0:
            limit = config.mad_multiplier * mad
            detections = [
                replace(item, valid=False, invalid_reason="latency_outlier")
                if item.valid and abs(float(item.latency_ms) - center) > limit
                else item
                for item in detections
            ]

    valid_latencies = np.asarray(
        [float(item.latency_ms) for item in detections if item.valid], dtype=np.float64
    )
    enough = len(valid_latencies) >= config.min_valid_events
    return CameraLatencyEstimate(
        camera=camera,
        status="valid" if enough else "insufficient_valid_events",
        median_ms=float(np.median(valid_latencies)) if len(valid_latencies) else None,
        mad_ms=(
            float(np.median(np.abs(valid_latencies - np.median(valid_latencies))))
            if len(valid_latencies)
            else None
        ),
        p95_ms=float(np.percentile(valid_latencies, 95)) if len(valid_latencies) else None,
        valid_events=len(valid_latencies),
        total_events=len(events),
        detections=tuple(detections),
    )


def simulate_brightness_samples(
    events: Sequence[DisplayEvent],
    protocol: FlashProtocolConfig,
    simulation: LatencySimulationConfig,
    latency_ms: float,
    seed_offset: int = 0,
) -> Tuple[BrightnessSample, ...]:
    """Generate a deterministic received-frame brightness signal with known latency."""

    if not events:
        raise ValueError("Simulation requires display events.")
    interval_ns = round(1_000_000_000 / simulation.fps)
    latency_ns = round(latency_ms * 1_000_000)
    tail_ns = round(protocol.tail_duration_ms * 1_000_000)
    end_ns = events[-1].display_timestamp_ns + tail_ns
    count = int((end_ns - simulation.base_unix_timestamp_ns) // interval_ns) + 1
    rng = np.random.default_rng(simulation.random_seed + seed_offset)
    samples = []
    for frame in range(count):
        received_timestamp = simulation.base_unix_timestamp_ns + frame * interval_ns
        scene_timestamp = received_timestamp - latency_ns
        light_state = 0
        for event in events:
            if event.display_timestamp_ns > scene_timestamp:
                break
            light_state = event.light_state
        base = (
            simulation.light_brightness if light_state else simulation.dark_brightness
        )
        brightness = float(base + rng.normal(0.0, simulation.noise_std))
        samples.append(BrightnessSample(frame, received_timestamp, brightness))
    return tuple(samples)


def brightness_from_frame(
    frame: np.ndarray, roi_norm: Tuple[float, float, float, float]
) -> float:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Brightness input frame must be uint8 BGR with shape (H, W, 3).")
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = roi_norm
    left = min(width - 1, max(0, round(x0 * width)))
    right = min(width, max(left + 1, round(x1 * width)))
    top = min(height - 1, max(0, round(y0 * height)))
    bottom = min(height, max(top + 1, round(y1 * height)))
    gray = np.mean(frame[top:bottom, left:right], axis=2)
    return float(np.mean(gray))


def ensure_estimates_valid(estimates: Iterable[CameraLatencyEstimate]) -> None:
    invalid = [estimate.camera for estimate in estimates if estimate.status != "valid"]
    if invalid:
        raise ValueError("Insufficient valid latency events for: %s" % ", ".join(invalid))
