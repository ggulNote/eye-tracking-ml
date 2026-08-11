import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ggulnote_ml.synchronization.calibration import (
    create_latency_paths,
    run_latency_simulation,
)
from ggulnote_ml.synchronization.config import load_latency_config
from ggulnote_ml.synchronization.latency import (
    BrightnessSample,
    brightness_from_frame,
    build_display_events,
    estimate_camera_latency,
    simulate_brightness_samples,
)


def test_latency_config_and_simulation_recover_known_delays_within_one_frame():
    config = load_latency_config(Path("configs/latency.yaml"))
    events = build_display_events(config.protocol, config.simulation.base_unix_timestamp_ns)
    webcam_samples = simulate_brightness_samples(
        events,
        config.protocol,
        config.simulation,
        config.simulation.webcam_latency_ms,
    )
    phonecam_samples = simulate_brightness_samples(
        events,
        config.protocol,
        config.simulation,
        config.simulation.phonecam_latency_ms,
        seed_offset=1,
    )

    webcam = estimate_camera_latency("webcam", webcam_samples, events, config.detection)
    phonecam = estimate_camera_latency("phonecam", phonecam_samples, events, config.detection)

    frame_ms = 1000.0 / config.simulation.fps
    assert webcam.status == "valid"
    assert phonecam.status == "valid"
    assert webcam.median_ms == pytest.approx(config.simulation.webcam_latency_ms, abs=frame_ms)
    assert phonecam.median_ms == pytest.approx(
        config.simulation.phonecam_latency_ms, abs=frame_ms
    )
    assert webcam.valid_events == config.protocol.transition_count
    assert phonecam.valid_events == config.protocol.transition_count


def test_detector_rejects_flat_brightness_signal():
    config = load_latency_config(Path("configs/latency.yaml"))
    events = build_display_events(config.protocol, config.simulation.base_unix_timestamp_ns)
    simulated = simulate_brightness_samples(
        events,
        config.protocol,
        config.simulation,
        config.simulation.webcam_latency_ms,
    )
    flat = tuple(replace(sample, brightness=50.0) for sample in simulated)

    estimate = estimate_camera_latency("webcam", flat, events, config.detection)

    assert estimate.status == "insufficient_valid_events"
    assert estimate.valid_events == 0
    assert {item.invalid_reason for item in estimate.detections} == {
        "brightness_change_below_threshold"
    }


def test_detector_accepts_low_contrast_face_reflection_signal():
    config = load_latency_config(Path("configs/latency.yaml"))
    events = build_display_events(config.protocol, config.simulation.base_unix_timestamp_ns)
    low_contrast_simulation = replace(
        config.simulation,
        dark_brightness=100.0,
        light_brightness=103.2,
        noise_std=0.05,
    )
    samples = simulate_brightness_samples(
        events,
        config.protocol,
        low_contrast_simulation,
        config.simulation.webcam_latency_ms,
    )
    detection = replace(
        config.detection,
        webcam=replace(config.detection.webcam, min_brightness_change=2.5),
    )

    estimate = estimate_camera_latency("webcam", samples, events, detection)

    assert estimate.status == "valid"
    assert estimate.valid_events >= config.detection.min_valid_events


def test_detector_rejects_non_monotonic_sample_timestamps():
    config = load_latency_config(Path("configs/latency.yaml"))
    events = build_display_events(config.protocol, config.simulation.base_unix_timestamp_ns)
    samples = (
        BrightnessSample(0, config.simulation.base_unix_timestamp_ns + 10, 20.0),
        BrightnessSample(1, config.simulation.base_unix_timestamp_ns + 5, 30.0),
        BrightnessSample(2, config.simulation.base_unix_timestamp_ns + 20, 40.0),
        BrightnessSample(3, config.simulation.base_unix_timestamp_ns + 30, 50.0),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        estimate_camera_latency("webcam", samples, events, config.detection)


def test_brightness_uses_configured_normalized_roi():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[2:8, 2:8] = 120

    brightness = brightness_from_frame(frame, (0.2, 0.2, 0.8, 0.8))

    assert brightness == pytest.approx(120.0)


def test_simulation_pipeline_writes_auditable_result_and_refuses_overwrite(tmp_path):
    config = load_latency_config(Path("configs/latency.yaml"))
    paths = create_latency_paths(tmp_path, "p00", measurement_id="contract")

    result = run_latency_simulation(config, paths)

    assert result["status"] == "valid"
    assert paths.final_json.is_file()
    assert paths.events_csv.is_file()
    assert paths.detections_csv.is_file()
    assert paths.webcam_brightness_csv.is_file()
    assert paths.phonecam_brightness_csv.is_file()
    with paths.final_json.open(encoding="utf-8") as file:
        saved = json.load(file)
    assert saved["cameras"]["webcam"]["valid_events"] == 15
    assert saved["cameras"]["phonecam"]["valid_events"] == 15
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        create_latency_paths(tmp_path, "p00", measurement_id="second")
