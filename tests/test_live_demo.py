from __future__ import annotations

import numpy as np
import pytest

from gaze_pipeline.cli import build_parser
from gaze_pipeline.live_demo import (
    AffineCalibration,
    LiveDemoError,
    normalized_to_pixel,
    parse_camera_source,
    pixel_to_normalized,
)


def test_centered_normalized_pixel_round_trip() -> None:
    pixel = normalized_to_pixel(np.asarray([0.0, 0.0]), 1920, 1080, clamp=True)
    normalized = pixel_to_normalized(pixel, 1920, 1080)

    assert pixel == (960, 540)
    assert normalized == pytest.approx([0.0, 0.0])


def test_normalized_to_pixel_clamps_out_of_screen_prediction() -> None:
    assert normalized_to_pixel(np.asarray([-2.0, 3.0]), 100, 50, clamp=True) == (0, 49)


def test_affine_calibration_recovers_translation_and_scale() -> None:
    calibration = AffineCalibration()
    predictions = [
        np.asarray([-0.3, -0.3]),
        np.asarray([0.3, -0.3]),
        np.asarray([-0.3, 0.3]),
        np.asarray([0.3, 0.3]),
    ]
    for prediction in predictions:
        target = np.asarray([1.1 * prediction[0] + 0.04, 0.9 * prediction[1] - 0.02])
        calibration.add(prediction, target)

    assert calibration.matrix is not None
    assert calibration.apply(np.asarray([0.1, -0.2])) == pytest.approx([0.15, -0.2])


def test_affine_calibration_requires_non_collinear_points() -> None:
    calibration = AffineCalibration()
    calibration.add(np.asarray([-0.2, 0.0]), np.asarray([-0.1, 0.0]))
    calibration.add(np.asarray([0.0, 0.0]), np.asarray([0.1, 0.0]))
    fitted = calibration.add(np.asarray([0.2, 0.0]), np.asarray([0.3, 0.0]))

    assert fitted is False
    assert calibration.matrix is None


def test_parse_camera_source_supports_device_index_and_url() -> None:
    assert parse_camera_source(" 2 ") == 2
    assert parse_camera_source("rtsp://phone/live") == "rtsp://phone/live"
    with pytest.raises(LiveDemoError):
        parse_camera_source("  ")


def test_demo_cli_parses_portable_options() -> None:
    args = build_parser().parse_args(
        [
            "demo",
            "--front-camera",
            "2",
            "--side-camera",
            "rtsp://phone/live",
            "--side-rotate",
            "90",
            "--verify-only",
        ]
    )

    assert args.command == "demo"
    assert args.front_camera == "2"
    assert args.side_camera == "rtsp://phone/live"
    assert args.side_rotate == 90
    assert args.verify_only is True


def test_demo_cli_accepts_apple_silicon_mps() -> None:
    args = build_parser().parse_args(["demo", "--device", "mps", "--verify-only"])

    assert args.device == "mps"
