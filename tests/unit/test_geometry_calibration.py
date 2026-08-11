from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.io import loadmat

from ggulnote_ml.capture.geometry import (
    CheckerboardSpec,
    ScreenSpec,
    calibrate_camera_intrinsics,
    checkerboard_object_points,
    generate_checkerboard_svg,
    write_camera_mat,
    write_screen_size_mat,
)
from ggulnote_ml.capture.geometry_config import load_geometry_calibration_config


def _board() -> CheckerboardSpec:
    return CheckerboardSpec(9, 6, 25.0, 297.0, 210.0)


def test_default_geometry_config_matches_fixed_macbook_setup():
    config = load_geometry_calibration_config(Path("configs/geometry_calibration.yaml"))

    assert config.setup_id == "macbook_air_m5_13_iphone16"
    assert config.screen.width_pixel == 1920
    assert config.screen.height_pixel == 1248
    assert config.checkerboard.spec.pattern_size == (9, 6)
    assert config.checkerboard.spec.squares_x == 10
    assert config.checkerboard.spec.squares_y == 7
    assert config.screen_verification_source == "apple_official_224ppi"
    assert config.checkerboard.verification_source == "unverified"


def test_checkerboard_svg_uses_exact_a4_and_square_dimensions(tmp_path):
    output = generate_checkerboard_svg(tmp_path / "board.svg", _board())
    text = output.read_text(encoding="utf-8")

    assert 'width="297.0000mm"' in text
    assert 'height="210.0000mm"' in text
    assert text.count("<rect") == 36  # one white page + 35 black squares
    assert 'width="25.0000" height="25.0000"' in text
    with pytest.raises(FileExistsError):
        generate_checkerboard_svg(output, _board())


def test_screen_size_mat_contract_and_aspect_ratio_validation(tmp_path):
    output = write_screen_size_mat(
        tmp_path / "screenSize.mat",
        ScreenSpec(1920, 1248, 290.3, 188.7),
    )
    values = loadmat(output)

    assert values["width_pixel"][0, 0] == 1920
    assert values["height_pixel"][0, 0] == 1248
    assert values["width_mm"][0, 0] == pytest.approx(290.3)
    assert values["height_mm"][0, 0] == pytest.approx(188.7)
    assert values["verification_source"][0] == "unspecified"
    with pytest.raises(ValueError, match="aspect ratios"):
        ScreenSpec(1920, 1080, 290.3, 188.7).validate()


def test_camera_intrinsics_from_synthetic_checkerboard_views(tmp_path):
    spec = _board()
    object_points = checkerboard_object_points(spec)
    expected_matrix = np.array(
        [[900.0, 0.0, 640.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    observations = []
    poses = (
        ((0.05, -0.10, 0.02), (-100.0, -70.0, 650.0)),
        ((-0.08, 0.12, -0.03), (-70.0, -90.0, 720.0)),
        ((0.15, 0.04, 0.08), (-120.0, -50.0, 800.0)),
        ((-0.12, -0.06, 0.10), (-40.0, -100.0, 690.0)),
        ((0.03, 0.18, -0.09), (-130.0, -60.0, 760.0)),
        ((-0.18, 0.02, 0.04), (-80.0, -40.0, 840.0)),
    )
    for rotation, translation in poses:
        points, _ = cv2.projectPoints(
            object_points,
            np.asarray(rotation, dtype=np.float64),
            np.asarray(translation, dtype=np.float64),
            expected_matrix,
            np.zeros((1, 5), dtype=np.float64),
        )
        observations.append(points.astype(np.float32))

    result = calibrate_camera_intrinsics(observations, (1280, 720), spec)

    assert result.camera_matrix.shape == (3, 3)
    assert result.distortion_coefficients.size >= 4
    assert result.rms_error_px < 0.01
    output = write_camera_mat(tmp_path / "Camera.mat", result, spec, 1.0)
    values = loadmat(output)
    assert values["cameraMatrix"].shape == (3, 3)
    assert values["rvecs"].shape == (len(poses), 3)
    assert values["tvecs"].shape == (len(poses), 3)
    assert values["retval"][0, 0] < 0.01
