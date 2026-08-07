import numpy as np
from scipy.io import savemat

from ggulnote_ml.capture.calibration_assets import inspect_calibration_assets


def test_calibration_assets_validate_only_consumed_variables(tmp_path):
    calibration = tmp_path / "Calibration"
    for camera_name in ("webcam", "phonecam"):
        camera_directory = calibration / camera_name
        camera_directory.mkdir(parents=True)
        savemat(
            camera_directory / "Camera.mat",
            {
                "cameraMatrix": np.eye(3),
                "distCoeffs": np.zeros((1, 5)),
                "retval": np.array([[0.1]]),
                "rvecs": np.zeros((2, 3)),
                "tvecs": np.zeros((2, 3)),
                "unused_variable": np.ones((1, 1)),
            },
        )
        savemat(
            camera_directory / "monitorPose.mat",
            {"rvects": np.zeros((1, 3)), "tvecs": np.zeros((1, 3))},
        )
    savemat(
        calibration / "screenSize.mat",
        {
            "width_pixel": np.array([[1920]]),
            "height_pixel": np.array([[1080]]),
            "width_mm": np.array([[310.0]]),
            "height_mm": np.array([[174.0]]),
        },
    )

    result = inspect_calibration_assets(calibration)

    assert result["required_assets_valid"]
    assert result["cameras"]["webcam_front"]["camera"]["valid"]
    assert "unused_variable" in result["cameras"]["webcam_front"]["camera"]["available_variables"]
    assert not result["stereo"]["exists"]
    assert result["stereo"]["optional"]
