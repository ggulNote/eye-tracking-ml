from __future__ import annotations

from typing import Any

import numpy as np

from ggulnote_ml.capture.calibration_assets import CameraIntrinsics
from ggulnote_ml.exceptions import ContractError


class FrameUndistorter:
    """Apply one camera's calibrated lens model while preserving frame size."""

    def __init__(self, intrinsics: CameraIntrinsics, cv2_module: Any) -> None:
        self.intrinsics = intrinsics
        self._cv2 = cv2_module

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ContractError("Frame undistortion input must be uint8 BGR [H,W,3].")
        height, width = frame.shape[:2]
        expected = (self.intrinsics.image_width, self.intrinsics.image_height)
        if (width, height) != expected:
            raise ContractError(
                "Frame size %dx%d does not match Camera.mat size %dx%d: %s"
                % (
                    width,
                    height,
                    expected[0],
                    expected[1],
                    self.intrinsics.path,
                )
            )
        corrected = self._cv2.undistort(
            frame,
            self.intrinsics.camera_matrix,
            self.intrinsics.distortion_coefficients,
            None,
            self.intrinsics.camera_matrix,
        )
        corrected = np.asarray(corrected)
        if corrected.shape != frame.shape or corrected.dtype != np.uint8:
            raise ContractError(
                "OpenCV undistortion must preserve uint8 BGR frame shape."
            )
        return np.ascontiguousarray(corrected)
