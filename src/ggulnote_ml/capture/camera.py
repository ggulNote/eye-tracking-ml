from __future__ import annotations

import time
from typing import Dict, List

import cv2
import numpy as np

from .config import CameraConfig
from .contracts import FramePacket


BACKENDS: Dict[str, int] = {
    "auto": cv2.CAP_ANY,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "dshow": cv2.CAP_DSHOW,
    "v4l2": cv2.CAP_V4L2,
}


class FrozenFrameGuard:
    """Fail when a camera repeatedly returns the exact same sensor frame."""

    def __init__(self, camera_name: str, max_identical_frames: int) -> None:
        if max_identical_frames <= 0:
            raise ValueError("max_identical_frames must be positive.")
        self.camera_name = camera_name
        self.max_identical_frames = max_identical_frames
        self._previous_frame = None
        self._identical_count = 0

    def observe(self, frame: np.ndarray) -> None:
        if self._previous_frame is not None and np.array_equal(frame, self._previous_frame):
            self._identical_count += 1
        else:
            self._identical_count = 0
        self._previous_frame = frame.copy()
        if self._identical_count >= self.max_identical_frames:
            raise RuntimeError(
                "Camera %s appears frozen: %d consecutive identical frames. "
                "Reconnect the camera before collecting data."
                % (self.camera_name, self._identical_count + 1)
            )

    def reset(self) -> None:
        self._previous_frame = None
        self._identical_count = 0


class CameraSource:
    """Read one OS camera device while preserving a fixed frame contract."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture = None
        self._frame_index = 0
        self._freeze_guard = FrozenFrameGuard(
            config.name, config.max_identical_frames
        )

    def __enter__(self) -> "CameraSource":
        capture = cv2.VideoCapture(self.config.device_index, BACKENDS[self.config.backend])
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                "Could not open camera %s at device index %d. Run --list-cameras and close other camera apps."
                % (self.config.name, self.config.device_index)
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._capture = capture
        for _ in range(self.config.warmup_frames):
            capture.read()
        return self

    def read(self, session_started_ns: int) -> FramePacket:
        if self._capture is None:
            raise RuntimeError("CameraSource must be used as a context manager.")
        ok = False
        frame = None
        attempts = self.config.read_retry_count + 1
        for attempt in range(attempts):
            ok, frame = self._capture.read()
            if ok and frame is not None:
                break
            if attempt + 1 < attempts and self.config.read_retry_delay_ms > 0:
                time.sleep(self.config.read_retry_delay_ms / 1000.0)
        captured_ns = time.monotonic_ns()
        unix_timestamp_ns = time.time_ns()
        if not ok or frame is None:
            raise RuntimeError(
                "Could not read a frame from camera %s after %d attempt(s). "
                "Check the USB/Continuity Camera connection."
                % (self.config.name, attempts)
            )
        if frame.shape[:2] != (self.config.height, self.config.width):
            frame = cv2.resize(frame, (self.config.width, self.config.height))
        self._freeze_guard.observe(frame)
        packet = FramePacket(
            frame_index=self._frame_index,
            captured_at_ms=(captured_ns - session_started_ns) / 1_000_000.0,
            unix_timestamp_ns=unix_timestamp_ns,
            frame=frame,
        )
        packet.validate()
        self._frame_index += 1
        return packet

    def reset_session(self) -> None:
        """Restart frame numbering after an unrecorded camera preflight."""

        self._frame_index = 0
        self._freeze_guard.reset()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def probe_camera_indices(max_devices: int, backend_name: str) -> List[int]:
    """Return device indices that can produce at least one readable frame."""
    available = []
    for index in range(max_devices):
        capture = cv2.VideoCapture(index, BACKENDS[backend_name])
        try:
            ok, frame = capture.read() if capture.isOpened() else (False, None)
            if ok and frame is not None:
                available.append(index)
        finally:
            capture.release()
    return available
