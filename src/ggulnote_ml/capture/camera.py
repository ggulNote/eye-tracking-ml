from __future__ import annotations

import time
from typing import Dict, List

import cv2

from .config import CameraConfig
from .contracts import FramePacket


BACKENDS: Dict[str, int] = {
    "auto": cv2.CAP_ANY,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "dshow": cv2.CAP_DSHOW,
    "v4l2": cv2.CAP_V4L2,
}


class CameraSource:
    """Read one OS camera device while preserving a fixed frame contract."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture = None
        self._frame_index = 0

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
        ok, frame = self._capture.read()
        captured_ns = time.monotonic_ns()
        unix_timestamp_ns = time.time_ns()
        if not ok or frame is None:
            raise RuntimeError("Could not read a frame from camera %s." % self.config.name)
        if frame.shape[:2] != (self.config.height, self.config.width):
            frame = cv2.resize(frame, (self.config.width, self.config.height))
        packet = FramePacket(
            frame_index=self._frame_index,
            captured_at_ms=(captured_ns - session_started_ns) / 1_000_000.0,
            unix_timestamp_ns=unix_timestamp_ns,
            frame=frame,
        )
        packet.validate()
        self._frame_index += 1
        return packet

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
