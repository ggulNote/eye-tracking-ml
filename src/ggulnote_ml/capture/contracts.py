from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """One camera frame using the shared capture-session clock.

    Input contract:
      frame_index: non-negative, per-camera sequential index
      captured_at_ms: non-negative milliseconds from the common session start
      frame: uint8 BGR array with shape (height, width, 3)
    """

    frame_index: int
    captured_at_ms: float
    frame: np.ndarray

    def validate(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative.")
        if self.captured_at_ms < 0:
            raise ValueError("captured_at_ms must be non-negative.")
        if self.frame.dtype != np.uint8:
            raise ValueError("frame dtype must be uint8, got %s." % self.frame.dtype)
        if self.frame.ndim != 3 or self.frame.shape[2] != 3:
            raise ValueError("frame shape must be (height, width, 3), got %s." % (self.frame.shape,))
