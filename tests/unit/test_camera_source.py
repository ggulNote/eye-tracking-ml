import time

import numpy as np
import pytest

from ggulnote_ml.capture.camera import CameraSource
from ggulnote_ml.capture.config import CameraConfig


class _FakeCapture:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def read(self):
        self.calls += 1
        return self.responses.pop(0)


def _config(retries):
    return CameraConfig(
        name="iphone",
        role="iphone_left",
        device_index=0,
        backend="auto",
        width=4,
        height=3,
        fps=30.0,
        warmup_frames=0,
        max_identical_frames=5,
        read_retry_count=retries,
        read_retry_delay_ms=0.0,
        mirror=False,
    )


def test_camera_source_recovers_from_transient_read_failure():
    frame = np.full((3, 4, 3), 127, dtype=np.uint8)
    source = CameraSource(_config(retries=2))
    source._capture = _FakeCapture(((False, None), (False, None), (True, frame)))

    packet = source.read(time.monotonic_ns())

    assert packet.frame_index == 0
    assert source._capture.calls == 3
    np.testing.assert_array_equal(packet.frame, frame)


def test_camera_source_reports_failure_after_configured_retries():
    source = CameraSource(_config(retries=2))
    source._capture = _FakeCapture(((False, None),) * 3)

    with pytest.raises(RuntimeError, match="after 3 attempt"):
        source.read(time.monotonic_ns())
