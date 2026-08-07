from __future__ import annotations

import csv
from pathlib import Path

import cv2

from .contracts import FramePacket


class SessionRecorder:
    """Write one camera's raw frames and shared-clock timestamps."""

    def __init__(self, video_path: Path, timestamps_path: Path, codec: str, fps: float) -> None:
        self.video_path = video_path
        self.timestamps_path = timestamps_path
        self.codec = codec
        self.fps = fps
        self._writer = None
        self._csv_file = None
        self._csv_writer = None

    def write(self, packet: FramePacket) -> None:
        height, width = packet.frame.shape[:2]
        if self._writer is None:
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (width, height))
            if not self._writer.isOpened():
                raise RuntimeError("Could not open video output: %s" % self.video_path)
            self._csv_file = self.timestamps_path.open("w", encoding="utf-8", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["frame_index", "captured_at_ms"])
        self._writer.write(packet.frame)
        self._csv_writer.writerow([packet.frame_index, "%.3f" % packet.captured_at_ms])

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
