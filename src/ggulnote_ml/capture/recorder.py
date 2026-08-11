from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import imageio_ffmpeg

from .contracts import FramePacket


@dataclass(frozen=True)
class RecordingSummary:
    """Auditable timing information for one encoded camera video."""

    schema_version: int
    timing_mode: str
    configured_fps: float
    encoded_fps: float
    measured_fps: float
    frame_count: int
    timestamp_span_ms: float
    timestamp_scale: float


class SessionRecorder:
    """Write camera frames and normalize MP4 time to the measured timestamps.

    Input
        ``FramePacket.frame``: BGR uint8 image with a constant width and height.
        ``FramePacket.unix_timestamp_ns``: strictly increasing Unix nanoseconds.
    Output
        One MP4 with one encoded frame per input packet, one timestamp CSV row per
        frame, and a JSON timing summary. Frame indices are never duplicated or
        removed while correcting playback time.
    """

    def __init__(
        self,
        video_path: Path,
        timestamps_path: Path,
        codec: str,
        fps: float,
        timing_mode: str,
    ) -> None:
        if fps <= 0:
            raise ValueError("Recorder fps must be positive.")
        if timing_mode not in {"configured", "measured"}:
            raise ValueError("Recorder timing_mode must be configured or measured.")
        self.video_path = video_path
        self.timestamps_path = timestamps_path
        self.codec = codec
        self.fps = fps
        self.timing_mode = timing_mode
        self.summary: Optional[RecordingSummary] = None
        self._writer = None
        self._csv_file = None
        self._csv_writer = None
        self._frame_count = 0
        self._first_timestamp_ns: Optional[int] = None
        self._last_timestamp_ns: Optional[int] = None
        self._closed = False
        self._writer_path = (
            video_path.with_name(".%s.unscaled%s" % (video_path.stem, video_path.suffix))
            if timing_mode == "measured"
            else video_path
        )

    def write(self, packet: FramePacket) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed recorder.")
        if (
            self._last_timestamp_ns is not None
            and packet.unix_timestamp_ns <= self._last_timestamp_ns
        ):
            raise ValueError("Frame timestamps must be strictly increasing.")
        height, width = packet.frame.shape[:2]
        if self._writer is None:
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            if self.video_path.exists() or self._writer_path.exists():
                raise FileExistsError("Video output already exists: %s" % self.video_path)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(
                str(self._writer_path), fourcc, self.fps, (width, height)
            )
            if not self._writer.isOpened():
                raise RuntimeError("Could not open video output: %s" % self._writer_path)
            self._csv_file = self.timestamps_path.open("x", encoding="utf-8", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["frame", "elapsed_ms", "timestamp"])
        self._writer.write(packet.frame)
        self._csv_writer.writerow(
            [packet.frame_index, "%.3f" % packet.captured_at_ms, packet.unix_timestamp_ns]
        )
        if self._first_timestamp_ns is None:
            self._first_timestamp_ns = packet.unix_timestamp_ns
        self._last_timestamp_ns = packet.unix_timestamp_ns
        self._frame_count += 1

    def close(self) -> None:
        if self._closed:
            return
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        self._closed = True
        if self._frame_count == 0:
            return

        timestamp_span_ns = int(self._last_timestamp_ns) - int(self._first_timestamp_ns)
        measured_fps = (
            (self._frame_count - 1) * 1_000_000_000.0 / timestamp_span_ns
            if self._frame_count > 1 and timestamp_span_ns > 0
            else self.fps
        )
        timestamp_scale = self.fps / measured_fps
        encoded_fps = self.fps
        if self.timing_mode == "measured":
            self._normalize_video_timestamps(timestamp_scale)
            encoded_fps = measured_fps

        self.summary = RecordingSummary(
            schema_version=1,
            timing_mode=self.timing_mode,
            configured_fps=self.fps,
            encoded_fps=encoded_fps,
            measured_fps=measured_fps,
            frame_count=self._frame_count,
            timestamp_span_ms=timestamp_span_ns / 1_000_000.0,
            timestamp_scale=timestamp_scale if self.timing_mode == "measured" else 1.0,
        )
        metadata_path = self._metadata_path()
        temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(asdict(self.summary), file, ensure_ascii=False, indent=2)
        temporary.replace(metadata_path)

    def _normalize_video_timestamps(self, timestamp_scale: float) -> None:
        if abs(timestamp_scale - 1.0) <= 1e-6:
            self._writer_path.replace(self.video_path)
            return
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-loglevel",
            "error",
            "-itsscale",
            "%.12f" % timestamp_scale,
            "-i",
            str(self._writer_path),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(self.video_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            if self.video_path.exists():
                self.video_path.unlink()
            raise RuntimeError(
                "Could not normalize MP4 timestamps for %s: %s"
                % (self.video_path, completed.stderr.strip())
            )
        self._writer_path.unlink()

    def _metadata_path(self) -> Path:
        stem = self.timestamps_path.stem
        prefix = stem[: -len("_timestamps")] if stem.endswith("_timestamps") else ""
        name = "%s_recording.json" % prefix if prefix else "recording.json"
        return self.timestamps_path.with_name(name)
