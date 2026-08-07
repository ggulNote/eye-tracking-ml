from __future__ import annotations

import argparse
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import cv2

from .camera import CameraSource, probe_camera_indices
from .config import load_capture_config
from .recorder import SessionRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview and record multiple gaze cameras.")
    parser.add_argument("--config", default="configs/capture.yaml")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--max-devices", type=int, default=10)
    return parser.parse_args()


def run_capture(config_path: Path) -> None:
    config = load_capture_config(config_path)
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    recorders = {}
    if config.recording.enabled:
        for camera in config.cameras:
            recorders[camera.name] = SessionRecorder(
                config.recording.output_directory / (session_id + "_" + camera.name + ".mp4"),
                config.recording.output_directory / (session_id + "_" + camera.name + "_timestamps.csv"),
                config.recording.video_codec,
                camera.fps,
            )

    print("Starting cameras: %s" % ", ".join(camera.name for camera in config.cameras))
    print("Press Q or Escape in a preview window to stop all cameras.")
    try:
        with ExitStack() as stack:
            sources = {
                camera.name: stack.enter_context(CameraSource(camera)) for camera in config.cameras
            }
            session_started_ns = time.monotonic_ns()
            while True:
                for camera in config.cameras:
                    packet = sources[camera.name].read(session_started_ns)
                    if camera.name in recorders:
                        # Mirroring is preview-only; training artifacts preserve raw orientation.
                        recorders[camera.name].write(packet)
                    if config.preview.enabled:
                        preview = cv2.flip(packet.frame, 1) if camera.mirror else packet.frame
                        cv2.imshow(config.preview.window_name_prefix + " - " + camera.name, preview)
                if config.preview.enabled and cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                    break
    finally:
        for recorder in recorders.values():
            recorder.close()
        cv2.destroyAllWindows()

    for name, recorder in recorders.items():
        print("[%s] video: %s" % (name, recorder.video_path))
        print("[%s] timestamps: %s" % (name, recorder.timestamps_path))


def main() -> None:
    args = parse_args()
    try:
        config = load_capture_config(Path(args.config))
        if args.list_cameras:
            if args.max_devices <= 0:
                raise ValueError("--max-devices must be positive.")
            indices = probe_camera_indices(args.max_devices, config.cameras[0].backend)
            print("Available camera indices: %s" % (indices if indices else "none"))
            return
        run_capture(Path(args.config))
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print("[capture error] %s" % error, file=sys.stderr)
        raise SystemExit(1) from error
