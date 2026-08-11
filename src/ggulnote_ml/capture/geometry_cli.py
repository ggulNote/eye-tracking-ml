from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import cv2

from .calibration_assets import inspect_calibration_assets
from .camera import CameraSource
from .config import CameraConfig, load_capture_config
from .geometry import (
    calibrate_camera_intrinsics,
    find_checkerboard_corners,
    generate_checkerboard_svg,
    write_calibration_summary,
    write_camera_mat,
    write_screen_size_mat,
)
from .geometry_config import GeometryCalibrationConfig, load_geometry_calibration_config


CAMERA_ALIASES = {
    "webcam": ("webcam_front", "webcam"),
    "phonecam": ("iphone_left", "phonecam"),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and capture geometry calibration assets for the fixed camera rig."
    )
    parser.add_argument("--config", default="configs/geometry_calibration.yaml")
    parser.add_argument("--capture-config", default="configs/capture.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    board = commands.add_parser("generate-board", help="Generate an exact-size A4 SVG checkerboard")
    board.add_argument("--output", type=Path)

    commands.add_parser("write-screen", help="Create screenSize.mat after ruler verification")

    intrinsics = commands.add_parser(
        "capture-intrinsics", help="Capture checkerboard views and create one Camera.mat"
    )
    intrinsics.add_argument("--camera", required=True, choices=sorted(CAMERA_ALIASES))

    commands.add_parser("inspect", help="Inspect generated MAT files")
    return parser.parse_args(argv)


def _camera_for_alias(capture_config_path: Path, alias: str) -> CameraConfig:
    capture = load_capture_config(capture_config_path)
    role, _ = CAMERA_ALIASES[alias]
    by_role = {camera.role: camera for camera in capture.cameras}
    return by_role[role]


def _capture_intrinsics(
    geometry: GeometryCalibrationConfig,
    capture_config_path: Path,
    alias: str,
) -> Path:
    if geometry.checkerboard.verification_source != "manual_measurement":
        raise ValueError(
            "Measure the printed checkerboard, update checkerboard.square_size_mm, "
            "and set verification_source=manual_measurement first."
        )
    camera = _camera_for_alias(capture_config_path, alias)
    role, directory_name = CAMERA_ALIASES[alias]
    camera_directory = geometry.output_directory / directory_name
    camera_mat = camera_directory / "Camera.mat"
    if camera_mat.exists():
        raise FileExistsError(
            "Calibration output already exists for %s and will not be overwritten: %s"
            % (alias, camera_directory)
        )
    run_id = datetime.now(timezone.utc).strftime("intrinsics_%Y%m%dT%H%M%S_%fZ")
    run_directory = camera_directory / "intrinsics_runs" / run_id
    captures_directory = run_directory / "captures"
    summary_path = run_directory / "result.json"
    captures_directory.mkdir(parents=True, exist_ok=True)

    image_points = []
    capture_paths = []
    image_size = None
    window = "Geometry calibration - %s" % alias
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    started_ns = time.monotonic_ns()
    try:
        with CameraSource(camera) as source:
            while len(image_points) < geometry.checkerboard.required_captures:
                packet = source.read(started_ns)
                frame = packet.frame
                image_size = (frame.shape[1], frame.shape[0])
                found, corners = find_checkerboard_corners(frame, geometry.checkerboard.spec)
                preview = frame.copy()
                if found and corners is not None:
                    cv2.drawChessboardCorners(
                        preview,
                        geometry.checkerboard.spec.pattern_size,
                        corners,
                        True,
                    )
                status = "%s %d/%d | SPACE: save | Q/ESC: finish" % (
                    "DETECTED" if found else "NOT FOUND",
                    len(image_points),
                    geometry.checkerboard.required_captures,
                )
                cv2.putText(
                    preview,
                    status,
                    (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if found else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    break
                if key == ord(" ") and found and corners is not None:
                    capture_path = captures_directory / ("frame_%03d.png" % len(image_points))
                    if not cv2.imwrite(str(capture_path), frame):
                        raise RuntimeError("Could not save calibration frame: %s" % capture_path)
                    image_points.append(corners.copy())
                    capture_paths.append(capture_path)
    finally:
        cv2.destroyWindow(window)

    if len(image_points) < geometry.checkerboard.minimum_captures:
        raise RuntimeError(
            "Only %d checkerboard views were saved; at least %d are required."
            % (len(image_points), geometry.checkerboard.minimum_captures)
        )
    if image_size is None:
        raise RuntimeError("No camera frame was captured.")
    result = calibrate_camera_intrinsics(
        image_points,
        image_size,
        geometry.checkerboard.spec,
    )
    written = write_camera_mat(
        camera_mat,
        result,
        geometry.checkerboard.spec,
        geometry.checkerboard.max_rms_error_px,
    )
    write_calibration_summary(summary_path, role, result, capture_paths)
    print("Camera: %s" % alias)
    print("RMS reprojection error: %.4f px" % result.rms_error_px)
    print("Saved: %s" % written)
    return written


def run(args: argparse.Namespace) -> None:
    geometry = load_geometry_calibration_config(Path(args.config))
    if args.command == "generate-board":
        output = args.output or (
            geometry.project_root
            / "outputs"
            / "calibration"
            / (
                "checkerboard_%dx%d_%.1fmm.svg"
                % (
                    geometry.checkerboard.spec.inner_corners_x,
                    geometry.checkerboard.spec.inner_corners_y,
                    geometry.checkerboard.spec.square_size_mm,
                )
            )
        )
        generated = generate_checkerboard_svg(output, geometry.checkerboard.spec)
        print("Checkerboard generated: %s" % generated)
        print("Print at 100% / Actual Size, then measure several squares with a ruler.")
        return
    if args.command == "write-screen":
        if geometry.screen_verification_source not in {
            "apple_official_224ppi",
            "manual_measurement",
        }:
            raise ValueError(
                "Measure the active screen area, update screen width_mm/height_mm, "
                "and set a verified screen verification_source first."
            )
        written = write_screen_size_mat(
            geometry.output_directory / "screenSize.mat",
            geometry.screen,
            geometry.screen_verification_source,
        )
        print("Screen calibration saved: %s" % written)
        return
    if args.command == "capture-intrinsics":
        _capture_intrinsics(geometry, Path(args.capture_config), args.camera)
        return
    if args.command == "inspect":
        print(inspect_calibration_assets(geometry.output_directory))
        return
    raise ValueError("Unknown geometry calibration command: %s" % args.command)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as error:
        print("[geometry calibration error] %s" % error, file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
