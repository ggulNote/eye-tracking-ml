from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .calibration_assets import inspect_calibration_assets
from .camera import CameraSource
from .config import CameraConfig, load_capture_config
from .geometry import (
    calibrate_camera_intrinsics,
    calibrate_camera_intrinsics_robust,
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
    intrinsics.add_argument(
        "--auto-capture",
        action="store_true",
        help=(
            "Save automatically after detection, then only after the checkerboard "
            "has moved enough to provide a different view"
        ),
    )

    commands.add_parser("inspect", help="Inspect generated MAT files")
    return parser.parse_args(argv)


def _camera_for_alias(capture_config_path: Path, alias: str) -> CameraConfig:
    capture = load_capture_config(capture_config_path)
    role, _ = CAMERA_ALIASES[alias]
    by_role = {camera.role: camera for camera in capture.cameras}
    return by_role[role]


def checkerboard_view_changed(
    previous: Optional[np.ndarray],
    current: np.ndarray,
    minimum_mean_motion_px: float = 25.0,
) -> bool:
    """Return true only when the detected board moved enough for a new view."""

    if minimum_mean_motion_px <= 0:
        raise ValueError("minimum_mean_motion_px must be positive.")
    current_points = np.asarray(current, dtype=np.float32).reshape(-1, 2)
    if previous is None:
        return True
    previous_points = np.asarray(previous, dtype=np.float32).reshape(-1, 2)
    if previous_points.shape != current_points.shape:
        return True
    mean_motion = float(
        np.linalg.norm(current_points - previous_points, axis=1).mean()
    )
    return mean_motion >= minimum_mean_motion_px


def checkerboard_view_stable(
    previous: Optional[np.ndarray],
    current: np.ndarray,
    maximum_mean_motion_px: float = 2.0,
) -> bool:
    if maximum_mean_motion_px <= 0:
        raise ValueError("maximum_mean_motion_px must be positive.")
    if previous is None:
        return False
    previous_points = np.asarray(previous, dtype=np.float32).reshape(-1, 2)
    current_points = np.asarray(current, dtype=np.float32).reshape(-1, 2)
    if previous_points.shape != current_points.shape:
        return False
    mean_motion = float(
        np.linalg.norm(current_points - previous_points, axis=1).mean()
    )
    return mean_motion <= maximum_mean_motion_px


def _capture_intrinsics(
    geometry: GeometryCalibrationConfig,
    capture_config_path: Path,
    alias: str,
    auto_capture: bool = False,
) -> Path:
    board_spec = geometry.checkerboard.spec_for_camera(alias)
    if (
        geometry.checkerboard.verification_source_for_camera(alias)
        != "manual_measurement"
    ):
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
    mouse_capture_requested = [False]

    def request_capture(event: int, _x: int, _y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_capture_requested[0] = True

    cv2.setMouseCallback(window, request_capture)
    previous_saved_corners = None
    previous_detected_corners = None
    stable_detection_frames = 0
    last_capture_time = 0.0
    started_ns = time.monotonic_ns()
    try:
        with CameraSource(camera) as source:
            while len(image_points) < geometry.checkerboard.required_captures:
                packet = source.read(started_ns)
                frame = packet.frame
                image_size = (frame.shape[1], frame.shape[0])
                found, corners = find_checkerboard_corners(frame, board_spec)
                if found and corners is not None:
                    stable_detection_frames = (
                        stable_detection_frames + 1
                        if checkerboard_view_stable(previous_detected_corners, corners)
                        else 0
                    )
                    previous_detected_corners = corners.copy()
                else:
                    stable_detection_frames = 0
                    previous_detected_corners = None
                preview = frame.copy()
                if found and corners is not None:
                    cv2.drawChessboardCorners(
                        preview,
                        board_spec.pattern_size,
                        corners,
                        True,
                    )
                input_help = (
                    "AUTO: move board" if auto_capture else "CLICK/SPACE/ENTER: save"
                )
                status = "%s %d/%d | %s | Q/ESC: finish" % (
                    "DETECTED" if found else "NOT FOUND",
                    len(image_points),
                    geometry.checkerboard.required_captures,
                    input_help,
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
                manual_capture = key in {ord(" "), 10, 13} or mouse_capture_requested[0]
                mouse_capture_requested[0] = False
                now = time.monotonic()
                automatic_capture = bool(
                    auto_capture
                    and found
                    and corners is not None
                    and now - last_capture_time >= 0.75
                    and stable_detection_frames >= 5
                    and checkerboard_view_changed(previous_saved_corners, corners)
                )
                if (manual_capture or automatic_capture) and found and corners is not None:
                    capture_path = captures_directory / ("frame_%03d.png" % len(image_points))
                    if not cv2.imwrite(str(capture_path), frame):
                        raise RuntimeError("Could not save calibration frame: %s" % capture_path)
                    image_points.append(corners.copy())
                    capture_paths.append(capture_path)
                    previous_saved_corners = corners.copy()
                    last_capture_time = now
    finally:
        cv2.destroyWindow(window)

    if len(image_points) < geometry.checkerboard.minimum_captures:
        raise RuntimeError(
            "Only %d checkerboard views were saved; at least %d are required."
            % (len(image_points), geometry.checkerboard.minimum_captures)
        )
    if image_size is None:
        raise RuntimeError("No camera frame was captured.")
    robust = calibrate_camera_intrinsics_robust(
        image_points,
        image_size,
        board_spec,
        geometry.checkerboard.minimum_captures,
        geometry.checkerboard.max_rms_error_px,
    )
    result = robust.result
    retained_capture_paths = [capture_paths[index] for index in robust.retained_indices]
    rejected_capture_paths = [capture_paths[index] for index in robust.rejected_indices]
    written = write_camera_mat(
        camera_mat,
        result,
        board_spec,
        geometry.checkerboard.max_rms_error_px,
    )
    write_calibration_summary(
        summary_path,
        role,
        result,
        retained_capture_paths,
        rejected_capture_paths,
    )
    print("Camera: %s" % alias)
    print("RMS reprojection error: %.4f px" % result.rms_error_px)
    print("Views used/rejected: %d/%d" % (len(retained_capture_paths), len(rejected_capture_paths)))
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
        _capture_intrinsics(
            geometry,
            Path(args.capture_config),
            args.camera,
            auto_capture=args.auto_capture,
        )
        return
    if args.command == "inspect":
        result = inspect_calibration_assets(geometry.output_directory)
        print("intrinsics_2d_ready: %s" % result["camera_intrinsics_valid"])
        print("calibrated_3d_ready: %s" % result["required_assets_valid"])
        print(result)
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
