from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .calibration_assets import (
    FULL_CALIBRATION_ASSET_PATHS,
    INTRINSICS_ASSET_PATHS,
    copy_calibration_assets,
    inspect_calibration_assets,
    load_camera_intrinsics,
)
from .camera import probe_camera_indices
from .collection import (
    run_camera_preflight,
    run_real_protocols,
    run_simulation_protocols,
)
from .config import CaptureConfig, load_capture_config
from .dataset import (
    create_participant_paths,
    load_optional_metadata,
    normalize_participant_id,
    suggest_next_participant_id,
    write_participant_json,
)
from .protocols import build_protocol_plans, select_protocols


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect synchronized webcam and left-iPhone gaze protocol data."
    )
    parser.add_argument("--config", default="configs/capture.yaml")
    parser.add_argument("--participant", help="Participant id such as p00 or 0")
    parser.add_argument(
        "--protocol",
        action="append",
        default=[],
        help="Protocol id to run; repeat it or use all (default: full participant-confirmed dot test)",
    )
    parser.add_argument("--simulate", action="store_true", help="Create videos and CSVs without cameras")
    parser.add_argument("--participant-metadata", type=Path, help="Optional participant metadata JSON")
    parser.add_argument("--dataset-root", type=Path, help="Override dataset root (useful for simulation/CI)")
    parser.add_argument(
        "--allow-missing-calibration",
        action="store_true",
        help="Allow an uncalibrated hardware smoke test and record it in participant.json",
    )
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument(
        "--check-cameras",
        action="store_true",
        help="Preview both cameras and require MediaPipe face+iris readiness without saving data",
    )
    parser.add_argument("--max-devices", type=int, default=10)
    parser.add_argument("--suggest-participant", action="store_true")
    return parser.parse_args(argv)


def _participant_from_args(value: Optional[str], dataset_root: Path) -> str:
    suggestion = suggest_next_participant_id(dataset_root)
    if value is not None:
        return normalize_participant_id(value)
    entered = input("Participant id [%s]: " % suggestion).strip()
    return normalize_participant_id(entered or suggestion)


def _apply_dataset_override(
    config: CaptureConfig, dataset_root: Optional[Path]
) -> CaptureConfig:
    if dataset_root is None:
        return config
    return replace(
        config,
        dataset=replace(config.dataset, root_directory=dataset_root.expanduser().resolve()),
    )


def run_collection(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).expanduser().resolve()
    config = _apply_dataset_override(load_capture_config(config_path), args.dataset_root)
    participant_id = _participant_from_args(args.participant, config.dataset.root_directory)
    calibration_source = config.dataset.calibration_source_directory
    calibration = inspect_calibration_assets(calibration_source)
    calibration_override = bool(args.allow_missing_calibration and not args.simulate)
    if config.dataset.geometry_mode == "intrinsics_2d":
        calibration_ready = calibration["camera_intrinsics_valid"]
    elif config.dataset.geometry_mode == "calibrated_3d":
        calibration_ready = calibration["required_assets_valid"]
    else:
        calibration_ready = True
    if (
        config.dataset.require_calibration_assets
        and not args.simulate
        and not calibration_override
        and not calibration_ready
    ):
        raise ValueError(
            "Required calibration assets are missing or invalid under %s. "
            "Use fixed_rig_2d only for deliberate collection without lens correction."
            % calibration_source
        )
    uses_calibration = config.dataset.geometry_mode != "fixed_rig_2d"
    if not args.simulate and uses_calibration and calibration_ready:
        camera_directories = {
            "webcam_front": "webcam",
            "iphone_left": "phonecam",
        }
        for camera in config.cameras:
            directory = camera_directories[camera.role]
            intrinsics = load_camera_intrinsics(
                calibration_source / directory / "Camera.mat"
            )
            if (intrinsics.image_width, intrinsics.image_height) != (
                camera.width,
                camera.height,
            ):
                raise ValueError(
                    "%s Camera.mat size %dx%d does not match capture size %dx%d."
                    % (
                        directory,
                        intrinsics.image_width,
                        intrinsics.image_height,
                        camera.width,
                        camera.height,
                    )
                )

    paths = create_participant_paths(config.dataset.root_directory, participant_id)
    calibration_copied = bool(
        not args.simulate and uses_calibration and calibration_ready
    )
    if calibration_copied:
        calibration_paths = (
            INTRINSICS_ASSET_PATHS
            if config.dataset.geometry_mode == "intrinsics_2d"
            else FULL_CALIBRATION_ASSET_PATHS
        )
        copy_calibration_assets(
            calibration_source,
            paths.calibration_directory,
            calibration_paths,
        )
        calibration = inspect_calibration_assets(paths.calibration_directory)

    plans = select_protocols(build_protocol_plans(config.protocols), args.protocol or ["all"])
    participant_metadata = load_optional_metadata(args.participant_metadata)
    config_snapshot = paths.events_directory / "capture_config.yaml"
    shutil.copyfile(config_path, config_snapshot)

    started_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": 4,
        "csv_schema_version": 2,
        "participant_id": participant_id,
        "status": "running",
        "mode": "simulation" if args.simulate else "camera",
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "camera_roles": {camera.role: camera.name for camera in config.cameras},
        "camera_config": [
            {
                "name": camera.name,
                "role": camera.role,
                "device_index": camera.device_index,
                "backend": camera.backend,
                "width": camera.width,
                "height": camera.height,
                "fps": camera.fps,
                "max_identical_frames": camera.max_identical_frames,
                "position": "front" if camera.role == "webcam_front" else "participant_left_30_45_deg",
            }
            for camera in config.cameras
        ],
        "screen": {
            "canvas_width_pixel": config.display.canvas_width,
            "canvas_height_pixel": config.display.canvas_height,
        },
        "protocols": [plan.protocol_id for plan in plans],
        "click_confirmation": {
            "input": "left_mouse_or_space",
            "train_required": config.protocols.train_static.confirmation_required,
            "vertical_required": config.protocols.vertical_click.confirmation_required,
            "evaluation_required": config.protocols.evaluation_static.confirmation_required,
        },
        "frame_capture": {
            "enabled": config.frame_capture.enabled,
            "image_format": config.frame_capture.image_format,
            "webcam_directory": "images/webcam",
            "phonecam_directory": "images/phonecam",
            "manifest": "labels/image_samples.csv",
            "scope": "one_best_pair_per_confirmed_target",
            "samples_per_target": config.frame_capture.samples_per_target,
            "quality_heuristic": (
                "mediapipe_face_iris_then_haar_sharpness_exposure"
            ),
        },
        "participant_metadata": participant_metadata,
        "geometry": {
            "mode": config.dataset.geometry_mode,
            "camera_intrinsics_available": bool(
                calibration["camera_intrinsics_valid"]
            ),
            "calibration_copied_to_participant": calibration_copied,
            "frame_undistortion_stage": "video_preprocessing",
            "calibration_source": str(calibration_source),
            "camera_position_must_remain_fixed": True,
        },
        "calibration_assets": calibration,
        "missing_calibration_override": calibration_override,
        "config_snapshot": str(config_snapshot.relative_to(paths.participant_directory)),
        "results": [],
    }
    write_participant_json(paths.participant_json, metadata)

    try:
        results = (
            run_simulation_protocols(config, paths, plans)
            if args.simulate
            else run_real_protocols(config, paths, plans)
        )
        metadata["results"] = [
            {
                "protocol_id": result.protocol_id,
                "status": result.status,
                "frame_pairs": result.frame_pairs,
                "labels": str(result.label_path.relative_to(paths.participant_directory)),
            }
            for result in results
        ]
        metadata["status"] = (
            "aborted" if any(result.status == "aborted" for result in results) else "completed"
        )
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = "%s: %s" % (type(error).__name__, error)
        raise
    finally:
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_participant_json(paths.participant_json, metadata)

    print("Participant data saved: %s" % paths.participant_directory)
    for result in metadata["results"]:
        print("[%s] %s, %d frame pairs" % (result["protocol_id"], result["status"], result["frame_pairs"]))
    return paths.participant_directory


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        config = _apply_dataset_override(load_capture_config(Path(args.config)), args.dataset_root)
        if args.suggest_participant:
            print(suggest_next_participant_id(config.dataset.root_directory))
            return
        if args.list_cameras:
            if args.max_devices <= 0:
                raise ValueError("--max-devices must be positive.")
            indices = probe_camera_indices(args.max_devices, config.cameras[0].backend)
            print("Available camera indices: %s" % (indices if indices else "none"))
            return
        if args.check_cameras:
            run_camera_preflight(config)
            print("Camera preflight passed for webcam and phonecam.")
            return
        run_collection(args)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as error:
        print("[collection error] %s" % error, file=sys.stderr)
        raise SystemExit(1) from error
