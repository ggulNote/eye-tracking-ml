from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .calibration_assets import copy_calibration_assets, inspect_calibration_assets
from .camera import probe_camera_indices
from .collection import run_real_protocols, run_simulation_protocols
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
        help="Protocol id to run; repeat it or use all (default: full 105-second dot test)",
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
    if (
        config.dataset.require_calibration_assets
        and not args.simulate
        and not calibration_override
        and not calibration["required_assets_valid"]
    ):
        raise ValueError(
            "Required calibration assets are missing or invalid under %s. "
            "Set dataset.require_calibration_assets=false only for collection tests."
            % calibration_source
        )

    paths = create_participant_paths(config.dataset.root_directory, participant_id)
    if not args.simulate:
        copy_calibration_assets(calibration_source, paths.calibration_directory)
        calibration = inspect_calibration_assets(paths.calibration_directory)

    plans = select_protocols(build_protocol_plans(config.protocols), args.protocol or ["all"])
    participant_metadata = load_optional_metadata(args.participant_metadata)
    config_snapshot = paths.events_directory / "capture_config.yaml"
    shutil.copyfile(config_path, config_snapshot)

    started_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": 2,
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
        "participant_metadata": participant_metadata,
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
        run_collection(args)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as error:
        print("[collection error] %s" % error, file=sys.stderr)
        raise SystemExit(1) from error
