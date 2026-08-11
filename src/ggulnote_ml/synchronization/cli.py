from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from ggulnote_ml.capture.config import CaptureConfig, load_capture_config

from .calibration import (
    copy_config_snapshots,
    create_latency_paths,
    run_latency_simulation,
    run_real_latency_measurement,
)
from .config import load_latency_config


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure webcam and phonecam display-to-video latency with screen flashes."
    )
    parser.add_argument("--capture-config", default="configs/capture.yaml")
    parser.add_argument("--latency-config", default="configs/latency.yaml")
    parser.add_argument("--participant", required=True, help="Participant id such as p00 or 0")
    parser.add_argument("--dataset-root", type=Path, help="Override dataset root")
    parser.add_argument("--measurement-id", help="Optional deterministic run id for tests")
    parser.add_argument("--simulate", action="store_true", help="Run without cameras or GUI")
    return parser.parse_args(argv)


def _override_dataset(config: CaptureConfig, dataset_root: Optional[Path]) -> CaptureConfig:
    if dataset_root is None:
        return config
    return replace(
        config,
        dataset=replace(config.dataset, root_directory=dataset_root.expanduser().resolve()),
    )


def run(args: argparse.Namespace) -> Path:
    capture_config_path = Path(args.capture_config).expanduser().resolve()
    latency_config_path = Path(args.latency_config).expanduser().resolve()
    capture_config = _override_dataset(
        load_capture_config(capture_config_path), args.dataset_root
    )
    latency_config = load_latency_config(latency_config_path)
    paths = create_latency_paths(
        capture_config.dataset.root_directory,
        args.participant,
        measurement_id=args.measurement_id,
    )
    copy_config_snapshots(paths, capture_config_path, latency_config_path)
    result = (
        run_latency_simulation(latency_config, paths)
        if args.simulate
        else run_real_latency_measurement(capture_config, latency_config, paths)
    )
    print("Latency calibration saved: %s" % paths.final_json)
    cameras = result["cameras"]
    for camera in ("webcam", "phonecam"):
        summary = cameras[camera]
        print(
            "[%s] median=%.3f ms, MAD=%.3f ms, p95=%.3f ms, valid=%d/%d"
            % (
                camera,
                summary["median_ms"],
                summary["mad_ms"],
                summary["p95_ms"],
                summary["valid_events"],
                summary["total_events"],
            )
        )
    return paths.final_json


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as error:
        print("[latency error] %s" % error, file=sys.stderr)
        raise SystemExit(1) from error
