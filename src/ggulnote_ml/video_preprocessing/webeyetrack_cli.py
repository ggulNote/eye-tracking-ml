from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from ggulnote_ml.video_preprocessing.webeyetrack_inputs import (
    discover_recordings,
    run_webeyetrack_input_preprocessing,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate WebEyeTrack eye ROI, head_vector, face_origin_3d, and "
            "screen-centered targets from completed recordings."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Process every completed recording")
    selection.add_argument("--participant", help="One participant name")
    parser.add_argument(
        "--head-pose",
        help="Required with --participant: neutral, head_up, or head_down",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/participants"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/webeyetrack_preprocessing.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing webeyetrack output directory",
    )
    args = parser.parse_args(argv)
    if bool(args.participant) != bool(args.head_pose):
        parser.error("--participant and --head-pose must be provided together.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    recordings = (
        discover_recordings(args.dataset_root)
        if args.all
        else ((args.participant, args.head_pose),)
    )
    if not recordings:
        raise FileNotFoundError("No completed recordings were found.")
    failures = []
    for participant, head_pose in recordings:
        try:
            result = run_webeyetrack_input_preprocessing(
                participant_id=participant,
                head_pose=head_pose,
                dataset_root=args.dataset_root,
                config_path=args.config,
                output_root=args.output_root,
                force=args.force,
            )
        except Exception as exc:
            if args.all and not args.force and isinstance(exc, FileExistsError):
                print("[SKIP] %s/%s: already processed" % (participant, head_pose))
                continue
            failures.append((participant, head_pose, str(exc)))
            print("[FAILED] %s/%s: %s" % (participant, head_pose, exc))
            continue
        print(
            "[OK] %s/%s: valid=%d/%d, training=%d, evaluation=%d -> %s"
            % (
                result.participant,
                result.head_pose,
                result.valid_rows,
                result.total_rows,
                result.training_rows,
                result.evaluation_rows,
                result.output_directory,
            )
        )
    if failures:
        print("WebEyeTrack preprocessing failed for %d recording(s)." % len(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
