from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from ggulnote_ml.video_preprocessing.pipeline import run_video_preprocessing


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export latency-synchronized frames, MediaPipe video features, "
            "and image-pipeline manifests."
        )
    )
    parser.add_argument("--participant", required=True, help="Participant id such as p00")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/participants"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/interim/dual_view"),
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.20,
        help="Mark an eye closed when its EAR is below this value",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_video_preprocessing(
        participant_id=args.participant,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        ear_threshold=args.ear_threshold,
    )
    print("Training manifest: %s" % result.training_manifest)
    print("Evaluation manifest: %s" % result.evaluation_manifest)
    print("Webcam features: %s" % result.webcam_features)
    print("Phonecam features: %s" % result.phonecam_features)
    print("Video training features: %s" % result.video_training_features)
    print("Video evaluation features: %s" % result.video_evaluation_features)
    print("Summary: %s" % result.summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
