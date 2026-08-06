from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from ggulnote_ml.config import load_config
from ggulnote_ml.exceptions import PipelineError
from ggulnote_ml.pipelines import run_evaluation, run_prediction, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ggulnote-ml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config", help="Validate config schema.")
    _add_config_argument(validate_parser)

    train_parser = subparsers.add_parser("train", help="Train and log an experiment.")
    _add_config_argument(train_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a saved model on test data.")
    _add_config_argument(evaluate_parser)
    evaluate_parser.add_argument("--model-path", type=Path, required=True)

    predict_parser = subparsers.add_parser("predict", help="Predict one test sample.")
    _add_config_argument(predict_parser)
    predict_parser.add_argument("--model-path", type=Path, required=True)
    predict_parser.add_argument("--sample-index", type=int, default=0)
    return parser


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        loaded = load_config(args.config)
        if args.command == "validate-config":
            payload = {
                "status": "ok",
                "config": str(loaded.source_path),
                "project_root": str(loaded.project_root),
            }
        elif args.command == "train":
            result = run_training(loaded.config, loaded.project_root)
            payload = {
                "run_id": result.run_id,
                "output_dir": str(result.output_dir),
                "model_path": str(result.model_path),
                "metrics": result.metrics,
            }
        elif args.command == "evaluate":
            payload = run_evaluation(
                loaded.config,
                loaded.project_root,
                args.model_path,
            )
        elif args.command == "predict":
            payload = run_prediction(
                loaded.config,
                loaded.project_root,
                args.model_path,
                args.sample_index,
            )
        else:
            raise AssertionError("Unhandled command: %s" % args.command)
    except (PipelineError, OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

