"""Command-line entry points for validation, preparation, training, and evaluation."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gaze_pipeline.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    load_and_validate_config,
)
from gaze_pipeline.tracking import MLflowTrackingError, log_data_preparation


class CommandError(RuntimeError):
    """An expected command failure that should not show a traceback."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gaze_pipeline",
        description=("Dual-view gaze pipeline의 데이터 준비, 학습, 평가를 config로 실행합니다."),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-config",
        help="YAML을 resolve한 뒤 데이터/모델/fusion 계약을 검사합니다.",
    )
    _add_config_arguments(validate_parser)
    validate_parser.add_argument(
        "--skip-path-checks",
        action="store_true",
        help="dataset/model 파일의 실제 존재 여부는 검사하지 않습니다.",
    )
    validate_parser.add_argument(
        "--require-model-entrypoints",
        action="store_true",
        help=(
            "enabled model의 package.module:callable entrypoint까지 요구합니다. "
            "기본 검사는 데이터 준비 전 config도 허용합니다."
        ),
    )
    validate_parser.set_defaults(handler=_run_validate_config)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="검증된 config로 canonical manifest와 subject-wise split을 만듭니다.",
    )
    _add_config_arguments(prepare_parser)
    prepare_parser.set_defaults(handler=_run_prepare)

    train_parser = subparsers.add_parser(
        "train",
        help="manifest를 준비하고 외부/내장 PyTorch 모델을 학습합니다.",
    )
    _add_config_arguments(train_parser)
    train_parser.set_defaults(handler=_run_train)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="저장된 pipeline checkpoint로 validation/test를 평가합니다.",
    )
    _add_config_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "평가할 .pt checkpoint. 생략하면 checkpoint.resume_from 또는 "
            "checkpoint.save_best 경로를 사용합니다."
        ),
    )
    evaluate_parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="평가할 split (기본값: test)",
    )
    evaluate_parser.set_defaults(handler=_run_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ConfigError as exc:
        print(f"설정 오류:\n{exc}", file=sys.stderr)
        return 2
    except CommandError as exc:
        print(f"실행 오류:\n{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("사용자가 실행을 중단했습니다.", file=sys.stderr)
        return 130


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"기준 YAML 경로 (기본값: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        action="append",
        default=[],
        help="기준 config 위에 merge할 profile YAML. 여러 번 지정할 수 있습니다.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="key=value",
        help="마지막에 적용할 dotted override. 예: data.dataloader.batch_size=64",
    )


def _run_validate_config(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser()
    config = load_and_validate_config(
        config_path,
        profiles=args.profile,
        overrides=args.overrides,
        check_paths=not args.skip_path_checks,
        require_model_entrypoints=args.require_model_entrypoints,
        base_dir=_project_base_dir(config_path),
    )

    experiment = config.get("experiment", {})
    experiment_name = (
        experiment.get("name", "<이름 없음>") if isinstance(experiment, Mapping) else "<이름 없음>"
    )
    run_name = (
        experiment.get("run_name", "<이름 없음>")
        if isinstance(experiment, Mapping)
        else "<이름 없음>"
    )
    print(f"Config 검증 성공: {config_path}")
    print(f"  experiment: {experiment_name}")
    print(f"  resolved run: {run_name}")
    if not args.require_model_entrypoints:
        print("  model entrypoint: 선택 검사(prepare 전 config 허용)")
    return 0


def _run_prepare(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser()
    config = load_and_validate_config(
        config_path,
        profiles=args.profile,
        overrides=args.overrides,
        check_paths=True,
        # Preparing manifests never imports or executes user model code.
        require_model_entrypoints=False,
        base_dir=_project_base_dir(config_path),
    )
    _preload_enabled_mlflow(config)

    try:
        from gaze_pipeline.data.pipeline import prepare_data
    except ImportError as exc:
        raise CommandError(
            "데이터 준비 모듈 'gaze_pipeline.data.pipeline.prepare_data'를 "
            "불러오지 못했습니다. package 설치 상태와 requirements를 확인하세요. "
            f"원인: {exc}"
        ) from exc

    try:
        result = prepare_data(config, config_path=config_path.resolve())
    except (OSError, ValueError) as exc:
        raise CommandError(f"데이터 manifest/split 준비에 실패했습니다: {exc}") from exc

    print("데이터 준비가 완료되었습니다.")
    _print_prepare_result(result)

    try:
        tracking_result = log_data_preparation(config, result)
    except MLflowTrackingError as exc:
        raise CommandError(
            "manifest와 split 생성은 완료되었지만 MLflow 기록만 실패했습니다. "
            "위에 출력된 로컬 결과는 삭제되거나 변경되지 않았습니다. "
            f"원인: {exc}"
        ) from exc
    if tracking_result is not None:
        print(f"  MLflow run_id: {tracking_result.run_id}")
    return 0


def _run_train(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser()
    config = _load_execution_config(args, config_path)
    _preload_enabled_mlflow(config)
    try:
        from gaze_pipeline.training import TrainingError, train_pipeline

        result = train_pipeline(config, config_path=config_path.resolve())
    except ImportError as exc:
        raise CommandError(
            "학습 모듈을 불러오지 못했습니다. make setup으로 PyTorch 의존성을 설치하세요. "
            f"원인: {exc}"
        ) from exc
    except TrainingError as exc:
        raise CommandError(f"학습에 실패했습니다: {exc}") from exc
    print("학습이 완료되었습니다.")
    _print_pipeline_result(result)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser()
    config = _load_execution_config(args, config_path)
    _preload_enabled_mlflow(config)
    try:
        from gaze_pipeline.training import TrainingError, evaluate_pipeline

        result = evaluate_pipeline(
            config,
            checkpoint=args.checkpoint,
            split=args.split,
            config_path=config_path.resolve(),
        )
    except ImportError as exc:
        raise CommandError(
            "평가 모듈을 불러오지 못했습니다. make setup으로 PyTorch 의존성을 설치하세요. "
            f"원인: {exc}"
        ) from exc
    except TrainingError as exc:
        raise CommandError(f"평가에 실패했습니다: {exc}") from exc
    print(f"{args.split} 평가가 완료되었습니다.")
    _print_pipeline_result(result)
    return 0


def _load_execution_config(args: argparse.Namespace, config_path: Path) -> dict[str, Any]:
    """Load paths for data/model execution while allowing built-in fallback models."""

    return load_and_validate_config(
        config_path,
        profiles=args.profile,
        overrides=args.overrides,
        check_paths=True,
        require_model_entrypoints=False,
        base_dir=_project_base_dir(config_path),
    )


def _preload_enabled_mlflow(config: Mapping[str, Any]) -> None:
    """Load MLflow before PyTorch-backed modules on Windows.

    MLflow imports PyArrow while the data and training packages import
    PyTorch.  Some Windows CUDA/PyArrow wheel combinations corrupt the native
    heap when PyArrow is loaded after PyTorch, while the reverse order is
    stable.  Keeping this at the CLI boundary also preserves the no-MLflow
    dependency path when tracking is disabled.
    """

    mlflow_config = config.get("mlflow")
    if not isinstance(mlflow_config, Mapping) or mlflow_config.get("enabled") is not True:
        return
    try:
        importlib.import_module("mlflow")
    except ImportError as exc:
        raise CommandError(
            "MLflow 기록이 켜져 있지만 mlflow package를 불러오지 못했습니다. "
            "현재 가상환경에 requirements.txt를 설치하세요."
        ) from exc


def _print_pipeline_result(result: Any) -> None:
    for name in (
        "checkpoint_path",
        "last_checkpoint_path",
        "model_path",
        "metrics_path",
        "predictions_path",
    ):
        value = getattr(result, name, None)
        if value is not None:
            print(f"  {name}: {value}")
    best_epoch = getattr(result, "best_epoch", None)
    if best_epoch is not None:
        print(f"  best_epoch: {best_epoch}")
    mlflow_run_id = getattr(result, "mlflow_run_id", None)
    if mlflow_run_id is not None:
        print(f"  MLflow run_id: {mlflow_run_id}")
    metrics = getattr(result, "metrics", None)
    if isinstance(metrics, Mapping):
        for name, value in sorted(metrics.items()):
            print(f"  metric {name}: {float(value):.6f}")


def _print_prepare_result(result: Any) -> None:
    if result is None:
        return

    manifest_paths = getattr(result, "manifest_paths", None)
    if isinstance(manifest_paths, Mapping):
        for split_name, path in sorted(manifest_paths.items(), key=lambda item: str(item[0])):
            print(f"  {split_name} manifest: {path}")

    split_counts = getattr(result, "split_counts", None)
    if isinstance(split_counts, Mapping):
        rendered = ", ".join(
            f"{name}={count}"
            for name, count in sorted(split_counts.items(), key=lambda item: str(item[0]))
        )
        print(f"  split counts: {rendered}")

    dataset_manifest_hash = getattr(result, "dataset_manifest_hash", None)
    if dataset_manifest_hash is None:
        dataset_manifest_hash = getattr(result, "dataset_hash", None)
    if dataset_manifest_hash:
        print(f"  dataset manifest hash: {dataset_manifest_hash}")

    if manifest_paths is None and split_counts is None and dataset_manifest_hash is None:
        print(f"  result: {result}")


def _project_base_dir(config_path: Path) -> Path:
    """Find the repository root for paths written as ``./data``/``./outputs``."""

    absolute = config_path.resolve(strict=False)
    for parent in absolute.parents:
        if parent.name == "configs":
            return parent.parent
    return Path.cwd()


__all__ = ["build_parser", "main"]
