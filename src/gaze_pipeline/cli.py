"""Command-line entry points available before the training runner exists."""

from __future__ import annotations

import argparse
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
        description=(
            "Dual-view gaze pipeline의 config를 검사하고 데이터 manifest/split을 준비합니다."
        ),
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
