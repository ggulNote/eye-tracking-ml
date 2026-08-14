"""Read-only MLflow installation and local SQLite run checker."""

from __future__ import annotations

import argparse
import importlib
import os
import sqlite3
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _redacted_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    netloc = parsed.netloc
    if "@" in netloc:
        netloc = f"<redacted>@{netloc.rsplit('@', 1)[1]}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _sqlite_path(uri: str, project_root: Path) -> Path | None:
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return None
    raw_path = unquote(uri[len(prefix) :].split("?", 1)[0])
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "<not-installed>"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--imports-only", action="store_true")
    mode.add_argument("--require-db", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = _project_root()
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{project_root / 'mlflow.db'}",
    )

    print(f"[OK] project root: {project_root}")
    print(f"[OK] python: {Path(sys.executable).resolve()}")
    print(f"[OK] mlflow: {_version('mlflow')}")
    print(f"[OK] psutil: {_version('psutil')}")
    print(f"[OK] tracking URI: {_redacted_uri(tracking_uri)}")

    missing = [name for name in ("mlflow", "psutil") if _version(name) == "<not-installed>"]
    if missing:
        print(f"[FAIL] missing packages: {', '.join(missing)}", file=sys.stderr)
        return 2
    for module_name in ("mlflow", "psutil"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            print(
                f"[FAIL] import {module_name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2
    if args.imports_only:
        print("[OK] MLflow runtime dependencies are installed.")
        return 0

    database_path = _sqlite_path(tracking_uri, project_root)
    if database_path is None:
        print(
            "[FAIL] read-only DB 검사는 local sqlite tracking URI만 지원합니다.",
            file=sys.stderr,
        )
        return 2
    print(f"[OK] MLflow DB: {database_path}")
    if not database_path.is_file():
        print(
            "[FAIL] MLflow DB가 없습니다. 먼저 make prepare, make train 또는 "
            "make demo-dual-train을 실행하세요.",
            file=sys.stderr,
        )
        return 2

    read_only_uri = f"file:{quote(str(database_path))}?mode=ro"
    try:
        with sqlite3.connect(read_only_uri, uri=True) as connection:
            table_rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
            tables = {str(row[0]) for row in table_rows}
            if not {"experiments", "runs"}.issubset(tables):
                print("[FAIL] MLflow schema가 없는 SQLite 파일입니다.", file=sys.stderr)
                return 2
            experiments = int(connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])
            status_rows = connection.execute(
                "SELECT status, COUNT(*) FROM runs GROUP BY status ORDER BY status"
            ).fetchall()
    except sqlite3.Error as exc:
        print(f"[FAIL] MLflow DB를 읽지 못했습니다: {exc}", file=sys.stderr)
        return 2

    statuses = {str(status): int(count) for status, count in status_rows}
    total_runs = sum(statuses.values())
    print(f"[OK] experiments: {experiments}")
    print(f"[OK] runs: {total_runs} {statuses}")
    if statuses.get("FINISHED", 0) < 1:
        print("[FAIL] FINISHED MLflow run이 없습니다.", file=sys.stderr)
        return 1
    print("[OK] MLflow setup and at least one completed run were verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
