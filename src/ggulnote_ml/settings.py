from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from ggulnote_ml.exceptions import ConfigurationError


DATA_ROOT_ENV = "GGULNOTE_DATA_ROOT"


def load_project_env(project_root: Path) -> Dict[str, str]:
    """Load simple KEY=VALUE entries from .env without overriding shell env."""

    env_path = project_root / ".env"
    if not env_path.is_file():
        return {}
    loaded: Dict[str, str] = {}
    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ConfigurationError(
                "Invalid .env entry at line %d; expected KEY=VALUE." % line_number
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ConfigurationError("Empty .env key at line %d." % line_number)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def resolve_data_root(project_root: Path, required: bool = True) -> Optional[Path]:
    load_project_env(project_root)
    configured = os.environ.get(DATA_ROOT_ENV, "").strip()
    if not configured:
        if required:
            raise ConfigurationError(
                "%s is not set. Copy .env.example to .env and configure a local data path."
                % DATA_ROOT_ENV
            )
        return None
    root = Path(configured).expanduser()
    if not root.is_absolute():
        raise ConfigurationError("%s must be an absolute path." % DATA_ROOT_ENV)
    return root.resolve()


def resolve_data_path(value: str, data_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()
