from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_sha(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def write_predictions_csv(
    path: Path,
    participant_ids: Iterable[str],
    session_ids: Iterable[str],
    targets: Any,
    predictions: Any,
) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["participant_id", "session_id", "target_x", "target_y", "predicted_x", "predicted_y"]
        )
        for participant_id, session_id, target, prediction in zip(
            participant_ids, session_ids, targets, predictions
        ):
            writer.writerow(
                [
                    participant_id,
                    session_id,
                    float(target[0]),
                    float(target[1]),
                    float(prediction[0]),
                    float(prediction[1]),
                ]
            )
