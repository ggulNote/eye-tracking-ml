from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PARTICIPANT_PATTERN = re.compile(r"^p(\d{2})$")
CAMERA_DIRECTORIES = ("webcam", "phonecam")


@dataclass(frozen=True)
class ParticipantPaths:
    """All outputs for one participant and exactly one collection run."""

    participant_id: str
    participant_directory: Path
    calibration_directory: Path
    webcam_directory: Path
    phone_directory: Path
    labels_directory: Path
    events_directory: Path
    participant_json: Path


def normalize_participant_id(value: str) -> str:
    text = value.strip().lower()
    if text.isdigit():
        text = "p%02d" % int(text)
    match = PARTICIPANT_PATTERN.fullmatch(text)
    if not match:
        raise ValueError("Participant id must be p00-p99 or an integer from 0 to 99.")
    number = int(match.group(1))
    if number > 99:
        raise ValueError("Participant id cannot exceed p99.")
    return "p%02d" % number


def existing_participant_ids(dataset_root: Path) -> Iterable[str]:
    if not dataset_root.exists():
        return ()
    return tuple(
        sorted(
            path.name
            for path in dataset_root.iterdir()
            if path.is_dir() and PARTICIPANT_PATTERN.fullmatch(path.name)
        )
    )


def suggest_next_participant_id(dataset_root: Path) -> str:
    numbers = [int(participant[1:]) for participant in existing_participant_ids(dataset_root)]
    next_number = max(numbers, default=-1) + 1
    if next_number > 99:
        raise ValueError("Participant id space is exhausted at p99.")
    return "p%02d" % next_number


def create_participant_paths(dataset_root: Path, participant_id: str) -> ParticipantPaths:
    """Create a participant dataset and refuse any existing participant directory."""

    participant_id = normalize_participant_id(participant_id)
    participant_directory = dataset_root.expanduser().resolve() / participant_id
    output_markers = (
        participant_directory / "webcam",
        participant_directory / "phonecam",
        participant_directory / "labels",
        participant_directory / "events",
        participant_directory / "participant.json",
    )
    if any(path.exists() for path in output_markers):
        raise FileExistsError(
            "Participant data already exists and will not be overwritten: %s"
            % participant_directory
        )

    paths = ParticipantPaths(
        participant_id=participant_id,
        participant_directory=participant_directory,
        calibration_directory=participant_directory / "Calibration",
        webcam_directory=participant_directory / "webcam",
        phone_directory=participant_directory / "phonecam",
        labels_directory=participant_directory / "labels",
        events_directory=participant_directory / "events",
        participant_json=participant_directory / "participant.json",
    )
    for path in (
        paths.calibration_directory / "webcam",
        paths.calibration_directory / "phonecam",
        paths.webcam_directory,
        paths.phone_directory,
        paths.labels_directory,
        paths.events_directory,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_participant_json(path: Path, metadata: Dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def load_optional_metadata(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Participant metadata file does not exist: %s" % resolved)
    with resolved.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Participant metadata must be a JSON object.")
    return data
