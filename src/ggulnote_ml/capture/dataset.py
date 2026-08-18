from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PARTICIPANT_PATTERN = re.compile(r"^p(\d{2})$")
INVALID_PARTICIPANT_CHARACTERS = re.compile(r"[\x00-\x1f\x7f/\\:]")
CAMERA_DIRECTORIES = ("webcam", "phonecam")
HEAD_POSES = ("neutral", "head_up", "head_down")
HEAD_POSE_ALIASES = {
    "neutral": "neutral",
    "middle": "neutral",
    "center": "neutral",
    "정면": "neutral",
    "중립": "neutral",
    "head_up": "head_up",
    "up": "head_up",
    "위": "head_up",
    "head_down": "head_down",
    "down": "head_down",
    "아래": "head_down",
}


@dataclass(frozen=True)
class ParticipantPaths:
    """All outputs for one participant and exactly one collection run."""

    participant_id: str
    head_pose: str
    participant_root_directory: Path
    participant_directory: Path
    calibration_directory: Path
    video_directory: Path
    web_video_directory: Path
    phone_video_directory: Path
    images_directory: Path
    web_images_directory: Path
    phone_images_directory: Path
    labels_directory: Path
    metadata_directory: Path
    feature_maps_directory: Path
    web_feature_directory: Path
    phone_feature_directory: Path
    participant_json: Path

    # Internal camera roles retain their historical webcam/phonecam names.  These
    # aliases keep the capture modules readable while the on-disk contract uses
    # the shorter user-facing web/phone directories.
    @property
    def webcam_directory(self) -> Path:
        return self.web_video_directory

    @property
    def phone_directory(self) -> Path:
        return self.phone_video_directory

    @property
    def webcam_images_directory(self) -> Path:
        return self.web_images_directory

    @property
    def events_directory(self) -> Path:
        return self.metadata_directory


def normalize_participant_id(value: str) -> str:
    """Return a filesystem-safe participant name while preserving Korean text.

    Existing ``p00``-style pilot folders remain valid, but new collections may
    use names such as ``안은제``.  The same normalized value is written to CSV
    participant columns and used as the participant directory name.
    """

    text = unicodedata.normalize("NFC", str(value)).strip()
    if not text:
        raise ValueError("Participant name cannot be empty.")
    if len(text) > 50:
        raise ValueError("Participant name must contain at most 50 characters.")
    if text in {".", ".."} or text.startswith("."):
        raise ValueError("Participant name cannot start with a dot.")
    if INVALID_PARTICIPANT_CHARACTERS.search(text):
        raise ValueError(
            "Participant name cannot contain '/', '\\', ':', or control characters."
        )

    # Keep legacy pilot IDs canonical on case-sensitive filesystems.
    legacy_match = PARTICIPANT_PATTERN.fullmatch(text.lower())
    if legacy_match:
        return "p%02d" % int(legacy_match.group(1))
    return text


def normalize_head_pose(value: str) -> str:
    """Normalize the controlled head-pose condition used for one recording."""

    text = unicodedata.normalize("NFC", str(value)).strip().lower().replace("-", "_")
    normalized = HEAD_POSE_ALIASES.get(text)
    if normalized is None:
        raise ValueError(
            "Head pose must be one of: %s." % ", ".join(HEAD_POSES)
        )
    return normalized


def participant_recording_directory(
    dataset_root: Path,
    participant_id: str,
    head_pose: Optional[str] = None,
) -> Path:
    """Resolve a new pose recording while retaining legacy flat-folder support."""

    participant = normalize_participant_id(participant_id)
    participant_root = dataset_root.expanduser().resolve() / participant
    if head_pose is None or str(head_pose).strip() == "":
        return participant_root
    return participant_root / normalize_head_pose(head_pose)


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
    resolved = dataset_root.expanduser().resolve()
    for number in range(100):
        participant_id = "p%02d" % number
        participant_directory = resolved / participant_id
        capture_outputs = (
            participant_directory / "video",
            participant_directory / "images",
            participant_directory / "labels",
            participant_directory / "metadata",
            participant_directory / "feature_maps",
            # Legacy markers remain recognized so an old participant is never
            # silently overwritten during migration.
            participant_directory / "webcam",
            participant_directory / "phonecam",
            participant_directory / "events",
            participant_directory / "participant.json",
        )
        if not any(path.exists() for path in capture_outputs):
            return participant_id
    raise ValueError("Participant id space is exhausted at p99.")


def create_participant_paths(
    dataset_root: Path,
    participant_id: str,
    head_pose: Optional[str] = None,
) -> ParticipantPaths:
    """Create a participant dataset and refuse any existing participant directory."""

    participant_id = normalize_participant_id(participant_id)
    normalized_head_pose = (
        "" if head_pose is None or str(head_pose).strip() == "" else normalize_head_pose(head_pose)
    )
    participant_root_directory = (
        dataset_root.expanduser().resolve() / participant_id
    )
    participant_directory = (
        participant_root_directory / normalized_head_pose
        if normalized_head_pose
        else participant_root_directory
    )
    output_markers = (
        participant_directory / "video",
        participant_directory / "images",
        participant_directory / "labels",
        participant_directory / "metadata",
        participant_directory / "feature_maps",
        participant_directory / "webcam",
        participant_directory / "phonecam",
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
        head_pose=normalized_head_pose,
        participant_root_directory=participant_root_directory,
        participant_directory=participant_directory,
        calibration_directory=participant_directory / "calibration",
        video_directory=participant_directory / "video",
        web_video_directory=participant_directory / "video" / "web",
        phone_video_directory=participant_directory / "video" / "phone",
        images_directory=participant_directory / "images",
        web_images_directory=participant_directory / "images" / "web",
        phone_images_directory=participant_directory / "images" / "phone",
        labels_directory=participant_directory / "labels",
        metadata_directory=participant_directory / "metadata",
        feature_maps_directory=participant_directory / "feature_maps",
        web_feature_directory=participant_directory / "feature_maps" / "web",
        phone_feature_directory=participant_directory / "feature_maps" / "phone",
        participant_json=participant_directory / "participant.json",
    )
    for path in (
        paths.calibration_directory / "web",
        paths.calibration_directory / "phone",
        paths.web_video_directory,
        paths.phone_video_directory,
        paths.web_images_directory,
        paths.phone_images_directory,
        paths.labels_directory,
        paths.metadata_directory,
        paths.web_feature_directory,
        paths.phone_feature_directory,
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
