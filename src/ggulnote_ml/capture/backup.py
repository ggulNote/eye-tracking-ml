from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
from uuid import uuid4

from .dataset import (
    normalize_head_pose,
    normalize_participant_id,
    participant_recording_directory,
)


FileFingerprint = Tuple[int, str]


@dataclass(frozen=True)
class BackupResult:
    source: Path
    destination: Path
    file_count: int
    total_bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(directory: Path) -> Dict[str, FileFingerprint]:
    return {
        path.relative_to(directory).as_posix(): (path.stat().st_size, _sha256(path))
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _validate_completed_recording(source: Path) -> None:
    participant_json = source / "participant.json"
    latency_json = source / "calibration" / "latency.json"
    summary_json = source / "feature_maps" / "summary.json"
    required = (participant_json, latency_json, summary_json)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Backup requires completed collection and postprocessing files: %s"
            % ", ".join(missing)
        )
    with participant_json.open(encoding="utf-8") as file:
        participant_metadata = json.load(file)
    with latency_json.open(encoding="utf-8") as file:
        latency_metadata = json.load(file)
    if participant_metadata.get("status") != "completed":
        raise ValueError("Only a completed recording can be backed up: %s" % source)
    if latency_metadata.get("status") != "valid":
        raise ValueError("Only a valid latency run can be backed up: %s" % latency_json)


def backup_participant_recording(
    dataset_root: Path,
    backup_root: Path,
    participant: str,
    head_pose: str,
) -> BackupResult:
    """Copy one completed pose recording and verify every byte with SHA-256."""

    normalized_participant = normalize_participant_id(participant)
    normalized_pose = normalize_head_pose(head_pose)
    source = participant_recording_directory(
        dataset_root,
        normalized_participant,
        normalized_pose,
    )
    if not source.is_dir():
        raise FileNotFoundError("Participant recording does not exist: %s" % source)
    _validate_completed_recording(source)

    resolved_backup_root = backup_root.expanduser().resolve()
    if not resolved_backup_root.is_dir():
        raise FileNotFoundError(
            "iCloud shared backup folder does not exist: %s" % resolved_backup_root
        )
    destination_parent = resolved_backup_root / normalized_participant
    destination = destination_parent / normalized_pose
    if destination.exists():
        raise FileExistsError(
            "Backup already exists and will not be overwritten: %s" % destination
        )
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Backup destination cannot be inside the source recording.")

    destination_parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_parent / (
        ".%s.partial-%s" % (normalized_pose, uuid4().hex)
    )
    try:
        shutil.copytree(source, temporary, copy_function=shutil.copy2)
        source_manifest = _manifest(source)
        copied_manifest = _manifest(temporary)
        if source_manifest != copied_manifest:
            raise RuntimeError(
                "Backup verification failed; local data was not deleted: %s" % temporary
            )
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return BackupResult(
        source=source,
        destination=destination,
        file_count=len(source_manifest),
        total_bytes=sum(size for size, _digest in source_manifest.values()),
    )


def delete_verified_local_recording(result: BackupResult) -> None:
    """Delete only the exact source pose after a verified backup exists."""

    if not result.source.is_dir() or not result.destination.is_dir():
        raise FileNotFoundError("Both local and backed-up recordings must exist before deletion.")
    if _manifest(result.source) != _manifest(result.destination):
        raise RuntimeError("Backup changed after verification; local data was not deleted.")
    shutil.rmtree(result.source)


def confirm_local_deletion() -> bool:
    answer = input("iCloud 복사·검증이 완료되었습니다. 로컬 데이터를 삭제하시겠습니까? [네/아니요]: ")
    return answer.strip().lower() in {"네", "예", "y", "yes"}
