import json
from pathlib import Path

import pytest

from ggulnote_ml.capture.backup import (
    backup_participant_recording,
    confirm_local_deletion,
    delete_verified_local_recording,
)


def _completed_recording(dataset_root: Path) -> Path:
    source = dataset_root / "안은제" / "neutral"
    (source / "calibration").mkdir(parents=True)
    (source / "feature_maps").mkdir()
    (source / "video" / "web").mkdir(parents=True)
    (source / "participant.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (source / "calibration" / "latency.json").write_text(
        json.dumps({"status": "valid"}), encoding="utf-8"
    )
    (source / "feature_maps" / "summary.json").write_text(
        json.dumps({"selected_pairs": 81}), encoding="utf-8"
    )
    (source / "video" / "web" / "capture.mp4").write_bytes(b"video-data")
    return source


def test_backup_verifies_copy_and_deletes_only_after_explicit_delete_call(tmp_path):
    dataset_root = tmp_path / "local"
    backup_root = tmp_path / "icloud"
    backup_root.mkdir()
    source = _completed_recording(dataset_root)

    result = backup_participant_recording(
        dataset_root, backup_root, "안은제", "neutral"
    )

    assert source.is_dir()
    assert result.destination == backup_root / "안은제" / "neutral"
    assert (result.destination / "video" / "web" / "capture.mp4").read_bytes() == b"video-data"
    assert result.file_count == 4

    delete_verified_local_recording(result)

    assert not source.exists()
    assert result.destination.is_dir()


def test_backup_refuses_incomplete_recording_and_missing_shared_folder(tmp_path):
    dataset_root = tmp_path / "local"
    source = dataset_root / "안은제" / "neutral"
    source.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="completed collection"):
        backup_participant_recording(
            dataset_root, tmp_path / "missing", "안은제", "neutral"
        )


def test_backup_refuses_overwrite(tmp_path):
    dataset_root = tmp_path / "local"
    backup_root = tmp_path / "icloud"
    backup_root.mkdir()
    _completed_recording(dataset_root)
    (backup_root / "안은제" / "neutral").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        backup_participant_recording(
            dataset_root, backup_root, "안은제", "neutral"
        )


@pytest.mark.parametrize("answer", ("네", "예", "y", "YES"))
def test_delete_confirmation_accepts_only_explicit_yes(monkeypatch, answer):
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)
    assert confirm_local_deletion()


@pytest.mark.parametrize("answer", ("", "아니요", "no"))
def test_delete_confirmation_defaults_to_keep_local(monkeypatch, answer):
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)
    assert not confirm_local_deletion()
