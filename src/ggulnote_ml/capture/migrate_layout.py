from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .dataset import existing_participant_ids, normalize_participant_id
from .frame_samples import IMAGE_SAMPLE_COLUMNS


LEGACY_LABEL_COLUMN_MAP = {
    "webcam_image": "web_image",
    "webcam_frame": "web_frame",
    "webcam_timestamp": "web_timestamp",
    "webcam_face_detected": "web_face_detected",
    "webcam_eyes_detected": "web_eyes_detected",
    "webcam_mediapipe_face_detected": "web_mediapipe_face_detected",
    "webcam_mediapipe_iris_detected": "web_mediapipe_iris_detected",
    "webcam_sharpness": "web_sharpness",
    "webcam_brightness": "web_brightness",
    "phonecam_image": "phone_image",
    "phonecam_frame": "phone_frame",
    "phonecam_timestamp": "phone_timestamp",
    "phonecam_face_detected": "phone_face_detected",
    "phonecam_eyes_detected": "phone_eyes_detected",
    "phonecam_mediapipe_face_detected": "phone_mediapipe_face_detected",
    "phonecam_mediapipe_iris_detected": "phone_mediapipe_iris_detected",
    "phonecam_sharpness": "phone_sharpness",
    "phonecam_brightness": "phone_brightness",
}


def _move(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if destination.exists():
        try:
            same_file = source.samefile(destination)
        except OSError:
            same_file = False
        if same_file and source.name != destination.name:
            temporary = source.parent / (source.name + ".layout_v6_tmp")
            source.rename(temporary)
            temporary.rename(destination)
            return
        raise FileExistsError("Migration destination already exists: %s" % destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)


def _rewrite_csv_values(path: Path, replacements: Dict[str, str]) -> None:
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = tuple(reader.fieldnames or ())
        rows = list(reader)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: _replace_text(value, replacements)
                    for name, value in row.items()
                }
            )
    temporary.replace(path)


def _replace_text(value: object, replacements: Dict[str, str]) -> str:
    text = "" if value is None else str(value)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _rewrite_json_values(path: Path, replacements: Dict[str, str]) -> None:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    def rewrite(value):
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            return _replace_text(value, replacements)
        return value

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(rewrite(data), file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _convert_labels(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    if destination.exists():
        raise FileExistsError("Canonical labels already exist: %s" % destination)
    with source.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=IMAGE_SAMPLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            converted = {
                LEGACY_LABEL_COLUMN_MAP.get(name, name): value
                for name, value in row.items()
            }
            for key in ("web_image", "phone_image"):
                converted[key] = (
                    converted.get(key, "")
                    .replace("images/webcam/", "images/web/")
                    .replace("images/phonecam/", "images/phone/")
                )
            writer.writerow({name: converted.get(name, "") for name in IMAGE_SAMPLE_COLUMNS})
    temporary.replace(destination)


def _merge_protocol_csvs(source_directory: Path, destination: Path) -> None:
    csv_paths = sorted(source_directory.glob("*.csv")) if source_directory.is_dir() else []
    if not csv_paths:
        return
    if destination.exists():
        raise FileExistsError("Protocol metadata already exists: %s" % destination)
    rows: List[Dict[str, str]] = []
    for path in csv_paths:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                row["x_norm"] = row.pop("x", row.get("x_norm", ""))
                row["y_norm"] = row.pop("y", row.get("y_norm", ""))
                rows.append(row)
    columns = (
        "protocol",
        "split",
        "segment",
        "repeat",
        "target",
        "x_norm",
        "y_norm",
        "direction",
        "confirmation_required",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in columns} for row in rows)


def _move_feature_outputs(
    participant: str, participant_directory: Path, interim_root: Path
) -> None:
    source = interim_root / participant
    destination = participant_directory / "feature_maps"
    for legacy_name, new_name in (("webcam", "web"), ("phonecam", "phone")):
        source_camera = source / legacy_name
        if not source_camera.is_dir():
            continue
        frames_destination = destination / new_name / "frames"
        frames_destination.mkdir(parents=True, exist_ok=True)
        for image in sorted(source_camera.glob("*.png")):
            _move(image, frames_destination / image.name.replace(legacy_name, new_name))
        _move(
            source_camera / "processed_features.csv",
            destination / new_name / "features.csv",
        )
        if not any(source_camera.iterdir()):
            source_camera.rmdir()
    manifest_root = interim_root / "manifests"
    manifest_names = {
        "%s_training.csv" % participant: "training.csv",
        "%s_evaluation.csv" % participant: "evaluation.csv",
        "%s_video_training.csv" % participant: "training_features.csv",
        "%s_video_evaluation.csv" % participant: "evaluation_features.csv",
        "%s_summary.json" % participant: "summary.json",
    }
    for legacy_name, new_name in manifest_names.items():
        _move(manifest_root / legacy_name, destination / new_name)
    replacements = {
        "%s/webcam/" % participant: "web/frames/",
        "%s/phonecam/" % participant: "phone/frames/",
        "_webcam_frame_": "_web_frame_",
        "_phonecam_frame_": "_phone_frame_",
    }
    for path in (
        destination / "web" / "features.csv",
        destination / "phone" / "features.csv",
        destination / "training.csv",
        destination / "evaluation.csv",
        destination / "training_features.csv",
        destination / "evaluation_features.csv",
    ):
        _rewrite_csv_values(path, replacements)
    _rewrite_json_values(destination / "summary.json", replacements)
    if source.is_dir() and not any(source.iterdir()):
        source.rmdir()


def _update_participant_json(participant_directory: Path) -> None:
    path = participant_directory / "participant.json"
    if not path.is_file():
        return
    metadata_directory = participant_directory / "metadata"
    backup = metadata_directory / "participant_legacy.json"
    if not backup.exists():
        shutil.copy2(path, backup)
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    data["schema_version"] = 6
    data["csv_schema_version"] = 3
    data["data_layout"] = {
        "calibration": "calibration",
        "video_web": "video/web",
        "video_phone": "video/phone",
        "images_web": "images/web",
        "images_phone": "images/phone",
        "labels": "labels/labels.csv",
        "frame_log": "metadata/frame_log.csv",
        "protocol": "metadata/protocol.csv",
        "feature_maps": "feature_maps",
    }
    data["config_snapshot"] = "metadata/capture_config.yaml"
    frame_capture = data.get("frame_capture")
    if isinstance(frame_capture, dict):
        frame_capture.update(
            {
                "web_directory": "images/web",
                "phone_directory": "images/phone",
                "labels": "labels/labels.csv",
            }
        )
        for legacy_key in ("webcam_directory", "phonecam_directory", "manifest"):
            frame_capture.pop(legacy_key, None)
    for result in data.get("results", []):
        if isinstance(result, dict):
            result.pop("labels", None)
            result["frame_log"] = "metadata/frame_log.csv"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def migrate_participant_layout(
    participant_directory: Path,
    interim_root: Optional[Path] = None,
) -> Path:
    """Convert the legacy capture tree to schema v6 without deleting payloads."""

    participant_directory = participant_directory.expanduser().resolve()
    participant = normalize_participant_id(participant_directory.name)
    if not participant_directory.is_dir():
        raise FileNotFoundError("Participant directory does not exist: %s" % participant_directory)

    has_capture_payload = any(
        path.exists()
        for path in (
            participant_directory / "participant.json",
            participant_directory / "labels",
            participant_directory / "webcam",
            participant_directory / "phonecam",
            participant_directory / "video",
        )
    )
    _move(participant_directory / "Calibration", participant_directory / "calibration")
    if not has_capture_payload:
        # A latency-only participant ID is still available for its first Dot
        # Test. Do not create capture markers that would make suggestion skip it.
        return participant_directory
    calibration = participant_directory / "calibration"
    _move(calibration / "webcam", calibration / "web")
    _move(calibration / "phonecam", calibration / "phone")
    _move(participant_directory / "webcam", participant_directory / "video" / "web")
    _move(participant_directory / "phonecam", participant_directory / "video" / "phone")
    _move(participant_directory / "images" / "webcam", participant_directory / "images" / "web")
    _move(participant_directory / "images" / "phonecam", participant_directory / "images" / "phone")

    metadata = participant_directory / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    labels = participant_directory / "labels"
    legacy_frame_log = labels / "labels.csv"
    legacy_selected_labels = labels / "image_samples.csv"
    if legacy_selected_labels.exists():
        _move(legacy_frame_log, metadata / "frame_log.csv")
        _move(legacy_selected_labels, metadata / "legacy_image_samples.csv")
        _convert_labels(metadata / "legacy_image_samples.csv", labels / "labels.csv")

    events = participant_directory / "events"
    if events.is_dir():
        _move(events / "capture_config.yaml", metadata / "capture_config.yaml")
        _merge_protocol_csvs(events, metadata / "protocol.csv")
        _move(events, metadata / "legacy_events")

    synchronized = participant_directory / "synchronized"
    _move(
        synchronized / "synchronized_frames.csv",
        participant_directory / "feature_maps" / "synchronized.csv",
    )
    _move(
        synchronized / "synchronization.json",
        participant_directory / "feature_maps" / "synchronization.json",
    )
    if synchronized.is_dir() and not any(synchronized.iterdir()):
        synchronized.rmdir()

    for directory in (
        participant_directory / "video" / "web",
        participant_directory / "video" / "phone",
        participant_directory / "images" / "web",
        participant_directory / "images" / "phone",
        participant_directory / "feature_maps" / "web",
        participant_directory / "feature_maps" / "phone",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if interim_root is not None:
        _move_feature_outputs(participant, participant_directory, interim_root)
    _update_participant_json(participant_directory)
    return participant_directory


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate participant folders to schema v6.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/participants"))
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim/dual_view"))
    parser.add_argument("--participant", action="append", help="p00; repeat for multiple")
    parser.add_argument("--all", action="store_true", help="Migrate every existing pXX folder")
    parser.add_argument("--apply", action="store_true", help="Perform the migration")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.dataset_root.expanduser().resolve()
    participants: Iterable[str]
    if args.all:
        participants = existing_participant_ids(root)
    elif args.participant:
        participants = tuple(normalize_participant_id(value) for value in args.participant)
    else:
        raise SystemExit("Use --participant p00 (repeatable) or --all.")
    participants = tuple(participants)
    if not args.apply:
        print("Dry run; no files changed. Participants: %s" % ", ".join(participants))
        print("Add --apply after reviewing the schema-v6 documentation.")
        return 0
    for participant in participants:
        migrated = migrate_participant_layout(
            root / participant,
            interim_root=args.interim_root.expanduser().resolve(),
        )
        print("Migrated: %s" % migrated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
