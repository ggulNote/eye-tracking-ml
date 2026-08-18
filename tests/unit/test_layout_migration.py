import csv
import json
from pathlib import Path

from ggulnote_ml.capture.frame_samples import IMAGE_SAMPLE_COLUMNS
from ggulnote_ml.capture.migrate_layout import (
    LEGACY_LABEL_COLUMN_MAP,
    migrate_participant_layout,
)
from ggulnote_ml.capture.dataset import suggest_next_participant_id


def test_latency_only_participant_remains_available_for_first_capture(tmp_path):
    participant = tmp_path / "participants" / "p00"
    (participant / "Calibration").mkdir(parents=True)
    (participant / "Calibration" / "latency.json").write_text("{}", encoding="utf-8")

    migrate_participant_layout(participant)

    assert (participant / "calibration" / "latency.json").is_file()
    assert not (participant / "video").exists()
    assert suggest_next_participant_id(tmp_path / "participants") == "p00"


def test_legacy_participant_is_migrated_to_single_label_layout(tmp_path):
    participant = tmp_path / "raw" / "participants" / "p00"
    (participant / "Calibration" / "webcam").mkdir(parents=True)
    (participant / "Calibration" / "phonecam").mkdir(parents=True)
    (participant / "webcam").mkdir()
    (participant / "phonecam").mkdir()
    (participant / "images" / "webcam").mkdir(parents=True)
    (participant / "images" / "phonecam").mkdir(parents=True)
    (participant / "labels").mkdir()
    (participant / "events").mkdir()
    (participant / "synchronized").mkdir()
    (participant / "Calibration" / "webcam" / "Camera.mat").write_bytes(b"web")
    (participant / "Calibration" / "phonecam" / "Camera.mat").write_bytes(b"phone")
    (participant / "webcam" / "capture.mp4").write_bytes(b"web video")
    (participant / "phonecam" / "capture.mp4").write_bytes(b"phone video")
    (participant / "images" / "webcam" / "s000000.jpg").write_bytes(b"web image")
    (participant / "images" / "phonecam" / "s000000.jpg").write_bytes(b"phone image")
    (participant / "labels" / "labels.csv").write_text(
        "participant,pair\np00,0\n", encoding="utf-8"
    )

    inverse = {new: old for old, new in LEGACY_LABEL_COLUMN_MAP.items()}
    legacy_columns = tuple(inverse.get(name, name) for name in IMAGE_SAMPLE_COLUMNS)
    legacy_row = {name: "" for name in legacy_columns}
    legacy_row.update(
        {
            "sample": "s000000",
            "participant": "p00",
            "protocol": "train_static_3x9",
            "split": "train",
            "pair": "7",
            "webcam_image": "images/webcam/s000000.jpg",
            "webcam_frame": "7",
            "webcam_timestamp": "100",
            "phonecam_image": "images/phonecam/s000000.jpg",
            "phonecam_frame": "7",
            "phonecam_timestamp": "110",
            "segment": "0",
            "target": "0",
            "direction": "static",
        }
    )
    with (participant / "labels" / "image_samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=legacy_columns)
        writer.writeheader()
        writer.writerow(legacy_row)
    (participant / "events" / "train.csv").write_text(
        "protocol,split,segment,repeat,target,x,y,direction,confirmation_required\n"
        "train_static_3x9,train,0,0,0,0.2,0.08,static,1\n",
        encoding="utf-8",
    )
    (participant / "synchronized" / "synchronized_frames.csv").write_text(
        "participant,pair\np00,0\n", encoding="utf-8"
    )
    (participant / "synchronized" / "synchronization.json").write_text(
        "{}", encoding="utf-8"
    )
    (participant / "participant.json").write_text(
        json.dumps({"schema_version": 4, "results": [{"labels": "labels/labels.csv"}]}),
        encoding="utf-8",
    )

    migrate_participant_layout(participant)

    assert (participant / "calibration" / "web" / "Camera.mat").is_file()
    assert (participant / "calibration" / "phone" / "Camera.mat").is_file()
    assert (participant / "video" / "web" / "capture.mp4").is_file()
    assert (participant / "video" / "phone" / "capture.mp4").is_file()
    assert (participant / "images" / "web" / "s000000.jpg").is_file()
    assert (participant / "images" / "phone" / "s000000.jpg").is_file()
    assert (participant / "metadata" / "frame_log.csv").is_file()
    assert (participant / "metadata" / "protocol.csv").is_file()
    assert (participant / "feature_maps" / "synchronized.csv").is_file()
    with (participant / "labels" / "labels.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        row = next(csv.DictReader(file))
    assert row["web_image"] == "images/web/s000000.jpg"
    assert row["phone_image"] == "images/phone/s000000.jpg"
    assert "webcam_image" not in row
    metadata = json.loads((participant / "participant.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 6
    assert metadata["data_layout"]["feature_maps"] == "feature_maps"
