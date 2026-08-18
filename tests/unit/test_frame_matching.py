import csv
import json
from pathlib import Path

import pytest

from ggulnote_ml.synchronization.config import load_latency_config
from ggulnote_ml.synchronization.matching import SYNC_COLUMNS, synchronize_labels


LABEL_COLUMNS = (
    "participant",
    "head_pose",
    "protocol",
    "split",
    "pair",
    "display_timestamp",
    "webcam_frame",
    "webcam_timestamp",
    "phonecam_frame",
    "phonecam_timestamp",
    "x_norm",
    "y_norm",
    "x_centered",
    "y_centered",
    "segment",
    "target",
    "direction",
    "usable",
)


def _write_latency(
    path: Path,
    participant: str = "p00",
    head_pose: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "valid",
                "participant": participant,
                "head_pose": head_pose,
                "cameras": {
                    "webcam": {"median_ms": 100.0},
                    "phonecam": {"median_ms": 120.0},
                },
            }
        ),
        encoding="utf-8",
    )


def _write_dynamic_labels(
    path: Path,
    participant: str = "p00",
    head_pose: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = 1_700_000_000_000_000_000
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for frame in range(4):
            display = base + frame * 100_000_000
            scene = display + 50_000_000
            writer.writerow(
                {
                    "participant": participant,
                    "head_pose": head_pose,
                    "protocol": "dynamic_vertical_3col",
                    "split": "train",
                    "pair": frame,
                    "display_timestamp": display,
                    "webcam_frame": frame,
                    "webcam_timestamp": scene + 100_000_000,
                    "phonecam_frame": frame,
                    "phonecam_timestamp": scene + 120_000_000,
                    "x_norm": "0.500000",
                    "y_norm": "%.6f" % (frame * 0.1),
                    "x_centered": "0.000000",
                    "y_centered": "%.6f" % (frame * 0.1 - 0.5),
                    "segment": 0,
                    "target": 0,
                    "direction": "top_to_bottom",
                    "usable": 1,
                }
            )


def test_synchronization_applies_camera_specific_latency_and_interpolates_dynamic_target(
    tmp_path,
):
    config = load_latency_config(Path("configs/latency.yaml"))
    labels = tmp_path / "labels.csv"
    latency = tmp_path / "latency.json"
    output = tmp_path / "synchronized_frames.csv"
    _write_dynamic_labels(labels)
    _write_latency(latency)

    summary = synchronize_labels(labels, latency, output, config.matching)

    assert summary["total_pairs"] == 4
    assert summary["valid_sync_pairs"] == 4
    assert summary["valid_pairs"] == 4
    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert tuple(rows[0]) == SYNC_COLUMNS
    assert [row["phonecam_frame"] for row in rows] == ["0", "1", "2", "3"]
    assert {float(row["corrected_time_diff_ms"]) for row in rows} == {0.0}
    assert float(rows[0]["y_norm"]) == pytest.approx(0.05)
    assert float(rows[1]["y_norm"]) == pytest.approx(0.15)
    assert rows[0]["target_interpolated"] == "1"
    assert rows[-1]["target_interpolated"] == "0"


def test_synchronization_refuses_output_overwrite(tmp_path):
    config = load_latency_config(Path("configs/latency.yaml"))
    labels = tmp_path / "labels.csv"
    latency = tmp_path / "latency.json"
    output = tmp_path / "synchronized_frames.csv"
    _write_dynamic_labels(labels)
    _write_latency(latency)
    synchronize_labels(labels, latency, output, config.matching)

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        synchronize_labels(labels, latency, output, config.matching)


def test_synchronization_rejects_participant_mismatch(tmp_path):
    config = load_latency_config(Path("configs/latency.yaml"))
    labels = tmp_path / "labels.csv"
    latency = tmp_path / "latency.json"
    _write_dynamic_labels(labels)
    _write_latency(latency, participant="p01")

    with pytest.raises(ValueError, match="does not match"):
        synchronize_labels(labels, latency, tmp_path / "out.csv", config.matching)


def test_synchronization_allows_explicit_shared_latency(tmp_path):
    config = load_latency_config(Path("configs/latency.yaml"))
    labels = tmp_path / "labels.csv"
    latency = tmp_path / "latency.json"
    output = tmp_path / "out.csv"
    _write_dynamic_labels(labels)
    _write_latency(latency, participant="p01")

    summary = synchronize_labels(
        labels,
        latency,
        output,
        config.matching,
        allow_shared_latency=True,
    )

    assert summary["participant"] == "p00"
    assert summary["valid_pairs"] == 4


def test_synchronization_preserves_named_participant_and_head_pose(tmp_path):
    config = load_latency_config(Path("configs/latency.yaml"))
    labels = tmp_path / "labels.csv"
    latency = tmp_path / "latency.json"
    output = tmp_path / "out.csv"
    _write_dynamic_labels(labels, participant="안은제", head_pose="neutral")
    _write_latency(latency, participant="안은제", head_pose="neutral")

    summary = synchronize_labels(
        labels,
        latency,
        output,
        config.matching,
        expected_head_pose="neutral",
    )

    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert summary["participant"] == "안은제"
    assert summary["head_pose"] == "neutral"
    assert {row["participant"] for row in rows} == {"안은제"}
    assert {row["head_pose"] for row in rows} == {"neutral"}
