import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ggulnote_ml.capture.collection import PairedLabelWriter, run_simulation_protocols
from ggulnote_ml.capture.config import load_capture_config
from ggulnote_ml.capture.contracts import FramePacket
from ggulnote_ml.capture.dataset import (
    create_participant_paths,
    normalize_participant_id,
    suggest_next_participant_id,
)
from ggulnote_ml.capture.protocols import (
    ProtocolPlan,
    ProtocolSegment,
    build_protocol_plans,
    normalized_coordinates,
)


def test_protocol_counts_randomization_and_splits():
    config = load_capture_config(Path("configs/capture.yaml"))
    intro, train, transition, dynamic, refix, evaluation = build_protocol_plans(config.protocols)

    assert intro.active_duration_ms == 5000
    assert len(train.segments) == 27
    assert transition.active_duration_ms == 3000
    assert len(dynamic.segments) == 12
    assert dynamic.active_duration_ms == 36000
    assert refix.active_duration_ms == 4000
    assert len(evaluation.segments) == 18
    assert train.split == "train"
    assert evaluation.split == "evaluation"
    assert dynamic.split == "train"
    assert [segment.direction for segment in dynamic.segments[::2]] == [
        "top_to_bottom",
        "bottom_to_top",
        "top_to_bottom",
        "bottom_to_top",
        "top_to_bottom",
        "bottom_to_top",
    ]
    assert [segment.target_index for segment in train.segments] != list(range(27))
    assert sum(plan.total_duration_ms for plan in (intro, train, transition, dynamic, refix, evaluation)) == 105000


def test_static_training_windows_match_protocol_contract():
    config = load_capture_config(Path("configs/capture.yaml"))
    train = build_protocol_plans(config.protocols)[1]
    start = train.initial_delay_ms

    assert train.state_at(start + 100).is_settling
    assert not train.state_at(start + 399).is_training_sample
    assert train.state_at(start + 400).is_training_sample
    assert train.state_at(start + 1049).is_training_sample
    assert not train.state_at(start + 1050).is_training_sample


def test_dynamic_target_interpolates_without_static_guide_requirement():
    config = load_capture_config(Path("configs/capture.yaml"))
    dynamic = build_protocol_plans(config.protocols)[3]
    halfway = dynamic.initial_delay_ms + dynamic.segments[0].duration_ms / 2
    state = dynamic.state_at(halfway)

    assert state.split == "train"
    assert state.target_x == pytest.approx(config.protocols.dynamic.columns[0])
    assert state.target_y == pytest.approx(0.5)
    assert state.show_guide_line
    assert state.is_training_sample
    transition_state = dynamic.state_at(dynamic.segments[0].duration_ms + 100)
    assert transition_state.direction == "transition"
    assert not transition_state.is_usable_window


def test_participant_suggestion_and_participant_no_overwrite(tmp_path):
    (tmp_path / "p00").mkdir()
    (tmp_path / "p02").mkdir()
    assert suggest_next_participant_id(tmp_path) == "p03"
    assert normalize_participant_id("7") == "p07"

    paths = create_participant_paths(tmp_path, "p03")
    assert paths.webcam_directory.name == "webcam"
    assert paths.phone_directory.name == "phonecam"
    with pytest.raises(FileExistsError):
        create_participant_paths(tmp_path, "p03")


def test_simulation_writes_videos_pair_labels_and_unix_timestamps(tmp_path):
    config = load_capture_config(Path("configs/capture.yaml"))
    config = replace(
        config,
        recording=replace(config.recording, enabled=True),
        simulation=replace(config.simulation, fps=10, width=64, height=48),
    )
    paths = create_participant_paths(tmp_path, "p00")
    plan = ProtocolPlan(
        protocol_id="tiny_static",
        split="train",
        initial_delay_ms=100,
        completion_duration_ms=100,
        segments=(ProtocolSegment(0, 0, 0, 0.5, 0.5, 0.5, 0.5, 200, "static"),),
        settling_duration_ms=50,
        usable_duration_ms=100,
        show_guide_line=False,
    )

    result = run_simulation_protocols(config, paths, (plan,))[0]

    assert result.status == "completed"
    assert (paths.webcam_directory / "capture.mp4").is_file()
    assert (paths.phone_directory / "capture.mp4").is_file()
    with result.label_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert rows[0]["participant"] == "p00"
    display_timestamps = [int(row["display_timestamp"]) for row in rows]
    assert display_timestamps[0] == config.simulation.base_unix_timestamp_ns
    assert display_timestamps == sorted(display_timestamps)
    assert all(
        int(row["webcam_timestamp"]) >= int(row["display_timestamp"]) for row in rows
    )
    assert int(rows[0]["webcam_timestamp"]) > 0
    assert float(rows[0]["time_diff_ms"]) == pytest.approx(5.0)
    assert "latency_ms" not in rows[0]
    assert set(row["split"] for row in rows) == {"train"}


def test_multiple_protocols_share_one_video_and_label_file(tmp_path):
    config = load_capture_config(Path("configs/capture.yaml"))
    config = replace(
        config,
        recording=replace(config.recording, enabled=True),
        simulation=replace(config.simulation, fps=10, width=64, height=48),
    )
    paths = create_participant_paths(tmp_path, "p00")
    train = ProtocolPlan(
        "tiny_train", "train", 0, 0,
        (ProtocolSegment(0, 0, 0, 0.5, 0.5, 0.5, 0.5, 100, "static"),),
        0, 100, False,
    )
    validation = replace(train, protocol_id="tiny_validation", split="validation")

    results = run_simulation_protocols(config, paths, (train, validation))

    assert len(results) == 2
    assert results[0].label_path == results[1].label_path == paths.labels_directory / "labels.csv"
    assert len(list(paths.webcam_directory.glob("*.mp4"))) == 1
    assert len(list(paths.phone_directory.glob("*.mp4"))) == 1
    with results[0].label_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["split"] for row in rows} == {"train", "validation"}


def test_label_writer_rejects_invalid_or_decreasing_display_timestamps(tmp_path):
    config = load_capture_config(Path("configs/capture.yaml"))
    paths = create_participant_paths(tmp_path, "p00")
    state = build_protocol_plans(config.protocols)[0].state_at(0)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    webcam = FramePacket(0, 0.0, 200, frame)
    phonecam = FramePacket(0, 0.0, 205, frame)
    writer = PairedLabelWriter(paths.labels_directory / "contract.csv")
    try:
        with pytest.raises(ValueError, match="positive Unix nanosecond"):
            writer.write(paths, 0, 0, webcam, phonecam, state, config.display)
        writer.write(paths, 0, 100, webcam, phonecam, state, config.display)
        with pytest.raises(ValueError, match="non-decreasing"):
            writer.write(paths, 1, 99, webcam, phonecam, state, config.display)
    finally:
        writer.close()


def test_coordinate_conventions():
    assert normalized_coordinates(0.0, 0.0, 101, 51) == (0, 0, -0.5, -0.5)
    assert normalized_coordinates(1.0, 1.0, 101, 51) == (100, 50, 0.5, 0.5)
