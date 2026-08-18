import csv
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from ggulnote_ml.capture import cli as capture_cli
from ggulnote_ml.capture.collection import (
    LABEL_COLUMNS,
    PairedLabelWriter,
    render_display_check,
    run_simulation_protocols,
)
from ggulnote_ml.capture.camera import FrozenFrameGuard
from ggulnote_ml.capture.config import load_capture_config
from ggulnote_ml.capture.contracts import FramePacket
from ggulnote_ml.capture.dataset import (
    create_participant_paths,
    normalize_head_pose,
    normalize_participant_id,
    suggest_next_participant_id,
)
from ggulnote_ml.capture.frame_samples import (
    IMAGE_SAMPLE_COLUMNS,
    FrameQualityScorer,
    FrameSampleWriter,
)
from ggulnote_ml.capture.protocols import (
    ProtocolFrameState,
    ProtocolPlan,
    ProtocolRunner,
    ProtocolSegment,
    build_protocol_plans,
    normalized_coordinates,
)
from ggulnote_ml.capture.recorder import SessionRecorder
from ggulnote_ml.video_preprocessing.mediapipe_landmarks import FaceIrisLandmarks


def test_collection_cli_prompts_once_then_runs_latency_before_dot_test(
    tmp_path, monkeypatch
):
    config = load_capture_config(Path("configs/capture.yaml"))
    config = replace(
        config,
        dataset=replace(config.dataset, root_directory=tmp_path),
    )
    answers = iter((" 안은제 ", "head-up"))
    events = []
    participant_directory = tmp_path / "안은제" / "head_up"
    participant_directory.mkdir(parents=True)
    (participant_directory / "participant.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(capture_cli, "load_capture_config", lambda _path: config)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(
        capture_cli,
        "_run_latency_before_collection",
        lambda args, loaded, participant, pose: events.append(
            ("latency", participant, pose)
        ),
    )
    monkeypatch.setattr(
        capture_cli,
        "run_collection",
        lambda args: (
            events.append(("collection", args.participant, args.head_pose)),
            participant_directory,
        )[1],
    )
    monkeypatch.setattr(
        capture_cli,
        "_run_postprocessing_after_collection",
        lambda args, loaded, participant, pose: events.append(
            ("postprocessing", participant, pose)
        ),
    )

    capture_cli.main(["--config", "configs/capture.yaml"])

    assert events == [
        ("latency", "안은제", "head_up"),
        ("collection", "안은제", "head_up"),
        ("postprocessing", "안은제", "head_up"),
    ]


def test_collection_cli_does_not_start_dot_test_when_latency_fails(
    tmp_path, monkeypatch
):
    config = load_capture_config(Path("configs/capture.yaml"))
    config = replace(
        config,
        dataset=replace(config.dataset, root_directory=tmp_path),
    )
    collection_started = []

    monkeypatch.setattr(capture_cli, "load_capture_config", lambda _path: config)
    monkeypatch.setattr(
        capture_cli,
        "_run_latency_before_collection",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("latency aborted")),
    )
    monkeypatch.setattr(
        capture_cli,
        "run_collection",
        lambda _args: collection_started.append(True),
    )

    with pytest.raises(SystemExit):
        capture_cli.main(
            ["--participant", "안은제", "--head-pose", "neutral"]
        )

    assert not collection_started


def test_collection_cli_skips_postprocessing_when_dot_test_is_aborted(
    tmp_path, monkeypatch
):
    config = load_capture_config(Path("configs/capture.yaml"))
    config = replace(
        config,
        dataset=replace(config.dataset, root_directory=tmp_path),
    )
    participant_directory = tmp_path / "안은제" / "head_down"
    participant_directory.mkdir(parents=True)
    (participant_directory / "participant.json").write_text(
        json.dumps({"status": "aborted"}),
        encoding="utf-8",
    )
    postprocessing_started = []

    monkeypatch.setattr(capture_cli, "load_capture_config", lambda _path: config)
    monkeypatch.setattr(
        capture_cli,
        "_run_latency_before_collection",
        lambda *_args: tmp_path / "latency.json",
    )
    monkeypatch.setattr(
        capture_cli,
        "run_collection",
        lambda _args: participant_directory,
    )
    monkeypatch.setattr(
        capture_cli,
        "_run_postprocessing_after_collection",
        lambda *_args: postprocessing_started.append(True),
    )

    capture_cli.main(
        ["--participant", "안은제", "--head-pose", "head_down"]
    )

    assert not postprocessing_started


def test_protocol_counts_randomization_and_splits():
    config = load_capture_config(Path("configs/capture.yaml"))
    intro, train, transition, vertical, refix, evaluation = build_protocol_plans(config.protocols)

    assert intro.active_duration_ms == 5000
    assert len(train.segments) == 27
    assert transition.active_duration_ms == 3000
    assert len(vertical.segments) == 36
    assert vertical.active_duration_ms == 50400
    assert refix.active_duration_ms == 4000
    assert len(evaluation.segments) == 18
    assert train.split == "train"
    assert evaluation.split == "evaluation"
    assert vertical.split == "train"
    assert [segment.direction for segment in vertical.segments[:6]] == [
        "top_to_bottom"
    ] * 6
    assert [segment.direction for segment in vertical.segments[6:12]] == [
        "bottom_to_top"
    ] * 6
    assert [segment.target_index for segment in train.segments] != list(range(27))
    # Simulation confirms each static point after 0.6s. Real duration is participant-controlled.
    assert sum(plan.total_duration_ms for plan in (intro, train, transition, vertical, refix, evaluation)) == 128400
    assert config.preview.preflight_duration_ms == 5000
    assert config.preview.camera_width_px == 640
    assert config.preview.landmark_required_cameras == ("webcam",)
    assert config.frame_capture.mediapipe_quality_cameras == ("webcam",)
    assert (config.display.canvas_width, config.display.canvas_height) == (1470, 956)
    assert config.preview.required_consecutive_detections == 5
    assert config.preview.landmark_detection_confidence == pytest.approx(0.3)


def test_display_check_marks_all_four_canvas_edges():
    config = load_capture_config(Path("configs/capture.yaml"))
    canvas = render_display_check(config.display)

    assert canvas.shape == (956, 1470, 3)
    assert tuple(canvas[0, 735]) == (0, 255, 0)
    assert tuple(canvas[955, 735]) == (0, 255, 0)
    assert tuple(canvas[478, 0]) == (0, 255, 0)
    assert tuple(canvas[478, 1469]) == (0, 255, 0)


def test_static_training_windows_match_protocol_contract():
    config = load_capture_config(Path("configs/capture.yaml"))
    train = build_protocol_plans(config.protocols)[1]
    runner = ProtocolRunner(train)

    assert runner.state_at(100).is_settling
    assert not runner.confirm(399, 1_700_000_000_000_000_000)
    assert not runner.state_at(399).is_training_sample
    assert runner.confirm(400, 1_700_000_000_400_000_000)
    assert runner.state_at(400).is_training_sample
    assert runner.state_at(1049).is_training_sample
    assert not runner.state_at(1050).is_training_sample
    next_target = runner.state_at(1200)
    assert next_target.segment_index == 1
    assert not next_target.is_confirmed


def test_vertical_click_targets_traverse_each_column_down_and_up():
    config = load_capture_config(Path("configs/capture.yaml"))
    vertical = build_protocol_plans(config.protocols)[3]
    expected_y = np.linspace(
        config.protocols.vertical_click.y_min,
        config.protocols.vertical_click.y_max,
        6,
    )

    for column_index, x in enumerate(config.protocols.vertical_click.columns):
        column_segments = vertical.segments[column_index * 12 : (column_index + 1) * 12]
        assert [segment.start_x for segment in column_segments] == pytest.approx([x] * 12)
        assert [segment.start_y for segment in column_segments[:6]] == pytest.approx(expected_y)
        assert [segment.start_y for segment in column_segments[6:]] == pytest.approx(
            expected_y[::-1]
        )
    assert vertical.confirmation_required
    assert not vertical.show_guide_line


def test_participant_suggestion_and_participant_no_overwrite(tmp_path):
    (tmp_path / "p00" / "labels").mkdir(parents=True)
    (tmp_path / "p01" / "Calibration").mkdir(parents=True)
    (tmp_path / "p02" / "labels").mkdir(parents=True)
    # A calibration-only p01 directory is still available for its first capture.
    assert suggest_next_participant_id(tmp_path) == "p01"
    assert normalize_participant_id("안은제") == "안은제"
    assert normalize_participant_id("P07") == "p07"

    paths = create_participant_paths(tmp_path, "p01")
    assert paths.webcam_directory == paths.video_directory / "web"
    assert paths.phone_directory == paths.video_directory / "phone"
    assert paths.webcam_images_directory == paths.images_directory / "web"
    assert paths.phone_images_directory == paths.images_directory / "phone"
    assert paths.calibration_directory.name == "calibration"
    assert paths.metadata_directory.name == "metadata"
    assert paths.feature_maps_directory.name == "feature_maps"
    with pytest.raises(FileExistsError):
        create_participant_paths(tmp_path, "p01")


@pytest.mark.parametrize("invalid_name", ("", "../은제", "은제/승연", ".숨김"))
def test_participant_name_rejects_empty_or_unsafe_paths(invalid_name):
    with pytest.raises(ValueError):
        normalize_participant_id(invalid_name)


def test_korean_participant_name_is_used_as_directory(tmp_path):
    paths = create_participant_paths(tmp_path, " 안은제 ", "neutral")

    assert paths.participant_id == "안은제"
    assert paths.head_pose == "neutral"
    assert paths.participant_root_directory == tmp_path / "안은제"
    assert paths.participant_directory == tmp_path / "안은제" / "neutral"
    assert paths.labels_directory.is_dir()


def test_same_participant_can_have_three_controlled_pose_recordings(tmp_path):
    paths = {
        pose: create_participant_paths(tmp_path, "안은제", pose)
        for pose in ("neutral", "head_up", "head_down")
    }

    assert set(path.participant_id for path in paths.values()) == {"안은제"}
    assert {path.participant_directory.name for path in paths.values()} == {
        "neutral",
        "head_up",
        "head_down",
    }
    with pytest.raises(FileExistsError):
        create_participant_paths(tmp_path, "안은제", "neutral")


def test_head_pose_aliases_are_normalized_and_unknown_pose_is_rejected():
    assert normalize_head_pose("middle") == "neutral"
    assert normalize_head_pose("위") == "head_up"
    assert normalize_head_pose("head-down") == "head_down"
    with pytest.raises(ValueError):
        normalize_head_pose("left")


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
    with result.frame_log_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert tuple(rows[0]) == LABEL_COLUMNS
    assert rows[0]["participant"] == "p00"
    display_timestamps = [int(row["display_timestamp"]) for row in rows]
    assert display_timestamps[0] == config.simulation.base_unix_timestamp_ns
    assert display_timestamps == sorted(display_timestamps)
    assert all(
        int(row["webcam_timestamp"]) >= int(row["display_timestamp"]) for row in rows
    )
    assert int(rows[0]["webcam_timestamp"]) > 0
    assert "time_diff_ms" not in rows[0]
    assert rows[0]["pair"] == "0"
    assert "latency_ms" not in rows[0]
    assert set(row["split"] for row in rows) == {"train"}


def test_multiple_protocols_share_one_video_and_frame_log(tmp_path):
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
    assert (
        results[0].frame_log_path
        == results[1].frame_log_path
        == paths.metadata_directory / "frame_log.csv"
    )
    assert len(list(paths.webcam_directory.glob("*.mp4"))) == 1
    assert len(list(paths.phone_directory.glob("*.mp4"))) == 1
    with results[0].frame_log_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["split"] for row in rows} == {"train", "validation"}


def test_confirmed_static_simulation_writes_paired_images_and_manifest(tmp_path):
    config = load_capture_config(Path("configs/capture.yaml"))
    config = replace(
        config,
        recording=replace(config.recording, enabled=False),
        simulation=replace(config.simulation, fps=10, width=64, height=48),
    )
    paths = create_participant_paths(tmp_path, "p00")
    plan = ProtocolPlan(
        protocol_id="click_static",
        split="train",
        initial_delay_ms=0,
        completion_duration_ms=0,
        segments=(ProtocolSegment(0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1400, "static"),),
        settling_duration_ms=400,
        usable_duration_ms=650,
        show_guide_line=False,
        confirmation_required=True,
        confirmation_transition_ms=150,
        simulation_confirm_after_ms=600,
    )

    result = run_simulation_protocols(config, paths, (plan,))[0]

    assert result.status == "completed"
    with (paths.labels_directory / "labels.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        samples = list(csv.DictReader(file))
    assert len(samples) == 1
    assert tuple(samples[0]) == IMAGE_SAMPLE_COLUMNS
    assert samples[0]["sample"] == "s000000"
    assert samples[0]["candidate_count"] == "7"
    assert all(row["protocol"] == "click_static" for row in samples)
    assert all(row["split"] == "train" for row in samples)
    assert len(list(paths.webcam_images_directory.glob("*.jpg"))) == 1
    assert len(list(paths.phone_images_directory.glob("*.jpg"))) == 1

    with result.frame_log_path.open(encoding="utf-8", newline="") as file:
        labels = list(csv.DictReader(file))
    selected_label = next(
        row for row in labels if row["pair"] == samples[0]["pair"]
    )
    assert selected_label["usable"] == "1"
    assert selected_label["confirmation_timestamp"]
    with (paths.metadata_directory / "protocol.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        event_rows = list(csv.DictReader(file))
    assert tuple(event_rows[0]) == (
        "participant",
        "head_pose",
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


def test_quality_scorer_prefers_sharp_well_exposed_frame():
    config = load_capture_config(Path("configs/capture.yaml"))
    scorer = FrameQualityScorer(config.frame_capture)
    flat = np.full((160, 240, 3), 128, dtype=np.uint8)
    checker = np.indices((160, 240)).sum(axis=0) % 2
    sharp = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)

    flat_quality = scorer.score(flat)
    sharp_quality = scorer.score(sharp)

    assert sharp_quality.sharpness > flat_quality.sharpness
    assert sharp_quality.score > flat_quality.score


class _FakeLandmarkExtractor:
    def __init__(self, results):
        self._results = iter(results)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def extract(self, _frame):
        return next(self._results)


def test_sample_writer_prioritizes_webcam_pair_ready_for_mediapipe(tmp_path):
    config = load_capture_config(Path("configs/capture.yaml"))
    paths = create_participant_paths(tmp_path, "p00")
    ready = FaceIrisLandmarks(True, True, ())
    missing = FaceIrisLandmarks(False, False, ())
    extractors = {
        "webcam": _FakeLandmarkExtractor((missing, ready)),
    }
    state = ProtocolFrameState(
        protocol_id="test",
        split="train",
        phase="target",
        segment_index=0,
        repeat_index=0,
        target_index=0,
        target_x=0.5,
        target_y=0.5,
        direction="static",
        is_settling=False,
        is_usable_window=True,
        is_training_sample=True,
        show_guide_line=False,
        label_latency_ms=0.0,
        confirmation_required=True,
        is_confirmed=True,
        confirmation_timestamp_ns=1_700_000_000_000_000_000,
        confirmation_offset_ms=500.0,
    )
    frame = np.full((80, 120, 3), 128, dtype=np.uint8)
    writer = FrameSampleWriter(
        paths,
        config.frame_capture,
        config.display,
        landmark_extractor_factory=lambda camera: extractors[camera],
    )
    try:
        for pair in (1, 2):
            packet = FramePacket(pair, float(pair), 1_700_000_000_000_000_000 + pair, frame)
            writer.observe(pair, packet.unix_timestamp_ns, packet, packet, state)
    finally:
        writer.close()

    with (paths.labels_directory / "labels.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        sample = next(csv.DictReader(file))
    assert sample["pair"] == "2"
    assert sample["web_mediapipe_iris_detected"] == "1"
    assert sample["phone_mediapipe_iris_detected"] == "0"


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


def test_recorder_uses_measured_timestamps_without_changing_frame_indices(tmp_path):
    video_path = tmp_path / "capture.mp4"
    timestamps_path = tmp_path / "timestamps.csv"
    recorder = SessionRecorder(video_path, timestamps_path, "mp4v", 30.0, "measured")
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    base_ns = 1_700_000_000_000_000_000
    for index in range(101):
        recorder.write(
            FramePacket(index, index * 100.0, base_ns + index * 100_000_000, frame)
        )
    recorder.close()

    capture = cv2.VideoCapture(str(video_path))
    try:
        encoded_fps = capture.get(cv2.CAP_PROP_FPS)
        encoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    assert encoded_frames == 101
    assert encoded_fps == pytest.approx(10.0, abs=0.1)
    with timestamps_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [int(row["frame"]) for row in rows] == list(range(101))
    with (tmp_path / "recording.json").open(encoding="utf-8") as file:
        summary = json.load(file)
    assert summary["configured_fps"] == 30.0
    assert summary["measured_fps"] == pytest.approx(10.0)
    assert summary["frame_count"] == 101


def test_frozen_frame_guard_rejects_stalled_camera_and_resets_on_change():
    guard = FrozenFrameGuard("phonecam", max_identical_frames=3)
    still = np.zeros((8, 8, 3), dtype=np.uint8)
    changed = still.copy()
    changed[0, 0, 0] = 1

    guard.observe(still)
    guard.observe(still)
    guard.observe(changed)
    guard.observe(still)
    guard.observe(still)
    guard.observe(still)
    with pytest.raises(RuntimeError, match="Camera phonecam appears frozen"):
        guard.observe(still)
