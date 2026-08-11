import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ggulnote_ml.exceptions import ContractError
from ggulnote_ml.synchronization.matching import SYNC_COLUMNS
from ggulnote_ml.video_preprocessing.contracts import (
    IMAGE_PIPELINE_REQUIRED_COLUMNS,
    VIDEO_FEATURE_NAMES,
)
from ggulnote_ml.video_preprocessing.mediapipe_landmarks import (
    FaceIrisLandmarks,
    NormalizedLandmark,
)
from ggulnote_ml.video_preprocessing.pipeline import run_video_preprocessing


class _FakeCapture:
    def __init__(self, frames):
        self._frames = frames
        self._index = 0
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame.copy()

    def release(self):
        self._opened = False


class _FakeCv2:
    def __init__(self, videos):
        self._videos = videos

    def VideoCapture(self, path):
        return _FakeCapture(self._videos[Path(path).resolve()])

    @staticmethod
    def imwrite(path, frame):
        Path(path).write_bytes(frame.tobytes())
        return True


class _FakeLandmarkExtractor:
    def __init__(self, results):
        self._results = results
        self._index = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def extract(self, frame):
        result = self._results[self._index]
        self._index += 1
        return result


def _synchronized_row(pair, split, training, valid=True):
    row = {column: "" for column in SYNC_COLUMNS}
    base = 1_700_000_000_000_000_000 + pair * 10_000_000
    row.update(
        {
            "participant": "p00",
            "pair": pair,
            "webcam_frame": pair,
            "phonecam_frame": pair,
            "webcam_timestamp": base + 100,
            "phonecam_timestamp": base + 200,
            "webcam_latency_ms": "100.0",
            "phonecam_latency_ms": "120.0",
            "webcam_corrected_timestamp": base,
            "phonecam_corrected_timestamp": base,
            "corrected_time_diff_ms": "0.0",
            "reference_timestamp": base,
            "target_timestamp": base - 1_000,
            "x_norm": "0.500000",
            "y_norm": "0.250000",
            "x_centered": "0.000000",
            "y_centered": "-0.250000",
            "protocol": "static_grid",
            "split": split,
            "segment": 0,
            "repeat": 0,
            "target": pair,
            "direction": "",
            "settling": 0,
            "usable": 1,
            "training": int(training),
            "target_interpolated": 0,
            "valid_sync": int(valid),
            "valid_target": int(valid),
            "valid": int(valid),
            "invalid_reason": "" if valid else "time_diff_exceeded",
        }
    )
    return row


def _build_source_dataset(tmp_path):
    dataset_root = tmp_path / "raw" / "participants"
    participant = dataset_root / "p00"
    (participant / "synchronized").mkdir(parents=True)
    (participant / "webcam").mkdir()
    (participant / "phonecam").mkdir()
    (participant / "labels").mkdir()
    (participant / "webcam" / "capture.mp4").write_bytes(b"webcam")
    (participant / "phonecam" / "capture.mp4").write_bytes(b"phonecam")
    (participant / "labels" / "labels.csv").write_text(
        "raw-label-sentinel", encoding="utf-8"
    )
    (participant / "participant.json").write_text(
        json.dumps(
            {
                "screen": {
                    "canvas_width_pixel": 1920,
                    "canvas_height_pixel": 1080,
                }
            }
        ),
        encoding="utf-8",
    )
    synchronized = participant / "synchronized" / "synchronized_frames.csv"
    with synchronized.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SYNC_COLUMNS)
        writer.writeheader()
        writer.writerow(_synchronized_row(0, "training", True))
        writer.writerow(_synchronized_row(1, "evaluation", False))
        writer.writerow(_synchronized_row(2, "training", True, valid=False))
    return dataset_root, participant, synchronized


def _fake_cv2(participant):
    frames = [
        np.full((2, 3, 3), value, dtype=np.uint8) for value in (10, 20, 30)
    ]
    return _FakeCv2(
        {
            (participant / "webcam" / "capture.mp4").resolve(): frames,
            (participant / "phonecam" / "capture.mp4").resolve(): frames,
        }
    )


def _detected_landmarks():
    landmarks = [NormalizedLandmark(0.5, 0.5, 0.0) for _ in range(478)]
    points = {
        362: (0.40, 0.50),
        385: (0.44, 0.47),
        387: (0.52, 0.47),
        263: (0.60, 0.50),
        373: (0.52, 0.53),
        380: (0.44, 0.53),
        33: (0.10, 0.50),
        160: (0.14, 0.47),
        158: (0.22, 0.47),
        133: (0.30, 0.50),
        153: (0.22, 0.53),
        144: (0.14, 0.53),
        473: (0.50, 0.50),
        468: (0.20, 0.50),
    }
    for index, (x, y) in points.items():
        landmarks[index] = NormalizedLandmark(x, y, 0.0)
    return FaceIrisLandmarks(True, True, tuple(landmarks))


def _landmark_extractor_factory(phone_results=None):
    detected = _detected_landmarks()
    missing = FaceIrisLandmarks(False, False, ())
    by_camera = {
        "webcam": (detected, detected),
        "phonecam": phone_results or (detected, missing),
    }
    return lambda camera: _FakeLandmarkExtractor(by_camera[camera])


def _read_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_video_preprocessing_exports_separate_main_compatible_manifests(tmp_path):
    dataset_root, participant, synchronized = _build_source_dataset(tmp_path)
    output_root = tmp_path / "interim" / "dual_view"
    original_sync = synchronized.read_bytes()

    result = run_video_preprocessing(
        "p00",
        dataset_root,
        output_root,
        cv2_module=_fake_cv2(participant),
        landmark_extractor_factory=_landmark_extractor_factory(),
    )

    training_rows = _read_rows(result.training_manifest)
    evaluation_rows = _read_rows(result.evaluation_manifest)
    assert len(training_rows) == 2
    assert len(evaluation_rows) == 2
    assert IMAGE_PIPELINE_REQUIRED_COLUMNS <= set(training_rows[0])
    assert {row["view"] for row in training_rows} == {"webcam", "phonecam"}
    assert {row["pair_id"] for row in training_rows} == {"p00_pair_00000000"}
    assert {row["pair_id"] for row in evaluation_rows} == {"p00_pair_00000001"}
    assert {row["collection_split"] for row in training_rows} == {"training"}
    assert {row["collection_split"] for row in evaluation_rows} == {"evaluation"}
    assert {row["source_frame"] for row in training_rows} == {"0"}
    assert {row["source_frame"] for row in evaluation_rows} == {"1"}
    assert {
        (row["view"], row["source_timestamp"], row["corrected_timestamp"])
        for row in training_rows
    } == {
        ("webcam", "1700000000000000100", "1700000000000000000"),
        ("phonecam", "1700000000000000200", "1700000000000000000"),
    }
    assert {row["target_x_px"] for row in training_rows} == {"960.000000"}
    assert {row["target_y_px"] for row in training_rows} == {"270.000000"}
    for row in training_rows + evaluation_rows:
        assert (output_root / row["image_path"]).is_file()

    webcam_features = _read_rows(result.webcam_features)
    phonecam_features = _read_rows(result.phonecam_features)
    video_training = _read_rows(result.video_training_features)
    video_evaluation = _read_rows(result.video_evaluation_features)
    assert len(webcam_features) == 2
    assert len(phonecam_features) == 2
    assert len(video_training) == 2
    assert len(video_evaluation) == 2
    assert set(VIDEO_FEATURE_NAMES) <= set(video_training[0])
    assert {row["camera"] for row in video_training} == {"webcam", "phonecam"}
    assert {row["collection_split"] for row in video_training} == {"training"}
    assert {row["collection_split"] for row in video_evaluation} == {"evaluation"}
    missing_face = [row for row in video_evaluation if row["camera"] == "phonecam"]
    assert missing_face[0]["face_detected"] == "0"
    assert missing_face[0]["landmark_count"] == "0"
    assert missing_face[0]["feature_valid"] == "0"
    assert missing_face[0]["invalid_reason"] == "face_not_detected"
    assert missing_face[0]["left_eye_center_x"] == ""

    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["training_pairs"] == 1
    assert summary["evaluation_pairs"] == 1
    assert summary["excluded_pairs"] == 1
    assert summary["valid_feature_rows"] == 3
    assert summary["training_feature_pairs"] == 1
    assert summary["face_not_detected_rows"] == 1
    assert summary["video_feature_order"] == list(VIDEO_FEATURE_NAMES)
    assert synchronized.read_bytes() == original_sync
    assert (participant / "labels" / "labels.csv").read_text(
        encoding="utf-8"
    ) == "raw-label-sentinel"


def test_video_preprocessing_refuses_to_overwrite_outputs(tmp_path):
    dataset_root, participant, _ = _build_source_dataset(tmp_path)
    output_root = tmp_path / "interim" / "dual_view"
    cv2_module = _fake_cv2(participant)
    run_video_preprocessing(
        "p00",
        dataset_root,
        output_root,
        cv2_module=cv2_module,
        landmark_extractor_factory=_landmark_extractor_factory(),
    )

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        run_video_preprocessing(
            "p00", dataset_root, output_root, cv2_module=_fake_cv2(participant)
        )


def test_video_training_csv_requires_both_camera_features(tmp_path):
    dataset_root, participant, _ = _build_source_dataset(tmp_path)
    output_root = tmp_path / "interim" / "dual_view"
    detected = _detected_landmarks()
    missing = FaceIrisLandmarks(False, False, ())

    result = run_video_preprocessing(
        "p00",
        dataset_root,
        output_root,
        cv2_module=_fake_cv2(participant),
        landmark_extractor_factory=_landmark_extractor_factory(
            phone_results=(missing, detected)
        ),
    )

    assert _read_rows(result.video_training_features) == []
    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["training_feature_pairs"] == 0


def test_video_preprocessing_refuses_output_inside_raw_dataset(tmp_path):
    dataset_root, participant, _ = _build_source_dataset(tmp_path)

    with pytest.raises(ContractError, match="must not be inside raw dataset_root"):
        run_video_preprocessing(
            "p00",
            dataset_root,
            dataset_root / "processed",
            cv2_module=_fake_cv2(participant),
        )
