import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from ggulnote_ml.capture.frame_samples import IMAGE_SAMPLE_COLUMNS
from ggulnote_ml.exceptions import ContractError
from ggulnote_ml.synchronization.matching import SYNC_COLUMNS
from ggulnote_ml.video_preprocessing.contracts import (
    DUAL_VIEW_MANIFEST_COLUMNS,
    IMAGE_PIPELINE_REQUIRED_COLUMNS,
    PROCESSED_FEATURE_COLUMNS,
    VIDEO_FEATURE_NAMES,
)
from ggulnote_ml.video_preprocessing.mediapipe_landmarks import (
    FaceIrisLandmarks,
    NormalizedLandmark,
)
from ggulnote_ml.video_preprocessing.loader import load_selected_samples
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
        self.undistort_calls = []

    def VideoCapture(self, path):
        return _FakeCapture(self._videos[Path(path).resolve()])

    @staticmethod
    def imwrite(path, frame):
        Path(path).write_bytes(frame.tobytes())
        return True

    def undistort(self, frame, camera_matrix, distortion, _new_matrix, output_matrix):
        self.undistort_calls.append(
            (frame.shape, camera_matrix.copy(), distortion.copy(), output_matrix.copy())
        )
        return frame.copy()


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


def _synchronized_row(
    pair,
    split,
    valid=True,
    participant="p00",
    head_pose="",
):
    row = {column: "" for column in SYNC_COLUMNS}
    base = 1_700_000_000_000_000_000 + pair * 10_000_000
    row.update(
        {
            "participant": participant,
            "head_pose": head_pose,
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
            "segment": pair,
            "target": pair,
            "direction": "static",
            "usable": 1,
            "target_interpolated": 0,
            "valid_sync": int(valid),
            "valid_target": int(valid),
            "valid": int(valid),
            "invalid_reason": "" if valid else "time_diff_exceeded",
        }
    )
    return row


def _image_sample_row(
    sample,
    pair,
    split,
    participant="p00",
    head_pose="",
):
    row = {column: "" for column in IMAGE_SAMPLE_COLUMNS}
    base = 1_700_000_000_000_000_000 + pair * 10_000_000
    row.update(
        {
            "sample": sample,
            "participant": participant,
            "head_pose": head_pose,
            "protocol": "static_grid",
            "split": split,
            "pair": pair,
            "display_timestamp": base - 1_000,
            "confirmation_timestamp": base - 500,
            "confirmation_offset_ms": "600.000",
            "candidate_count": 5,
            "pair_quality_score": "100.000000",
            "web_image": "images/web/%s.jpg" % sample,
            "web_frame": pair,
            "web_timestamp": base + 100,
            "web_face_detected": 1,
            "web_eyes_detected": 2,
            "web_sharpness": "100.000000",
            "web_brightness": "128.000000",
            "phone_image": "images/phone/%s.jpg" % sample,
            "phone_frame": pair,
            "phone_timestamp": base + 200,
            "phone_face_detected": 1,
            "phone_eyes_detected": 2,
            "phone_sharpness": "100.000000",
            "phone_brightness": "128.000000",
            "x_px": 960,
            "y_px": 270,
            "x_norm": "0.500000",
            "y_norm": "0.250000",
            "x_centered": "0.000000",
            "y_centered": "-0.250000",
            "segment": pair,
            "target": pair,
            "direction": "static",
        }
    )
    return row


def _build_source_dataset(tmp_path, participant_id="p00", head_pose=""):
    dataset_root = tmp_path / "raw" / "participants"
    participant = dataset_root / participant_id
    if head_pose:
        participant = participant / head_pose
    (participant / "feature_maps").mkdir(parents=True)
    (participant / "video" / "web").mkdir(parents=True)
    (participant / "video" / "phone").mkdir(parents=True)
    (participant / "labels").mkdir()
    for camera in ("web", "phone"):
        calibration = participant / "calibration" / camera
        calibration.mkdir(parents=True)
        savemat(
            calibration / "Camera.mat",
            {
                "cameraMatrix": np.array(
                    [[100.0, 0.0, 1.0], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]]
                ),
                "distCoeffs": np.zeros((1, 5)),
                "retval": np.array([[0.2]]),
                "image_width": np.array([[3]]),
                "image_height": np.array([[2]]),
            },
        )
    (participant / "video" / "web" / "capture.mp4").write_bytes(b"webcam")
    (participant / "video" / "phone" / "capture.mp4").write_bytes(b"phonecam")
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
    synchronized = participant / "feature_maps" / "synchronized.csv"
    with synchronized.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SYNC_COLUMNS)
        writer.writeheader()
        writer.writerow(
            _synchronized_row(0, "train", participant=participant_id, head_pose=head_pose)
        )
        writer.writerow(
            _synchronized_row(
                1,
                "evaluation",
                participant=participant_id,
                head_pose=head_pose,
            )
        )
        writer.writerow(
            _synchronized_row(
                2,
                "train",
                valid=False,
                participant=participant_id,
                head_pose=head_pose,
            )
        )
    image_samples = participant / "labels" / "labels.csv"
    with image_samples.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=IMAGE_SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerow(
            _image_sample_row(
                "s000000",
                0,
                "train",
                participant=participant_id,
                head_pose=head_pose,
            )
        )
        writer.writerow(
            _image_sample_row(
                "s000001",
                1,
                "evaluation",
                participant=participant_id,
                head_pose=head_pose,
            )
        )
    return dataset_root, participant, synchronized


def test_selected_samples_accepts_legacy_pair_frame_column(tmp_path):
    path = tmp_path / "image_samples.csv"
    columns = tuple(
        "pair_frame" if column == "pair" else column
        for column in IMAGE_SAMPLE_COLUMNS
    )
    row = _image_sample_row("s000000", 7, "train")
    row["pair_frame"] = row.pop("pair")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)

    samples = load_selected_samples(path)

    assert len(samples) == 1
    assert samples[0].source_pair == 7


def _fake_cv2(participant):
    frames = [
        np.full((2, 3, 3), value, dtype=np.uint8) for value in (10, 20, 30)
    ]
    return _FakeCv2(
        {
            (participant / "video" / "web" / "capture.mp4").resolve(): frames,
            (participant / "video" / "phone" / "capture.mp4").resolve(): frames,
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

    fake_cv2 = _fake_cv2(participant)
    result = run_video_preprocessing(
        "p00",
        dataset_root,
        output_root,
        cv2_module=fake_cv2,
        landmark_extractor_factory=_landmark_extractor_factory(),
    )

    training_rows = _read_rows(result.training_manifest)
    evaluation_rows = _read_rows(result.evaluation_manifest)
    assert len(training_rows) == 2
    assert len(evaluation_rows) == 2
    assert tuple(training_rows[0]) == DUAL_VIEW_MANIFEST_COLUMNS
    assert tuple(evaluation_rows[0]) == DUAL_VIEW_MANIFEST_COLUMNS
    assert IMAGE_PIPELINE_REQUIRED_COLUMNS <= set(training_rows[0])
    assert {row["view"] for row in training_rows} == {"webcam", "phonecam"}
    assert {row["pair_id"] for row in training_rows} == {"p00_s000000"}
    assert {row["pair_id"] for row in evaluation_rows} == {"p00_s000001"}
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
        assert (result.output_root / row["image_path"]).is_file()
        assert "session_id" not in row

    webcam_features = _read_rows(result.webcam_features)
    phonecam_features = _read_rows(result.phonecam_features)
    video_training = _read_rows(result.video_training_features)
    video_evaluation = _read_rows(result.video_evaluation_features)
    assert len(webcam_features) == 2
    assert phonecam_features == []
    assert len(video_training) == 1
    assert len(video_evaluation) == 1
    assert tuple(webcam_features[0]) == PROCESSED_FEATURE_COLUMNS
    assert set(VIDEO_FEATURE_NAMES) <= set(video_training[0])
    assert {row["camera"] for row in video_training} == {"webcam"}
    assert {row["collection_split"] for row in video_training} == {"training"}
    assert {row["collection_split"] for row in video_evaluation} == {"evaluation"}
    assert all(row["intrinsics_applied"] == "1" for row in video_training + video_evaluation)
    assert {
        row["camera_matrix_path"] for row in video_training + video_evaluation
    } == {"calibration/web/Camera.mat"}
    assert len(fake_cv2.undistort_calls) == 4
    assert all("training" not in row for row in video_training + video_evaluation)
    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["training_pairs"] == 1
    assert summary["evaluation_pairs"] == 1
    assert summary["selected_pairs"] == 2
    assert summary["unselected_synchronized_pairs"] == 1
    assert summary["valid_feature_rows"] == 2
    assert summary["training_feature_pairs"] == 1
    assert summary["face_not_detected_rows"] == 0
    assert summary["mediapipe_feature_cameras"] == ["webcam"]
    assert summary["video_feature_order"] == list(VIDEO_FEATURE_NAMES)
    assert summary["intrinsics_mode"] == "required"
    assert summary["intrinsics"]["webcam"]["applied"]
    assert summary["intrinsics"]["phonecam"]["rms_error_px"] == pytest.approx(0.2)
    assert synchronized.read_bytes() == original_sync
    assert len(_read_rows(participant / "labels" / "labels.csv")) == 2


def test_video_preprocessing_preserves_named_participant_and_head_pose(tmp_path):
    dataset_root, participant, _ = _build_source_dataset(
        tmp_path,
        participant_id="안은제",
        head_pose="neutral",
    )

    result = run_video_preprocessing(
        "안은제",
        dataset_root,
        head_pose="neutral",
        cv2_module=_fake_cv2(participant),
        landmark_extractor_factory=_landmark_extractor_factory(),
    )

    training_rows = _read_rows(result.training_manifest)
    feature_rows = _read_rows(result.webcam_features)
    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert result.output_root == participant / "feature_maps"
    assert {row["subject_id"] for row in training_rows} == {"안은제"}
    assert {row["head_pose"] for row in training_rows} == {"neutral"}
    assert {row["pair_id"] for row in training_rows} == {
        "안은제_neutral_s000000"
    }
    assert {row["head_pose"] for row in feature_rows} == {"neutral"}
    assert summary["participant"] == "안은제"
    assert summary["head_pose"] == "neutral"


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


def test_video_training_csv_requires_all_configured_feature_cameras(tmp_path):
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
        feature_cameras=("webcam", "phonecam"),
    )

    assert _read_rows(result.video_training_features) == []
    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["training_feature_pairs"] == 0


def test_video_preprocessing_refuses_output_inside_raw_dataset(tmp_path):
    dataset_root, participant, _ = _build_source_dataset(tmp_path)

    with pytest.raises(ContractError, match="must be outside raw dataset_root"):
        run_video_preprocessing(
            "p00",
            dataset_root,
            dataset_root / "processed",
            cv2_module=_fake_cv2(participant),
        )


def test_video_preprocessing_requires_camera_mat_by_default(tmp_path):
    dataset_root, participant, _ = _build_source_dataset(tmp_path)
    (participant / "calibration" / "phone" / "Camera.mat").unlink()

    with pytest.raises(FileNotFoundError, match="Camera calibration does not exist"):
        run_video_preprocessing(
            "p00",
            dataset_root,
            tmp_path / "interim" / "dual_view",
            cv2_module=_fake_cv2(participant),
        )
