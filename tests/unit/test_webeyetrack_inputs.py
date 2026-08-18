import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.io import savemat

from ggulnote_ml.video_preprocessing.mediapipe_landmarks import (
    FaceIrisLandmarks,
    NormalizedLandmark,
)
from ggulnote_ml.video_preprocessing.webeyetrack_inputs import (
    WEBEYETRACK_INPUT_COLUMNS,
    create_eye_patch,
    discover_recordings,
    estimate_metric_head_pose,
    load_webeyetrack_preprocessing_config,
    run_webeyetrack_input_preprocessing,
)
from ggulnote_ml.exceptions import ContractError


CONFIG_PATH = Path("configs/webeyetrack_preprocessing.yaml")


class _FakeExtractor:
    def __init__(self, result):
        self._result = result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def extract(self, frame):
        return self._result


def _synthetic_landmarks(config, width=640, height=480):
    camera_matrix = np.asarray(
        [[800.0, 0.0, width / 2], [0.0, 800.0, height / 2], [0.0, 0.0, 1.0]]
    )
    rotation_cv = np.diag([1.0, -1.0, -1.0])
    rotation_vector, _ = cv2.Rodrigues(rotation_cv)
    translation = np.asarray([[0.0], [0.0], [55.0]])
    projected, _ = cv2.projectPoints(
        config.metric_head_pose.canonical_points_cm,
        rotation_vector,
        translation,
        camera_matrix,
        np.zeros((1, 5)),
    )
    landmarks = [NormalizedLandmark(0.5, 0.5, 0.0) for _ in range(478)]
    for index, point in zip(
        config.metric_head_pose.landmark_ids, projected.reshape(-1, 2)
    ):
        landmarks[index] = NormalizedLandmark(
            float(point[0] / width), float(point[1] / height), 0.0
        )

    def open_eye(indices):
        first = landmarks[indices[0]]
        fourth = landmarks[indices[3]]
        x_mid = (first.x + fourth.x) / 2.0
        y_mid = (first.y + fourth.y) / 2.0
        horizontal = abs(first.x - fourth.x)
        vertical = max(horizontal * 0.35, 0.015)
        landmarks[indices[1]] = NormalizedLandmark(
            x_mid - horizontal * 0.15, y_mid - vertical, 0.0
        )
        landmarks[indices[2]] = NormalizedLandmark(
            x_mid + horizontal * 0.15, y_mid - vertical, 0.0
        )
        landmarks[indices[4]] = NormalizedLandmark(
            x_mid + horizontal * 0.15, y_mid + vertical, 0.0
        )
        landmarks[indices[5]] = NormalizedLandmark(
            x_mid - horizontal * 0.15, y_mid + vertical, 0.0
        )

    open_eye((362, 385, 387, 263, 373, 380))
    open_eye((33, 160, 158, 133, 153, 144))
    landmarks[103] = NormalizedLandmark(0.28, 0.22, 0.0)
    landmarks[150] = NormalizedLandmark(0.30, 0.78, 0.0)
    landmarks[379] = NormalizedLandmark(0.70, 0.78, 0.0)
    landmarks[332] = NormalizedLandmark(0.72, 0.22, 0.0)
    landmarks[151] = NormalizedLandmark(0.50, 0.37, 0.0)
    landmarks[195] = NormalizedLandmark(0.50, 0.62, 0.0)
    return tuple(landmarks), camera_matrix


def _write_manifest(path, participant, head_pose, split, image_path, pair_id):
    columns = (
        "sample_id",
        "subject_id",
        "head_pose",
        "view",
        "image_path",
        "pair_id",
        "target_x_px",
        "target_y_px",
        "screen_width_px",
        "screen_height_px",
        "collection_split",
        "protocol",
        "source_frame",
        "source_timestamp",
        "corrected_timestamp",
        "reference_timestamp",
        "target_timestamp",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "%s_webcam" % pair_id,
                "subject_id": participant,
                "head_pose": head_pose,
                "view": "webcam",
                "image_path": image_path,
                "pair_id": pair_id,
                "target_x_px": 735,
                "target_y_px": 239,
                "screen_width_px": 1470,
                "screen_height_px": 956,
                "collection_split": split,
                "protocol": "static",
                "source_frame": 1,
                "source_timestamp": 1,
                "corrected_timestamp": 1,
                "reference_timestamp": 1,
                "target_timestamp": 1,
            }
        )


def test_metric_pose_and_eye_patch_match_model_contract():
    config = load_webeyetrack_preprocessing_config(CONFIG_PATH)
    landmarks, camera_matrix = _synthetic_landmarks(config)
    pose = estimate_metric_head_pose(
        landmarks,
        640,
        480,
        camera_matrix,
        config.metric_head_pose,
        cv2,
    )
    assert pose.head_vector.shape == (3,)
    np.testing.assert_allclose(pose.head_vector, [0.0, 0.0, -1.0], atol=1e-5)
    assert 45.0 < pose.face_origin_3d[2] < 60.0
    assert pose.reprojection_error_px < 1e-4
    frame = np.full((480, 640, 3), 127, dtype=np.uint8)
    patch = create_eye_patch(frame, landmarks, config.eye_patch, cv2)
    assert patch.shape == (128, 512, 3)
    assert patch.dtype == np.uint8


def test_complete_recording_generates_training_ready_webeyetrack_inputs(tmp_path):
    config = load_webeyetrack_preprocessing_config(CONFIG_PATH)
    landmarks, camera_matrix = _synthetic_landmarks(config)
    dataset_root = tmp_path / "participants"
    recording = dataset_root / "안은제" / "neutral"
    feature_maps = recording / "feature_maps"
    frames = feature_maps / "web" / "frames"
    frames.mkdir(parents=True)
    calibration = recording / "calibration" / "web"
    calibration.mkdir(parents=True)
    savemat(
        calibration / "Camera.mat",
        {
            "cameraMatrix": camera_matrix,
            "distCoeffs": np.zeros((1, 5)),
            "retval": np.asarray([[0.2]]),
            "image_width": np.asarray([[640]]),
            "image_height": np.asarray([[480]]),
        },
    )
    frame = np.full((480, 640, 3), 127, dtype=np.uint8)
    for name in ("train.png", "evaluation.png"):
        assert cv2.imwrite(str(frames / name), frame)
    _write_manifest(
        feature_maps / "training.csv",
        "안은제",
        "neutral",
        "training",
        "web/frames/train.png",
        "train_pair",
    )
    _write_manifest(
        feature_maps / "evaluation.csv",
        "안은제",
        "neutral",
        "evaluation",
        "web/frames/evaluation.png",
        "evaluation_pair",
    )
    result = run_webeyetrack_input_preprocessing(
        participant_id="안은제",
        head_pose="neutral",
        dataset_root=dataset_root,
        config_path=CONFIG_PATH,
        landmark_extractor_factory=lambda: _FakeExtractor(
            FaceIrisLandmarks(True, True, landmarks)
        ),
    )
    assert result.total_rows == 2
    assert result.valid_rows == 2
    assert result.training_rows == 1
    assert result.evaluation_rows == 1
    with result.all_inputs_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert tuple(rows[0]) == WEBEYETRACK_INPUT_COLUMNS
    assert {row["valid"] for row in rows} == {"1"}
    assert {row["target_x_centered"] for row in rows} == {"0.00000000"}
    assert all((result.output_directory / row["eye_patch_path"]).is_file() for row in rows)
    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["eye_patch_contract"]["shape"] == [128, 512, 3]
    assert summary["valid_rows"] == 2
    assert summary["quality_gate"]["splits"]["training"]["valid_fraction"] == 1.0
    assert discover_recordings(dataset_root) == (("안은제", "neutral"),)


def test_low_valid_fraction_rejects_partial_output_and_preserves_source(tmp_path):
    config = load_webeyetrack_preprocessing_config(CONFIG_PATH)
    _, camera_matrix = _synthetic_landmarks(config)
    dataset_root = tmp_path / "participants"
    recording = dataset_root / "안은제" / "neutral"
    feature_maps = recording / "feature_maps"
    frames = feature_maps / "web" / "frames"
    frames.mkdir(parents=True)
    calibration = recording / "calibration" / "web"
    calibration.mkdir(parents=True)
    savemat(
        calibration / "Camera.mat",
        {
            "cameraMatrix": camera_matrix,
            "distCoeffs": np.zeros((1, 5)),
            "retval": np.asarray([[0.2]]),
            "image_width": np.asarray([[640]]),
            "image_height": np.asarray([[480]]),
        },
    )
    frame_path = frames / "frame.png"
    assert cv2.imwrite(str(frame_path), np.full((480, 640, 3), 127, dtype=np.uint8))
    for split in ("training", "evaluation"):
        _write_manifest(
            feature_maps / ("%s.csv" % split),
            "안은제",
            "neutral",
            split,
            "web/frames/frame.png",
            "%s_pair" % split,
        )
    missing_face = FaceIrisLandmarks(False, False, ())
    try:
        run_webeyetrack_input_preprocessing(
            participant_id="안은제",
            head_pose="neutral",
            dataset_root=dataset_root,
            config_path=CONFIG_PATH,
            landmark_extractor_factory=lambda: _FakeExtractor(missing_face),
        )
    except ContractError as exc:
        assert "valid fraction" in str(exc)
    else:
        raise AssertionError("Low-quality input should be rejected.")
    assert frame_path.is_file()
    assert not (feature_maps / "webeyetrack").exists()
    assert not list(feature_maps.glob(".webeyetrack.partial-*"))
