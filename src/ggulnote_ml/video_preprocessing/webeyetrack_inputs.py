from __future__ import annotations

import csv
import json
import os
import shutil
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from ggulnote_ml.capture.calibration_assets import load_camera_intrinsics
from ggulnote_ml.capture.dataset import (
    normalize_head_pose,
    normalize_participant_id,
    participant_recording_directory,
)
from ggulnote_ml.exceptions import ContractError, OptionalDependencyError
from ggulnote_ml.video_preprocessing.eye_state import calculate_eye_state
from ggulnote_ml.video_preprocessing.mediapipe_landmarks import (
    FaceIrisLandmarks,
    MediaPipeFaceIrisExtractor,
    NormalizedLandmark,
)


WEBEYETRACK_INPUT_COLUMNS = (
    "schema_version",
    "sample_id",
    "participant",
    "head_pose",
    "pair_id",
    "collection_split",
    "protocol",
    "source_image_path",
    "eye_patch_path",
    "target_x_centered",
    "target_y_centered",
    "head_vector_x",
    "head_vector_y",
    "head_vector_z",
    "face_origin_x_cm",
    "face_origin_y_cm",
    "face_origin_z_cm",
    "pose_reprojection_error_px",
    "quality_score",
    "left_ear",
    "right_ear",
    "valid",
    "invalid_reason",
)

REQUIRED_MANIFEST_COLUMNS = frozenset(
    {
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
    }
)


@dataclass(frozen=True)
class EyePatchConfig:
    width: int
    height: int
    face_crop_size: int
    padding_x: float
    padding_y: float
    image_format: str


@dataclass(frozen=True)
class MetricHeadPoseConfig:
    method: str
    landmark_ids: Tuple[int, ...]
    canonical_points_cm: np.ndarray
    left_eye_corner_ids: Tuple[int, int]
    right_eye_corner_ids: Tuple[int, int]
    min_depth_cm: float
    max_depth_cm: float
    max_mean_reprojection_error_px: float


@dataclass(frozen=True)
class WebEyeTrackPreprocessingConfig:
    schema_version: str
    camera: str
    ear_threshold: float
    eye_patch: EyePatchConfig
    metric_head_pose: MetricHeadPoseConfig
    output_directory_name: str
    keep_invalid_rows: bool
    minimum_valid_fraction_per_split: float


@dataclass(frozen=True)
class MetricHeadPose:
    head_vector: np.ndarray
    face_origin_3d: np.ndarray
    reprojection_error_px: float
    quality_score: float


@dataclass(frozen=True)
class WebEyeTrackPreprocessingResult:
    participant: str
    head_pose: str
    output_directory: Path
    all_inputs_csv: Path
    training_inputs_csv: Path
    evaluation_inputs_csv: Path
    summary_json: Path
    total_rows: int
    valid_rows: int
    training_rows: int
    evaluation_rows: int


def load_webeyetrack_preprocessing_config(
    path: Path,
) -> WebEyeTrackPreprocessingConfig:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("WebEyeTrack preprocessing config does not exist: %s" % resolved)
    with resolved.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, Mapping):
        raise ValueError("WebEyeTrack preprocessing config must be a mapping.")
    source = _mapping(raw, "source")
    eye = _mapping(raw, "eye_patch")
    pose = _mapping(raw, "metric_head_pose")
    output = _mapping(raw, "output")
    landmark_ids = tuple(int(value) for value in pose["landmark_ids"])
    canonical_points = np.asarray(pose["canonical_points_cm"], dtype=np.float64)
    if canonical_points.shape != (len(landmark_ids), 3):
        raise ValueError(
            "metric_head_pose canonical_points_cm must have shape [%d,3]."
            % len(landmark_ids)
        )
    if len(landmark_ids) < 6 or len(set(landmark_ids)) != len(landmark_ids):
        raise ValueError("metric_head_pose requires at least six unique landmark_ids.")
    left_eye = tuple(int(value) for value in pose["left_eye_corner_ids"])
    right_eye = tuple(int(value) for value in pose["right_eye_corner_ids"])
    if len(left_eye) != 2 or len(right_eye) != 2:
        raise ValueError("Each eye must contain exactly two horizontal corner IDs.")
    config = WebEyeTrackPreprocessingConfig(
        schema_version=str(raw["schema_version"]),
        camera=str(source["camera"]).strip().lower(),
        ear_threshold=float(source["ear_threshold"]),
        eye_patch=EyePatchConfig(
            width=int(eye["width"]),
            height=int(eye["height"]),
            face_crop_size=int(eye["face_crop_size"]),
            padding_x=float(eye["padding_x"]),
            padding_y=float(eye["padding_y"]),
            image_format=str(eye["image_format"]).strip().lower(),
        ),
        metric_head_pose=MetricHeadPoseConfig(
            method=str(pose["method"]),
            landmark_ids=landmark_ids,
            canonical_points_cm=canonical_points,
            left_eye_corner_ids=(left_eye[0], left_eye[1]),
            right_eye_corner_ids=(right_eye[0], right_eye[1]),
            min_depth_cm=float(pose["min_depth_cm"]),
            max_depth_cm=float(pose["max_depth_cm"]),
            max_mean_reprojection_error_px=float(
                pose["max_mean_reprojection_error_px"]
            ),
        ),
        output_directory_name=str(output["directory_name"]).strip(),
        keep_invalid_rows=bool(output["keep_invalid_rows"]),
        minimum_valid_fraction_per_split=float(
            output["minimum_valid_fraction_per_split"]
        ),
    )
    _validate_config(config)
    return config


def create_eye_patch(
    bgr_frame: np.ndarray,
    landmarks: Sequence[NormalizedLandmark],
    config: EyePatchConfig,
    cv2_module: Any,
) -> np.ndarray:
    """Create the public WebEyeTrack-style two-eye homography crop."""

    if bgr_frame.dtype != np.uint8 or bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
        raise ContractError("Eye patch input must be uint8 BGR [H,W,3].")
    required = (4, 103, 150, 151, 195, 332, 379)
    if len(landmarks) <= max(required):
        raise ContractError("Eye patch landmarks are incomplete.")
    height, width = bgr_frame.shape[:2]
    points = np.asarray(
        [[item.x * width, item.y * height] for item in landmarks], dtype=np.float64
    )
    source = np.asarray(
        [points[103], points[150], points[379], points[332]], dtype=np.float32
    )
    center = points[4].astype(np.float32)
    source = source + np.asarray(
        [config.padding_x, config.padding_y], dtype=np.float32
    ) * (source - center)
    crop_size = config.face_crop_size
    destination = np.asarray(
        [[0, 0], [0, crop_size], [crop_size, crop_size], [crop_size, 0]],
        dtype=np.float32,
    )
    homography, _ = cv2_module.findHomography(source, destination)
    if homography is None or not np.isfinite(homography).all():
        raise ContractError("Could not compute a finite eye-patch homography.")
    warped = cv2_module.warpPerspective(
        bgr_frame, homography, (crop_size, crop_size)
    )
    homogeneous = np.column_stack((points[:, :2], np.ones(len(points)))).T
    transformed = homography @ homogeneous
    denominators = transformed[2]
    if np.any(np.abs(denominators) < 1e-8):
        raise ContractError("Eye-patch homography produced points at infinity.")
    transformed = (transformed[:2] / denominators).T
    top = max(0, int(round(transformed[151, 1])))
    bottom = min(crop_size, int(round(transformed[195, 1])))
    if bottom - top < 2:
        raise ContractError("Eye-patch crop has an invalid vertical extent.")
    eye_region = warped[top:bottom, :]
    patch = cv2_module.resize(eye_region, (config.width, config.height))
    if patch.shape != (config.height, config.width, 3) or patch.dtype != np.uint8:
        raise ContractError("Eye patch must be uint8 [128,512,3].")
    return np.ascontiguousarray(patch)


def estimate_metric_head_pose(
    landmarks: Sequence[NormalizedLandmark],
    image_width: int,
    image_height: int,
    camera_matrix: np.ndarray,
    config: MetricHeadPoseConfig,
    cv2_module: Any,
) -> MetricHeadPose:
    """Estimate WebEyeTrack auxiliary inputs in calibrated camera coordinates."""

    all_required = set(config.landmark_ids)
    all_required.update(config.left_eye_corner_ids)
    all_required.update(config.right_eye_corner_ids)
    if len(landmarks) <= max(all_required):
        raise ContractError("Metric head-pose landmarks are incomplete.")
    image_points = np.asarray(
        [
            [landmarks[index].x * image_width, landmarks[index].y * image_height]
            for index in config.landmark_ids
        ],
        dtype=np.float64,
    )
    if not np.isfinite(image_points).all():
        raise ContractError("Metric head-pose image points are not finite.")
    distortion = np.zeros((1, 5), dtype=np.float64)
    success, rotation_vector, translation = cv2_module.solvePnP(
        config.canonical_points_cm,
        image_points,
        np.asarray(camera_matrix, dtype=np.float64),
        distortion,
        flags=cv2_module.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise ContractError("OpenCV solvePnP could not estimate head pose.")
    rotation_cv, _ = cv2_module.Rodrigues(rotation_vector)
    projected, _ = cv2_module.projectPoints(
        config.canonical_points_cm,
        rotation_vector,
        translation,
        camera_matrix,
        distortion,
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    mean_error = float(np.mean(errors))
    # OpenCV and MediaPipe use opposite Y/Z axes for a neutral canonical face.
    # This fixed proper rotation converts solvePnP R into the convention used by
    # WebEyeTrack's get_head_vector routine.
    neutral_correction = np.diag([1.0, -1.0, -1.0])
    rotation_head = rotation_cv @ neutral_correction
    head_vector = _webeyetrack_head_vector(rotation_head)
    point_by_id = {
        landmark_id: point
        for landmark_id, point in zip(
            config.landmark_ids, config.canonical_points_cm
        )
    }
    try:
        left_origin = np.mean(
            [point_by_id[index] for index in config.left_eye_corner_ids], axis=0
        )
        right_origin = np.mean(
            [point_by_id[index] for index in config.right_eye_corner_ids], axis=0
        )
    except KeyError as exc:
        raise ContractError(
            "Eye corner IDs must also be included in metric_head_pose.landmark_ids."
        ) from exc
    canonical_face_origin = (left_origin + right_origin) / 2.0
    face_origin = rotation_cv @ canonical_face_origin + translation.reshape(3)
    depth = float(face_origin[2])
    values = np.concatenate((head_vector, face_origin, [mean_error]))
    if not np.isfinite(values).all():
        raise ContractError("Metric head-pose output contains NaN or infinity.")
    if not config.min_depth_cm <= depth <= config.max_depth_cm:
        raise ContractError(
            "Estimated face depth %.3f cm is outside [%.3f, %.3f] cm."
            % (depth, config.min_depth_cm, config.max_depth_cm)
        )
    if mean_error > config.max_mean_reprojection_error_px:
        raise ContractError(
            "Head-pose reprojection error %.3f px exceeds %.3f px."
            % (mean_error, config.max_mean_reprojection_error_px)
        )
    quality = max(
        0.0,
        min(1.0, 1.0 - mean_error / config.max_mean_reprojection_error_px),
    )
    return MetricHeadPose(
        head_vector=head_vector.astype(np.float32),
        face_origin_3d=face_origin.astype(np.float32),
        reprojection_error_px=mean_error,
        quality_score=quality,
    )


def run_webeyetrack_input_preprocessing(
    participant_id: str,
    head_pose: str,
    dataset_root: Path = Path("data/raw/participants"),
    config_path: Path = Path("configs/webeyetrack_preprocessing.yaml"),
    output_root: Optional[Path] = None,
    force: bool = False,
    cv2_module: Any = None,
    landmark_extractor_factory: Any = None,
) -> WebEyeTrackPreprocessingResult:
    config = load_webeyetrack_preprocessing_config(config_path)
    participant = normalize_participant_id(participant_id)
    normalized_pose = normalize_head_pose(head_pose)
    raw_root = dataset_root.expanduser().resolve()
    recording = participant_recording_directory(
        raw_root, participant, normalized_pose
    )
    feature_maps = recording / "feature_maps"
    if not feature_maps.is_dir():
        raise FileNotFoundError(
            "Run synchronization and video preprocessing first: %s" % feature_maps
        )
    if output_root is None:
        destination = feature_maps / config.output_directory_name
    else:
        destination = (
            output_root.expanduser().resolve()
            / participant
            / normalized_pose
            / config.output_directory_name
        )
    if destination.exists() and not force:
        raise FileExistsError(
            "WebEyeTrack inputs already exist; use --force to rebuild: %s"
            % destination
        )
    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise OptionalDependencyError(
                "WebEyeTrack preprocessing requires: pip install -e '.[video,landmarks]'."
            ) from exc
    camera_directory = "web" if config.camera == "webcam" else "phone"
    intrinsics = load_camera_intrinsics(
        recording / "calibration" / camera_directory / "Camera.mat"
    )
    manifest_rows = _load_front_manifest_rows(
        feature_maps,
        config.camera,
        participant,
        normalized_pose,
    )
    temporary = destination.parent / (
        ".%s.partial-%s" % (destination.name, uuid.uuid4().hex)
    )
    temporary.mkdir(parents=True, exist_ok=False)
    eye_patch_directory = temporary / "eye_roi"
    eye_patch_directory.mkdir()
    if landmark_extractor_factory is None:
        landmark_extractor_factory = MediaPipeFaceIrisExtractor
    output_rows = []
    try:
        with ExitStack() as stack:
            extractor = stack.enter_context(landmark_extractor_factory())
            for manifest in manifest_rows:
                output_rows.append(
                    _process_manifest_row(
                        manifest,
                        feature_maps,
                        eye_patch_directory,
                        config,
                        intrinsics.camera_matrix,
                        extractor,
                        cv2_module,
                    )
                )
        all_rows = (
            output_rows
            if config.keep_invalid_rows
            else [row for row in output_rows if row["valid"] == 1]
        )
        training_rows = [
            row
            for row in output_rows
            if row["valid"] == 1 and row["collection_split"] == "training"
        ]
        evaluation_rows = [
            row
            for row in output_rows
            if row["valid"] == 1 and row["collection_split"] == "evaluation"
        ]
        split_quality = _validate_split_quality(
            output_rows,
            minimum_valid_fraction=config.minimum_valid_fraction_per_split,
        )
        _write_rows(temporary / "inputs.csv", all_rows)
        _write_rows(temporary / "training.csv", training_rows)
        _write_rows(temporary / "evaluation.csv", evaluation_rows)
        valid_rows = [row for row in output_rows if row["valid"] == 1]
        distances = [float(row["face_origin_z_cm"]) for row in valid_rows]
        errors = [float(row["pose_reprojection_error_px"]) for row in valid_rows]
        summary = {
            "schema_version": config.schema_version,
            "participant": participant,
            "head_pose": normalized_pose,
            "source_camera": config.camera,
            "source_feature_maps": str(feature_maps),
            "camera_matrix_path": str(intrinsics.path),
            "eye_patch_contract": {
                "shape": [config.eye_patch.height, config.eye_patch.width, 3],
                "stored_color_order": "standard PNG; decode as RGB for the model",
                "model_dtype": "float32",
                "model_range": [0.0, 1.0],
            },
            "model_input_contract": {
                "head_vector": "float32[3] unit vector",
                "face_origin_3d": "float32[3] camera coordinates in cm",
                "target": "float32[2] screen-centered normalized [-0.5,0.5]",
            },
            "metric_head_pose_method": config.metric_head_pose.method,
            "quality_gate": {
                "minimum_valid_fraction_per_split": (
                    config.minimum_valid_fraction_per_split
                ),
                "splits": split_quality,
            },
            "total_rows": len(output_rows),
            "valid_rows": len(valid_rows),
            "invalid_rows": len(output_rows) - len(valid_rows),
            "training_rows": len(training_rows),
            "evaluation_rows": len(evaluation_rows),
            "face_depth_cm": _range_summary(distances),
            "pose_reprojection_error_px": _range_summary(errors),
            "invalid_reasons": _reason_counts(output_rows),
        }
        with (temporary / "summary.json").open("x", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        _replace_directory_atomically(temporary, destination, force)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return WebEyeTrackPreprocessingResult(
        participant=participant,
        head_pose=normalized_pose,
        output_directory=destination,
        all_inputs_csv=destination / "inputs.csv",
        training_inputs_csv=destination / "training.csv",
        evaluation_inputs_csv=destination / "evaluation.csv",
        summary_json=destination / "summary.json",
        total_rows=len(output_rows),
        valid_rows=sum(row["valid"] == 1 for row in output_rows),
        training_rows=sum(
            row["valid"] == 1 and row["collection_split"] == "training"
            for row in output_rows
        ),
        evaluation_rows=sum(
            row["valid"] == 1 and row["collection_split"] == "evaluation"
            for row in output_rows
        ),
    )


def discover_recordings(dataset_root: Path) -> Tuple[Tuple[str, str], ...]:
    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        return ()
    recordings = []
    for participant_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        for pose_directory in sorted(
            path for path in participant_directory.iterdir() if path.is_dir()
        ):
            if (
                (pose_directory / "feature_maps" / "training.csv").is_file()
                and (pose_directory / "feature_maps" / "evaluation.csv").is_file()
            ):
                recordings.append((participant_directory.name, pose_directory.name))
    return tuple(recordings)


def _process_manifest_row(
    manifest: Mapping[str, str],
    feature_maps: Path,
    eye_patch_directory: Path,
    config: WebEyeTrackPreprocessingConfig,
    camera_matrix: np.ndarray,
    extractor: Any,
    cv2_module: Any,
) -> Dict[str, object]:
    source_path = feature_maps / manifest["image_path"]
    frame = cv2_module.imread(str(source_path))
    base = _base_output_row(manifest, config, source_path)
    if frame is None:
        return _invalid_row(base, "source_image_unreadable")
    if frame.shape[:2] != (
        int(manifest.get("source_height_px") or frame.shape[0]),
        int(manifest.get("source_width_px") or frame.shape[1]),
    ):
        return _invalid_row(base, "source_image_shape_mismatch")
    result: FaceIrisLandmarks = extractor.extract(frame)
    if not result.face_detected:
        return _invalid_row(base, "face_not_detected")
    if not result.iris_detected:
        return _invalid_row(base, "iris_not_detected")
    try:
        eye_state = calculate_eye_state(result.landmarks, config.ear_threshold)
    except ContractError:
        return _invalid_row(base, "invalid_landmark_geometry")
    base["left_ear"] = "%.8f" % eye_state.left_ear
    base["right_ear"] = "%.8f" % eye_state.right_ear
    if eye_state.eye_closed:
        return _invalid_row(base, "eye_closed")
    try:
        pose = estimate_metric_head_pose(
            result.landmarks,
            frame.shape[1],
            frame.shape[0],
            camera_matrix,
            config.metric_head_pose,
            cv2_module,
        )
    except ContractError:
        return _invalid_row(base, "metric_head_pose_invalid")
    try:
        patch = create_eye_patch(frame, result.landmarks, config.eye_patch, cv2_module)
    except ContractError:
        return _invalid_row(base, "eye_patch_invalid")
    filename = "%s.%s" % (manifest["pair_id"], config.eye_patch.image_format)
    patch_path = eye_patch_directory / filename
    if not cv2_module.imwrite(str(patch_path), patch):
        return _invalid_row(base, "eye_patch_write_failed")
    base.update(
        {
            "eye_patch_path": "eye_roi/%s" % filename,
            "head_vector_x": "%.8f" % pose.head_vector[0],
            "head_vector_y": "%.8f" % pose.head_vector[1],
            "head_vector_z": "%.8f" % pose.head_vector[2],
            "face_origin_x_cm": "%.8f" % pose.face_origin_3d[0],
            "face_origin_y_cm": "%.8f" % pose.face_origin_3d[1],
            "face_origin_z_cm": "%.8f" % pose.face_origin_3d[2],
            "pose_reprojection_error_px": "%.8f" % pose.reprojection_error_px,
            "quality_score": "%.8f" % pose.quality_score,
            "valid": 1,
            "invalid_reason": "",
        }
    )
    return base


def _base_output_row(
    manifest: Mapping[str, str],
    config: WebEyeTrackPreprocessingConfig,
    source_path: Path,
) -> Dict[str, object]:
    width = float(manifest["screen_width_px"])
    height = float(manifest["screen_height_px"])
    if width <= 0 or height <= 0:
        raise ContractError("Screen dimensions must be positive.")
    target_x = float(manifest["target_x_px"]) / width - 0.5
    target_y = float(manifest["target_y_px"]) / height - 0.5
    if not all(isfinite(value) and -0.5 <= value <= 0.5 for value in (target_x, target_y)):
        raise ContractError("Centered gaze target must be finite in [-0.5,0.5].")
    return {
        "schema_version": config.schema_version,
        "sample_id": manifest["sample_id"],
        "participant": manifest["subject_id"],
        "head_pose": manifest["head_pose"],
        "pair_id": manifest["pair_id"],
        "collection_split": manifest["collection_split"],
        "protocol": manifest["protocol"],
        "source_image_path": str(source_path),
        "eye_patch_path": "",
        "target_x_centered": "%.8f" % target_x,
        "target_y_centered": "%.8f" % target_y,
        "head_vector_x": "",
        "head_vector_y": "",
        "head_vector_z": "",
        "face_origin_x_cm": "",
        "face_origin_y_cm": "",
        "face_origin_z_cm": "",
        "pose_reprojection_error_px": "",
        "quality_score": "0.00000000",
        "left_ear": "",
        "right_ear": "",
        "valid": 0,
        "invalid_reason": "",
    }


def _invalid_row(base: Dict[str, object], reason: str) -> Dict[str, object]:
    base["valid"] = 0
    base["invalid_reason"] = reason
    return base


def _load_front_manifest_rows(
    feature_maps: Path,
    camera: str,
    participant: str,
    head_pose: str,
) -> Tuple[Dict[str, str], ...]:
    expected_view = "webcam" if camera == "webcam" else "phonecam"
    rows = []
    for filename in ("training.csv", "evaluation.csv"):
        path = feature_maps / filename
        if not path.is_file():
            raise FileNotFoundError("Image manifest does not exist: %s" % path)
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise ContractError(
                    "%s is missing columns: %s" % (path, ", ".join(sorted(missing)))
                )
            for row in reader:
                if row["view"] != expected_view:
                    continue
                if row["subject_id"] != participant or row["head_pose"] != head_pose:
                    raise ContractError(
                        "Manifest participant/head_pose does not match requested recording."
                    )
                rows.append(dict(row))
    if not rows:
        raise ContractError("No %s image rows were found." % expected_view)
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ContractError("WebEyeTrack source sample_id values must be unique.")
    return tuple(rows)


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=WEBEYETRACK_INPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _webeyetrack_head_vector(rotation: np.ndarray) -> np.ndarray:
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    yaw = np.arctan2(rotation[2, 1], rotation[2, 2])
    roll = np.arctan2(rotation[1, 0], rotation[0, 0])
    head_pitch = -yaw
    head_yaw = pitch
    z = -np.cos(head_pitch) * np.cos(head_yaw)
    x = np.cos(head_pitch) * np.sin(head_yaw)
    y = np.sin(head_pitch)
    vector = np.asarray([x, y, z], dtype=np.float64)
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    vector = np.asarray(
        [
            cos_roll * vector[0] - sin_roll * vector[1],
            sin_roll * vector[0] + cos_roll * vector[1],
            vector[2],
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    if not isfinite(norm) or norm <= 0:
        raise ContractError("Head vector is not a finite unit direction.")
    return vector / norm


def _replace_directory_atomically(temporary: Path, destination: Path, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        os.replace(temporary, destination)
        return
    if not force:
        raise FileExistsError("Output already exists: %s" % destination)
    backup = destination.parent / (
        ".%s.previous-%s" % (destination.name, uuid.uuid4().hex)
    )
    os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except Exception:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup)


def _validate_config(config: WebEyeTrackPreprocessingConfig) -> None:
    if config.camera not in {"webcam", "phonecam"}:
        raise ValueError("source.camera must be webcam or phonecam.")
    if not 0.0 < config.ear_threshold < 1.0:
        raise ValueError("source.ear_threshold must be in (0,1).")
    eye = config.eye_patch
    if min(eye.width, eye.height, eye.face_crop_size) <= 0:
        raise ValueError("Eye-patch dimensions must be positive.")
    if (eye.width, eye.height) != (512, 128):
        raise ValueError("webeyetrack_input_v1 requires a 512x128 eye patch.")
    if eye.image_format not in {"png", "jpg", "jpeg"}:
        raise ValueError("eye_patch.image_format must be png, jpg, or jpeg.")
    pose = config.metric_head_pose
    if not np.isfinite(pose.canonical_points_cm).all():
        raise ValueError("Canonical face points must be finite.")
    if not 0 < pose.min_depth_cm < pose.max_depth_cm:
        raise ValueError("Metric head-pose depth bounds are invalid.")
    if pose.max_mean_reprojection_error_px <= 0:
        raise ValueError("Maximum pose reprojection error must be positive.")
    if not config.output_directory_name:
        raise ValueError("output.directory_name must not be empty.")
    if not 0.0 < config.minimum_valid_fraction_per_split <= 1.0:
        raise ValueError(
            "output.minimum_valid_fraction_per_split must be in (0,1]."
        )


def _mapping(parent: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = parent.get(name)
    if not isinstance(value, Mapping):
        raise ValueError("Config section %s must be a mapping." % name)
    return value


def _range_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "median": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _validate_split_quality(
    rows: Sequence[Mapping[str, object]],
    minimum_valid_fraction: float,
) -> Dict[str, Dict[str, float]]:
    quality: Dict[str, Dict[str, float]] = {}
    for split in ("training", "evaluation"):
        split_rows = [row for row in rows if row["collection_split"] == split]
        if not split_rows:
            raise ContractError("WebEyeTrack input has no %s samples." % split)
        valid_count = sum(row["valid"] == 1 for row in split_rows)
        fraction = valid_count / len(split_rows)
        quality[split] = {
            "total_rows": len(split_rows),
            "valid_rows": valid_count,
            "valid_fraction": fraction,
        }
        if fraction < minimum_valid_fraction:
            raise ContractError(
                "%s valid fraction %.3f is below the configured minimum %.3f. "
                "Raw recording and earlier feature maps were preserved."
                % (split, fraction, minimum_valid_fraction)
            )
    return quality


def _reason_counts(rows: Iterable[Mapping[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        reason = str(row.get("invalid_reason") or "")
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    return counts
