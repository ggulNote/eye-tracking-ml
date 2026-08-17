from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .calibration_assets import (
    FULL_CALIBRATION_ASSET_PATHS,
    INTRINSICS_ASSET_PATHS,
    copy_calibration_assets,
    inspect_calibration_assets,
    load_camera_intrinsics,
)
from .backup import (
    backup_participant_recording,
    confirm_local_deletion,
    delete_verified_local_recording,
)
from .camera import probe_camera_indices
from .collection import (
    run_camera_preflight,
    run_display_check,
    run_real_protocols,
    run_simulation_protocols,
)
from .config import CaptureConfig, load_capture_config
from .dataset import (
    create_participant_paths,
    load_optional_metadata,
    normalize_head_pose,
    normalize_participant_id,
    participant_recording_directory,
    suggest_next_participant_id,
    write_participant_json,
)
from .protocols import build_protocol_plans, select_protocols
from ggulnote_ml.synchronization.calibration import (
    copy_config_snapshots as copy_latency_config_snapshots,
    create_latency_paths,
    run_latency_simulation,
    run_real_latency_measurement,
)
from ggulnote_ml.synchronization.config import load_latency_config
from ggulnote_ml.synchronization.matching import (
    synchronize_labels,
    write_synchronization_summary,
)
from ggulnote_ml.video_preprocessing.pipeline import run_video_preprocessing
from ggulnote_ml.video_preprocessing.mediapipe_landmarks import FaceIrisLandmarks
from ggulnote_ml.video_preprocessing.webeyetrack_inputs import (
    run_webeyetrack_input_preprocessing,
)


class _SimulationFaceIrisExtractor:
    """Headless extractor used only to validate simulated output contracts."""

    def __enter__(self) -> "_SimulationFaceIrisExtractor":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None

    def extract(self, _frame: object) -> FaceIrisLandmarks:
        return FaceIrisLandmarks(False, False, ())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect synchronized webcam and left-iPhone gaze protocol data."
    )
    parser.add_argument("--config", default="configs/capture.yaml")
    parser.add_argument("--latency-config", default="configs/latency.yaml")
    parser.add_argument("--participant", help="Participant name, for example 안은제")
    parser.add_argument(
        "--head-pose",
        help="Controlled recording condition: neutral, head_up, or head_down",
    )
    parser.add_argument(
        "--protocol",
        action="append",
        default=[],
        help="Protocol id to run; repeat it or use all (default: full participant-confirmed dot test)",
    )
    parser.add_argument("--simulate", action="store_true", help="Create videos and CSVs without cameras")
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help=(
            "Skip the latency protocol and run only the dot test. Use this only "
            "when the same participant/head-pose directory already has a valid "
            "calibration/latency.json."
        ),
    )
    parser.add_argument("--participant-metadata", type=Path, help="Optional participant metadata JSON")
    parser.add_argument("--dataset-root", type=Path, help="Override dataset root (useful for simulation/CI)")
    parser.add_argument(
        "--backup-root",
        type=Path,
        help=(
            "After successful postprocessing, copy and SHA-256 verify this pose "
            "under an existing iCloud shared folder, then ask before local deletion."
        ),
    )
    parser.add_argument(
        "--allow-missing-calibration",
        action="store_true",
        help="Allow an uncalibrated hardware smoke test and record it in participant.json",
    )
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument(
        "--check-cameras",
        action="store_true",
        help="Preview both cameras and require MediaPipe face+iris readiness without saving data",
    )
    parser.add_argument(
        "--check-display",
        action="store_true",
        help="Show the edge-to-edge dot-test canvas without cameras or saved participant data",
    )
    parser.add_argument("--max-devices", type=int, default=10)
    parser.add_argument("--suggest-participant", action="store_true")
    return parser.parse_args(argv)


def _participant_from_args(value: Optional[str]) -> str:
    if value is not None:
        return normalize_participant_id(value)
    entered = input("이름 적어주세요: ").strip()
    return normalize_participant_id(entered)


def _head_pose_from_args(value: Optional[str]) -> str:
    if value is not None:
        return normalize_head_pose(value)
    entered = input(
        "headpose를 입력해주세요 [neutral/head_up/head_down]: "
    ).strip()
    return normalize_head_pose(entered)


def _apply_dataset_override(
    config: CaptureConfig, dataset_root: Optional[Path]
) -> CaptureConfig:
    if dataset_root is None:
        return config
    return replace(
        config,
        dataset=replace(config.dataset, root_directory=dataset_root.expanduser().resolve()),
    )


def _run_latency_before_collection(
    args: argparse.Namespace,
    config: CaptureConfig,
    participant_id: str,
    head_pose: str,
) -> Path:
    """Measure one latency calibration for the exact participant/pose recording.

    This function intentionally runs before any dot-test output is created.  A
    failed or aborted latency protocol therefore prevents collection from
    continuing with missing or stale timing correction.
    """

    capture_config_path = Path(args.config).expanduser().resolve()
    latency_config_path = Path(args.latency_config).expanduser().resolve()
    latency_config = load_latency_config(latency_config_path)
    paths = create_latency_paths(
        config.dataset.root_directory,
        participant_id,
        head_pose=head_pose,
    )
    copy_latency_config_snapshots(
        paths,
        capture_config_path,
        latency_config_path,
    )
    result = (
        run_latency_simulation(latency_config, paths)
        if args.simulate
        else run_real_latency_measurement(config, latency_config, paths)
    )

    print("Latency calibration saved: %s" % paths.final_json)
    cameras = result["cameras"]
    for camera in ("webcam", "phonecam"):
        summary = cameras[camera]
        print(
            "[%s] median=%.3f ms, MAD=%.3f ms, p95=%.3f ms, valid=%d/%d"
            % (
                camera,
                summary["median_ms"],
                summary["mad_ms"],
                summary["p95_ms"],
                summary["valid_events"],
                summary["total_events"],
            )
        )
    return paths.final_json


def _load_collection_status(participant_directory: Path) -> str:
    metadata_path = participant_directory / "participant.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Collection metadata does not exist after capture: %s" % metadata_path
        )
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    status = str(metadata.get("status", "")).strip().lower()
    if status not in {"completed", "aborted", "failed", "running"}:
        raise ValueError("Unknown collection status in %s: %s" % (metadata_path, status))
    return status


def _run_postprocessing_after_collection(
    args: argparse.Namespace,
    config: CaptureConfig,
    participant_id: str,
    head_pose: str,
) -> Path:
    """Create latency-corrected pairs and front-camera MediaPipe features."""

    participant_directory = participant_recording_directory(
        config.dataset.root_directory,
        participant_id,
        head_pose,
    )
    feature_maps_directory = participant_directory / "feature_maps"
    synchronized_path = feature_maps_directory / "synchronized.csv"
    synchronization_summary_path = feature_maps_directory / "synchronization.json"
    latency_path = participant_directory / "calibration" / "latency.json"
    frame_log_path = participant_directory / "metadata" / "frame_log.csv"
    latency_config = load_latency_config(
        Path(args.latency_config).expanduser().resolve()
    )

    synchronization_summary = synchronize_labels(
        frame_log_path,
        latency_path,
        synchronized_path,
        latency_config.matching,
        expected_head_pose=head_pose,
    )
    write_synchronization_summary(
        synchronization_summary_path,
        synchronization_summary,
    )
    print("Synchronized frames saved: %s" % synchronized_path)
    print(
        "Valid synchronized pairs: %d/%d"
        % (
            synchronization_summary["valid_pairs"],
            synchronization_summary["total_pairs"],
        )
    )

    print("[4/5] 정면 MediaPipe 특징 추출을 시작합니다.")
    # Synthetic frames do not share the real Camera.mat resolution.  Real
    # intrinsics-based collections always require and apply both camera files.
    intrinsics_mode = (
        "off"
        if args.simulate or config.dataset.geometry_mode == "fixed_rig_2d"
        else "required"
    )
    preprocessing_outputs = (
        feature_maps_directory / "training.csv",
        feature_maps_directory / "evaluation.csv",
        feature_maps_directory / "training_features.csv",
        feature_maps_directory / "evaluation_features.csv",
        feature_maps_directory / "summary.json",
        feature_maps_directory / "web" / "features.csv",
        feature_maps_directory / "phone" / "features.csv",
        feature_maps_directory / "web" / "frames",
        feature_maps_directory / "phone" / "frames",
    )
    had_preprocessing_outputs = any(path.exists() for path in preprocessing_outputs)
    try:
        preprocessing = run_video_preprocessing(
            participant_id=participant_id,
            dataset_root=config.dataset.root_directory,
            head_pose=head_pose,
            ear_threshold=config.postprocessing.ear_threshold,
            intrinsics_mode=intrinsics_mode,
            feature_cameras=config.postprocessing.feature_cameras,
            landmark_extractor_factory=(
                (lambda _camera: _SimulationFaceIrisExtractor())
                if args.simulate
                else None
            ),
        )
    except Exception:
        # A newly started postprocessing run may fail while initializing an
        # optional dependency. Remove only outputs created by this failed run
        # so the original video, labels, and synchronization can be retried.
        if not had_preprocessing_outputs:
            for path in preprocessing_outputs:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
        raise
    print("Postprocessing summary: %s" % preprocessing.summary_json)
    if not config.postprocessing.webeyetrack_enabled:
        print("설정에 따라 WebEyeTrack 입력 생성을 건너뜁니다.")
        return preprocessing.summary_json
    if args.simulate:
        print("simulation에서는 실제 얼굴 3D 기하 계산을 건너뜁니다.")
        return preprocessing.summary_json
    print("[5/5] WebEyeTrack 눈 ROI와 3D head pose를 생성합니다.")
    webeyetrack = run_webeyetrack_input_preprocessing(
        participant_id=participant_id,
        head_pose=head_pose,
        dataset_root=config.dataset.root_directory,
        config_path=config.postprocessing.webeyetrack_config,
    )
    print(
        "WebEyeTrack inputs: valid=%d/%d, training=%d, evaluation=%d"
        % (
            webeyetrack.valid_rows,
            webeyetrack.total_rows,
            webeyetrack.training_rows,
            webeyetrack.evaluation_rows,
        )
    )
    print("WebEyeTrack summary: %s" % webeyetrack.summary_json)
    return webeyetrack.summary_json


def run_collection(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).expanduser().resolve()
    config = _apply_dataset_override(load_capture_config(config_path), args.dataset_root)
    participant_id = _participant_from_args(args.participant)
    head_pose = _head_pose_from_args(args.head_pose)
    calibration_source = config.dataset.calibration_source_directory
    calibration = inspect_calibration_assets(calibration_source)
    calibration_override = bool(args.allow_missing_calibration and not args.simulate)
    if config.dataset.geometry_mode == "intrinsics_2d":
        calibration_ready = calibration["camera_intrinsics_valid"]
    elif config.dataset.geometry_mode == "calibrated_3d":
        calibration_ready = calibration["required_assets_valid"]
    else:
        calibration_ready = True
    if (
        config.dataset.require_calibration_assets
        and not args.simulate
        and not calibration_override
        and not calibration_ready
    ):
        raise ValueError(
            "Required calibration assets are missing or invalid under %s. "
            "Use fixed_rig_2d only for deliberate collection without lens correction."
            % calibration_source
        )
    uses_calibration = config.dataset.geometry_mode != "fixed_rig_2d"
    if not args.simulate and uses_calibration and calibration_ready:
        camera_directories = {
            "webcam_front": "webcam",
            "iphone_left": "phonecam",
        }
        for camera in config.cameras:
            directory = camera_directories[camera.role]
            intrinsics = load_camera_intrinsics(
                calibration_source / directory / "Camera.mat"
            )
            if (intrinsics.image_width, intrinsics.image_height) != (
                camera.width,
                camera.height,
            ):
                raise ValueError(
                    "%s Camera.mat size %dx%d does not match capture size %dx%d."
                    % (
                        directory,
                        intrinsics.image_width,
                        intrinsics.image_height,
                        camera.width,
                        camera.height,
                    )
                )

    paths = create_participant_paths(
        config.dataset.root_directory,
        participant_id,
        head_pose,
    )
    calibration_copied = bool(
        not args.simulate and uses_calibration and calibration_ready
    )
    if calibration_copied:
        calibration_paths = (
            INTRINSICS_ASSET_PATHS
            if config.dataset.geometry_mode == "intrinsics_2d"
            else FULL_CALIBRATION_ASSET_PATHS
        )
        copy_calibration_assets(
            calibration_source,
            paths.calibration_directory,
            calibration_paths,
            camera_directory_aliases={"webcam": "web", "phonecam": "phone"},
        )
        calibration = inspect_calibration_assets(paths.calibration_directory)

    plans = select_protocols(build_protocol_plans(config.protocols), args.protocol or ["all"])
    participant_metadata = load_optional_metadata(args.participant_metadata)
    config_snapshot = paths.metadata_directory / "capture_config.yaml"
    shutil.copyfile(config_path, config_snapshot)

    started_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": 7,
        "csv_schema_version": 4,
        "participant_id": participant_id,
        "head_pose": head_pose,
        "recording_id": head_pose,
        "status": "running",
        "mode": "simulation" if args.simulate else "camera",
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "camera_roles": {camera.role: camera.name for camera in config.cameras},
        "camera_config": [
            {
                "name": camera.name,
                "role": camera.role,
                "device_index": camera.device_index,
                "backend": camera.backend,
                "width": camera.width,
                "height": camera.height,
                "fps": camera.fps,
                "max_identical_frames": camera.max_identical_frames,
                "position": "front" if camera.role == "webcam_front" else "participant_left_30_45_deg",
            }
            for camera in config.cameras
        ],
        "screen": {
            "canvas_width_pixel": config.display.canvas_width,
            "canvas_height_pixel": config.display.canvas_height,
            "window_fit": "free_ratio_fullscreen",
        },
        "data_layout": {
            "calibration": "calibration",
            "video_web": "video/web",
            "video_phone": "video/phone",
            "images_web": "images/web",
            "images_phone": "images/phone",
            "labels": "labels/labels.csv",
            "frame_log": "metadata/frame_log.csv",
            "protocol": "metadata/protocol.csv",
            "feature_maps": "feature_maps",
        },
        "protocols": [plan.protocol_id for plan in plans],
        "click_confirmation": {
            "input": "left_mouse_or_space",
            "train_required": config.protocols.train_static.confirmation_required,
            "vertical_required": config.protocols.vertical_click.confirmation_required,
            "evaluation_required": config.protocols.evaluation_static.confirmation_required,
        },
        "frame_capture": {
            "enabled": config.frame_capture.enabled,
            "image_format": config.frame_capture.image_format,
            "web_directory": "images/web",
            "phone_directory": "images/phone",
            "labels": "labels/labels.csv",
            "scope": "one_best_pair_per_confirmed_target",
            "samples_per_target": config.frame_capture.samples_per_target,
            "quality_heuristic": (
                "configured_mediapipe_then_haar_sharpness_exposure"
            ),
            "mediapipe_quality_cameras": list(
                config.frame_capture.mediapipe_quality_cameras
            ),
        },
        "mediapipe_preflight_cameras": list(
            config.preview.landmark_required_cameras
        ),
        "participant_metadata": participant_metadata,
        "geometry": {
            "mode": config.dataset.geometry_mode,
            "camera_intrinsics_available": bool(
                calibration["camera_intrinsics_valid"]
            ),
            "calibration_copied_to_participant": calibration_copied,
            "frame_undistortion_stage": "video_preprocessing",
            "calibration_source": str(calibration_source),
            "camera_position_must_remain_fixed": True,
        },
        "calibration_assets": calibration,
        "missing_calibration_override": calibration_override,
        "config_snapshot": str(config_snapshot.relative_to(paths.participant_directory)),
        "results": [],
    }
    write_participant_json(paths.participant_json, metadata)

    try:
        results = (
            run_simulation_protocols(config, paths, plans)
            if args.simulate
            else run_real_protocols(config, paths, plans)
        )
        metadata["results"] = [
            {
                "protocol_id": result.protocol_id,
                "status": result.status,
                "frame_pairs": result.frame_pairs,
                "frame_log": str(
                    result.frame_log_path.relative_to(paths.participant_directory)
                ),
            }
            for result in results
        ]
        metadata["status"] = (
            "aborted" if any(result.status == "aborted" for result in results) else "completed"
        )
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = "%s: %s" % (type(error).__name__, error)
        raise
    finally:
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_participant_json(paths.participant_json, metadata)

    print("Participant data saved: %s" % paths.participant_directory)
    for result in metadata["results"]:
        print("[%s] %s, %d frame pairs" % (result["protocol_id"], result["status"], result["frame_pairs"]))
    return paths.participant_directory


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        config = _apply_dataset_override(load_capture_config(Path(args.config)), args.dataset_root)
        if args.suggest_participant:
            print(suggest_next_participant_id(config.dataset.root_directory))
            return
        if args.list_cameras:
            if args.max_devices <= 0:
                raise ValueError("--max-devices must be positive.")
            indices = probe_camera_indices(args.max_devices, config.cameras[0].backend)
            print("Available camera indices: %s" % (indices if indices else "none"))
            return
        if args.check_cameras:
            run_camera_preflight(config)
            print("Camera preflight passed for webcam and phonecam.")
            return
        if args.check_display:
            image_rect = run_display_check(config)
            print(
                "Display check closed. Rendered image rectangle: x=%d, y=%d, width=%d, height=%d"
                % image_rect
            )
            return
        participant_id = _participant_from_args(args.participant)
        head_pose = _head_pose_from_args(args.head_pose)
        # Resolve the prompts once and reuse the exact normalized identifiers
        # for both latency calibration and dot-test collection.
        args.participant = participant_id
        args.head_pose = head_pose

        if not args.capture_only:
            print("[1/4] 레이턴시 측정을 시작합니다.")
            _run_latency_before_collection(args, config, participant_id, head_pose)
        else:
            latency_path = (
                config.dataset.root_directory
                / participant_id
                / head_pose
                / "calibration"
                / "latency.json"
            )
            if not latency_path.is_file():
                raise FileNotFoundError(
                    "--capture-only requires an existing latency calibration: %s"
                    % latency_path
                )
            print("기존 레이턴시 보정을 사용합니다: %s" % latency_path)

        print("[2/4] Dot Test 촬영을 시작합니다.")
        participant_directory = run_collection(args)
        collection_status = _load_collection_status(participant_directory)
        if collection_status != "completed":
            print(
                "촬영 상태가 %s이므로 동기화와 MediaPipe 후처리를 실행하지 않습니다."
                % collection_status
            )
            return
        if not config.postprocessing.enabled:
            print("설정에 따라 촬영 후 자동 후처리를 건너뜁니다.")
            return

        print("[3/4] 레이턴시 보정 프레임 동기화를 시작합니다.")
        _run_postprocessing_after_collection(
            args,
            config,
            participant_id,
            head_pose,
        )
        if args.backup_root is not None:
            print("iCloud 공유 폴더로 검증 복사를 시작합니다.")
            backup = backup_participant_recording(
                config.dataset.root_directory,
                args.backup_root,
                participant_id,
                head_pose,
            )
            print(
                "iCloud 복사 완료: %s (%d개 파일, %.2f MB)"
                % (
                    backup.destination,
                    backup.file_count,
                    backup.total_bytes / (1024 * 1024),
                )
            )
            if confirm_local_deletion():
                delete_verified_local_recording(backup)
                print("로컬 데이터 삭제 완료: %s" % backup.source)
            else:
                print("로컬 데이터를 유지합니다: %s" % backup.source)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as error:
        print("[collection error] %s" % error, file=sys.stderr)
        raise SystemExit(1) from error
