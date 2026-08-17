from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


SUPPORTED_BACKENDS = {"auto", "avfoundation", "dshow", "v4l2"}


@dataclass(frozen=True)
class CameraConfig:
    name: str
    role: str
    device_index: int
    backend: str
    width: int
    height: int
    fps: float
    warmup_frames: int
    max_identical_frames: int
    read_retry_count: int
    read_retry_delay_ms: float
    mirror: bool


@dataclass(frozen=True)
class PreviewConfig:
    enabled: bool
    window_name_prefix: str
    preflight_duration_ms: float
    camera_width_px: int
    landmark_required_cameras: Tuple[str, ...]
    required_consecutive_detections: int
    landmark_detection_confidence: float


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool
    video_codec: str
    timing_mode: str


@dataclass(frozen=True)
class FrameCaptureConfig:
    enabled: bool
    image_format: str
    jpeg_quality: int
    samples_per_target: int
    scoring_width_px: int
    ideal_brightness: float
    sharpness_reference: float
    eye_open_weight: float
    face_weight: float
    sharpness_weight: float
    brightness_weight: float
    face_scale_factor: float
    face_min_neighbors: int
    eye_scale_factor: float
    eye_min_neighbors: int
    mediapipe_quality_cameras: Tuple[str, ...]
    mediapipe_ready_weight: float


@dataclass(frozen=True)
class DatasetConfig:
    root_directory: Path
    calibration_source_directory: Path
    geometry_mode: str
    require_calibration_assets: bool


@dataclass(frozen=True)
class PostprocessingConfig:
    enabled: bool
    ear_threshold: float
    feature_cameras: Tuple[str, ...]
    webeyetrack_enabled: bool
    webeyetrack_config: Path


@dataclass(frozen=True)
class DisplayConfig:
    window_name: str
    fullscreen: bool
    canvas_width: int
    canvas_height: int
    point_radius_px: int
    confirmation_ring_radius_px: int
    background_bgr: Tuple[int, int, int]
    point_bgr: Tuple[int, int, int]
    guide_bgr: Tuple[int, int, int]
    confirmation_ring_bgr: Tuple[int, int, int]
    confirmed_ring_bgr: Tuple[int, int, int]


@dataclass(frozen=True)
class StaticProtocolConfig:
    protocol_id: str
    split: str
    columns: int
    rows: int
    repeats: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    confirmation_required: bool
    minimum_fixation_ms: float
    capture_duration_ms: float
    transition_duration_ms: float
    simulation_confirm_after_ms: float
    random_seed: int
    y_positions: Tuple[float, ...]


@dataclass(frozen=True)
class VerticalClickProtocolConfig:
    protocol_id: str
    split: str
    columns: Tuple[float, ...]
    rows: int
    y_min: float
    y_max: float
    confirmation_required: bool
    minimum_fixation_ms: float
    capture_duration_ms: float
    transition_duration_ms: float
    simulation_confirm_after_ms: float


@dataclass(frozen=True)
class ProtocolsConfig:
    intro_center_ms: float
    transition_ms: float
    center_refix_ms: float
    completion_duration_ms: float
    train_static: StaticProtocolConfig
    evaluation_static: StaticProtocolConfig
    vertical_click: VerticalClickProtocolConfig


@dataclass(frozen=True)
class SimulationConfig:
    fps: float
    width: int
    height: int
    phone_delay_ms: float
    base_unix_timestamp_ns: int


@dataclass(frozen=True)
class CaptureConfig:
    project_root: Path
    cameras: Tuple[CameraConfig, ...]
    preview: PreviewConfig
    recording: RecordingConfig
    frame_capture: FrameCaptureConfig
    dataset: DatasetConfig
    postprocessing: PostprocessingConfig
    display: DisplayConfig
    protocols: ProtocolsConfig
    simulation: SimulationConfig


def _required(section: Dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ValueError("Missing capture config value: %s.%s" % (section_name, key))
    return section[key]


def _mapping(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError("%s must be a mapping." % key)
    return value


def _parse_bgr(value: Any, name: str) -> Tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("%s must be a three-item BGR list." % name)
    color = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError("%s channels must be between 0 and 255." % name)
    return color


def _parse_camera_keys(value: Any, name: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("%s must be a list containing webcam and/or phonecam." % name)
    cameras = tuple(str(camera).strip().lower() for camera in value)
    if len(set(cameras)) != len(cameras):
        raise ValueError("%s must not contain duplicate camera names." % name)
    unsupported = set(cameras) - {"webcam", "phonecam"}
    if unsupported:
        raise ValueError(
            "%s contains unsupported cameras: %s"
            % (name, ", ".join(sorted(unsupported)))
        )
    return cameras


def _static_config(raw: Dict[str, Any], name: str) -> StaticProtocolConfig:
    config = StaticProtocolConfig(
        protocol_id=str(_required(raw, "protocol_id", name)),
        split=str(_required(raw, "split", name)),
        columns=int(_required(raw, "columns", name)),
        rows=int(_required(raw, "rows", name)),
        repeats=int(_required(raw, "repeats", name)),
        x_min=float(_required(raw, "x_min", name)),
        x_max=float(_required(raw, "x_max", name)),
        y_min=float(_required(raw, "y_min", name)),
        y_max=float(_required(raw, "y_max", name)),
        confirmation_required=bool(_required(raw, "confirmation_required", name)),
        minimum_fixation_ms=float(_required(raw, "minimum_fixation_ms", name)),
        capture_duration_ms=float(_required(raw, "capture_duration_ms", name)),
        transition_duration_ms=float(_required(raw, "transition_duration_ms", name)),
        simulation_confirm_after_ms=float(
            _required(raw, "simulation_confirm_after_ms", name)
        ),
        random_seed=int(_required(raw, "random_seed", name)),
        y_positions=tuple(float(value) for value in raw.get("y_positions", [])),
    )
    if config.columns <= 0 or config.rows <= 0 or config.repeats <= 0:
        raise ValueError("%s grid dimensions and repeats must be positive." % name)
    if not 0 <= config.x_min < config.x_max <= 1 or not 0 <= config.y_min < config.y_max <= 1:
        raise ValueError("%s coordinate bounds must be ordered within [0, 1]." % name)
    if (
        config.minimum_fixation_ms < 0
        or config.capture_duration_ms <= 0
        or config.transition_duration_ms < 0
    ):
        raise ValueError("%s durations are invalid." % name)
    if not config.minimum_fixation_ms <= config.simulation_confirm_after_ms:
        raise ValueError(
            "%s.simulation_confirm_after_ms must be at least minimum_fixation_ms." % name
        )
    if config.y_positions:
        if len(config.y_positions) != config.rows:
            raise ValueError("%s.y_positions length must equal rows." % name)
        if any(not 0 <= value <= 1 for value in config.y_positions):
            raise ValueError("%s.y_positions must be within [0, 1]." % name)
    return config


def load_capture_config(config_path: Path) -> CaptureConfig:
    resolved = config_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Capture config does not exist: %s" % resolved)
    with resolved.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("Capture config root must be a mapping.")

    raw_cameras = raw.get("cameras")
    if not isinstance(raw_cameras, list) or len(raw_cameras) != 2:
        raise ValueError("Exactly two cameras are required: webcam_front and iphone_left.")
    cameras = []
    for position, item in enumerate(raw_cameras):
        section = "cameras[%d]" % position
        if not isinstance(item, dict):
            raise ValueError("%s must be a mapping." % section)
        camera = CameraConfig(
            name=str(_required(item, "name", section)).strip(),
            role=str(_required(item, "role", section)).strip(),
            device_index=int(_required(item, "device_index", section)),
            backend=str(_required(item, "backend", section)).lower(),
            width=int(_required(item, "width", section)),
            height=int(_required(item, "height", section)),
            fps=float(_required(item, "fps", section)),
            warmup_frames=int(_required(item, "warmup_frames", section)),
            max_identical_frames=int(_required(item, "max_identical_frames", section)),
            read_retry_count=int(_required(item, "read_retry_count", section)),
            read_retry_delay_ms=float(
                _required(item, "read_retry_delay_ms", section)
            ),
            mirror=bool(_required(item, "mirror", section)),
        )
        if camera.role not in {"webcam_front", "iphone_left"}:
            raise ValueError("%s.role must be webcam_front or iphone_left." % section)
        if camera.device_index < 0 or camera.width <= 0 or camera.height <= 0 or camera.fps <= 0:
            raise ValueError("%s camera dimensions, fps, and index are invalid." % section)
        if camera.warmup_frames < 0 or camera.max_identical_frames <= 0:
            raise ValueError("%s camera warmup and freeze limits are invalid." % section)
        if camera.read_retry_count < 0 or camera.read_retry_delay_ms < 0:
            raise ValueError("%s camera read retry settings are invalid." % section)
        if camera.backend not in SUPPORTED_BACKENDS:
            raise ValueError("%s.backend must be one of %s." % (section, sorted(SUPPORTED_BACKENDS)))
        cameras.append(camera)
    if {camera.role for camera in cameras} != {"webcam_front", "iphone_left"}:
        raise ValueError("One webcam_front and one iphone_left camera are required.")
    if len({camera.device_index for camera in cameras}) != 2:
        raise ValueError("Camera device indices must be unique.")

    project_root = resolved.parent.parent
    dataset_raw = _mapping(raw, "dataset")
    dataset_root = Path(str(_required(dataset_raw, "root_directory", "dataset")))
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    calibration_source = Path(
        str(_required(dataset_raw, "calibration_source_directory", "dataset"))
    )
    if not calibration_source.is_absolute():
        calibration_source = project_root / calibration_source
    geometry_mode = str(
        _required(dataset_raw, "geometry_mode", "dataset")
    ).strip().lower()
    if geometry_mode not in {"fixed_rig_2d", "intrinsics_2d", "calibrated_3d"}:
        raise ValueError(
            "dataset.geometry_mode must be fixed_rig_2d, intrinsics_2d, or calibrated_3d."
        )
    require_calibration_assets = bool(
        _required(dataset_raw, "require_calibration_assets", "dataset")
    )
    if geometry_mode in {"intrinsics_2d", "calibrated_3d"} and not require_calibration_assets:
        raise ValueError(
            "%s mode requires dataset.require_calibration_assets=true." % geometry_mode
        )

    preview_raw = _mapping(raw, "preview")
    preview_duration_ms = float(
        _required(preview_raw, "preflight_duration_ms", "preview")
    )
    preview_width_px = int(_required(preview_raw, "camera_width_px", "preview"))
    preview_consecutive_detections = int(
        _required(
            preview_raw,
            "required_consecutive_detections",
            "preview",
        )
    )
    preview_landmark_confidence = float(
        _required(
            preview_raw,
            "landmark_detection_confidence",
            "preview",
        )
    )
    preview_landmark_cameras = _parse_camera_keys(
        _required(
            preview_raw,
            "landmark_required_cameras",
            "preview",
        ),
        "preview.landmark_required_cameras",
    )
    if (
        preview_duration_ms < 0
        or preview_width_px <= 0
        or preview_consecutive_detections <= 0
        or not 0.0 < preview_landmark_confidence <= 1.0
    ):
        raise ValueError("Preview duration and camera width are invalid.")
    recording_raw = _mapping(raw, "recording")
    codec = str(_required(recording_raw, "video_codec", "recording"))
    if len(codec) != 4:
        raise ValueError("recording.video_codec must contain exactly four characters.")
    timing_mode = str(_required(recording_raw, "timing_mode", "recording")).strip().lower()
    if timing_mode not in {"configured", "measured"}:
        raise ValueError("recording.timing_mode must be configured or measured.")

    frame_capture_raw = _mapping(raw, "frame_capture")
    image_format = str(
        _required(frame_capture_raw, "image_format", "frame_capture")
    ).strip().lower()
    jpeg_quality = int(_required(frame_capture_raw, "jpeg_quality", "frame_capture"))
    if image_format not in {"jpg", "png"}:
        raise ValueError("frame_capture.image_format must be jpg or png.")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("frame_capture.jpeg_quality must be between 1 and 100.")
    frame_capture = FrameCaptureConfig(
        enabled=bool(_required(frame_capture_raw, "enabled", "frame_capture")),
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        samples_per_target=int(
            _required(frame_capture_raw, "samples_per_target", "frame_capture")
        ),
        scoring_width_px=int(
            _required(frame_capture_raw, "scoring_width_px", "frame_capture")
        ),
        ideal_brightness=float(
            _required(frame_capture_raw, "ideal_brightness", "frame_capture")
        ),
        sharpness_reference=float(
            _required(frame_capture_raw, "sharpness_reference", "frame_capture")
        ),
        eye_open_weight=float(
            _required(frame_capture_raw, "eye_open_weight", "frame_capture")
        ),
        face_weight=float(_required(frame_capture_raw, "face_weight", "frame_capture")),
        sharpness_weight=float(
            _required(frame_capture_raw, "sharpness_weight", "frame_capture")
        ),
        brightness_weight=float(
            _required(frame_capture_raw, "brightness_weight", "frame_capture")
        ),
        face_scale_factor=float(
            _required(frame_capture_raw, "face_scale_factor", "frame_capture")
        ),
        face_min_neighbors=int(
            _required(frame_capture_raw, "face_min_neighbors", "frame_capture")
        ),
        eye_scale_factor=float(
            _required(frame_capture_raw, "eye_scale_factor", "frame_capture")
        ),
        eye_min_neighbors=int(
            _required(frame_capture_raw, "eye_min_neighbors", "frame_capture")
        ),
        mediapipe_quality_cameras=_parse_camera_keys(
            _required(
                frame_capture_raw,
                "mediapipe_quality_cameras",
                "frame_capture",
            ),
            "frame_capture.mediapipe_quality_cameras",
        ),
        mediapipe_ready_weight=float(
            _required(
                frame_capture_raw,
                "mediapipe_ready_weight",
                "frame_capture",
            )
        ),
    )
    if frame_capture.samples_per_target != 1:
        raise ValueError("frame_capture.samples_per_target must be 1 for best-frame mode.")
    if frame_capture.scoring_width_px <= 0:
        raise ValueError("frame_capture.scoring_width_px must be positive.")
    if not 0 <= frame_capture.ideal_brightness <= 255:
        raise ValueError("frame_capture.ideal_brightness must be within [0, 255].")
    if frame_capture.sharpness_reference <= 0:
        raise ValueError("frame_capture.sharpness_reference must be positive.")
    if any(
        weight < 0
        for weight in (
            frame_capture.eye_open_weight,
            frame_capture.face_weight,
            frame_capture.sharpness_weight,
            frame_capture.brightness_weight,
        )
    ):
        raise ValueError("frame_capture quality weights must be non-negative.")
    if frame_capture.face_scale_factor <= 1 or frame_capture.eye_scale_factor <= 1:
        raise ValueError("frame_capture cascade scale factors must be greater than 1.")
    if frame_capture.face_min_neighbors <= 0 or frame_capture.eye_min_neighbors <= 0:
        raise ValueError("frame_capture cascade neighbor counts must be positive.")
    if frame_capture.mediapipe_ready_weight <= 0:
        raise ValueError("frame_capture.mediapipe_ready_weight must be positive.")

    postprocessing_raw = _mapping(raw, "postprocessing")
    webeyetrack_config = Path(
        str(
            _required(
                postprocessing_raw,
                "webeyetrack_config",
                "postprocessing",
            )
        )
    )
    if not webeyetrack_config.is_absolute():
        webeyetrack_config = project_root / webeyetrack_config
    postprocessing = PostprocessingConfig(
        enabled=bool(_required(postprocessing_raw, "enabled", "postprocessing")),
        ear_threshold=float(
            _required(postprocessing_raw, "ear_threshold", "postprocessing")
        ),
        feature_cameras=_parse_camera_keys(
            _required(postprocessing_raw, "feature_cameras", "postprocessing"),
            "postprocessing.feature_cameras",
        ),
        webeyetrack_enabled=bool(
            _required(
                postprocessing_raw,
                "webeyetrack_enabled",
                "postprocessing",
            )
        ),
        webeyetrack_config=webeyetrack_config.resolve(),
    )
    if not 0.0 < postprocessing.ear_threshold < 1.0:
        raise ValueError("postprocessing.ear_threshold must be within (0, 1).")
    if not postprocessing.feature_cameras:
        raise ValueError("postprocessing.feature_cameras must not be empty.")
    if postprocessing.webeyetrack_enabled and not postprocessing.webeyetrack_config.is_file():
        raise FileNotFoundError(
            "WebEyeTrack preprocessing config does not exist: %s"
            % postprocessing.webeyetrack_config
        )

    display_raw = _mapping(raw, "display")
    canvas_width = int(_required(display_raw, "canvas_width", "display"))
    canvas_height = int(_required(display_raw, "canvas_height", "display"))
    radius = int(_required(display_raw, "point_radius_px", "display"))
    ring_radius = int(
        _required(display_raw, "confirmation_ring_radius_px", "display")
    )
    if canvas_width <= 0 or canvas_height <= 0 or radius <= 0 or ring_radius < radius:
        raise ValueError("Display canvas dimensions and point radius must be positive.")

    protocols_raw = _mapping(raw, "protocols")
    train = _static_config(_mapping(protocols_raw, "train_static"), "protocols.train_static")
    evaluation = _static_config(
        _mapping(protocols_raw, "evaluation_static"), "protocols.evaluation_static"
    )
    vertical_raw = _mapping(protocols_raw, "vertical_click")
    columns = tuple(
        float(value)
        for value in _required(vertical_raw, "columns", "protocols.vertical_click")
    )
    vertical_click = VerticalClickProtocolConfig(
        protocol_id=str(
            _required(vertical_raw, "protocol_id", "protocols.vertical_click")
        ),
        split=str(_required(vertical_raw, "split", "protocols.vertical_click")),
        columns=columns,
        rows=int(_required(vertical_raw, "rows", "protocols.vertical_click")),
        y_min=float(_required(vertical_raw, "y_min", "protocols.vertical_click")),
        y_max=float(_required(vertical_raw, "y_max", "protocols.vertical_click")),
        confirmation_required=bool(
            _required(vertical_raw, "confirmation_required", "protocols.vertical_click")
        ),
        minimum_fixation_ms=float(
            _required(vertical_raw, "minimum_fixation_ms", "protocols.vertical_click")
        ),
        capture_duration_ms=float(
            _required(vertical_raw, "capture_duration_ms", "protocols.vertical_click")
        ),
        transition_duration_ms=float(
            _required(vertical_raw, "transition_duration_ms", "protocols.vertical_click")
        ),
        simulation_confirm_after_ms=float(
            _required(vertical_raw, "simulation_confirm_after_ms", "protocols.vertical_click")
        ),
    )
    if len(columns) != 3 or any(not 0 <= value <= 1 for value in columns):
        raise ValueError("Vertical click protocol requires three normalized columns.")
    if vertical_click.rows <= 1:
        raise ValueError("Vertical click protocol requires at least two rows.")
    if not 0 <= vertical_click.y_min < vertical_click.y_max <= 1:
        raise ValueError("Vertical click protocol coordinate bounds are invalid.")
    if (
        vertical_click.minimum_fixation_ms < 0
        or vertical_click.capture_duration_ms <= 0
        or vertical_click.transition_duration_ms < 0
        or vertical_click.simulation_confirm_after_ms
        < vertical_click.minimum_fixation_ms
    ):
        raise ValueError("Vertical click protocol durations are invalid.")

    simulation_raw = _mapping(raw, "simulation")
    simulation = SimulationConfig(
        fps=float(_required(simulation_raw, "fps", "simulation")),
        width=int(_required(simulation_raw, "width", "simulation")),
        height=int(_required(simulation_raw, "height", "simulation")),
        phone_delay_ms=float(_required(simulation_raw, "phone_delay_ms", "simulation")),
        base_unix_timestamp_ns=int(
            _required(simulation_raw, "base_unix_timestamp_ns", "simulation")
        ),
    )
    if simulation.fps <= 0 or simulation.width <= 0 or simulation.height <= 0:
        raise ValueError("Simulation fps and dimensions must be positive.")

    return CaptureConfig(
        project_root=project_root,
        cameras=tuple(cameras),
        preview=PreviewConfig(
            enabled=bool(_required(preview_raw, "enabled", "preview")),
            window_name_prefix=str(_required(preview_raw, "window_name_prefix", "preview")),
            preflight_duration_ms=preview_duration_ms,
            camera_width_px=preview_width_px,
            landmark_required_cameras=preview_landmark_cameras,
            required_consecutive_detections=preview_consecutive_detections,
            landmark_detection_confidence=preview_landmark_confidence,
        ),
        recording=RecordingConfig(
            enabled=bool(_required(recording_raw, "enabled", "recording")),
            video_codec=codec,
            timing_mode=timing_mode,
        ),
        frame_capture=frame_capture,
        dataset=DatasetConfig(
            root_directory=dataset_root.resolve(),
            calibration_source_directory=calibration_source.resolve(),
            geometry_mode=geometry_mode,
            require_calibration_assets=require_calibration_assets,
        ),
        postprocessing=postprocessing,
        display=DisplayConfig(
            window_name=str(_required(display_raw, "window_name", "display")),
            fullscreen=bool(_required(display_raw, "fullscreen", "display")),
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            point_radius_px=radius,
            confirmation_ring_radius_px=ring_radius,
            background_bgr=_parse_bgr(
                _required(display_raw, "background_bgr", "display"), "display.background_bgr"
            ),
            point_bgr=_parse_bgr(_required(display_raw, "point_bgr", "display"), "display.point_bgr"),
            guide_bgr=_parse_bgr(_required(display_raw, "guide_bgr", "display"), "display.guide_bgr"),
            confirmation_ring_bgr=_parse_bgr(
                _required(display_raw, "confirmation_ring_bgr", "display"),
                "display.confirmation_ring_bgr",
            ),
            confirmed_ring_bgr=_parse_bgr(
                _required(display_raw, "confirmed_ring_bgr", "display"),
                "display.confirmed_ring_bgr",
            ),
        ),
        protocols=ProtocolsConfig(
            intro_center_ms=float(_required(protocols_raw, "intro_center_ms", "protocols")),
            transition_ms=float(_required(protocols_raw, "transition_ms", "protocols")),
            center_refix_ms=float(_required(protocols_raw, "center_refix_ms", "protocols")),
            completion_duration_ms=float(
                _required(protocols_raw, "completion_duration_ms", "protocols")
            ),
            train_static=train,
            evaluation_static=evaluation,
            vertical_click=vertical_click,
        ),
        simulation=simulation,
    )
