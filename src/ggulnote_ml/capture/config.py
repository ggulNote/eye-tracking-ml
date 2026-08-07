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
    mirror: bool


@dataclass(frozen=True)
class PreviewConfig:
    enabled: bool
    window_name_prefix: str


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool
    video_codec: str


@dataclass(frozen=True)
class DatasetConfig:
    root_directory: Path
    require_calibration_assets: bool


@dataclass(frozen=True)
class DisplayConfig:
    window_name: str
    fullscreen: bool
    canvas_width: int
    canvas_height: int
    point_radius_px: int
    background_bgr: Tuple[int, int, int]
    point_bgr: Tuple[int, int, int]
    guide_bgr: Tuple[int, int, int]


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
    point_duration_ms: float
    settling_duration_ms: float
    usable_duration_ms: float
    random_seed: int
    y_positions: Tuple[float, ...]


@dataclass(frozen=True)
class DynamicProtocolConfig:
    protocol_id: str
    split: str
    columns: Tuple[float, ...]
    y_min: float
    y_max: float
    movement_duration_ms: float
    transition_duration_ms: float
    edge_exclusion_ms: float
    latency_ms: float
    show_guide_line: bool


@dataclass(frozen=True)
class ProtocolsConfig:
    intro_center_ms: float
    transition_ms: float
    center_refix_ms: float
    completion_duration_ms: float
    train_static: StaticProtocolConfig
    evaluation_static: StaticProtocolConfig
    dynamic: DynamicProtocolConfig


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
    dataset: DatasetConfig
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
        point_duration_ms=float(_required(raw, "point_duration_ms", name)),
        settling_duration_ms=float(_required(raw, "settling_duration_ms", name)),
        usable_duration_ms=float(_required(raw, "usable_duration_ms", name)),
        random_seed=int(_required(raw, "random_seed", name)),
        y_positions=tuple(float(value) for value in raw.get("y_positions", [])),
    )
    if config.columns <= 0 or config.rows <= 0 or config.repeats <= 0:
        raise ValueError("%s grid dimensions and repeats must be positive." % name)
    if not 0 <= config.x_min < config.x_max <= 1 or not 0 <= config.y_min < config.y_max <= 1:
        raise ValueError("%s coordinate bounds must be ordered within [0, 1]." % name)
    if config.point_duration_ms <= 0 or config.settling_duration_ms < 0 or config.usable_duration_ms <= 0:
        raise ValueError("%s durations are invalid." % name)
    if config.settling_duration_ms + config.usable_duration_ms > config.point_duration_ms:
        raise ValueError("%s settling + usable duration cannot exceed point duration." % name)
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
            mirror=bool(_required(item, "mirror", section)),
        )
        if camera.role not in {"webcam_front", "iphone_left"}:
            raise ValueError("%s.role must be webcam_front or iphone_left." % section)
        if camera.device_index < 0 or camera.width <= 0 or camera.height <= 0 or camera.fps <= 0:
            raise ValueError("%s camera dimensions, fps, and index are invalid." % section)
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

    preview_raw = _mapping(raw, "preview")
    recording_raw = _mapping(raw, "recording")
    codec = str(_required(recording_raw, "video_codec", "recording"))
    if len(codec) != 4:
        raise ValueError("recording.video_codec must contain exactly four characters.")

    display_raw = _mapping(raw, "display")
    canvas_width = int(_required(display_raw, "canvas_width", "display"))
    canvas_height = int(_required(display_raw, "canvas_height", "display"))
    radius = int(_required(display_raw, "point_radius_px", "display"))
    if canvas_width <= 0 or canvas_height <= 0 or radius <= 0:
        raise ValueError("Display canvas dimensions and point radius must be positive.")

    protocols_raw = _mapping(raw, "protocols")
    train = _static_config(_mapping(protocols_raw, "train_static"), "protocols.train_static")
    evaluation = _static_config(
        _mapping(protocols_raw, "evaluation_static"), "protocols.evaluation_static"
    )
    dynamic_raw = _mapping(protocols_raw, "dynamic")
    columns = tuple(float(value) for value in _required(dynamic_raw, "columns", "protocols.dynamic"))
    dynamic = DynamicProtocolConfig(
        protocol_id=str(_required(dynamic_raw, "protocol_id", "protocols.dynamic")),
        split=str(_required(dynamic_raw, "split", "protocols.dynamic")),
        columns=columns,
        y_min=float(_required(dynamic_raw, "y_min", "protocols.dynamic")),
        y_max=float(_required(dynamic_raw, "y_max", "protocols.dynamic")),
        movement_duration_ms=float(
            _required(dynamic_raw, "movement_duration_ms", "protocols.dynamic")
        ),
        transition_duration_ms=float(
            _required(dynamic_raw, "transition_duration_ms", "protocols.dynamic")
        ),
        edge_exclusion_ms=float(
            _required(dynamic_raw, "edge_exclusion_ms", "protocols.dynamic")
        ),
        latency_ms=float(_required(dynamic_raw, "latency_ms", "protocols.dynamic")),
        show_guide_line=bool(_required(dynamic_raw, "show_guide_line", "protocols.dynamic")),
    )
    if len(columns) != 3 or any(not 0 <= value <= 1 for value in columns):
        raise ValueError("Dynamic protocol requires three normalized columns.")
    if not 0 <= dynamic.y_min < dynamic.y_max <= 1 or dynamic.movement_duration_ms <= 0:
        raise ValueError("Dynamic protocol bounds or duration are invalid.")
    if dynamic.transition_duration_ms < 0 or dynamic.edge_exclusion_ms < 0:
        raise ValueError("Dynamic transition and edge exclusion must be non-negative.")
    if dynamic.edge_exclusion_ms * 2 >= dynamic.movement_duration_ms:
        raise ValueError("Dynamic edge exclusion must leave a usable movement interval.")
    if dynamic.latency_ms < 0:
        raise ValueError("Dynamic latency must be non-negative.")

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
        ),
        recording=RecordingConfig(
            enabled=bool(_required(recording_raw, "enabled", "recording")),
            video_codec=codec,
        ),
        dataset=DatasetConfig(
            root_directory=dataset_root.resolve(),
            require_calibration_assets=bool(
                _required(dataset_raw, "require_calibration_assets", "dataset")
            ),
        ),
        display=DisplayConfig(
            window_name=str(_required(display_raw, "window_name", "display")),
            fullscreen=bool(_required(display_raw, "fullscreen", "display")),
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            point_radius_px=radius,
            background_bgr=_parse_bgr(
                _required(display_raw, "background_bgr", "display"), "display.background_bgr"
            ),
            point_bgr=_parse_bgr(_required(display_raw, "point_bgr", "display"), "display.point_bgr"),
            guide_bgr=_parse_bgr(_required(display_raw, "guide_bgr", "display"), "display.guide_bgr"),
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
            dynamic=dynamic,
        ),
        simulation=simulation,
    )
