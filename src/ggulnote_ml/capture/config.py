from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


SUPPORTED_BACKENDS = {"auto", "avfoundation", "dshow", "v4l2"}


@dataclass(frozen=True)
class CameraConfig:
    name: str
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
    output_directory: Path
    video_codec: str


@dataclass(frozen=True)
class CaptureConfig:
    project_root: Path
    cameras: Tuple[CameraConfig, ...]
    preview: PreviewConfig
    recording: RecordingConfig


def _required(section: Dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ValueError("Missing capture config value: %s.%s" % (section_name, key))
    return section[key]


def load_capture_config(config_path: Path) -> CaptureConfig:
    """Load and validate the complete capture contract before opening hardware."""
    resolved = config_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Capture config does not exist: %s" % resolved)
    with resolved.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("Capture config root must be a mapping.")

    raw_cameras = raw.get("cameras")
    if not isinstance(raw_cameras, list) or not raw_cameras:
        raise ValueError("cameras must contain at least one camera configuration.")

    cameras = []
    for position, item in enumerate(raw_cameras):
        section = "cameras[%d]" % position
        if not isinstance(item, dict):
            raise ValueError("%s must be a mapping." % section)
        camera = CameraConfig(
            name=str(_required(item, "name", section)).strip(),
            device_index=int(_required(item, "device_index", section)),
            backend=str(_required(item, "backend", section)).lower(),
            width=int(_required(item, "width", section)),
            height=int(_required(item, "height", section)),
            fps=float(_required(item, "fps", section)),
            warmup_frames=int(_required(item, "warmup_frames", section)),
            mirror=bool(_required(item, "mirror", section)),
        )
        if not camera.name or not camera.name.replace("_", "").isalnum():
            raise ValueError("%s.name must contain only letters, numbers, or underscores." % section)
        if camera.device_index < 0:
            raise ValueError("%s.device_index must be non-negative." % section)
        if camera.backend not in SUPPORTED_BACKENDS:
            raise ValueError("%s.backend must be one of %s." % (section, sorted(SUPPORTED_BACKENDS)))
        if camera.width <= 0 or camera.height <= 0 or camera.fps <= 0:
            raise ValueError("%s width, height, and fps must be positive." % section)
        if camera.warmup_frames < 0:
            raise ValueError("%s.warmup_frames must be non-negative." % section)
        cameras.append(camera)

    if len({camera.name for camera in cameras}) != len(cameras):
        raise ValueError("Camera names must be unique.")
    if len({camera.device_index for camera in cameras}) != len(cameras):
        raise ValueError("Camera device indices must be unique.")

    preview = raw.get("preview")
    recording = raw.get("recording")
    if not isinstance(preview, dict) or not isinstance(recording, dict):
        raise ValueError("preview and recording must be mappings.")
    codec = str(_required(recording, "video_codec", "recording"))
    if len(codec) != 4:
        raise ValueError("recording.video_codec must contain exactly four characters.")

    # configs/ lives directly under the repository root by contract.
    project_root = resolved.parent.parent
    output = Path(str(_required(recording, "output_directory", "recording")))
    if not output.is_absolute():
        output = project_root / output
    return CaptureConfig(
        project_root=project_root,
        cameras=tuple(cameras),
        preview=PreviewConfig(
            enabled=bool(_required(preview, "enabled", "preview")),
            window_name_prefix=str(_required(preview, "window_name_prefix", "preview")),
        ),
        recording=RecordingConfig(
            enabled=bool(_required(recording, "enabled", "recording")),
            output_directory=output.resolve(),
            video_codec=codec,
        ),
    )
