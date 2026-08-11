from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .geometry import CheckerboardSpec, ScreenSpec


SCREEN_VERIFICATION_SOURCES = {
    "unverified",
    "apple_official_224ppi",
    "manual_measurement",
}
CHECKERBOARD_VERIFICATION_SOURCES = {"unverified", "manual_measurement"}


@dataclass(frozen=True)
class CheckerboardCaptureConfig:
    spec: CheckerboardSpec
    verification_source: str
    required_captures: int
    minimum_captures: int
    max_rms_error_px: float


@dataclass(frozen=True)
class GeometryCalibrationConfig:
    project_root: Path
    setup_id: str
    output_directory: Path
    screen: ScreenSpec
    screen_verification_source: str
    checkerboard: CheckerboardCaptureConfig


def _mapping(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError("%s must be a mapping." % key)
    return value


def _required(raw: Dict[str, Any], key: str, section: str) -> Any:
    if key not in raw:
        raise ValueError("Missing geometry config value: %s.%s" % (section, key))
    return raw[key]


def load_geometry_calibration_config(path: Path) -> GeometryCalibrationConfig:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Geometry calibration config does not exist: %s" % resolved)
    with resolved.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("Geometry calibration config root must be a mapping.")

    project_root = resolved.parent.parent
    setup = _mapping(raw, "setup")
    setup_id = str(_required(setup, "id", "setup")).strip()
    if not setup_id:
        raise ValueError("setup.id cannot be empty.")
    output = Path(str(_required(setup, "output_directory", "setup")))
    if not output.is_absolute():
        output = project_root / output

    screen_raw = _mapping(raw, "screen")
    screen = ScreenSpec(
        width_pixel=int(_required(screen_raw, "width_pixel", "screen")),
        height_pixel=int(_required(screen_raw, "height_pixel", "screen")),
        width_mm=float(_required(screen_raw, "width_mm", "screen")),
        height_mm=float(_required(screen_raw, "height_mm", "screen")),
    )
    screen.validate()

    board_raw = _mapping(raw, "checkerboard")
    board = CheckerboardSpec(
        inner_corners_x=int(_required(board_raw, "inner_corners_x", "checkerboard")),
        inner_corners_y=int(_required(board_raw, "inner_corners_y", "checkerboard")),
        square_size_mm=float(_required(board_raw, "square_size_mm", "checkerboard")),
        page_width_mm=float(_required(board_raw, "page_width_mm", "checkerboard")),
        page_height_mm=float(_required(board_raw, "page_height_mm", "checkerboard")),
    )
    board.validate()
    required_captures = int(_required(board_raw, "required_captures", "checkerboard"))
    minimum_captures = int(_required(board_raw, "minimum_captures", "checkerboard"))
    max_rms_error_px = float(_required(board_raw, "max_rms_error_px", "checkerboard"))
    if minimum_captures < 3 or required_captures < minimum_captures:
        raise ValueError(
            "checkerboard capture counts must satisfy required >= minimum >= 3."
        )
    if max_rms_error_px <= 0:
        raise ValueError("checkerboard.max_rms_error_px must be positive.")

    screen_verification_source = str(
        _required(screen_raw, "verification_source", "screen")
    ).strip()
    checkerboard_verification_source = str(
        _required(board_raw, "verification_source", "checkerboard")
    ).strip()
    if screen_verification_source not in SCREEN_VERIFICATION_SOURCES:
        raise ValueError(
            "screen.verification_source must be one of %s."
            % sorted(SCREEN_VERIFICATION_SOURCES)
        )
    if checkerboard_verification_source not in CHECKERBOARD_VERIFICATION_SOURCES:
        raise ValueError(
            "checkerboard.verification_source must be one of %s."
            % sorted(CHECKERBOARD_VERIFICATION_SOURCES)
        )

    return GeometryCalibrationConfig(
        project_root=project_root,
        setup_id=setup_id,
        output_directory=output.expanduser().resolve(),
        screen=screen,
        screen_verification_source=screen_verification_source,
        checkerboard=CheckerboardCaptureConfig(
            spec=board,
            verification_source=checkerboard_verification_source,
            required_captures=required_captures,
            minimum_captures=minimum_captures,
            max_rms_error_px=max_rms_error_px,
        ),
    )
