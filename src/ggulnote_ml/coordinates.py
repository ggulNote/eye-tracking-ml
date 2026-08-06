from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ggulnote_ml.exceptions import ContractError


def to_top_left_normalized(
    values: Sequence[float],
    coordinate_space: str,
    screen_size_px: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Convert manifest coordinates to the pipeline's canonical [0,1] space."""

    target = np.asarray(values, dtype=np.float32)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ContractError("Gaze coordinates must be two finite values.")
    if coordinate_space == "top_left_normalized":
        normalized = target
    elif coordinate_space == "screen_centered_normalized":
        normalized = target + np.float32(0.5)
    elif coordinate_space == "screen_pixels":
        if screen_size_px is None:
            raise ContractError("screen_pixels coordinates require screen_width_px and screen_height_px.")
        size = np.asarray(screen_size_px, dtype=np.float32)
        if size.shape != (2,) or np.any(size <= 0):
            raise ContractError("Screen pixel dimensions must be two positive values.")
        normalized = target / size
    else:
        raise ContractError("Unsupported target_coordinate_space: %s" % coordinate_space)
    if np.any(normalized < 0.0) or np.any(normalized > 1.0):
        raise ContractError("Converted gaze coordinates must be inside the screen [0,1].")
    return normalized.astype(np.float32, copy=False)


def to_screen_centered_normalized(top_left_normalized: np.ndarray) -> np.ndarray:
    """Convert [...,2] [0,1] targets to WebEyeTrack's [-0.5,0.5] output space."""

    values = np.asarray(top_left_normalized, dtype=np.float32)
    if values.shape[-1:] != (2,) or not np.isfinite(values).all():
        raise ContractError("Targets must end with two finite coordinates.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ContractError("Top-left normalized targets must be in [0,1].")
    return (values - np.float32(0.5)).astype(np.float32, copy=False)


def to_top_left_from_centered(centered_normalized: np.ndarray) -> np.ndarray:
    """Convert WebEyeTrack output coordinates back to top-left normalized space."""

    values = np.asarray(centered_normalized, dtype=np.float32)
    if values.shape[-1:] != (2,) or not np.isfinite(values).all():
        raise ContractError("Targets must end with two finite coordinates.")
    if np.any(values < -0.5) or np.any(values > 0.5):
        raise ContractError("Screen-centered normalized targets must be in [-0.5,0.5].")
    return (values + np.float32(0.5)).astype(np.float32, copy=False)
