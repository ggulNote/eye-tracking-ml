from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from ggulnote_ml.config import EvaluationConfig
from ggulnote_ml.exceptions import ContractError


def gaze_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    config: EvaluationConfig,
) -> Dict[str, float]:
    if targets.shape != predictions.shape or targets.ndim != 2 or targets.shape[1] != 2:
        raise ContractError(
            "Metrics require matching [B,2] targets and predictions; got %s and %s."
            % (targets.shape, predictions.shape)
        )
    difference = predictions.astype(np.float64) - targets.astype(np.float64)
    absolute = np.abs(difference)
    normalized_distance = np.linalg.norm(difference, axis=1)
    pixel_difference = difference * np.asarray(
        [config.screen_width, config.screen_height], dtype=np.float64
    )
    pixel_distance = np.linalg.norm(pixel_difference, axis=1)

    metrics = {
        "mae_x": float(absolute[:, 0].mean()),
        "mae_y": float(absolute[:, 1].mean()),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "normalized_distance_mean": float(normalized_distance.mean()),
        "normalized_distance_median": float(np.median(normalized_distance)),
        "normalized_distance_p95": float(np.percentile(normalized_distance, 95)),
        "pixel_distance_mean": float(pixel_distance.mean()),
        "pixel_distance_median": float(np.median(pixel_distance)),
        "pixel_distance_p95": float(np.percentile(pixel_distance, 95)),
    }
    metrics.update(_threshold_metrics(normalized_distance, config.distance_thresholds))
    return metrics


def _threshold_metrics(distances: np.ndarray, thresholds: Iterable[float]) -> Dict[str, float]:
    result = {}
    for threshold in thresholds:
        key = "within_%s" % str(threshold).replace(".", "_")
        result[key] = float(np.mean(distances <= threshold))
    return result

