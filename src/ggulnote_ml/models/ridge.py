from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ggulnote_ml.exceptions import ContractError
from ggulnote_ml.models.base import GazeRegressor


class RidgeGazeRegressor(GazeRegressor):
    name = "ridge"

    def __init__(self, alpha: float, version: str) -> None:
        self.alpha = float(alpha)
        self.version = version
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_scale: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.intercept: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        x = self._to_matrix(features).astype(np.float64)
        y = self._validate_targets(targets, x.shape[0]).astype(np.float64)
        self.feature_mean = x.mean(axis=0, keepdims=True)
        self.feature_scale = x.std(axis=0, keepdims=True)
        self.feature_scale[self.feature_scale < 1e-8] = 1.0
        x_scaled = (x - self.feature_mean) / self.feature_scale
        target_mean = y.mean(axis=0)
        centered_y = y - target_mean

        sample_count, feature_count = x_scaled.shape
        if feature_count <= sample_count:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                regularized = x_scaled.T @ x_scaled + self.alpha * np.eye(
                    feature_count, dtype=np.float64
                )
                right_hand_side = x_scaled.T @ centered_y
            if not np.isfinite(regularized).all() or not np.isfinite(right_hand_side).all():
                raise ContractError("Ridge fitting produced a non-finite linear system.")
            self.weights = np.linalg.solve(regularized, right_hand_side)
        else:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                regularized = x_scaled @ x_scaled.T + self.alpha * np.eye(
                    sample_count, dtype=np.float64
                )
            if not np.isfinite(regularized).all():
                raise ContractError("Ridge fitting produced a non-finite dual system.")
            dual = np.linalg.solve(regularized, centered_y)
            # Some macOS Accelerate builds leave floating-point flags set after
            # LAPACK solve. Verify finiteness explicitly instead of emitting a
            # false-positive warning on the following matrix multiplication.
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                self.weights = x_scaled.T @ dual
        self.weights = self.weights.astype(np.float64, copy=False)
        self.intercept = target_mean.astype(np.float64, copy=False)
        if not np.isfinite(self.weights).all() or not np.isfinite(self.intercept).all():
            raise ContractError("Ridge fitting produced non-finite model parameters.")

    def predict(self, features: np.ndarray) -> np.ndarray:
        if any(
            state is None
            for state in (self.feature_mean, self.feature_scale, self.weights, self.intercept)
        ):
            raise ContractError("Model must be fitted or loaded before prediction.")
        x = self._to_matrix(features).astype(np.float64)
        if x.shape[1] != self.weights.shape[0]:
            raise ContractError(
                "Model expects %d pooled features; got %d."
                % (self.weights.shape[0], x.shape[1])
            )
        x_scaled = (x - self.feature_mean) / self.feature_scale
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            predictions = x_scaled @ self.weights + self.intercept
        if not np.isfinite(predictions).all():
            raise ContractError("Model prediction produced NaN or infinity.")
        return np.clip(predictions, 0.0, 1.0).astype(np.float32, copy=False)

    def save(self, path: Path) -> None:
        if any(
            state is None
            for state in (self.feature_mean, self.feature_scale, self.weights, self.intercept)
        ):
            raise ContractError("Cannot save an unfitted model.")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            alpha=np.asarray(self.alpha, dtype=np.float32),
            version=np.asarray(self.version),
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            weights=self.weights,
            intercept=self.intercept,
        )

    @classmethod
    def load(cls, path: Path) -> "RidgeGazeRegressor":
        with np.load(path, allow_pickle=False) as state:
            model = cls(alpha=float(state["alpha"]), version=str(state["version"]))
            model.feature_mean = state["feature_mean"].astype(np.float64)
            model.feature_scale = state["feature_scale"].astype(np.float64)
            model.weights = state["weights"].astype(np.float64)
            model.intercept = state["intercept"].astype(np.float64)
        return model

    @staticmethod
    def _to_matrix(features: np.ndarray) -> np.ndarray:
        if features.ndim != 3:
            raise ContractError("Model input must have shape [B,T,F].")
        if features.shape[0] == 0:
            raise ContractError("Model input batch cannot be empty.")
        if not np.isfinite(features).all():
            raise ContractError("Model input contains NaN or infinity.")
        return features.astype(np.float32, copy=False).mean(axis=1)

    @staticmethod
    def _validate_targets(targets: np.ndarray, sample_count: int) -> np.ndarray:
        if targets.shape != (sample_count, 2):
            raise ContractError("Training targets must have shape [B,2].")
        if not np.isfinite(targets).all():
            raise ContractError("Training targets contain NaN or infinity.")
        return targets.astype(np.float32, copy=False)
