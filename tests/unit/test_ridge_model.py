from pathlib import Path

import numpy as np

from ggulnote_ml.models.ridge import RidgeGazeRegressor


def test_ridge_model_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(20, 3, 5)).astype(np.float32)
    targets = rng.uniform(size=(20, 2)).astype(np.float32)
    model = RidgeGazeRegressor(alpha=1.0, version="test")
    model.fit(features, targets)
    before = model.predict(features)

    model_path = tmp_path / "model.npz"
    model.save(model_path)
    restored = RidgeGazeRegressor.load(model_path)
    after = restored.predict(features)

    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)
    assert after.shape == (20, 2)

