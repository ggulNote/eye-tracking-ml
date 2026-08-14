from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from gaze_pipeline.training import (
    TrainingError,
    YAxisResidualFusion,
    _Components,
    _export_model,
    _load_weights_checkpoint,
    _read_pipeline_checkpoint,
    _restore_training_checkpoint,
    _save_checkpoint,
    compute_gaze_metrics,
)


def _mark_unsafe_pickle_execution(marker_path: str) -> None:
    Path(marker_path).write_text("executed", encoding="utf-8")


class _UnsupportedPickleValue:
    def __init__(self, marker_path: Path) -> None:
        self.marker_path = marker_path

    def __reduce__(self):
        return _mark_unsafe_pickle_execution, (str(self.marker_path),)


def test_y_axis_fusion_never_changes_x_and_masks_invalid_side() -> None:
    fusion = YAxisResidualFusion(residual_weight=0.5, learnable=False)
    front = torch.tensor([[0.20, -0.10], [-0.30, 0.40]])
    residual = torch.tensor([[0.40], [-0.20]])

    output = fusion(front, residual, side_valid=torch.tensor([True, False]))

    assert torch.equal(output[:, 0], front[:, 0])
    assert torch.allclose(output, torch.tensor([[0.20, 0.10], [-0.30, 0.40]]))


def test_y_axis_fusion_learnable_weight_receives_gradient() -> None:
    fusion = YAxisResidualFusion(residual_weight=1.0, learnable=True)
    output = fusion(torch.zeros((2, 2)), torch.ones((2, 1)))

    output.sum().backward()

    assert fusion.residual_weight.grad is not None
    assert float(fusion.residual_weight.grad) == pytest.approx(2.0)


def test_gaze_metrics_report_normalized_pixel_cm_and_subject_macro() -> None:
    prediction = torch.tensor([[0.10, 0.00], [0.00, 0.20]])
    target = torch.zeros_like(prediction)

    metrics = compute_gaze_metrics(
        prediction,
        target,
        subjects=["p1", "p2"],
        screen_sizes_px=torch.tensor([[1000.0, 500.0], [1000.0, 500.0]]),
        screen_sizes_mm=torch.tensor([[300.0, 200.0], [300.0, 200.0]]),
    )

    assert metrics["euclidean_normalized_mean"] == pytest.approx(0.15)
    assert metrics["euclidean_pixel_mean"] == pytest.approx(100.0)
    assert metrics["euclidean_cm_mean"] == pytest.approx(3.5)
    assert metrics["subject_macro_euclidean_normalized"] == pytest.approx(0.15)
    assert math.isfinite(metrics["p95_euclidean_cm"])


def test_pipeline_checkpoint_round_trip(tmp_path: Path) -> None:
    front = torch.nn.Linear(2, 2)
    side = torch.nn.Linear(2, 1)
    fusion = YAxisResidualFusion(residual_weight=0.75, learnable=True)
    components = _Components(front=front, side=side, fusion=fusion)
    path = tmp_path / "weights.pt"
    original_front = {key: value.detach().clone() for key, value in front.state_dict().items()}

    _save_checkpoint(
        path,
        components=components,
        config={"model": {}},
        epoch=3,
        metrics={"euclidean_normalized_mean": 0.1},
        optimizer=None,
        scheduler=None,
        weights_only=True,
    )
    with torch.no_grad():
        for parameter in front.parameters():
            parameter.zero_()
        fusion.residual_weight.zero_()

    _load_weights_checkpoint(path, components, device=torch.device("cpu"))

    for key, value in front.state_dict().items():
        assert torch.equal(value, original_front[key])
    assert float(fusion.residual_weight.detach()) == pytest.approx(0.75)


def test_resume_checkpoint_restores_optimizer_scheduler_and_rng(tmp_path: Path) -> None:
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    front = torch.nn.Linear(2, 2)
    components = _Components(front=front, side=None, fusion=None)
    optimizer = torch.optim.AdamW(front.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    loss = front(torch.ones((2, 2))).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    original_front = {key: value.detach().clone() for key, value in front.state_dict().items()}
    expected_lr = optimizer.param_groups[0]["lr"]
    path = tmp_path / "last_checkpoint.pt"

    _save_checkpoint(
        path,
        components=components,
        config={"model": {"backend": "pytorch"}},
        epoch=7,
        metrics={"euclidean_normalized_mean": 0.1},
        optimizer=optimizer,
        scheduler=scheduler,
        weights_only=False,
        include_optimizer=True,
        include_scheduler=True,
        include_rng_state=True,
    )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(4)

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    with torch.no_grad():
        for parameter in front.parameters():
            parameter.zero_()
    optimizer.param_groups[0]["lr"] = 0.9

    restored_epoch = _restore_training_checkpoint(
        path,
        components=components,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
    )

    assert restored_epoch == 7
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    for key, value in front.state_dict().items():
        assert torch.equal(value, original_front[key])
    assert random.random() == pytest.approx(expected_python)
    assert float(np.random.random()) == pytest.approx(expected_numpy)
    assert torch.equal(torch.rand(4), expected_torch)


def test_pipeline_checkpoint_rejects_unsupported_pickle_without_execution(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "unsafe.pt"
    marker = tmp_path / "pickle-executed.txt"
    torch.save(
        {
            "format_version": 1,
            "components": {},
            "unsupported": _UnsupportedPickleValue(marker),
        },
        checkpoint,
    )

    with pytest.raises(TrainingError, match="weights-only"):
        _read_pipeline_checkpoint(checkpoint, torch.device("cpu"))

    assert not marker.exists()


def test_checkpoint_and_export_store_only_safe_contract_and_hash_lineage(
    tmp_path: Path,
) -> None:
    secret = "top-secret-api-token"
    private_root = "/private/users/alice/model-source"
    pretrained_path = "/private/users/alice/front_weights.pt"
    front = torch.nn.Linear(2, 2)
    components = _Components(front=front, side=None, fusion=None)
    config = {
        "api_token": secret,
        "paths": {"data_root": "/private/users/alice/face-data"},
        "task": {
            "type": "screen_point_regression",
            "output_key": "gaze_xy",
            "output_dim": 2,
            "output_order": ["x", "y"],
        },
        "model": {
            "backend": "pytorch",
            "front": {
                "enabled": True,
                "source_dir": private_root,
                "entrypoint": "private_models.front:create_model",
                "init_args": {
                    "api_token": secret,
                    "cache_path": "/private/users/alice/cache",
                },
                "pretrained": {
                    "path": pretrained_path,
                    "sha256": "a" * 64,
                    "strict": True,
                },
                "input_contract": {
                    "image_key": "front_image",
                    "shape": ["B", 3, 224, 224],
                    "private_path": private_root,
                },
                "output_contract": {"gaze_key": "gaze_xy", "gaze_shape": ["B", 2]},
            },
        },
        "fusion": {"enabled": False, "method": "y_axis_residual"},
        "model_export": {
            "enabled": True,
            "dir": str(tmp_path / "models"),
            "filename": "final_weights.pt",
            "include_model_contract": True,
            "include_resolved_config": True,
        },
    }
    lineage = {
        "dataset_manifest_hash": "b" * 64,
        "resolved_config_hash": "c" * 64,
        "artifact_hashes": {"train": "d" * 64, "validation": "e" * 64},
    }
    best_path = tmp_path / "best_weights.pt"
    last_path = tmp_path / "last_checkpoint.pt"

    for checkpoint_path, weights_only in ((best_path, True), (last_path, False)):
        _save_checkpoint(
            checkpoint_path,
            components=components,
            config=config,
            epoch=1,
            metrics={"euclidean_normalized_mean": 0.1},
            optimizer=None,
            scheduler=None,
            weights_only=weights_only,
            lineage=lineage,
        )
    final_path = _export_model(config, components, base_dir=tmp_path, lineage=lineage)
    assert final_path is not None

    for artifact_path in (best_path, last_path, final_path):
        payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
        rendered = repr(payload)
        raw = artifact_path.read_bytes()
        assert secret not in rendered
        assert private_root not in rendered
        assert pretrained_path not in rendered
        assert b"top-secret-api-token" not in raw
        assert b"/private/users/alice" not in raw
        assert "source_dir" not in rendered
        assert payload["lineage"]["dataset_manifest_hash"] == "b" * 64
        assert payload["lineage"]["resolved_config_hash"] == "c" * 64
        assert payload["model_contract"]["model"]["front"]["pretrained"] == {
            "strict": True,
            "sha256": "a" * 64,
        }


def test_export_flags_omit_contract_and_resolved_config_hash(tmp_path: Path) -> None:
    components = _Components(front=torch.nn.Linear(2, 2), side=None, fusion=None)
    config = {
        "model": {"backend": "pytorch", "front": {"enabled": True}},
        "model_export": {
            "enabled": True,
            "dir": str(tmp_path),
            "filename": "minimal.pt",
            "include_model_contract": False,
            "include_resolved_config": False,
        },
    }

    path = _export_model(
        config,
        components,
        base_dir=tmp_path,
        lineage={
            "dataset_manifest_hash": "a" * 64,
            "resolved_config_hash": "b" * 64,
        },
    )

    assert path is not None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert "model_contract" not in payload
    assert payload["lineage"] == {"dataset_manifest_hash": "a" * 64}
