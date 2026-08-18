"""Config-driven PyTorch training, evaluation, fusion, and checkpoint export.

The data package owns image/annotation contracts.  This module deliberately
keeps the model boundary small: a branch runtime receives a batch mapping and
returns standardized tensors.  Consequently a new PyTorch model can be wired
through YAML without changing the loop itself.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from gaze_pipeline.config import resolve_model_forward_keys
from gaze_pipeline.data import GazeImageDataset, build_dataloader, prepare_data
from gaze_pipeline.model_runtime import ModelRuntimeError, build_model_runtime
from gaze_pipeline.tracking import MLflowTrackingError
from gaze_pipeline.training_tracking import training_run

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SAFE_LINEAGE_KEYS = frozenset({"dataset_manifest_hash", "resolved_config_hash", "artifact_hashes"})
_SENSITIVE_KEY_MARKERS = (
    "secret",
    "token",
    "password",
    "credential",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "authorization",
    "cookie",
)
_PATH_KEY_SUFFIXES = ("_path", "_root", "_dir")
_CONTRACT_BRANCH_KEYS = ("input_contract", "output_contract")
_OMIT = object()


class TrainingError(RuntimeError):
    """Raised for a user-correctable training or evaluation failure."""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Stable paths and metrics returned by train/evaluate commands."""

    metrics: Mapping[str, float]
    checkpoint_path: Path | None = None
    last_checkpoint_path: Path | None = None
    model_path: Path | None = None
    metrics_path: Path | None = None
    predictions_path: Path | None = None
    best_epoch: int | None = None
    mlflow_run_id: str | None = None


class YAxisResidualFusion(nn.Module):
    """Keep front x unchanged and add only a side-derived y residual."""

    def __init__(self, *, residual_weight: float = 1.0, learnable: bool = True) -> None:
        super().__init__()
        initial = torch.tensor(float(residual_weight), dtype=torch.float32)
        if learnable:
            self.residual_weight = nn.Parameter(initial)
        else:
            self.register_buffer("residual_weight", initial)

    def forward(
        self,
        front_gaze_xy: torch.Tensor,
        delta_y_side: torch.Tensor,
        *,
        side_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        front = _as_prediction(front_gaze_xy, width=2, name="front gaze_xy")
        residual = _as_prediction(delta_y_side, width=1, name="side delta_y_side")
        if front.shape[0] != residual.shape[0]:
            raise TrainingError("front/side prediction의 batch 크기가 다릅니다.")
        if side_valid is not None:
            valid = side_valid.to(device=residual.device, dtype=torch.bool).reshape(-1, 1)
            if valid.shape[0] != residual.shape[0]:
                raise TrainingError("side validity mask의 batch 크기가 다릅니다.")
            residual = torch.where(valid, residual, torch.zeros_like(residual))
        final = front.clone()
        final[:, 1:2] = front[:, 1:2] + self.residual_weight * residual
        return final


@dataclass(slots=True)
class _Components:
    front: nn.Module | None
    side: nn.Module | None
    fusion: YAxisResidualFusion | None

    def modules(self) -> list[nn.Module]:
        return [module for module in (self.front, self.side, self.fusion) if module is not None]

    def parameters(self) -> Iterable[nn.Parameter]:
        for module in self.modules():
            yield from module.parameters()

    def train(self, mode: bool = True) -> None:
        for module in self.modules():
            module.train(mode)

    def state_dict(self) -> dict[str, Any]:
        return {
            name: module.state_dict()
            for name, module in (
                ("front", self.front),
                ("side", self.side),
                ("fusion", self.fusion),
            )
            if module is not None
        }

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        for name, module in (
            ("front", self.front),
            ("side", self.side),
            ("fusion", self.fusion),
        ):
            if module is None:
                continue
            branch_state = state.get(name)
            if not isinstance(branch_state, Mapping):
                if strict:
                    raise TrainingError(f"checkpoint에 {name} state_dict가 없습니다.")
                continue
            module.load_state_dict(branch_state, strict=strict)


@dataclass(slots=True)
class _EpochOutput:
    loss: float
    metrics: dict[str, float]
    predictions: torch.Tensor
    targets: torch.Tensor
    subjects: list[str]
    screen_sizes_px: torch.Tensor
    screen_sizes_mm: torch.Tensor


def train_pipeline(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    preparation: Any | None = None,
) -> PipelineResult:
    """Prepare loaders, train enabled branches, and export reproducible artifacts."""

    _validate_training_runtime_config(config)
    _seed_everything(config)
    device = _resolve_device(config)
    base_dir = _project_base_dir(config_path)
    prepared = preparation or prepare_data(config, config_path=config_path)
    loaders = _build_loaders(config, prepared)
    train_loader = loaders.get("train")
    if train_loader is None:
        raise TrainingError("train split에 학습 가능한 sample이 없습니다.")

    checkpoint_config = _mapping(config.get("checkpoint"), "checkpoint")
    checkpoint_dir = _configured_path(
        checkpoint_config.get("dir"), base_dir, field="checkpoint.dir"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_config = _mapping(checkpoint_config.get("save_best"), "checkpoint.save_best")
    last_config = _mapping(checkpoint_config.get("save_last"), "checkpoint.save_last")
    best_path = checkpoint_dir / str(best_config.get("filename", "best_weights.pt"))
    last_path = checkpoint_dir / str(last_config.get("filename", "last_checkpoint.pt"))
    monitor_mode = str(best_config.get("mode", "min")).lower()
    best_value = math.inf if monitor_mode == "min" else -math.inf
    best_epoch: int | None = None
    stale_epochs = 0
    early = _mapping(
        _mapping(config.get("training"), "training").get("early_stopping"),
        "training.early_stopping",
    )
    max_epochs = int(_mapping(config.get("training"), "training").get("max_epochs", 1))
    validate_every = int(
        _mapping(config.get("training"), "training").get("validate_every_n_epochs", 1)
    )

    final_output: _EpochOutput | None = None
    validation_loader = loaders.get("validation")
    lineage = _preparation_lineage(prepared)
    tracker_params = {"device": str(device), **_tracking_lineage(lineage)}
    mlflow_run_id: str | None = None
    model_path: Path | None = None
    start_epoch = 1
    metric_path: Path | None = None
    try:
        tracker_context = training_run(config, extra_params=tracker_params)
        with tracker_context as tracker:
            mlflow_run_id = tracker.run_id
            components = _build_components(config, base_dir=base_dir, device=device)
            _materialize_components(components, train_loader, config, device)
            parameters = [
                parameter for parameter in components.parameters() if parameter.requires_grad
            ]
            if not parameters:
                raise TrainingError(
                    "학습 가능한 parameter가 없습니다. 모델 freeze 설정을 확인하세요."
                )
            optimizer = _build_optimizer(parameters, config)
            scheduler = _build_scheduler(optimizer, config)
            resume_path = _optional_path(checkpoint_config.get("resume_from"), base_dir)
            if resume_path is not None:
                start_epoch = (
                    _restore_training_checkpoint(
                        resume_path,
                        components=components,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        device=device,
                    )
                    + 1
                )
            for epoch in range(start_epoch, max_epochs + 1):
                _set_dataset_epoch(train_loader, epoch - 1)
                train_output = _run_epoch(
                    components,
                    train_loader,
                    config,
                    device=device,
                    optimizer=optimizer,
                )
                should_validate = validation_loader is not None and epoch % validate_every == 0
                validation_output = (
                    _run_epoch(
                        components,
                        validation_loader,
                        config,
                        device=device,
                        optimizer=None,
                    )
                    if should_validate
                    else None
                )
                reference = validation_output or train_output
                final_output = reference
                monitored_name, monitored_value = _selection_metric(reference.metrics, config)
                selection_due = validation_loader is None or validation_output is not None
                if selection_due:
                    improved = _is_improved(
                        monitored_value,
                        best_value,
                        mode=monitor_mode,
                        min_delta=float(early.get("min_delta", 0.0)),
                    )
                    if improved:
                        best_value = monitored_value
                        best_epoch = epoch
                        stale_epochs = 0
                        if bool(best_config.get("enabled", True)):
                            _save_checkpoint(
                                best_path,
                                components=components,
                                config=config,
                                epoch=epoch,
                                metrics=reference.metrics,
                                optimizer=None,
                                scheduler=None,
                                weights_only=True,
                                lineage=lineage,
                            )
                            tracker.log_checkpoint(best_path, kind="best")
                    elif should_validate:
                        stale_epochs += 1

                if scheduler is not None and (
                    selection_due
                    or not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                ):
                    _step_scheduler(scheduler, monitored_value)
                if bool(last_config.get("enabled", True)):
                    _save_checkpoint(
                        last_path,
                        components=components,
                        config=config,
                        epoch=epoch,
                        metrics=reference.metrics,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        weights_only=False,
                        include_optimizer=bool(last_config.get("include_optimizer", True)),
                        include_scheduler=bool(last_config.get("include_scheduler", True)),
                        include_rng_state=bool(last_config.get("include_rng_state", True)),
                        lineage=lineage,
                    )
                tracker.log_epoch(
                    epoch,
                    train_loss=train_output.loss,
                    val_loss=validation_output.loss if validation_output is not None else None,
                    train_metrics=train_output.metrics,
                    val_metrics=(
                        validation_output.metrics if validation_output is not None else {}
                    ),
                )
                print(
                    f"epoch {epoch}/{max_epochs} "
                    f"train_loss={train_output.loss:.6f} "
                    f"{monitored_name}={monitored_value:.6f}"
                )
                if (
                    bool(early.get("enabled", False))
                    and should_validate
                    and stale_epochs >= int(early.get("patience_epochs", 1))
                ):
                    print(f"early stopping: {stale_epochs} validation epochs 동안 개선 없음")
                    break

            if bool(last_config.get("enabled", True)) and last_path.exists():
                tracker.log_checkpoint(last_path, kind="last")
            model_path = _export_model(
                config,
                components,
                base_dir=base_dir,
                lineage=lineage,
            )
            if model_path is not None:
                tracker.log_model_artifact(model_path)
            if final_output is None:
                raise TrainingError("학습 epoch가 한 번도 실행되지 않았습니다.")
            metric_path = _write_metrics(
                config,
                final_output.metrics,
                base_dir,
                name="train_summary.json",
            )
            tracker.log_artifact(metric_path, "training/summary", "summary")
    except (
        ModelRuntimeError,
        MLflowTrackingError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        if isinstance(exc, TrainingError):
            raise
        raise TrainingError(str(exc)) from exc

    if final_output is None:
        raise TrainingError("학습 epoch가 한 번도 실행되지 않았습니다.")
    if metric_path is None:
        raise TrainingError("학습 summary가 생성되지 않았습니다.")
    return PipelineResult(
        metrics=final_output.metrics,
        checkpoint_path=best_path if best_path.exists() else None,
        last_checkpoint_path=last_path if last_path.exists() else None,
        model_path=model_path,
        metrics_path=metric_path,
        best_epoch=best_epoch,
        mlflow_run_id=mlflow_run_id,
    )


def evaluate_pipeline(
    config: Mapping[str, Any],
    *,
    checkpoint: str | Path | None = None,
    split: str = "test",
    config_path: str | Path | None = None,
    preparation: Any | None = None,
) -> PipelineResult:
    """Load a checkpoint and write split metrics plus per-sample predictions."""

    _validate_training_runtime_config(config)
    split_name = "validation" if split in {"val", "valid"} else str(split)
    if split_name not in {"validation", "test"}:
        raise TrainingError("evaluate --split은 validation 또는 test여야 합니다.")
    _seed_everything(config)
    device = _resolve_device(config)
    base_dir = _project_base_dir(config_path)
    prepared = preparation or prepare_data(config, config_path=config_path)
    loaders = _build_loaders(config, prepared)
    loader = loaders.get(split_name)
    if loader is None:
        raise TrainingError(f"{split_name} split에 평가 가능한 sample이 없습니다.")
    checkpoint_path = _evaluation_checkpoint_path(config, checkpoint, base_dir)
    lineage = _preparation_lineage(prepared)
    tracker_params = {
        **_tracking_lineage(lineage),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
    }
    mlflow_run_id: str | None = None
    try:
        with training_run(
            config,
            extra_params=tracker_params,
            run_type="evaluation",
            extra_tags={"command": "evaluate", "split": split_name},
        ) as tracker:
            mlflow_run_id = tracker.run_id
            components = _build_components(config, base_dir=base_dir, device=device)
            _materialize_components(components, loader, config, device)
            _load_weights_checkpoint(checkpoint_path, components, device=device)
            output = _run_epoch(components, loader, config, device=device, optimizer=None)
            metrics_path = _write_metrics(
                config,
                output.metrics,
                base_dir,
                name=f"{split_name}_metrics.json",
            )
            predictions_path = _write_predictions(
                config,
                output,
                base_dir,
                name=f"{split_name}_predictions.csv",
            )
            tracker.log_metrics(
                {
                    f"{split_name}/loss": output.loss,
                    **{f"{split_name}/{key}": value for key, value in output.metrics.items()},
                },
                step=0,
            )
            tracker.log_artifact(
                checkpoint_path,
                "evaluation/input_checkpoint",
                "evaluated_checkpoint",
            )
            tracker.log_artifact(
                metrics_path,
                f"evaluation/{split_name}/metrics",
                "metrics",
            )
            if _mlflow_log_predictions(config):
                remote_predictions = _write_predictions(
                    config,
                    output,
                    base_dir,
                    name=f"{split_name}_predictions_anonymized.csv",
                    include_subject_id=False,
                )
                tracker.log_artifact(
                    remote_predictions,
                    f"evaluation/{split_name}/predictions",
                    "predictions",
                )
    except (
        ModelRuntimeError,
        MLflowTrackingError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        if isinstance(exc, TrainingError):
            raise
        raise TrainingError(str(exc)) from exc
    return PipelineResult(
        metrics=output.metrics,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        mlflow_run_id=mlflow_run_id,
    )


def _build_components(
    config: Mapping[str, Any], *, base_dir: Path, device: torch.device
) -> _Components:
    model_config = _mapping(config.get("model"), "model")
    front_config = _mapping(model_config.get("front"), "model.front")
    side_config = _mapping(model_config.get("side"), "model.side")
    front = (
        build_model_runtime(
            "front",
            front_config,
            base_dir=base_dir,
            device=device,
            forward_keys=resolve_model_forward_keys(config, "front"),
        )
        if bool(front_config.get("enabled", False))
        else None
    )
    side = (
        build_model_runtime(
            "side",
            side_config,
            base_dir=base_dir,
            device=device,
            forward_keys=resolve_model_forward_keys(config, "side"),
        )
        if bool(side_config.get("enabled", False))
        else None
    )
    fusion_config = _mapping(config.get("fusion"), "fusion")
    fusion = None
    if bool(fusion_config.get("enabled", False)):
        method = str(fusion_config.get("method", "")).lower()
        if method != "y_axis_residual":
            raise TrainingError("현재 실행 가능한 fusion.method는 y_axis_residual뿐입니다.")
        if front is None or side is None:
            raise TrainingError("y_axis_residual fusion에는 front와 side model이 모두 필요합니다.")
        if fusion_config.get("entrypoint") not in {None, ""}:
            raise TrainingError(
                "fusion.entrypoint는 아직 지원하지 않습니다. 내장 y_axis_residual을 사용하세요."
            )
        fusion = YAxisResidualFusion(
            residual_weight=float(fusion_config.get("residual_weight", 1.0)),
            learnable=bool(fusion_config.get("learnable_weight", True)),
        ).to(device)
    return _Components(front=front, side=side, fusion=fusion)


def _build_loaders(config: Mapping[str, Any], prepared: Any) -> dict[str, DataLoader[Any]]:
    model = _mapping(config.get("model"), "model")
    front_enabled = bool(_mapping(model.get("front"), "model.front").get("enabled", False))
    side_enabled = bool(_mapping(model.get("side"), "model.side").get("enabled", False))
    paired = front_enabled and side_enabled
    view = None if paired else ("front" if front_enabled else "side")
    data_root = _mapping(config.get("data"), "data").get("dataset_root")
    loaders: dict[str, DataLoader[Any]] = {}
    for split_name in ("train", "validation", "test"):
        split_counts = getattr(prepared, "split_counts", {})
        if isinstance(split_counts, Mapping) and int(split_counts.get(split_name, 0)) == 0:
            continue
        manifest_paths = getattr(prepared, "manifest_paths", {})
        manifest_path = (
            manifest_paths.get(split_name) if isinstance(manifest_paths, Mapping) else None
        )
        if manifest_path is None:
            raise TrainingError(f"prepared result에 {split_name} manifest가 없습니다.")
        try:
            dataset = GazeImageDataset(
                manifest_path,
                config,
                split=split_name,
                data_root=data_root,
                view=view,
                paired=paired,
            )
        except (OSError, ValueError) as exc:
            if split_name == "train":
                raise TrainingError(f"train Dataset 생성 실패: {exc}") from exc
            continue
        loaders[split_name] = build_dataloader(dataset, config, split=split_name)
    return loaders


def _materialize_components(
    components: _Components,
    loader: DataLoader[Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> None:
    """Materialize lazy fallback heads before optimizer/checkpoint construction."""

    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise TrainingError("DataLoader가 비어 있습니다.") from exc
    batch = _move_batch(batch, device)
    components.train(False)
    with torch.no_grad():
        _forward_pipeline(components, batch, config)


def _forward_pipeline(
    components: _Components,
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    front_output: Mapping[str, Any] = {}
    side_output: Mapping[str, Any] = {}
    if components.front is not None:
        front_output = components.front(batch)  # type: ignore[operator]
        front_gaze = _front_gaze(front_output)
        outputs["front_gaze_xy"] = front_gaze
    if components.side is not None:
        side_output = components.side(batch)  # type: ignore[operator]
        outputs["delta_y_side"] = _side_residual(side_output)
    if components.fusion is not None:
        side_valid = _validity_mask(batch, "side", outputs["delta_y_side"].shape[0])
        if "pair_mask" in batch:
            side_valid = side_valid & torch.as_tensor(
                batch["pair_mask"], device=side_valid.device, dtype=torch.bool
            ).reshape(-1)
        outputs["gaze_xy"] = components.fusion(
            outputs["front_gaze_xy"], outputs["delta_y_side"], side_valid=side_valid
        )
    elif "front_gaze_xy" in outputs:
        outputs["gaze_xy"] = outputs["front_gaze_xy"]
    elif "delta_y_side" in outputs:
        zeros = torch.zeros_like(outputs["delta_y_side"])
        outputs["gaze_xy"] = torch.cat((zeros, outputs["delta_y_side"]), dim=1)
    else:  # guarded by config validation, kept defensive
        raise TrainingError("활성화된 모델 branch가 없습니다.")
    for key, value in {**front_output, **side_output}.items():
        if torch.is_tensor(value) and key not in outputs:
            outputs[key] = value
    return outputs


def _run_epoch(
    components: _Components,
    loader: DataLoader[Any],
    config: Mapping[str, Any],
    *,
    device: torch.device,
    optimizer: Optimizer | None,
) -> _EpochOutput:
    training = optimizer is not None
    components.train(training)
    training_config = _mapping(config.get("training"), "training")
    accumulation = int(training_config.get("gradient_accumulation_steps", 1))
    clip_norm = float(training_config.get("gradient_clip_norm", 0.0))
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0
    prediction_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    subject_ids: list[str] = []
    screen_px: list[torch.Tensor] = []
    screen_mm: list[torch.Tensor] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, raw_batch in enumerate(loader, start=1):
            batch = _move_batch(raw_batch, device)
            target = _as_prediction(batch["target_gaze_xy"], width=2, name="target_gaze_xy")
            outputs = _forward_pipeline(components, batch, config)
            loss, metric_prediction, metric_mask = _pipeline_loss(outputs, batch, config)
            if training:
                (loss / accumulation).backward()
                if step % accumulation == 0:
                    if clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [
                                parameter
                                for parameter in components.parameters()
                                if parameter.grad is not None
                            ],
                            clip_norm,
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            valid_count = int(metric_mask.sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * valid_count
            total_samples += valid_count
            prediction_chunks.append(metric_prediction[metric_mask].detach().cpu())
            target_chunks.append(target[metric_mask].detach().cpu())
            selected_indices = (
                metric_mask.detach().cpu().nonzero(as_tuple=False).reshape(-1).tolist()
            )
            metadata = raw_batch.get("metadata", [])
            for index in selected_indices:
                sample_metadata = metadata[index] if index < len(metadata) else {}
                subject_ids.append(_metadata_subject(sample_metadata))
                px, mm = _metadata_screen_sizes(sample_metadata)
                screen_px.append(px)
                screen_mm.append(mm)
        if training and total_samples and len(loader) % accumulation:
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in components.parameters()
                        if parameter.grad is not None
                    ],
                    clip_norm,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    if not prediction_chunks or sum(chunk.shape[0] for chunk in prediction_chunks) == 0:
        raise TrainingError("유효한 gaze target/prediction sample이 없습니다.")
    predictions = torch.cat(prediction_chunks, dim=0)
    targets = torch.cat(target_chunks, dim=0)
    px_tensor = torch.stack(screen_px) if screen_px else torch.empty((0, 2))
    mm_tensor = torch.stack(screen_mm) if screen_mm else torch.empty((0, 2))
    metrics = compute_gaze_metrics(
        predictions,
        targets,
        subjects=subject_ids,
        screen_sizes_px=px_tensor,
        screen_sizes_mm=mm_tensor,
        threshold_rates=_mapping(config.get("metrics"), "metrics").get("threshold_rates"),
    )
    metric_config = _mapping(config.get("metrics"), "metrics")
    requested = metric_config.get("report")
    if isinstance(requested, Sequence) and not isinstance(requested, str | bytes):
        required = {
            str(metric_config.get("selection_metric", "euclidean_normalized_mean")),
            str(
                metric_config.get("fallback_selection_metric", "subject_macro_euclidean_normalized")
            ),
        }
        selected_names = {str(name) for name in requested} | required
        metrics = {name: value for name, value in metrics.items() if name in selected_names}
    return _EpochOutput(
        loss=total_loss / max(total_samples, 1),
        metrics=metrics,
        predictions=predictions,
        targets=targets,
        subjects=subject_ids,
        screen_sizes_px=px_tensor,
        screen_sizes_mm=mm_tensor,
    )


def _pipeline_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = _as_prediction(batch["target_gaze_xy"], width=2, name="target_gaze_xy")
    loss_config = _mapping(config.get("loss"), "loss")
    primary = _mapping(loss_config.get("primary"), "loss.primary")
    front_mask = _validity_mask(batch, "front", target.shape[0])
    side_mask = _validity_mask(batch, "side", target.shape[0])
    finite_target = torch.isfinite(target).all(dim=1)
    front_mask &= finite_target
    side_mask &= finite_target
    if "pair_mask" in batch:
        side_mask &= torch.as_tensor(
            batch["pair_mask"], device=target.device, dtype=torch.bool
        ).reshape(-1)

    final_prediction = _as_prediction(outputs["gaze_xy"], width=2, name="gaze_xy")
    # The configured missing-branch policy keeps a valid Front prediction when
    # Side is invalid/missing. Fusion already replaces that residual with zero.
    final_mask = front_mask if "front_gaze_xy" in outputs else side_mask
    total = _masked_xy_loss(final_prediction, target, final_mask, primary)
    total = total * float(primary.get("weight", 1.0))

    auxiliary = _mapping(loss_config.get("branch_auxiliary"), "loss.branch_auxiliary")
    if bool(auxiliary.get("enabled", False)):
        front_weight = float(auxiliary.get("front_weight", 0.0))
        if front_weight and "front_gaze_xy" in outputs:
            total = total + front_weight * _masked_xy_loss(
                outputs["front_gaze_xy"], target, front_mask, primary
            )
        side_weight = float(auxiliary.get("side_weight", 0.0))
        if side_weight and "delta_y_side" in outputs:
            baseline_y = (
                outputs["front_gaze_xy"][:, 1:2].detach()
                if "front_gaze_xy" in outputs
                else torch.zeros_like(target[:, 1:2])
            )
            residual_target = target[:, 1:2] - baseline_y
            total = total + side_weight * _masked_xy_loss(
                outputs["delta_y_side"], residual_target, side_mask, primary
            )
    return total, final_prediction, final_mask


def _masked_xy_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    prediction = prediction.reshape(target.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    if prediction.shape != target.shape:
        shapes = f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        raise TrainingError(f"loss prediction/target shape 불일치: {shapes}")
    mask = mask.reshape(-1).bool() & torch.isfinite(prediction).all(dim=1)
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    error = prediction[mask] - target[mask]
    name = str(config.get("name", "huber_xy")).lower()
    if name in {"huber", "huber_xy", "smooth_l1"}:
        delta = float(config.get("delta", 1.0) or 1.0)
        absolute = error.abs()
        elementwise = torch.where(
            absolute <= delta,
            0.5 * error.square() / delta,
            absolute - 0.5 * delta,
        )
    elif name in {"mse", "mse_xy", "weighted_l2_xy", "l2"}:
        elementwise = error.square()
    else:
        raise TrainingError(f"지원하지 않는 loss.primary.name입니다: {name}")
    axis_weights = config.get("axis_weights")
    if isinstance(axis_weights, Mapping) and elementwise.shape[1] == 2:
        weights = elementwise.new_tensor(
            [float(axis_weights.get("x", 1.0)), float(axis_weights.get("y", 1.0))]
        )
        elementwise = elementwise * weights
    if name == "weighted_l2_xy" and target.shape[1] == 2:
        sample_weights = _inverse_frequency_weights(target[mask], config)
        elementwise = elementwise * sample_weights.unsqueeze(1)
    return elementwise.mean()


def _inverse_frequency_weights(target: torch.Tensor, config: Mapping[str, Any]) -> torch.Tensor:
    """Return mean-one inverse occupancy weights for configured screen cells."""

    grid = config.get("frequency_grid_size", [30, 30])
    if not isinstance(grid, Sequence) or isinstance(grid, str | bytes) or len(grid) != 2:
        raise TrainingError("loss.primary.frequency_grid_size는 [x_bins, y_bins]여야 합니다.")
    x_bins, y_bins = int(grid[0]), int(grid[1])
    if x_bins <= 0 or y_bins <= 0:
        raise TrainingError("loss.primary.frequency_grid_size 값은 양수여야 합니다.")
    normalized = (target + 0.5).clamp(0.0, 1.0 - torch.finfo(target.dtype).eps)
    x_index = torch.floor(normalized[:, 0] * x_bins).long()
    y_index = torch.floor(normalized[:, 1] * y_bins).long()
    flat_index = y_index * x_bins + x_index
    counts = torch.bincount(flat_index, minlength=x_bins * y_bins).to(target.dtype)
    weights = counts[flat_index].reciprocal()
    return weights / weights.mean().clamp_min(torch.finfo(weights.dtype).eps)


def compute_gaze_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    subjects: Sequence[str] | None = None,
    screen_sizes_px: torch.Tensor | None = None,
    screen_sizes_mm: torch.Tensor | None = None,
    threshold_rates: Any = None,
) -> dict[str, float]:
    """Compute normalized metrics plus pixel/cm metrics when screen data exists."""

    prediction = _as_prediction(predictions.float(), width=2, name="predictions").cpu()
    target = _as_prediction(targets.float(), width=2, name="targets").cpu()
    if prediction.shape != target.shape or prediction.shape[0] == 0:
        raise TrainingError("metric prediction과 target은 비어 있지 않은 같은 [N,2]여야 합니다.")
    error = prediction - target
    distance = torch.linalg.vector_norm(error, dim=1)
    metrics: dict[str, float] = {
        "euclidean_normalized_mean": float(distance.mean()),
        "euclidean_normalized_median": float(distance.median()),
        "mae_x_normalized": float(error[:, 0].abs().mean()),
        "mae_y_normalized": float(error[:, 1].abs().mean()),
        "rmse_normalized": float(error.square().mean().sqrt()),
        "out_of_bounds_rate": float(
            ((prediction < -0.5) | (prediction > 0.5)).any(dim=1).float().mean()
        ),
    }
    subject_values = list(subjects or ["unknown"] * prediction.shape[0])
    metrics["subject_macro_euclidean_normalized"] = _subject_macro(distance, subject_values)

    px_distance: torch.Tensor | None = None
    cm_distance_all: torch.Tensor | None = None
    px = _valid_size_tensor(screen_sizes_px, prediction.shape[0])
    if px is not None:
        px_error = error * px
        px_distance = torch.linalg.vector_norm(px_error, dim=1)
        valid_px = torch.isfinite(px_distance)
        if bool(valid_px.any()):
            metrics["euclidean_pixel_mean"] = float(px_distance[valid_px].mean())
            metrics["subject_macro_euclidean_pixel"] = _subject_macro(
                px_distance, subject_values, valid=valid_px
            )
    mm = _valid_size_tensor(screen_sizes_mm, prediction.shape[0])
    if mm is not None:
        mm_distance = torch.linalg.vector_norm(error * mm, dim=1)
        valid_mm = torch.isfinite(mm_distance)
        if bool(valid_mm.any()):
            cm_distance = mm_distance[valid_mm] / 10.0
            metrics["euclidean_cm_mean"] = float(cm_distance.mean())
            metrics["p90_euclidean_cm"] = float(torch.quantile(cm_distance, 0.90))
            metrics["p95_euclidean_cm"] = float(torch.quantile(cm_distance, 0.95))
            all_cm = torch.full_like(mm_distance, torch.nan)
            all_cm[valid_mm] = mm_distance[valid_mm] / 10.0
            cm_distance_all = all_cm
            metrics["subject_macro_euclidean_cm"] = _subject_macro(
                all_cm, subject_values, valid=valid_mm
            )
    if threshold_rates is not None:
        if not isinstance(threshold_rates, Sequence) or isinstance(threshold_rates, str | bytes):
            raise TrainingError("metrics.threshold_rates는 mapping list여야 합니다.")
        available = {
            "normalized": distance,
            "pixel": px_distance,
            "cm": cm_distance_all,
        }
        for index, specification in enumerate(threshold_rates):
            if not isinstance(specification, Mapping):
                raise TrainingError(f"metrics.threshold_rates[{index}]는 mapping이어야 합니다.")
            if not bool(specification.get("enabled", True)):
                continue
            name = str(specification.get("name", "")).strip()
            unit = str(specification.get("unit", "normalized")).lower()
            threshold = float(specification.get("threshold", 0.0))
            values = available.get(unit)
            if not name or unit not in available or threshold <= 0 or not math.isfinite(threshold):
                raise TrainingError(
                    f"metrics.threshold_rates[{index}]의 name/unit/threshold가 올바르지 않습니다."
                )
            if values is None:
                continue
            valid = torch.isfinite(values)
            if bool(valid.any()):
                metrics[name] = float((values[valid] <= threshold).float().mean())
    return metrics


def _subject_macro(
    values: torch.Tensor,
    subjects: Sequence[str],
    *,
    valid: torch.Tensor | None = None,
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    valid_mask = torch.ones(values.shape[0], dtype=torch.bool) if valid is None else valid.bool()
    for index, value in enumerate(values):
        if index >= len(subjects) or not bool(valid_mask[index]) or not bool(torch.isfinite(value)):
            continue
        grouped[str(subjects[index])].append(float(value))
    if not grouped:
        return float("nan")
    return float(np.mean([np.mean(group) for group in grouped.values()]))


def _front_gaze(output: Mapping[str, Any]) -> torch.Tensor:
    value = output.get("gaze_xy")
    if value is None and output.get("x_front") is not None and output.get("y_front") is not None:
        value = torch.cat((output["x_front"], output["y_front"]), dim=1)
    if value is None:
        raise TrainingError("Front adapter 출력에 gaze_xy [B,2]가 없습니다.")
    return _as_prediction(value, width=2, name="front gaze_xy")


def _side_residual(output: Mapping[str, Any]) -> torch.Tensor:
    value = output.get("delta_y_side")
    if value is None:
        raise TrainingError("Side adapter 출력에 delta_y_side [B,1]이 없습니다.")
    return _as_prediction(value, width=1, name="side delta_y_side")


def _as_prediction(value: Any, *, width: int, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TrainingError(f"{name}은 torch.Tensor여야 합니다.")
    if value.ndim == 1 and width == 1:
        value = value.unsqueeze(1)
    if value.ndim != 2 or value.shape[1] != width:
        raise TrainingError(f"{name} shape은 [B,{width}]여야 합니다: {tuple(value.shape)}")
    return value


def _validity_mask(batch: Mapping[str, Any], branch: str, batch_size: int) -> torch.Tensor:
    reference = batch.get(f"{branch}_image")
    if reference is None:
        reference = batch.get("target_gaze_xy")
    device = reference.device if torch.is_tensor(reference) else torch.device("cpu")
    value = batch.get(f"{branch}_gaze_valid")
    if value is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    return torch.as_tensor(value, device=device, dtype=torch.bool).reshape(-1)


def _build_optimizer(parameters: list[nn.Parameter], config: Mapping[str, Any]) -> Optimizer:
    optimizer_config = _mapping(config.get("optimizer"), "optimizer")
    name = str(optimizer_config.get("name", "AdamW")).lower()
    kwargs: dict[str, Any] = {
        "lr": float(optimizer_config.get("learning_rate", 1e-4)),
        "weight_decay": float(optimizer_config.get("weight_decay", 0.0)),
    }
    betas = optimizer_config.get("betas")
    if isinstance(betas, Sequence) and not isinstance(betas, str | bytes) and len(betas) == 2:
        kwargs["betas"] = (float(betas[0]), float(betas[1]))
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    raise TrainingError("optimizer.name은 Adam 또는 AdamW여야 합니다.")


def _build_scheduler(optimizer: Optimizer, config: Mapping[str, Any]) -> Any | None:
    scheduler_config = _mapping(config.get("scheduler"), "scheduler")
    if not bool(scheduler_config.get("enabled", False)):
        return None
    name = str(scheduler_config.get("name", "")).lower()
    if name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_config.get("mode", "min")),
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience_epochs", 5)),
            min_lr=float(scheduler_config.get("min_learning_rate", 0.0)),
        )
    if name == "exponentiallr":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=float(scheduler_config.get("gamma", 0.95))
        )
    raise TrainingError("scheduler.name은 ReduceLROnPlateau 또는 ExponentialLR이어야 합니다.")


def _step_scheduler(scheduler: Any, monitored_value: float) -> None:
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(monitored_value)
    else:
        scheduler.step()


def _save_checkpoint(
    path: Path,
    *,
    components: _Components,
    config: Mapping[str, Any],
    epoch: int,
    metrics: Mapping[str, float],
    optimizer: Optimizer | None,
    scheduler: Any | None,
    weights_only: bool,
    include_optimizer: bool = False,
    include_scheduler: bool = False,
    include_rng_state: bool = False,
    lineage: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "epoch": int(epoch),
        "components": components.state_dict(),
        "metrics": {str(name): float(value) for name, value in metrics.items()},
        "model_contract": _safe_model_contract(config),
    }
    safe_lineage = _safe_checkpoint_lineage(lineage)
    if safe_lineage:
        payload["lineage"] = safe_lineage
    if not weights_only:
        if include_optimizer:
            payload["optimizer_state_dict"] = (
                optimizer.state_dict() if optimizer is not None else None
            )
        if include_scheduler:
            payload["scheduler_state_dict"] = (
                scheduler.state_dict() if scheduler is not None else None
            )
        if include_rng_state:
            payload["rng_state"] = _rng_state()
    torch.save(payload, path)


def _restore_training_checkpoint(
    path: Path,
    *,
    components: _Components,
    optimizer: Optimizer,
    scheduler: Any | None,
    device: torch.device,
) -> int:
    payload = _read_pipeline_checkpoint(path, device)
    components.load_state_dict(_checkpoint_components(payload), strict=True)
    optimizer_state = payload.get("optimizer_state_dict")
    if isinstance(optimizer_state, Mapping):
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None and isinstance(scheduler_state, Mapping):
        scheduler.load_state_dict(scheduler_state)
    _restore_rng_state(payload.get("rng_state"))
    return int(payload.get("epoch", 0))


def _load_weights_checkpoint(path: Path, components: _Components, *, device: torch.device) -> None:
    payload = _read_pipeline_checkpoint(path, device)
    components.load_state_dict(_checkpoint_components(payload), strict=True)


def _read_pipeline_checkpoint(path: Path, device: torch.device) -> Mapping[str, Any]:
    if not path.exists() or not path.is_file():
        raise TrainingError(f"checkpoint 파일이 없습니다: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise TrainingError(
            f"checkpoint를 weights-only 방식으로 안전하게 읽지 못했습니다: {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TrainingError("pipeline checkpoint 최상위 값은 mapping이어야 합니다.")
    return payload


def _checkpoint_components(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    components = payload.get("components")
    if not isinstance(components, Mapping):
        raise TrainingError("checkpoint에 components state_dict가 없습니다.")
    return components


def _export_model(
    config: Mapping[str, Any],
    components: _Components,
    *,
    base_dir: Path,
    lineage: Mapping[str, Any] | None = None,
) -> Path | None:
    export = _mapping(config.get("model_export"), "model_export")
    if not bool(export.get("enabled", True)):
        return None
    directory = _configured_path(export.get("dir"), base_dir, field="model_export.dir")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / str(export.get("filename", "final_weights.pt"))
    payload: dict[str, Any] = {
        "format_version": 1,
        "components": components.state_dict(),
    }
    if bool(export.get("include_model_contract", True)):
        payload["model_contract"] = _safe_model_contract(config)
    safe_lineage = _safe_checkpoint_lineage(lineage)
    if not bool(export.get("include_resolved_config", True)):
        safe_lineage.pop("resolved_config_hash", None)
    if safe_lineage:
        payload["lineage"] = safe_lineage
    torch.save(payload, path)
    return path


def _safe_model_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the inference contract without code, credential, or filesystem fields."""

    contract: dict[str, Any] = {"schema_version": 1}
    task = config.get("task")
    if isinstance(task, Mapping):
        allowed_task_keys = (
            "type",
            "output_key",
            "output_dim",
            "output_order",
            "output_dtype",
            "coordinate_system",
            "output_activation",
            "uncertainty",
        )
        safe_task = _contract_allowlist(task, allowed_task_keys)
        if safe_task:
            contract["task"] = safe_task

    model = _mapping(config.get("model"), "model")
    safe_model: dict[str, Any] = {}
    backend = _safe_contract_value(model.get("backend"), key="backend")
    if backend is not _OMIT:
        safe_model["backend"] = backend
    for branch_name in ("front", "side"):
        branch = model.get(branch_name)
        if not isinstance(branch, Mapping):
            continue
        safe_branch: dict[str, Any] = {"enabled": bool(branch.get("enabled", False))}
        safe_branch.update(_contract_allowlist(branch, _CONTRACT_BRANCH_KEYS))
        pretrained = branch.get("pretrained")
        if isinstance(pretrained, Mapping):
            safe_pretrained: dict[str, Any] = {"strict": bool(pretrained.get("strict", True))}
            digest = pretrained.get("sha256")
            if digest is not None:
                safe_pretrained["sha256"] = _validated_sha256(
                    digest, field=f"model.{branch_name}.pretrained.sha256"
                )
            safe_branch["pretrained"] = safe_pretrained
        safe_model[branch_name] = safe_branch
    contract["model"] = safe_model

    fusion = config.get("fusion")
    if isinstance(fusion, Mapping):
        allowed_fusion_keys = (
            "enabled",
            "stage",
            "method",
            "input_keys",
            "output_key",
            "output_shape",
            "residual_weight",
            "learnable_weight",
            "missing_branch_policy",
        )
        safe_fusion = _contract_allowlist(fusion, allowed_fusion_keys)
        if safe_fusion:
            contract["fusion"] = safe_fusion
    return contract


def _contract_allowlist(source: Mapping[str, Any], allowed_keys: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in allowed_keys:
        if key not in source:
            continue
        safe = _safe_contract_value(source[key], key=key)
        if safe is not _OMIT:
            result[key] = safe
    return result


def _safe_contract_value(value: Any, *, key: str | None = None) -> Any:
    if key is not None and (_is_sensitive_key(key) or _is_path_key(key)):
        return _OMIT
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if not isinstance(child_key, str):
                continue
            safe = _safe_contract_value(child_value, key=child_key)
            if safe is not _OMIT:
                result[child_key] = safe
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        result_list: list[Any] = []
        for child in value:
            safe = _safe_contract_value(child)
            if safe is not _OMIT:
                result_list.append(safe)
        return result_list
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        text = value.strip()
        if (
            text.startswith(("/", "~/", "./", "../"))
            or re.match(r"^[A-Za-z]:[\\/]", text)
            or "://" in text
        ):
            return _OMIT
        return value
    return _OMIT


def _safe_checkpoint_lineage(lineage: Mapping[str, Any] | None) -> dict[str, Any]:
    if lineage is None:
        return {}
    if not isinstance(lineage, Mapping):
        raise TrainingError("checkpoint lineage는 mapping이어야 합니다.")
    unknown = sorted(str(key) for key in lineage if key not in _SAFE_LINEAGE_KEYS)
    if unknown:
        raise TrainingError(f"지원하지 않는 checkpoint lineage key입니다: {unknown}")

    safe: dict[str, Any] = {}
    for key in ("dataset_manifest_hash", "resolved_config_hash"):
        value = lineage.get(key)
        if value is not None:
            safe[key] = _validated_sha256(value, field=f"lineage.{key}")
    artifact_hashes = lineage.get("artifact_hashes")
    if artifact_hashes is not None:
        if not isinstance(artifact_hashes, Mapping):
            raise TrainingError("lineage.artifact_hashes는 name -> SHA-256 mapping이어야 합니다.")
        safe_artifacts: dict[str, str] = {}
        for raw_name, digest in artifact_hashes.items():
            if not isinstance(raw_name, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*", raw_name
            ):
                raise TrainingError("lineage.artifact_hashes 이름이 안전하지 않습니다.")
            safe_artifacts[raw_name] = _validated_sha256(
                digest, field=f"lineage.artifact_hashes.{raw_name}"
            )
        safe["artifact_hashes"] = safe_artifacts
    return safe


def _validated_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TrainingError(f"{field}는 64자리 SHA-256 hex 문자열이어야 합니다.")
    return value.lower()


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(marker in normalized for marker in _SENSITIVE_KEY_MARKERS)


def _is_path_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return normalized in {"path", "root", "dir", "source_dir"} or normalized.endswith(
        _PATH_KEY_SUFFIXES
    )


def _selection_metric(metrics: Mapping[str, float], config: Mapping[str, Any]) -> tuple[str, float]:
    metric_config = _mapping(config.get("metrics"), "metrics")
    requested = str(metric_config.get("selection_metric", "euclidean_normalized_mean"))
    fallback = str(
        metric_config.get("fallback_selection_metric", "subject_macro_euclidean_normalized")
    )
    for name in (requested, fallback, "euclidean_normalized_mean"):
        value = metrics.get(name)
        if value is not None and math.isfinite(float(value)):
            return name, float(value)
    raise TrainingError("checkpoint 선택에 사용할 유효한 metric이 없습니다.")


def _is_improved(value: float, best: float, *, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return value < best - min_delta
    if mode == "max":
        return value > best + min_delta
    raise TrainingError("checkpoint/early stopping mode는 min 또는 max여야 합니다.")


def _evaluation_checkpoint_path(
    config: Mapping[str, Any], checkpoint: str | Path | None, base_dir: Path
) -> Path:
    if checkpoint is not None:
        return _configured_path(checkpoint, base_dir, field="--checkpoint")
    checkpoint_config = _mapping(config.get("checkpoint"), "checkpoint")
    resume = checkpoint_config.get("resume_from")
    if resume:
        return _configured_path(resume, base_dir, field="checkpoint.resume_from")
    directory = _configured_path(checkpoint_config.get("dir"), base_dir, field="checkpoint.dir")
    best = _mapping(checkpoint_config.get("save_best"), "checkpoint.save_best")
    return directory / str(best.get("filename", "best_weights.pt"))


def _write_metrics(
    config: Mapping[str, Any], metrics: Mapping[str, float], base_dir: Path, *, name: str
) -> Path:
    paths = _mapping(config.get("paths"), "paths")
    directory = _configured_path(paths.get("metric_dir"), base_dir, field="paths.metric_dir")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        json.dumps(dict(metrics), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_predictions(
    config: Mapping[str, Any],
    output: _EpochOutput,
    base_dir: Path,
    *,
    name: str,
    include_subject_id: bool = True,
) -> Path:
    paths = _mapping(config.get("paths"), "paths")
    directory = _configured_path(
        paths.get("prediction_dir"), base_dir, field="paths.prediction_dir"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["target_x", "target_y", "prediction_x", "prediction_y"]
        if include_subject_id:
            header.insert(0, "subject_id")
        writer.writerow(header)
        for index, (target, prediction) in enumerate(
            zip(output.targets, output.predictions, strict=True)
        ):
            row: list[Any] = [
                float(target[0]),
                float(target[1]),
                float(prediction[0]),
                float(prediction[1]),
            ]
            if include_subject_id:
                row.insert(0, output.subjects[index] if index < len(output.subjects) else "")
            writer.writerow(row)
    return path


def _mlflow_log_predictions(config: Mapping[str, Any]) -> bool:
    mlflow = _mapping(config.get("mlflow"), "mlflow")
    value = mlflow.get("log_predictions", False)
    if not isinstance(value, bool):
        raise TrainingError("mlflow.log_predictions는 true 또는 false여야 합니다.")
    return value


def _preparation_lineage(prepared: Any) -> dict[str, Any]:
    """Return path-free dataset/config hashes for checkpoints and MLflow params."""

    lineage: dict[str, Any] = {}
    for attribute in ("dataset_manifest_hash", "resolved_config_hash"):
        value = getattr(prepared, attribute, None)
        if isinstance(value, str) and value:
            lineage[attribute] = value
    artifact_hashes = getattr(prepared, "artifact_hashes", None)
    if isinstance(artifact_hashes, Mapping):
        safe_artifacts: dict[str, str] = {}
        for name, value in artifact_hashes.items():
            if isinstance(name, str) and isinstance(value, str) and value:
                safe_artifacts[name] = value
        if safe_artifacts:
            lineage["artifact_hashes"] = safe_artifacts
    return lineage


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise TrainingError(f"checkpoint 파일이 없습니다: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracking_lineage(lineage: Mapping[str, Any]) -> dict[str, str]:
    """Flatten safe lineage hashes for MLflow parameters."""

    flattened: dict[str, str] = {}
    for key in ("dataset_manifest_hash", "resolved_config_hash"):
        value = lineage.get(key)
        if isinstance(value, str):
            flattened[key] = value
    artifacts = lineage.get("artifact_hashes")
    if isinstance(artifacts, Mapping):
        for name, value in artifacts.items():
            if isinstance(name, str) and isinstance(value, str):
                flattened[f"artifact_{name}_hash"] = value
    return flattened


def _metadata_subject(metadata: Any) -> str:
    if not isinstance(metadata, Mapping):
        return "unknown"
    return str(metadata.get("subject_id") or "unknown")


def _metadata_screen_sizes(metadata: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(metadata, Mapping):
        nan = torch.full((2,), torch.nan)
        return nan, nan.clone()
    source = metadata.get("front") if isinstance(metadata.get("front"), Mapping) else metadata
    px = torch.as_tensor(source.get("screen_size_px", [math.nan, math.nan])).float().reshape(2)
    mm = torch.as_tensor(source.get("screen_size_mm", [math.nan, math.nan])).float().reshape(2)
    return px, mm


def _valid_size_tensor(value: torch.Tensor | None, count: int) -> torch.Tensor | None:
    if value is None or value.numel() == 0:
        return None
    tensor = value.float().reshape(-1, 2)
    if tensor.shape[0] != count:
        return None
    positive = (tensor > 0).all(dim=1, keepdim=True)
    return torch.where(positive, tensor, torch.full_like(tensor, torch.nan))


def _move_batch(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_batch(child, device) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_batch(child, device) for child in value)
    if isinstance(value, list):
        return value
    return value


def _set_dataset_epoch(loader: DataLoader[Any], epoch: int) -> None:
    setter = getattr(loader.dataset, "set_epoch", None)
    if callable(setter):
        setter(epoch)


def _resolve_device(config: Mapping[str, Any]) -> torch.device:
    training = _mapping(config.get("training"), "training")
    if int(training.get("devices", 1)) != 1:
        raise TrainingError("현재 training.devices=1만 지원합니다.")
    accelerator = str(training.get("accelerator", "auto")).lower()
    if accelerator == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if accelerator in {"cpu", "cuda", "mps"}:
        if accelerator == "cuda" and not torch.cuda.is_available():
            raise TrainingError("CUDA를 요청했지만 사용할 수 없습니다.")
        if accelerator == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise TrainingError("MPS를 요청했지만 사용할 수 없습니다.")
        return torch.device(accelerator)
    raise TrainingError("training.accelerator는 auto, cpu, cuda, mps 중 하나여야 합니다.")


def _seed_everything(config: Mapping[str, Any]) -> None:
    experiment = _mapping(config.get("experiment"), "experiment")
    seed = int(experiment.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if bool(experiment.get("deterministic", False)):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _rng_state() -> dict[str, Any]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    return {
        "format_version": 1,
        "python": {
            "version": int(python_state[0]),
            "state": torch.tensor(python_state[1], dtype=torch.int64),
            "gauss_next": (None if python_state[2] is None else float(python_state[2])),
        },
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": torch.from_numpy(
                np.asarray(numpy_state[1], dtype=np.uint32).astype(np.int64, copy=True)
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state().detach().cpu(),
        "cuda": (
            [state.detach().cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _restore_rng_state(value: Any) -> None:
    if not isinstance(value, Mapping):
        return
    python_state = value.get("python")
    if isinstance(python_state, Mapping) and torch.is_tensor(python_state.get("state")):
        state_tensor = python_state["state"].detach().cpu().reshape(-1)
        random.setstate(
            (
                int(python_state.get("version", 3)),
                tuple(int(item) for item in state_tensor.tolist()),
                (
                    None
                    if python_state.get("gauss_next") is None
                    else float(python_state["gauss_next"])
                ),
            )
        )
    numpy_state = value.get("numpy")
    if isinstance(numpy_state, Mapping) and torch.is_tensor(numpy_state.get("state")):
        state_array = np.asarray(
            numpy_state["state"].detach().cpu().reshape(-1).tolist(),
            dtype=np.uint32,
        )
        np.random.set_state(
            (
                str(numpy_state.get("bit_generator", "MT19937")),
                state_array,
                int(numpy_state.get("position", 0)),
                int(numpy_state.get("has_gauss", 0)),
                float(numpy_state.get("cached_gaussian", 0.0)),
            )
        )
    torch_state = value.get("torch")
    if torch.is_tensor(torch_state):
        torch.set_rng_state(torch_state.detach().cpu())
    cuda_state = value.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, Sequence):
        states = [state.detach().cpu() for state in cuda_state if torch.is_tensor(state)]
        if states:
            torch.cuda.set_rng_state_all(states)


def _validate_training_runtime_config(config: Mapping[str, Any]) -> None:
    model = _mapping(config.get("model"), "model")
    enabled = [
        branch
        for branch in ("front", "side")
        if bool(_mapping(model.get(branch), f"model.{branch}").get("enabled", False))
    ]
    if not enabled:
        raise TrainingError("model.front 또는 model.side 중 하나는 enabled=true여야 합니다.")
    training = _mapping(config.get("training"), "training")
    if int(training.get("max_epochs", 0)) <= 0:
        raise TrainingError("training.max_epochs는 1 이상이어야 합니다.")
    if int(training.get("gradient_accumulation_steps", 1)) <= 0:
        raise TrainingError("training.gradient_accumulation_steps는 1 이상이어야 합니다.")
    if int(training.get("validate_every_n_epochs", 1)) <= 0:
        raise TrainingError("training.validate_every_n_epochs는 1 이상이어야 합니다.")
    if int(training.get("precision", 32)) != 32:
        raise TrainingError("현재 training.precision=32만 지원합니다.")
    fusion = _mapping(config.get("fusion"), "fusion")
    if bool(fusion.get("enabled", False)) and str(fusion.get("method")) != "y_axis_residual":
        raise TrainingError("현재 fusion.method는 y_axis_residual만 실행할 수 있습니다.")
    if bool(fusion.get("enabled", False)) and fusion.get("entrypoint") not in {None, ""}:
        raise TrainingError("현재 fusion.entrypoint는 지원하지 않습니다.")
    if (
        bool(fusion.get("enabled", False))
        and str(fusion.get("missing_branch_policy", "use_available_branch"))
        != "use_available_branch"
    ):
        raise TrainingError(
            "y_axis_residual은 fusion.missing_branch_policy=use_available_branch만 지원합니다."
        )


def _configured_path(value: Any, base_dir: Path, *, field: str) -> Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise TrainingError(f"{field} 경로가 필요합니다.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve(strict=False)


def _optional_path(value: Any, base_dir: Path) -> Path | None:
    return None if value in {None, ""} else _configured_path(value, base_dir, field="path")


def _project_base_dir(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    path = Path(config_path).expanduser().resolve(strict=False)
    for parent in path.parents:
        if parent.name == "configs":
            return parent.parent
    return path.parent


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingError(f"{field}는 mapping이어야 합니다.")
    return value


__all__ = [
    "PipelineResult",
    "TrainingError",
    "YAxisResidualFusion",
    "compute_gaze_metrics",
    "evaluate_pipeline",
    "train_pipeline",
]
