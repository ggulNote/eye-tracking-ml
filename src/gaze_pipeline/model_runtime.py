"""Config-driven PyTorch model loading and branch output normalization.

This module intentionally keeps the external-model boundary small.  A model
factory is imported from ``package.module:callable`` and receives only the
configured ``init_args``.  An optional adapter may translate canonical batch
keys to model arguments and model-specific return values to a mapping.  When
no factory is configured, small built-in models make smoke training possible.

Only state dictionaries from ``.pt``/``.pth`` files are accepted.  PyTorch is
always invoked with ``weights_only=True`` so loading a configured weight file
does not fall back to arbitrary pickle object construction.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import re
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn

_ENTRYPOINT_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth"})
_TORCH_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "bfloat16": torch.bfloat16,
    "double": torch.float64,
    "float": torch.float32,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "half": torch.float16,
    "int": torch.int32,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "uint8": torch.uint8,
}


class ModelRuntimeError(RuntimeError):
    """Raised when an external model, adapter, weight, or output is invalid."""


class ModelImportError(ModelRuntimeError):
    """Raised when a configured Python entrypoint cannot be imported."""


class ModelCheckpointError(ModelRuntimeError):
    """Raised when a PyTorch state dictionary cannot be loaded safely."""


class ModelContractError(ModelRuntimeError):
    """Raised when model inputs or standardized outputs violate a contract."""


def _resolve_path(path: str | Path, base_dir: str | Path | None) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute() and base_dir is not None:
        resolved = Path(base_dir).expanduser() / resolved
    return resolved.resolve()


@contextmanager
def _temporary_import_path(source_dir: Path | None):
    if source_dir is None:
        yield
        return
    source_text = str(source_dir)
    sys.path.insert(0, source_text)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        try:
            sys.path.remove(source_text)
        except ValueError:  # pragma: no cover - defensive against imported code
            pass


def import_entrypoint(
    entrypoint: str,
    *,
    source_dir: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> Any:
    """Import and return ``package.module:attribute``.

    ``source_dir`` is placed first on ``sys.path`` only for the duration of the
    import.  It must already exist and be a directory; this loader never clones
    or executes installer commands.
    """

    if not isinstance(entrypoint, str) or not _ENTRYPOINT_RE.fullmatch(entrypoint):
        raise ModelImportError(
            f"invalid entrypoint {entrypoint!r}; expected package.module:callable"
        )

    resolved_source: Path | None = None
    if source_dir is not None:
        resolved_source = _resolve_path(source_dir, base_dir)
        if not resolved_source.is_dir():
            raise ModelImportError(f"model source_dir is not a directory: {resolved_source}")

    module_name, attribute_path = entrypoint.split(":", 1)
    try:
        with _temporary_import_path(resolved_source):
            value: ModuleType | Any = importlib.import_module(module_name)
            for attribute in attribute_path.split("."):
                value = getattr(value, attribute)
    except Exception as exc:
        raise ModelImportError(f"could not import model entrypoint {entrypoint!r}: {exc}") from exc
    return value


def _build_from_factory(factory: Any, init_args: Mapping[str, Any], *, label: str) -> nn.Module:
    if isinstance(factory, nn.Module):
        if init_args:
            raise ModelImportError(f"{label} is already an nn.Module; init_args must be empty")
        return factory
    if not callable(factory):
        raise ModelImportError(f"{label} must resolve to a callable or nn.Module")
    try:
        model = factory(**dict(init_args))
    except Exception as exc:
        raise ModelImportError(f"{label} factory failed: {exc}") from exc
    if not isinstance(model, nn.Module):
        raise ModelImportError(f"{label} factory must return torch.nn.Module, got {type(model)!r}")
    return model


def _call_with_supported_context(factory: Any, context: Mapping[str, Any], *, label: str) -> Any:
    """Instantiate an adapter factory without guessing positional arguments."""

    if not callable(factory):
        raise ModelImportError(f"{label} must resolve to a callable")
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        kwargs: dict[str, Any] = {}
    else:
        accepts_extra = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        kwargs = {
            name: value
            for name, value in context.items()
            if accepts_extra or name in signature.parameters
        }
    try:
        return factory(**kwargs)
    except Exception as exc:
        raise ModelImportError(f"{label} factory failed: {exc}") from exc


def _checkpoint_state_dict(payload: Any, path: Path) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ModelCheckpointError(
            f"checkpoint must contain a state_dict mapping, got {type(payload)!r}: {path}"
        )
    for wrapper_key in ("state_dict", "model_state_dict", "model"):
        candidate = payload.get(wrapper_key)
        if isinstance(candidate, Mapping):
            payload = candidate
            break
    if not payload or not all(isinstance(key, str) for key in payload):
        raise ModelCheckpointError(f"checkpoint state_dict has no valid string keys: {path}")
    return payload


def _strip_data_parallel_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return dict(state_dict)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pytorch_state_dict(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
    sha256: str | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """Safely load a raw or wrapped PyTorch state dictionary into ``model``."""

    checkpoint_path = _resolve_path(path, base_dir)
    if checkpoint_path.suffix.lower() not in _CHECKPOINT_SUFFIXES:
        raise ModelCheckpointError(
            f"unsupported pretrained format {checkpoint_path.suffix!r}; use .pt or .pth"
        )
    if not checkpoint_path.is_file():
        raise ModelCheckpointError(f"pretrained state_dict does not exist: {checkpoint_path}")
    if sha256 is not None:
        expected = str(sha256).lower()
        actual = _sha256(checkpoint_path)
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ModelCheckpointError("pretrained sha256 must be a 64-character hex digest")
        if actual != expected:
            details = f"expected {expected}, got {actual}"
            raise ModelCheckpointError(
                f"pretrained sha256 mismatch for {checkpoint_path}: {details}"
            )

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelCheckpointError(
            f"could not safely load pretrained state_dict {checkpoint_path}: {exc}"
        ) from exc
    state_dict = _strip_data_parallel_prefix(_checkpoint_state_dict(payload, checkpoint_path))
    try:
        model.load_state_dict(state_dict, strict=bool(strict))
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ModelCheckpointError(
            f"pretrained state_dict is incompatible with the model ({checkpoint_path}): {exc}"
        ) from exc
    return checkpoint_path


def _fixed_contract_width(contract: Mapping[str, Any], key: str, default: int) -> int:
    shape = contract.get(key)
    if isinstance(shape, Sequence) and not isinstance(shape, str | bytes) and len(shape) == 2:
        width = shape[1]
        if isinstance(width, int) and not isinstance(width, bool) and width > 0:
            return width
    return default


def _input_channels(input_contract: Mapping[str, Any]) -> int:
    shape = input_contract.get("shape")
    if isinstance(shape, Sequence) and not isinstance(shape, str | bytes) and len(shape) == 4:
        channels = shape[1]
        if isinstance(channels, int) and not isinstance(channels, bool) and channels > 0:
            return channels
    return 3


class _SimpleImageEncoder(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(24, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, image: Tensor) -> Tensor:
        if not isinstance(image, Tensor) or image.ndim != 4:
            shape = getattr(image, "shape", None)
            raise ModelContractError(f"fallback model image must have shape [B,C,H,W], got {shape}")
        return self.network(image)


class SimpleFrontModel(nn.Module):
    """Small trainable front-view fallback for end-to-end smoke runs."""

    def __init__(self, image_key: str, in_channels: int = 3, embedding_dim: int = 32) -> None:
        super().__init__()
        self.image_key = image_key
        self.encoder = _SimpleImageEncoder(in_channels, embedding_dim)
        self.gaze_head = nn.Linear(embedding_dim, 2)

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        if self.image_key not in inputs:
            raise ModelContractError(f"fallback front model requires batch key {self.image_key!r}")
        embedding = self.encoder(inputs[self.image_key])
        gaze = self.gaze_head(embedding)
        return {
            "gaze_xy": gaze,
            "x_front": gaze[:, 0:1],
            "y_front": gaze[:, 1:2],
            "front_embedding": embedding,
        }


class SimpleSideModel(nn.Module):
    """Small trainable side-view fallback that predicts only a y residual."""

    def __init__(self, image_key: str, in_channels: int = 3, embedding_dim: int = 32) -> None:
        super().__init__()
        self.image_key = image_key
        self.encoder = _SimpleImageEncoder(in_channels, embedding_dim)
        self.residual_head = nn.Linear(embedding_dim, 1)

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        if self.image_key not in inputs:
            raise ModelContractError(f"fallback side model requires batch key {self.image_key!r}")
        embedding = self.encoder(inputs[self.image_key])
        return {
            "delta_y_side": self.residual_head(embedding),
            "side_embedding": embedding,
        }


class DefaultModelAdapter:
    """Pass declared batch keys as kwargs and normalize common raw outputs."""

    def __init__(
        self,
        *,
        branch: str,
        forward_keys: Sequence[str],
        output_contract: Mapping[str, Any],
    ) -> None:
        self.branch = branch
        self.forward_keys = tuple(forward_keys)
        self.output_contract = dict(output_contract)

    def to_model_inputs(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        missing = [key for key in self.forward_keys if key not in batch]
        if missing:
            raise ModelContractError(
                f"model.{self.branch} forward input is missing batch keys: {missing}"
            )
        return {key: batch[key] for key in self.forward_keys}

    def to_standard_outputs(self, raw_output: Any) -> dict[str, Any]:
        if isinstance(raw_output, Mapping):
            return dict(raw_output)
        if isinstance(raw_output, Tensor):
            return {self._primary_key(): raw_output}
        if isinstance(raw_output, Sequence) and not isinstance(raw_output, str | bytes):
            values = list(raw_output)
            if not values:
                raise ModelContractError(f"model.{self.branch} returned an empty output sequence")
            outputs: dict[str, Any] = {self._primary_key(): values[0]}
            embedding_key = self.output_contract.get("embedding_key")
            if len(values) > 1 and isinstance(embedding_key, str) and embedding_key:
                outputs[embedding_key] = values[1]
            if len(values) > 2:
                raise ModelContractError(
                    f"model.{self.branch} returned extra positional outputs; configure an adapter"
                )
            return outputs
        raise ModelContractError(
            f"model.{self.branch} output must be a mapping, tensor, or sequence; "
            f"got {type(raw_output)!r}"
        )

    def _primary_key(self) -> str:
        if self.branch == "side":
            key = self.output_contract.get("delta_y_key")
            if isinstance(key, str) and key:
                return key
        key = self.output_contract.get("gaze_key")
        if isinstance(key, str) and key:
            return key
        return "delta_y_side" if self.branch == "side" else "gaze_xy"


def _normalize_model_inputs(value: Any, *, branch: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if isinstance(value, Mapping):
        return (), dict(value)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], str | bytes)
        and isinstance(value[1], Mapping)
    ):
        return tuple(value[0]), dict(value[1])
    raise ModelContractError(
        f"model.{branch} adapter.to_model_inputs must return kwargs mapping or (args, kwargs)"
    )


def _shape_matches(actual: Sequence[int], expected: Any, batch_size: int | None) -> bool:
    if not isinstance(expected, Sequence) or isinstance(expected, str | bytes):
        return True
    if len(actual) != len(expected):
        return False
    for index, (actual_dim, expected_dim) in enumerate(zip(actual, expected, strict=True)):
        if isinstance(expected_dim, str):
            if expected_dim == "B" and batch_size is not None and actual_dim != batch_size:
                return False
            continue
        if isinstance(expected_dim, int | float) and not isinstance(expected_dim, bool):
            if actual_dim != int(expected_dim):
                return False
        elif expected_dim is not None:
            return False
        if index == 0 and batch_size is not None and actual_dim != batch_size:
            return False
    return True


def _declared_dtype(value: Any, *, field: str) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if not isinstance(value, str):
        raise ModelContractError(f"{field} must be a PyTorch dtype name, got {value!r}")
    normalized = value.strip().lower().removeprefix("torch.")
    dtype = _TORCH_DTYPES.get(normalized)
    if dtype is None:
        raise ModelContractError(f"{field} uses unsupported dtype {value!r}")
    return dtype


def _validity_rows(
    batch: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    *,
    branch: str,
    batch_size: int,
) -> Tensor | None:
    validity_key = input_contract.get("validity_key")
    if not isinstance(validity_key, str) or not validity_key:
        return None
    value = batch.get(validity_key)
    if value is None:
        return None
    try:
        mask = torch.as_tensor(value, dtype=torch.bool).reshape(-1)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ModelContractError(
            f"model.{branch} input validity key {validity_key!r} must be boolean-like"
        ) from exc
    if mask.shape[0] != batch_size:
        raise ModelContractError(
            f"model.{branch} input validity key {validity_key!r} has shape "
            f"{tuple(mask.shape)}, expected ['B'] with B={batch_size}"
        )
    return mask


def _value_range(value: Any, *, field: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
    ):
        raise ModelContractError(f"{field} must be a numeric [minimum, maximum] pair")
    try:
        lower, upper = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ModelContractError(f"{field} must be a numeric [minimum, maximum] pair") from exc
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ModelContractError(f"{field} must contain finite values with minimum <= maximum")
    return lower, upper


def _validate_input_tensor(
    value: Any,
    contract: Mapping[str, Any],
    *,
    branch: str,
    key: str,
    batch_size: int,
    valid_rows: Tensor | None,
) -> Tensor:
    label = f"model.{branch} input {key!r}"
    if not isinstance(value, Tensor):
        raise ModelContractError(f"{label} must be a torch.Tensor, got {type(value)!r}")

    expected_shape = contract.get("shape")
    if expected_shape is not None:
        if (
            not isinstance(expected_shape, Sequence)
            or isinstance(expected_shape, str | bytes)
            or not expected_shape
            or expected_shape[0] != "B"
        ):
            raise ModelContractError(
                f"{label} contract shape must be a non-empty list beginning with 'B'"
            )
        if not _shape_matches(tuple(value.shape), expected_shape, batch_size):
            raise ModelContractError(
                f"{label} has shape {tuple(value.shape)}, expected {expected_shape}"
            )

    expected_dtype = _declared_dtype(contract.get("dtype"), field=f"{label} dtype")
    if expected_dtype is not None and value.dtype != expected_dtype:
        raise ModelContractError(f"{label} has dtype {value.dtype}, expected {expected_dtype}")

    checked = value.detach()
    if valid_rows is not None:
        if checked.ndim == 0:
            raise ModelContractError(f"{label} cannot apply a batch validity mask to a scalar")
        checked = checked[valid_rows.to(device=checked.device)]
    if checked.numel() == 0:
        return value

    if checked.is_floating_point() or checked.is_complex():
        if not bool(torch.isfinite(checked).all().item()):
            scope = "valid rows" if valid_rows is not None else "tensor"
            raise ModelContractError(f"{label} contains NaN or infinity in {scope}")

    declared_range = _value_range(contract.get("value_range"), field=f"{label} value_range")
    if declared_range is not None:
        if checked.is_complex():
            raise ModelContractError(f"{label} cannot apply value_range to a complex tensor")
        lower, upper = declared_range
        actual_min = float(checked.amin().cpu())
        actual_max = float(checked.amax().cpu())
        tolerance = 1e-6
        if actual_min < lower - tolerance or actual_max > upper + tolerance:
            raise ModelContractError(
                f"{label} values [{actual_min:.6g}, {actual_max:.6g}] are outside "
                f"declared range [{lower:.6g}, {upper:.6g}]"
            )
    return value


def _keyed_auxiliary_contracts(value: Any) -> dict[str, Mapping[str, Any]]:
    contracts: dict[str, Mapping[str, Any]] = {}

    def visit(node: Any) -> None:
        if not isinstance(node, Mapping):
            return
        key = node.get("key")
        if (
            isinstance(key, str)
            and key
            and any(field in node for field in ("shape", "dtype", "value_range"))
        ):
            contracts[key] = node
        for child in node.values():
            if isinstance(child, Mapping):
                visit(child)

    visit(value)
    return contracts


def validate_model_inputs(
    batch: Mapping[str, Any],
    *,
    branch: str,
    input_contract: Mapping[str, Any],
    forward_keys: Sequence[str],
) -> int:
    """Validate canonical batch tensors before an external adapter transforms them.

    The primary image contract is always required.  Auxiliary contracts are
    validated only for selected ``forward_keys`` whose key and tensor contract
    are explicit.  Invalid rows may carry auxiliary sentinels such as NaN when
    the branch ``validity_key`` masks those rows, but valid rows must be finite.
    """

    if not isinstance(batch, Mapping):
        raise ModelContractError(f"model.{branch} batch must be a mapping")
    image_key = input_contract.get("image_key")
    if not isinstance(image_key, str) or not image_key:
        raise ModelContractError(f"model.{branch}.input_contract.image_key is required")
    if image_key not in batch:
        raise ModelContractError(f"model.{branch} input is missing primary image key {image_key!r}")
    image = batch[image_key]
    if not isinstance(image, Tensor):
        raise ModelContractError(
            f"model.{branch} input {image_key!r} must be a torch.Tensor, got {type(image)!r}"
        )
    if image.ndim == 0 or image.shape[0] <= 0:
        raise ModelContractError(
            f"model.{branch} input {image_key!r} must have a non-empty batch dimension"
        )
    batch_size = int(image.shape[0])
    _validate_input_tensor(
        image,
        input_contract,
        branch=branch,
        key=image_key,
        batch_size=batch_size,
        valid_rows=None,
    )

    valid_rows = _validity_rows(
        batch,
        input_contract,
        branch=branch,
        batch_size=batch_size,
    )
    auxiliary = _keyed_auxiliary_contracts(input_contract.get("auxiliary_keys"))
    for key in forward_keys:
        if key == image_key:
            continue
        if key not in batch:
            raise ModelContractError(f"model.{branch} input is missing forward key {key!r}")
        if key not in auxiliary:
            value = batch[key]
            if not isinstance(value, Tensor):
                raise ModelContractError(
                    f"model.{branch} input {key!r} must be a torch.Tensor, got {type(value)!r}"
                )
            if value.ndim == 0 or int(value.shape[0]) != batch_size:
                raise ModelContractError(
                    f"model.{branch} input {key!r} must have batch dimension B={batch_size}, "
                    f"got {tuple(value.shape)}"
                )
            continue
        _validate_input_tensor(
            batch[key],
            auxiliary[key],
            branch=branch,
            key=key,
            batch_size=batch_size,
            valid_rows=valid_rows,
        )
    return batch_size


def _validate_output(
    outputs: Mapping[str, Any],
    *,
    key: str,
    shape: Any,
    branch: str,
    batch_size: int | None,
    required: bool,
) -> None:
    if key not in outputs:
        if required:
            raise ModelContractError(f"model.{branch} output is missing required key {key!r}")
        return
    value = outputs[key]
    if not isinstance(value, Tensor):
        raise ModelContractError(f"model.{branch} output {key!r} must be a torch.Tensor")
    if not _shape_matches(tuple(value.shape), shape, batch_size):
        raise ModelContractError(
            f"model.{branch} output {key!r} has shape {tuple(value.shape)}, expected {shape}"
        )


def validate_standard_outputs(
    outputs: Mapping[str, Any],
    *,
    branch: str,
    output_contract: Mapping[str, Any],
    batch_size: int | None = None,
) -> dict[str, Tensor]:
    """Validate required primary output and any optional output that is present."""

    if not isinstance(outputs, Mapping):
        raise ModelContractError(f"model.{branch} standardized output must be a mapping")
    branch = branch.lower()
    if branch not in {"front", "side"}:
        raise ModelContractError(f"unsupported model branch {branch!r}")

    delta_key = output_contract.get("delta_y_key")
    gaze_key = output_contract.get("gaze_key")
    if branch == "side" and isinstance(delta_key, str) and delta_key:
        primary_key = delta_key
        primary_shape = output_contract.get("delta_y_shape", ["B", 1])
    else:
        primary_key = gaze_key if isinstance(gaze_key, str) and gaze_key else "gaze_xy"
        primary_shape = output_contract.get("gaze_shape", ["B", 2])
    standardized = {key: value for key, value in outputs.items() if isinstance(value, Tensor)}
    canonical_key = "delta_y_side" if branch == "side" else "gaze_xy"
    if primary_key not in standardized and canonical_key in standardized:
        standardized[primary_key] = standardized[canonical_key]
    _validate_output(
        standardized,
        key=primary_key,
        shape=primary_shape,
        branch=branch,
        batch_size=batch_size,
        required=True,
    )

    optional_contracts = (
        ("embedding_key", "embedding_shape"),
        ("uncertainty_key", "uncertainty_shape"),
        ("quality_key", "quality_shape"),
        ("reconstruction_key", "reconstruction_shape"),
    )
    for key_field, shape_field in optional_contracts:
        key = output_contract.get(key_field)
        if isinstance(key, str) and key:
            _validate_output(
                standardized,
                key=key,
                shape=output_contract.get(shape_field),
                branch=branch,
                batch_size=batch_size,
                required=False,
            )
    # The runtime boundary is canonical even when an external adapter uses a
    # model-specific contract key.  This lets the generic trainer stay model
    # agnostic while preserving the original adapter output for diagnostics.
    standardized.setdefault(canonical_key, standardized[primary_key])
    return standardized


class ModelRuntime(nn.Module):
    """Own a branch model, its adapter, and runtime I/O contract checks."""

    def __init__(
        self,
        *,
        branch: str,
        model: nn.Module,
        adapter: Any,
        input_contract: Mapping[str, Any],
        output_contract: Mapping[str, Any],
        forward_keys: Sequence[str],
        loaded_pretrained_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.branch = branch
        self.model = model
        self.adapter = adapter
        self.input_contract = dict(input_contract)
        self.output_contract = dict(output_contract)
        self.forward_keys = tuple(forward_keys)
        self.loaded_pretrained_path = loaded_pretrained_path
        for method in ("to_model_inputs", "to_standard_outputs"):
            if not callable(getattr(adapter, method, None)):
                raise ModelImportError(f"model.{branch} adapter must define callable {method}()")

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        batch_size = validate_model_inputs(
            batch,
            branch=self.branch,
            input_contract=self.input_contract,
            forward_keys=self.forward_keys,
        )
        try:
            adapted_inputs = self.adapter.to_model_inputs(batch)
        except ModelRuntimeError:
            raise
        except Exception as exc:
            raise ModelContractError(
                f"model.{self.branch} adapter.to_model_inputs failed: {exc}"
            ) from exc
        args, kwargs = _normalize_model_inputs(adapted_inputs, branch=self.branch)
        try:
            raw_output = self.model(*args, **kwargs)
        except ModelRuntimeError:
            raise
        except Exception as exc:
            raise ModelRuntimeError(f"model.{self.branch} forward failed: {exc}") from exc
        try:
            standardized = self.adapter.to_standard_outputs(raw_output)
        except ModelRuntimeError:
            raise
        except Exception as exc:
            raise ModelContractError(
                f"model.{self.branch} adapter.to_standard_outputs failed: {exc}"
            ) from exc
        return validate_standard_outputs(
            standardized,
            branch=self.branch,
            output_contract=self.output_contract,
            batch_size=batch_size,
        )


def _default_forward_keys(input_contract: Mapping[str, Any]) -> tuple[str, ...]:
    declared = input_contract.get("forward_keys")
    if declared == "auto":
        raise ModelContractError(
            "forward_keys='auto' requires the resolved forward_keys from the full config"
        )
    if isinstance(declared, Sequence) and not isinstance(declared, str | bytes):
        keys = tuple(str(key) for key in declared)
        if keys:
            return keys
    image_key = input_contract.get("image_key")
    if not isinstance(image_key, str) or not image_key:
        raise ModelContractError("model input_contract.image_key must be a non-empty string")
    keys = [image_key]
    auxiliary = input_contract.get("auxiliary_keys")
    if isinstance(auxiliary, Mapping):
        for value in auxiliary.values():
            if isinstance(value, Mapping):
                key = value.get("key")
                if isinstance(key, str) and key:
                    keys.append(key)
    return tuple(keys)


def _fallback_model(
    branch: str,
    input_contract: Mapping[str, Any],
    output_contract: Mapping[str, Any],
    init_args: Mapping[str, Any],
) -> nn.Module:
    allowed = {"embedding_dim", "in_channels"}
    unexpected = sorted(set(init_args) - allowed)
    if unexpected:
        raise ModelImportError(
            f"built-in {branch} fallback does not support init_args: {unexpected}; "
            "configure model.entrypoint for a custom constructor"
        )
    image_key = input_contract.get("image_key")
    if not isinstance(image_key, str) or not image_key:
        raise ModelContractError(f"model.{branch}.input_contract.image_key is required")
    embedding_dim = int(
        init_args.get(
            "embedding_dim",
            _fixed_contract_width(output_contract, "embedding_shape", 32),
        )
    )
    in_channels = int(init_args.get("in_channels", _input_channels(input_contract)))
    if embedding_dim <= 0 or in_channels <= 0:
        raise ModelImportError("fallback in_channels and embedding_dim must be positive")
    model_type = SimpleFrontModel if branch == "front" else SimpleSideModel
    return model_type(image_key, in_channels=in_channels, embedding_dim=embedding_dim)


def build_model_runtime(
    branch: str,
    branch_config: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    forward_keys: Sequence[str] | None = None,
) -> ModelRuntime:
    """Build a configured front/side runtime, including optional weights.

    ``forward_keys`` should be supplied by the full-config feature resolver for
    contracts that use ``forward_keys: auto``.  Other contracts are resolved
    locally from ``input_contract``.
    """

    branch = str(branch).lower()
    if branch not in {"front", "side"}:
        raise ModelRuntimeError(f"unsupported model branch {branch!r}; use 'front' or 'side'")
    if not isinstance(branch_config, Mapping):
        raise ModelRuntimeError(f"model.{branch} config must be a mapping")
    input_contract = branch_config.get("input_contract")
    output_contract = branch_config.get("output_contract")
    if not isinstance(input_contract, Mapping) or not isinstance(output_contract, Mapping):
        raise ModelContractError(f"model.{branch} requires input_contract and output_contract")
    resolved_keys = (
        tuple(forward_keys) if forward_keys is not None else _default_forward_keys(input_contract)
    )
    if not resolved_keys or len(resolved_keys) != len(set(resolved_keys)):
        raise ModelContractError(f"model.{branch} forward_keys must be non-empty and unique")

    init_args = branch_config.get("init_args", {})
    if not isinstance(init_args, Mapping):
        raise ModelImportError(f"model.{branch}.init_args must be a mapping")
    source_dir = branch_config.get("source_dir")
    entrypoint = branch_config.get("entrypoint")
    if entrypoint is None:
        model = _fallback_model(branch, input_contract, output_contract, init_args)
    else:
        factory = import_entrypoint(entrypoint, source_dir=source_dir, base_dir=base_dir)
        model = _build_from_factory(factory, init_args, label=f"model.{branch}.entrypoint")

    adapter_entrypoint = branch_config.get("adapter_entrypoint")
    if adapter_entrypoint is None:
        adapter: Any = DefaultModelAdapter(
            branch=branch,
            forward_keys=resolved_keys,
            output_contract=output_contract,
        )
    else:
        adapter_factory = import_entrypoint(
            adapter_entrypoint,
            source_dir=source_dir,
            base_dir=base_dir,
        )
        adapter = _call_with_supported_context(
            adapter_factory,
            {
                "model": model,
                "branch": branch,
                "input_contract": dict(input_contract),
                "output_contract": dict(output_contract),
                "forward_keys": resolved_keys,
            },
            label=f"model.{branch}.adapter_entrypoint",
        )

    loaded_path: Path | None = None
    pretrained = branch_config.get("pretrained")
    if pretrained is not None and not isinstance(pretrained, Mapping):
        raise ModelCheckpointError(f"model.{branch}.pretrained must be a mapping")
    if isinstance(pretrained, Mapping) and pretrained.get("path") is not None:
        loaded_path = load_pytorch_state_dict(
            model,
            pretrained["path"],
            strict=bool(pretrained.get("strict", True)),
            sha256=pretrained.get("sha256"),
            base_dir=base_dir,
        )

    runtime = ModelRuntime(
        branch=branch,
        model=model,
        adapter=adapter,
        input_contract=input_contract,
        output_contract=output_contract,
        forward_keys=resolved_keys,
        loaded_pretrained_path=loaded_path,
    )
    if device is not None:
        runtime.to(device)
    return runtime


__all__ = [
    "DefaultModelAdapter",
    "ModelCheckpointError",
    "ModelContractError",
    "ModelImportError",
    "ModelRuntime",
    "ModelRuntimeError",
    "SimpleFrontModel",
    "SimpleSideModel",
    "build_model_runtime",
    "import_entrypoint",
    "load_pytorch_state_dict",
    "validate_model_inputs",
    "validate_standard_outputs",
]
