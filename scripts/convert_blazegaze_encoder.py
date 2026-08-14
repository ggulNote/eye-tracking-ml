"""Convert the pinned WebEyeTrack BlazeGaze Keras encoder weights to PyTorch.

TensorFlow is intentionally imported only by the CLI conversion path.  The
layout and explicit layer-mapping helpers can be imported and tested with only
NumPy and PyTorch installed.

Example::

    python scripts/convert_blazegaze_encoder.py \
      --source /absolute/path/blazegaze.keras \
      --output /absolute/path/blazegaze_side_encoder.pt \
      --source-commit 14719ad861467c98890058f7c41a94638ae1db2b

The output contains only an encoder ``state_dict`` plus inert lineage
metadata.  Side projection, auxiliary, embedding, gaze, and quality weights
are never read from or written to this artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gaze_pipeline.models.side.blazegaze_transfer import (
    ENCODER_STATE_DICT_FORMAT,
    SOURCE_COMMIT,
    SOURCE_FUNCTION,
    SOURCE_LICENSE,
    SOURCE_REPOSITORY,
    BlazeGazeConvEncoder,
)

MAPPING_VERSION = "webeyetrack-14719ad-cnn-encoder.v1"
SUPPORTED_SOURCE_LAYER_TYPES = frozenset({"Conv2D", "DepthwiseConv2D", "BatchNormalization"})


@dataclass(frozen=True)
class SourceLayerWeights:
    """Framework-neutral Keras layer description used by the pure converter."""

    name: str
    layer_type: str
    tensors: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class TensorMapping:
    """One explicit source tensor to target tensor conversion."""

    source_tensor: str | None
    source_shape: tuple[int, ...] | None
    target_key: str
    target_shape: tuple[int, ...]
    layout: str


@dataclass(frozen=True)
class LayerMapping:
    """Pinned Keras layer role/type and all of its target state_dict entries."""

    role: str
    source_layer: str
    source_type: str
    tensors: tuple[TensorMapping, ...]


def convert_conv2d_kernel(kernel: np.ndarray) -> np.ndarray:
    """Convert a Keras Conv2D kernel from HWIO to PyTorch OIHW layout."""

    array = np.asarray(kernel)
    if array.ndim != 4:
        raise ValueError(f"Conv2D kernel must be rank 4 HWIO, got shape {array.shape}")
    return np.ascontiguousarray(array.transpose(3, 2, 0, 1))


def convert_depthwise_conv2d_kernel(kernel: np.ndarray) -> np.ndarray:
    """Convert Keras HWIM depthwise weights to PyTorch ``(I*M),1,H,W``."""

    array = np.asarray(kernel)
    if array.ndim != 4:
        raise ValueError(f"DepthwiseConv2D kernel must be rank 4 HWIM, got shape {array.shape}")
    height, width, input_channels, multiplier = array.shape
    converted = array.transpose(2, 3, 0, 1).reshape(input_channels * multiplier, 1, height, width)
    return np.ascontiguousarray(converted)


def convert_batch_norm_tensors(
    gamma: np.ndarray,
    beta: np.ndarray,
    moving_mean: np.ndarray,
    moving_variance: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map Keras BatchNorm tensors to PyTorch names without changing layout."""

    arrays = {
        "weight": np.asarray(gamma),
        "bias": np.asarray(beta),
        "running_mean": np.asarray(moving_mean),
        "running_var": np.asarray(moving_variance),
    }
    shapes = {name: value.shape for name, value in arrays.items()}
    if any(value.ndim != 1 for value in arrays.values()) or len(set(shapes.values())) != 1:
        raise ValueError(f"BatchNorm tensors must be equal-length rank-1 arrays, got {shapes}")
    return {name: np.ascontiguousarray(value).copy() for name, value in arrays.items()}


def _conv(
    role: str,
    source_layer: str,
    target_prefix: str,
    source_shape: tuple[int, int, int, int],
) -> LayerMapping:
    height, width, input_channels, output_channels = source_shape
    return LayerMapping(
        role,
        source_layer,
        "Conv2D",
        (
            TensorMapping(
                "kernel",
                source_shape,
                f"{target_prefix}.weight",
                (output_channels, input_channels, height, width),
                "conv2d_hwio_to_oihw",
            ),
            TensorMapping(
                "bias",
                (output_channels,),
                f"{target_prefix}.bias",
                (output_channels,),
                "identity",
            ),
        ),
    )


def _depthwise(
    role: str,
    source_layer: str,
    target_prefix: str,
    source_shape: tuple[int, int, int, int],
) -> LayerMapping:
    height, width, input_channels, multiplier = source_shape
    output_channels = input_channels * multiplier
    return LayerMapping(
        role,
        source_layer,
        "DepthwiseConv2D",
        (
            TensorMapping(
                "kernel",
                source_shape,
                f"{target_prefix}.weight",
                (output_channels, 1, height, width),
                "depthwise_hwim_to_im1hw",
            ),
            TensorMapping(
                "bias",
                (output_channels,),
                f"{target_prefix}.bias",
                (output_channels,),
                "identity",
            ),
        ),
    )


def _batch_norm(role: str, source_layer: str, target_prefix: str, channels: int) -> LayerMapping:
    return LayerMapping(
        role,
        source_layer,
        "BatchNormalization",
        (
            TensorMapping("gamma", (channels,), f"{target_prefix}.weight", (channels,), "identity"),
            TensorMapping("beta", (channels,), f"{target_prefix}.bias", (channels,), "identity"),
            TensorMapping(
                "moving_mean",
                (channels,),
                f"{target_prefix}.running_mean",
                (channels,),
                "identity",
            ),
            TensorMapping(
                "moving_variance",
                (channels,),
                f"{target_prefix}.running_var",
                (channels,),
                "identity",
            ),
            TensorMapping(
                None,
                None,
                f"{target_prefix}.num_batches_tracked",
                (),
                "zero_int64",
            ),
        ),
    )


# This table is deliberately verbose.  Each role is bound to the canonical
# Keras name produced by the pinned source, its exact layer type and source
# shapes, and exact PyTorch state_dict keys/shapes.  No name-only or
# order-only fallback is allowed.
EXPLICIT_ENCODER_MAPPING: tuple[LayerMapping, ...] = (
    _conv("first_conv", "conv2d", "first_conv.0.conv", (5, 5, 3, 24)),
    _depthwise("single_1.depthwise", "depthwise_conv2d", "single_1.depthwise.conv", (5, 5, 24, 1)),
    _conv("single_1.pointwise", "conv2d_1", "single_1.pointwise.conv", (1, 1, 24, 24)),
    _depthwise(
        "single_2.depthwise",
        "depthwise_conv2d_1",
        "single_2.depthwise.conv",
        (5, 5, 24, 1),
    ),
    _conv("single_2.pointwise", "conv2d_2", "single_2.pointwise.conv", (1, 1, 24, 24)),
    _depthwise(
        "single_3.depthwise",
        "depthwise_conv2d_2",
        "single_3.depthwise.conv",
        (5, 5, 24, 1),
    ),
    _conv("single_3.pointwise", "conv2d_3", "single_3.pointwise.conv", (1, 1, 24, 48)),
    _conv(
        "single_3.residual_projection",
        "conv2d_4",
        "single_3.residual_projection.conv",
        (1, 1, 24, 48),
    ),
    _depthwise(
        "single_4.depthwise",
        "depthwise_conv2d_3",
        "single_4.depthwise.conv",
        (5, 5, 48, 1),
    ),
    _conv("single_4.pointwise", "conv2d_5", "single_4.pointwise.conv", (1, 1, 48, 48)),
    _depthwise(
        "single_5.depthwise",
        "depthwise_conv2d_4",
        "single_5.depthwise.conv",
        (5, 5, 48, 1),
    ),
    _conv("single_5.pointwise", "conv2d_6", "single_5.pointwise.conv", (1, 1, 48, 48)),
    _depthwise(
        "double_1.depthwise_1",
        "depthwise_conv2d_5",
        "double_1.depthwise_1.conv",
        (5, 5, 48, 1),
    ),
    _conv("double_1.pointwise_1", "conv2d_7", "double_1.pointwise_1.conv", (1, 1, 48, 24)),
    _depthwise(
        "double_1.depthwise_2",
        "depthwise_conv2d_6",
        "double_1.depthwise_2.conv",
        (5, 5, 24, 1),
    ),
    _conv("double_1.pointwise_2", "conv2d_8", "double_1.pointwise_2.conv", (1, 1, 24, 96)),
    _conv(
        "double_1.residual_projection",
        "conv2d_9",
        "double_1.residual_projection.conv",
        (1, 1, 48, 96),
    ),
    _depthwise(
        "double_2.depthwise_1",
        "depthwise_conv2d_7",
        "double_2.depthwise_1.conv",
        (5, 5, 96, 1),
    ),
    _conv("double_2.pointwise_1", "conv2d_10", "double_2.pointwise_1.conv", (1, 1, 96, 24)),
    _depthwise(
        "double_2.depthwise_2",
        "depthwise_conv2d_8",
        "double_2.depthwise_2.conv",
        (5, 5, 24, 1),
    ),
    _conv("double_2.pointwise_2", "conv2d_11", "double_2.pointwise_2.conv", (1, 1, 24, 96)),
    _depthwise(
        "double_3.depthwise_1",
        "depthwise_conv2d_9",
        "double_3.depthwise_1.conv",
        (5, 5, 96, 1),
    ),
    _conv("double_3.pointwise_1", "conv2d_12", "double_3.pointwise_1.conv", (1, 1, 96, 24)),
    _depthwise(
        "double_3.depthwise_2",
        "depthwise_conv2d_10",
        "double_3.depthwise_2.conv",
        (5, 5, 24, 1),
    ),
    _conv("double_3.pointwise_2", "conv2d_13", "double_3.pointwise_2.conv", (1, 1, 24, 96)),
    _depthwise(
        "double_4.depthwise_1",
        "depthwise_conv2d_11",
        "double_4.depthwise_1.conv",
        (5, 5, 96, 1),
    ),
    _conv("double_4.pointwise_1", "conv2d_14", "double_4.pointwise_1.conv", (1, 1, 96, 24)),
    _depthwise(
        "double_4.depthwise_2",
        "depthwise_conv2d_12",
        "double_4.depthwise_2.conv",
        (5, 5, 24, 1),
    ),
    _conv("double_4.pointwise_2", "conv2d_15", "double_4.pointwise_2.conv", (1, 1, 24, 96)),
    _conv(
        "double_4.residual_projection",
        "conv2d_16",
        "double_4.residual_projection.conv",
        (1, 1, 96, 96),
    ),
    _depthwise(
        "double_5.depthwise_1",
        "depthwise_conv2d_13",
        "double_5.depthwise_1.conv",
        (5, 5, 96, 1),
    ),
    _conv("double_5.pointwise_1", "conv2d_17", "double_5.pointwise_1.conv", (1, 1, 96, 24)),
    _depthwise(
        "double_5.depthwise_2",
        "depthwise_conv2d_14",
        "double_5.depthwise_2.conv",
        (5, 5, 24, 1),
    ),
    _conv("double_5.pointwise_2", "conv2d_18", "double_5.pointwise_2.conv", (1, 1, 24, 96)),
    _depthwise(
        "double_6.depthwise_1",
        "depthwise_conv2d_15",
        "double_6.depthwise_1.conv",
        (5, 5, 96, 1),
    ),
    _conv("double_6.pointwise_1", "conv2d_19", "double_6.pointwise_1.conv", (1, 1, 96, 24)),
    _depthwise(
        "double_6.depthwise_2",
        "depthwise_conv2d_16",
        "double_6.depthwise_2.conv",
        (5, 5, 24, 1),
    ),
    _conv("double_6.pointwise_2", "conv2d_20", "double_6.pointwise_2.conv", (1, 1, 24, 96)),
    _conv("squeeze_1.conv", "conv2d_21", "squeeze_1.0.conv", (3, 3, 96, 64)),
    _batch_norm("squeeze_1.batch_norm", "batch_normalization", "squeeze_1.2", 64),
    _conv("squeeze_2.conv", "conv2d_22", "squeeze_2.0.conv", (3, 3, 64, 32)),
    _batch_norm("squeeze_2.batch_norm", "batch_normalization_1", "squeeze_2.2", 32),
)


def convert_explicit_encoder_layers(
    source_layers: Sequence[SourceLayerWeights],
) -> dict[str, np.ndarray]:
    """Validate and convert only the explicitly mapped ``cnn_encoder`` layers.

    This function has no TensorFlow or file-system dependency.  Every mapped
    role validates source name, source layer type, source tensor shape,
    transformed target shape, and target key uniqueness.
    """

    layers_by_name = {layer.name: layer for layer in source_layers}
    if len(layers_by_name) != len(source_layers):
        raise ValueError("source encoder contains duplicate supported layer names")

    expected_names = {mapping.source_layer for mapping in EXPLICIT_ENCODER_MAPPING}
    actual_names = set(layers_by_name)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise ValueError(
            "supported cnn_encoder layers do not match the explicit mapping: "
            f"missing={missing}, extra={extra}"
        )

    converted: dict[str, np.ndarray] = {}
    for mapping in EXPLICIT_ENCODER_MAPPING:
        layer = layers_by_name[mapping.source_layer]
        if layer.layer_type != mapping.source_type:
            raise ValueError(
                f"{mapping.role}: source layer {layer.name!r} must be "
                f"{mapping.source_type}, got {layer.layer_type}"
            )
        expected_tensors = {
            tensor.source_tensor for tensor in mapping.tensors if tensor.source_tensor is not None
        }
        actual_tensors = set(layer.tensors)
        if actual_tensors != expected_tensors:
            raise ValueError(
                f"{mapping.role}: source tensors must be {sorted(expected_tensors)}, "
                f"got {sorted(actual_tensors)}"
            )

        batch_norm_values: dict[str, np.ndarray] | None = None
        if mapping.source_type == "BatchNormalization":
            batch_norm_values = convert_batch_norm_tensors(
                layer.tensors["gamma"],
                layer.tensors["beta"],
                layer.tensors["moving_mean"],
                layer.tensors["moving_variance"],
            )

        for tensor in mapping.tensors:
            if tensor.source_tensor is None:
                value = np.zeros(tensor.target_shape, dtype=np.int64)
            else:
                source = np.asarray(layer.tensors[tensor.source_tensor])
                if source.shape != tensor.source_shape:
                    raise ValueError(
                        f"{mapping.role}/{tensor.source_tensor}: source shape must be "
                        f"{tensor.source_shape}, got {source.shape}"
                    )
                if not np.issubdtype(source.dtype, np.floating):
                    raise ValueError(
                        f"{mapping.role}/{tensor.source_tensor}: source dtype must be floating, "
                        f"got {source.dtype}"
                    )
                if tensor.layout == "conv2d_hwio_to_oihw":
                    value = convert_conv2d_kernel(source)
                elif tensor.layout == "depthwise_hwim_to_im1hw":
                    value = convert_depthwise_conv2d_kernel(source)
                elif batch_norm_values is not None:
                    target_leaf = tensor.target_key.rsplit(".", 1)[-1]
                    value = batch_norm_values[target_leaf]
                elif tensor.layout == "identity":
                    value = np.ascontiguousarray(source).copy()
                else:
                    raise AssertionError(f"unsupported mapping layout: {tensor.layout}")
                value = value.astype(np.float32, copy=False)

            if value.shape != tensor.target_shape:
                raise ValueError(
                    f"{mapping.role}/{tensor.target_key}: target shape must be "
                    f"{tensor.target_shape}, got {value.shape}"
                )
            if tensor.target_key in converted:
                raise ValueError(f"duplicate target state_dict key: {tensor.target_key}")
            converted[tensor.target_key] = value

    return converted


def numpy_state_dict_to_torch(
    state_dict: Mapping[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    """Copy a NumPy encoder state_dict into CPU tensors."""

    return {
        key: torch.from_numpy(np.array(value, copy=True, order="C")).cpu()
        for key, value in state_dict.items()
    }


def build_payload(
    state_dict: Mapping[str, torch.Tensor],
    *,
    source_checkpoint_sha256: str,
    converted_tensors: Sequence[str],
    skipped_tensors: Sequence[str],
) -> dict[str, Any]:
    """Build the inert encoder-only payload accepted by the model-local loader."""

    return {
        "format": ENCODER_STATE_DICT_FORMAT,
        "source_commit": SOURCE_COMMIT,
        "state_dict": {key: value.detach().cpu() for key, value in state_dict.items()},
        "metadata": {
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "source_function": SOURCE_FUNCTION,
            "source_license": SOURCE_LICENSE,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "mapping_version": MAPPING_VERSION,
            "converted_tensors": list(converted_tensors),
            "skipped_tensors": list(skipped_tensors),
        },
    }


def _import_tensorflow() -> Any:
    try:
        return importlib.import_module("tensorflow")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "TensorFlow is an optional converter-only dependency and is not installed. "
            "Create or activate an isolated conversion environment, run "
            "`python -m pip install tensorflow`, then rerun this command. "
            "Do not add TensorFlow to the main runtime requirements.txt."
        ) from error


def _variable_name(variable: Any) -> str:
    raw_name = str(getattr(variable, "path", "") or getattr(variable, "name", ""))
    return raw_name.rsplit("/", 1)[-1].split(":", 1)[0]


def _find_cnn_encoder(model: Any, tensorflow: Any) -> Any:
    if getattr(model, "name", None) == "cnn_encoder":
        encoder = model
    else:
        try:
            encoder = model.get_layer("cnn_encoder")
        except (ValueError, AttributeError) as error:
            raise ValueError(
                "source .keras must contain the pinned model named 'cnn_encoder'"
            ) from error
    if not isinstance(encoder, tensorflow.keras.Model):
        raise ValueError("source layer 'cnn_encoder' must be a Keras Model")
    return encoder


def _extract_supported_layers(encoder: Any) -> list[SourceLayerWeights]:
    records: list[SourceLayerWeights] = []
    for layer in encoder.layers:
        layer_type = layer.__class__.__name__
        if layer_type not in SUPPORTED_SOURCE_LAYER_TYPES:
            continue
        tensors: dict[str, np.ndarray] = {}
        for variable in layer.weights:
            name = _variable_name(variable)
            if name in tensors:
                raise ValueError(f"{layer.name}: duplicate Keras weight name {name!r}")
            tensors[name] = np.asarray(variable.numpy())
        records.append(SourceLayerWeights(layer.name, layer_type, tensors))
    return records


def _leaf_weight_references(model: Any) -> set[str]:
    references: set[str] = set()

    def visit(container: Any, prefix: str) -> None:
        for layer in container.layers:
            layer_path = f"{prefix}/{layer.name}" if prefix else layer.name
            nested_layers = getattr(layer, "layers", None)
            if nested_layers is not None:
                visit(layer, layer_path)
                continue
            for variable in layer.weights:
                references.add(f"{layer_path}/{_variable_name(variable)}")

    root_prefix = "" if getattr(model, "name", None) != "cnn_encoder" else "cnn_encoder"
    visit(model, root_prefix)
    return references


def load_keras_encoder_source(source_path: Path) -> tuple[Any, list[SourceLayerWeights], list[str]]:
    """Safely load the Keras file and return explicit encoder records/skips."""

    tensorflow = _import_tensorflow()
    try:
        model = tensorflow.keras.models.load_model(source_path, compile=False, safe_mode=True)
    except TypeError as error:
        raise RuntimeError(
            "The installed TensorFlow/Keras does not support safe_mode loading; "
            "upgrade TensorFlow before converting the official .keras checkpoint."
        ) from error
    encoder = _find_cnn_encoder(model, tensorflow)
    records = _extract_supported_layers(encoder)

    mapped_sources = {
        f"cnn_encoder/{mapping.source_layer}/{tensor.source_tensor}"
        for mapping in EXPLICIT_ENCODER_MAPPING
        for tensor in mapping.tensors
        if tensor.source_tensor is not None
    }
    all_sources = _leaf_weight_references(model)
    skipped = sorted(all_sources - mapped_sources)
    return encoder, records, skipped


def _converted_tensor_descriptions() -> list[str]:
    descriptions: list[str] = []
    for mapping in EXPLICIT_ENCODER_MAPPING:
        for tensor in mapping.tensors:
            source = (
                f"cnn_encoder/{mapping.source_layer}/{tensor.source_tensor}"
                if tensor.source_tensor is not None
                else "constant:int64_zero"
            )
            descriptions.append(f"{source} -> {tensor.target_key}")
    return descriptions


def _validate_against_pytorch_encoder(state_dict: Mapping[str, torch.Tensor]) -> None:
    encoder = BlazeGazeConvEncoder()
    expected = encoder.state_dict()
    actual_keys = set(state_dict)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        raise ValueError(
            "converted keys do not match BlazeGazeConvEncoder: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    for key, tensor in state_dict.items():
        if tensor.shape != expected[key].shape:
            raise ValueError(
                f"{key}: converted target shape must be {tuple(expected[key].shape)}, "
                f"got {tuple(tensor.shape)}"
            )
    encoder.load_state_dict(dict(state_dict), strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_payload(payload: Mapping[str, Any], output_path: Path, *, force: bool) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists (pass --force to replace it): {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def convert_checkpoint(source_path: Path, output_path: Path, *, force: bool = False) -> None:
    """Convert one official checkpoint after strict source and target validation."""

    source_path = source_path.expanduser().resolve(strict=True)
    output_path = output_path.expanduser().resolve(strict=False)
    if source_path.suffix != ".keras":
        raise ValueError(f"source must be a .keras file: {source_path}")
    if output_path.suffix != ".pt":
        raise ValueError(f"output must be a .pt file: {output_path}")
    if source_path == output_path:
        raise ValueError("source and output paths must differ")

    _encoder, source_layers, skipped_tensors = load_keras_encoder_source(source_path)
    numpy_state_dict = convert_explicit_encoder_layers(source_layers)
    state_dict = numpy_state_dict_to_torch(numpy_state_dict)
    _validate_against_pytorch_encoder(state_dict)
    checkpoint_sha256 = _sha256(source_path)
    payload = build_payload(
        state_dict,
        source_checkpoint_sha256=checkpoint_sha256,
        converted_tensors=_converted_tensor_descriptions(),
        skipped_tensors=skipped_tensors,
    )
    _save_payload(payload, output_path, force=force)
    print(f"[OK] source SHA-256: {checkpoint_sha256}")
    print(f"[OK] converted encoder tensors: {len(state_dict)}")
    print(f"[OK] skipped source tensors: {len(skipped_tensors)}")
    print(f"[OK] output: {output_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="official BlazeGaze .keras path")
    parser.add_argument("--output", type=Path, required=True, help="encoder-only .pt output path")
    parser.add_argument(
        "--source-commit",
        required=True,
        choices=(SOURCE_COMMIT,),
        help="must match the pinned WebEyeTrack source commit",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        convert_checkpoint(args.source, args.output, force=args.force)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
