from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gaze_pipeline.models.side.blazegaze_transfer import (
    ENCODER_STATE_DICT_FORMAT,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    BlazeGazeConvEncoder,
)
from scripts import convert_blazegaze_encoder as converter


def _synthetic_source_layers() -> list[converter.SourceLayerWeights]:
    layers: list[converter.SourceLayerWeights] = []
    offset = 1.0
    for mapping in converter.EXPLICIT_ENCODER_MAPPING:
        tensors: dict[str, np.ndarray] = {}
        for tensor in mapping.tensors:
            if tensor.source_tensor is None:
                continue
            assert tensor.source_shape is not None
            size = int(np.prod(tensor.source_shape))
            tensors[tensor.source_tensor] = (
                np.arange(size, dtype=np.float32).reshape(tensor.source_shape) + offset
            )
            offset += size
        layers.append(
            converter.SourceLayerWeights(
                name=mapping.source_layer,
                layer_type=mapping.source_type,
                tensors=tensors,
            )
        )
    return layers


def test_conv2d_hwio_to_oihw_layout() -> None:
    source = np.arange(2 * 3 * 2 * 4, dtype=np.float32).reshape(2, 3, 2, 4)

    converted = converter.convert_conv2d_kernel(source)

    assert converted.shape == (4, 2, 2, 3)
    for height in range(2):
        for width in range(3):
            for input_channel in range(2):
                for output_channel in range(4):
                    assert (
                        converted[output_channel, input_channel, height, width]
                        == source[height, width, input_channel, output_channel]
                    )


def test_depthwise_hwim_to_im1hw_layout_with_multiplier() -> None:
    source = np.arange(2 * 3 * 2 * 3, dtype=np.float32).reshape(2, 3, 2, 3)

    converted = converter.convert_depthwise_conv2d_kernel(source)

    assert converted.shape == (6, 1, 2, 3)
    for input_channel in range(2):
        for multiplier in range(3):
            target_channel = input_channel * 3 + multiplier
            np.testing.assert_array_equal(
                converted[target_channel, 0], source[:, :, input_channel, multiplier]
            )


def test_batch_norm_mapping_and_shape_validation() -> None:
    gamma = np.array([1.0, 2.0], dtype=np.float32)
    beta = np.array([3.0, 4.0], dtype=np.float32)
    mean = np.array([5.0, 6.0], dtype=np.float32)
    variance = np.array([7.0, 8.0], dtype=np.float32)

    converted = converter.convert_batch_norm_tensors(gamma, beta, mean, variance)

    assert set(converted) == {"weight", "bias", "running_mean", "running_var"}
    np.testing.assert_array_equal(converted["weight"], gamma)
    np.testing.assert_array_equal(converted["bias"], beta)
    np.testing.assert_array_equal(converted["running_mean"], mean)
    np.testing.assert_array_equal(converted["running_var"], variance)

    with pytest.raises(ValueError, match="equal-length rank-1"):
        converter.convert_batch_norm_tensors(gamma, beta[:1], mean, variance)


def test_explicit_mapping_matches_strict_pytorch_encoder_state_dict() -> None:
    numpy_state_dict = converter.convert_explicit_encoder_layers(_synthetic_source_layers())
    torch_state_dict = converter.numpy_state_dict_to_torch(numpy_state_dict)
    expected = BlazeGazeConvEncoder().state_dict()

    assert set(torch_state_dict) == set(expected)
    assert len(torch_state_dict) == 90
    assert all(tuple(torch_state_dict[key].shape) == tuple(expected[key].shape) for key in expected)
    assert all(tensor.device.type == "cpu" for tensor in torch_state_dict.values())
    assert all(
        tensor.item() == 0
        for key, tensor in torch_state_dict.items()
        if key.endswith("num_batches_tracked")
    )
    assert not any(
        key.startswith(("flatten_projection", "auxiliary_projector", "output_head"))
        for key in torch_state_dict
    )

    encoder = BlazeGazeConvEncoder()
    encoder.load_state_dict(torch_state_dict, strict=True)


def test_mapping_rejects_wrong_layer_type() -> None:
    layers = _synthetic_source_layers()
    layers[0] = replace(layers[0], layer_type="DepthwiseConv2D")

    with pytest.raises(ValueError, match="first_conv.*must be Conv2D"):
        converter.convert_explicit_encoder_layers(layers)


def test_mapping_rejects_wrong_source_shape() -> None:
    layers = _synthetic_source_layers()
    first = layers[0]
    layers[0] = replace(
        first,
        tensors={**first.tensors, "kernel": np.zeros((3, 3, 3, 24), dtype=np.float32)},
    )

    with pytest.raises(ValueError, match="first_conv/kernel: source shape must be"):
        converter.convert_explicit_encoder_layers(layers)


def test_mapping_rejects_missing_or_unmapped_supported_layer() -> None:
    layers = _synthetic_source_layers()

    with pytest.raises(ValueError, match="missing=.*conv2d"):
        converter.convert_explicit_encoder_layers(layers[1:])

    layers.append(
        converter.SourceLayerWeights(
            name="unmapped_projection",
            layer_type="Conv2D",
            tensors={},
        )
    )
    with pytest.raises(ValueError, match="extra=.*unmapped_projection"):
        converter.convert_explicit_encoder_layers(layers)


def test_encoder_payload_is_weights_only_loadable_and_records_lineage(tmp_path: Path) -> None:
    state_dict = converter.numpy_state_dict_to_torch(
        converter.convert_explicit_encoder_layers(_synthetic_source_layers())
    )
    converted_tensors = ["cnn_encoder/conv2d/kernel -> first_conv.0.conv.weight"]
    skipped_tensors = ["gaze_mlp/dense/kernel"]
    payload = converter.build_payload(
        state_dict,
        source_checkpoint_sha256="a" * 64,
        converted_tensors=converted_tensors,
        skipped_tensors=skipped_tensors,
    )
    output = tmp_path / "synthetic_encoder.pt"
    torch.save(payload, output)

    restored = torch.load(output, map_location="cpu", weights_only=True)

    assert restored["format"] == ENCODER_STATE_DICT_FORMAT
    assert restored["source_commit"] == SOURCE_COMMIT
    assert restored["metadata"] == {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_function": "python/webeyetrack/blazegaze.py:get_cnn_encoder",
        "source_license": "MIT",
        "source_checkpoint_sha256": "a" * 64,
        "mapping_version": converter.MAPPING_VERSION,
        "converted_tensors": converted_tensors,
        "skipped_tensors": skipped_tensors,
    }
    BlazeGazeConvEncoder().load_state_dict(restored["state_dict"], strict=True)


def test_tensorflow_import_error_explains_converter_only_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_tensorflow(name: str) -> None:
        assert name == "tensorflow"
        raise ModuleNotFoundError("No module named 'tensorflow'")

    monkeypatch.setattr(converter.importlib, "import_module", missing_tensorflow)

    with pytest.raises(RuntimeError, match="optional converter-only dependency") as error:
        converter._import_tensorflow()
    assert "python -m pip install tensorflow" in str(error.value)
    assert "requirements.txt" in str(error.value)


def test_official_keras_side_width_final_feature_parity_when_opted_in() -> None:
    source_value = os.environ.get("WEBEYETRACK_WEIGHTS")
    if not source_value:
        pytest.skip("set WEBEYETRACK_WEIGHTS to opt in to TensorFlow parity")
    tensorflow = pytest.importorskip("tensorflow")
    source_path = Path(source_value).expanduser().resolve(strict=True)

    source_encoder, source_layers, _skipped = converter.load_keras_encoder_source(source_path)
    side_input = tensorflow.keras.Input(shape=(128, 256, 3))
    side_encoder = tensorflow.keras.models.clone_model(source_encoder, input_tensors=[side_input])
    side_encoder.set_weights(source_encoder.get_weights())
    keras_features = tensorflow.keras.Model(
        side_encoder.inputs,
        side_encoder.get_layer("batch_normalization_1").output,
    )

    values = np.linspace(0.0, 1.0, num=128 * 256 * 3, dtype=np.float32).reshape(1, 128, 256, 3)
    expected = keras_features([values], training=False).numpy()

    state_dict = converter.numpy_state_dict_to_torch(
        converter.convert_explicit_encoder_layers(source_layers)
    )
    pytorch_encoder = BlazeGazeConvEncoder().eval()
    pytorch_encoder.load_state_dict(state_dict, strict=True)
    inputs = torch.from_numpy(values.transpose(0, 3, 1, 2).copy())
    with torch.inference_mode():
        actual = pytorch_encoder(inputs).permute(0, 2, 3, 1).numpy()

    assert expected.shape == actual.shape == (1, 2, 4, 32)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
