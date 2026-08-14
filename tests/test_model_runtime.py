from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import torch

from gaze_pipeline.model_runtime import (
    ModelCheckpointError,
    ModelContractError,
    ModelImportError,
    build_model_runtime,
    import_entrypoint,
    load_pytorch_state_dict,
)


def _front_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "source_dir": None,
        "entrypoint": None,
        "adapter_entrypoint": None,
        "init_args": {},
        "pretrained": {"path": None, "sha256": None, "strict": True},
        "input_contract": {
            "image_key": "front_image",
            "shape": ["B", 3, 16, 16],
            "dtype": "float32",
            "value_range": [0.0, 1.0],
        },
        "output_contract": {
            "gaze_key": "gaze_xy",
            "gaze_shape": ["B", 2],
            "embedding_key": "front_embedding",
            "embedding_shape": ["B", 8],
        },
    }
    config.update(overrides)
    return config


def _side_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "source_dir": None,
        "entrypoint": None,
        "adapter_entrypoint": None,
        "init_args": {},
        "pretrained": {"path": None, "sha256": None, "strict": True},
        "input_contract": {
            "image_key": "side_image",
            "shape": ["B", 3, 16, 16],
            "dtype": "float32",
            "value_range": [0.0, 1.0],
        },
        "output_contract": {
            "gaze_key": None,
            "gaze_shape": None,
            "delta_y_key": "delta_y_side",
            "delta_y_shape": ["B", 1],
            "embedding_key": "side_embedding",
            "embedding_shape": ["B", 6],
        },
    }
    config.update(overrides)
    return config


def test_fallback_front_and_side_models_obey_y_residual_contract() -> None:
    front = build_model_runtime("front", _front_config())
    side = build_model_runtime("side", _side_config())

    front_output = front({"front_image": torch.rand(3, 3, 16, 16)})
    side_output = side({"side_image": torch.rand(3, 3, 16, 16)})

    assert front_output["gaze_xy"].shape == (3, 2)
    assert front_output["front_embedding"].shape == (3, 8)
    assert front_output["x_front"].shape == (3, 1)
    assert front_output["y_front"].shape == (3, 1)
    assert side_output["delta_y_side"].shape == (3, 1)
    assert side_output["side_embedding"].shape == (3, 6)


def test_dynamic_model_and_adapter_are_loaded_from_source_dir(tmp_path: Path) -> None:
    package = tmp_path / "external_demo"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "components.py").write_text(
        """
import torch
from torch import nn

class External(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(bias)))
    def forward(self, image):
        batch = image.shape[0]
        return torch.stack((self.bias.expand(batch), -self.bias.expand(batch)), dim=1)

def create_model(bias=0.0):
    return External(bias)

class Adapter:
    def __init__(self, output_contract):
        self.output_contract = output_contract
    def to_model_inputs(self, batch):
        return ([batch[\"front_image\"]], {})
    def to_standard_outputs(self, raw):
        return {self.output_contract[\"gaze_key\"]: raw}

def create_adapter(output_contract):
    return Adapter(output_contract)
""".strip(),
        encoding="utf-8",
    )
    config = _front_config(
        source_dir=str(tmp_path),
        entrypoint="external_demo.components:create_model",
        adapter_entrypoint="external_demo.components:create_adapter",
        init_args={"bias": 0.25},
    )

    runtime = build_model_runtime("front", config)
    output = runtime({"front_image": torch.rand(2, 3, 16, 16)})

    assert torch.allclose(output["gaze_xy"], torch.tensor([[0.25, -0.25]]).expand(2, 2))
    assert str(tmp_path) not in sys.path


def test_import_entrypoint_rejects_invalid_or_missing_source(tmp_path: Path) -> None:
    with pytest.raises(ModelImportError, match="package.module:callable"):
        import_entrypoint("invalid")
    with pytest.raises(ModelImportError, match="not a directory"):
        import_entrypoint("anything:create", source_dir=tmp_path / "missing")


def test_wrapped_state_dict_is_loaded_with_hash_and_path_is_recorded(tmp_path: Path) -> None:
    source = torch.nn.Linear(4, 2)
    checkpoint = tmp_path / "weights.pt"
    package = tmp_path / "weight_demo"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "factory.py").write_text(
        """
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
    def forward(self, front_image):
        return self.linear(front_image)

def create():
    return Model()
""".strip(),
        encoding="utf-8",
    )
    # Match the external model's state_dict key prefix.
    torch.save(
        {"state_dict": {f"linear.{key}": value for key, value in source.state_dict().items()}},
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = _front_config(
        source_dir=str(tmp_path),
        entrypoint="weight_demo.factory:create",
        input_contract={"image_key": "front_image", "shape": ["B", 4]},
        pretrained={"path": "weights.pt", "sha256": digest, "strict": True},
    )

    runtime = build_model_runtime("front", config, base_dir=tmp_path)

    assert runtime.loaded_pretrained_path == checkpoint.resolve()
    assert torch.equal(runtime.model.linear.weight, source.weight)
    assert torch.equal(runtime.model.linear.bias, source.bias)


def test_state_dict_loader_rejects_unsafe_format_and_wrong_hash(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    keras = tmp_path / "weights.keras"
    keras.write_bytes(b"not a pytorch state dict")
    checkpoint = tmp_path / "weights.pth"
    torch.save(model.state_dict(), checkpoint)

    with pytest.raises(ModelCheckpointError, match="use .pt or .pth"):
        load_pytorch_state_dict(model, keras)
    with pytest.raises(ModelCheckpointError, match="sha256 mismatch"):
        load_pytorch_state_dict(model, checkpoint, sha256="0" * 64)


def test_contract_rejects_missing_input_and_wrong_primary_shape(tmp_path: Path) -> None:
    with pytest.raises(ModelContractError, match="front_image"):
        build_model_runtime("front", _front_config())({"different": torch.rand(1, 3, 8, 8)})

    package = tmp_path / "bad_output"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "factory.py").write_text(
        """
import torch
from torch import nn

class Bad(nn.Module):
    def forward(self, front_image):
        return torch.zeros((front_image.shape[0], 3))

def create():
    return Bad()
""".strip(),
        encoding="utf-8",
    )
    runtime = build_model_runtime(
        "front",
        _front_config(source_dir=str(tmp_path), entrypoint="bad_output.factory:create"),
    )

    with pytest.raises(ModelContractError, match=r"expected \['B', 2\]"):
        runtime({"front_image": torch.rand(2, 3, 16, 16)})


def test_optional_declared_output_is_not_required() -> None:
    config = _front_config(
        output_contract={
            "gaze_key": "gaze_xy",
            "gaze_shape": ["B", 2],
            "reconstruction_key": "reconstructed_eye_region",
        }
    )
    runtime = build_model_runtime("front", config)

    output = runtime({"front_image": torch.rand(1, 3, 16, 16)})

    assert output["gaze_xy"].shape == (1, 2)
    assert "reconstructed_eye_region" not in output


def test_custom_contract_key_is_also_exposed_as_canonical_trainer_key() -> None:
    config = _front_config(
        output_contract={
            "gaze_key": "external_prediction",
            "gaze_shape": ["B", 2],
        }
    )
    runtime = build_model_runtime("front", config)

    output = runtime({"front_image": torch.rand(2, 3, 16, 16)})

    assert torch.equal(output["gaze_xy"], output["external_prediction"])


def test_auto_forward_keys_must_be_resolved_from_full_config() -> None:
    config = _side_config(
        input_contract={
            "forward_keys": "auto",
            "image_key": "side_image",
            "shape": ["B", 3, 16, 16],
        }
    )

    with pytest.raises(ModelContractError, match="resolved forward_keys"):
        build_model_runtime("side", config)

    runtime = build_model_runtime("side", config, forward_keys=("side_image",))
    assert runtime({"side_image": torch.rand(2, 3, 16, 16)})["delta_y_side"].shape == (2, 1)


@pytest.mark.parametrize(
    "image",
    (
        torch.rand(2, 1, 16, 16),
        torch.rand(2, 3, 15, 16),
        torch.rand(2, 3, 16),
    ),
)
def test_primary_image_contract_rejects_rank_channel_and_spatial_mismatch(
    image: torch.Tensor,
) -> None:
    runtime = build_model_runtime("front", _front_config())

    with pytest.raises(ModelContractError, match=r"front_image.*shape"):
        runtime({"front_image": image})


def test_primary_image_contract_rejects_dtype_nonfinite_and_range_mismatch() -> None:
    runtime = build_model_runtime("front", _front_config())

    with pytest.raises(ModelContractError, match=r"front_image.*dtype"):
        runtime({"front_image": torch.rand(2, 3, 16, 16, dtype=torch.float64)})

    nonfinite = torch.rand(2, 3, 16, 16)
    nonfinite[0, 0, 0, 0] = torch.nan
    with pytest.raises(ModelContractError, match=r"front_image.*NaN or infinity"):
        runtime({"front_image": nonfinite, "front_gaze_valid": torch.tensor([False, True])})

    outside = torch.rand(2, 3, 16, 16)
    outside[1, 2, 3, 4] = 1.25
    with pytest.raises(ModelContractError, match=r"front_image.*outside declared range"):
        runtime({"front_image": outside})


def test_selected_auxiliary_contract_checks_shape_dtype_and_batch_size() -> None:
    config = _front_config(
        input_contract={
            "image_key": "front_image",
            "shape": ["B", 3, 16, 16],
            "dtype": "float32",
            "value_range": [0.0, 1.0],
            "validity_key": "front_gaze_valid",
            "auxiliary_keys": {
                "head_vector": {
                    "key": "front_head_vector",
                    "shape": ["B", 3],
                    "dtype": "float32",
                }
            },
        }
    )
    runtime = build_model_runtime(
        "front",
        config,
        forward_keys=("front_image", "front_head_vector"),
    )
    image = torch.rand(2, 3, 16, 16)

    output = runtime(
        {
            "front_image": image,
            "front_head_vector": torch.rand(2, 3),
            "front_gaze_valid": torch.tensor([True, True]),
        }
    )
    assert output["gaze_xy"].shape == (2, 2)

    with pytest.raises(ModelContractError, match=r"front_head_vector.*shape"):
        runtime(
            {
                "front_image": image,
                "front_head_vector": torch.rand(3, 3),
                "front_gaze_valid": torch.tensor([True, True]),
            }
        )
    with pytest.raises(ModelContractError, match=r"front_head_vector.*dtype"):
        runtime(
            {
                "front_image": image,
                "front_head_vector": torch.rand(2, 3, dtype=torch.float64),
                "front_gaze_valid": torch.tensor([True, True]),
            }
        )


def test_nested_auto_auxiliary_contract_allows_masked_sentinel_and_checks_valid_rows() -> None:
    config = _side_config(
        input_contract={
            "forward_keys": "auto",
            "image_key": "side_image",
            "shape": ["B", 3, 16, 16],
            "dtype": "float32",
            "value_range": [0.0, 1.0],
            "validity_key": "side_gaze_valid",
            "auxiliary_keys": {
                "side_eyeangle": {
                    "key": "side_eye_angles",
                    "shape": ["B", 2],
                    "dtype": "float32",
                    "value_range": [-1.0, 1.0],
                }
            },
        }
    )
    runtime = build_model_runtime(
        "side",
        config,
        forward_keys=("side_image", "side_eye_angles"),
    )
    batch = {
        "side_image": torch.rand(2, 3, 16, 16),
        "side_eye_angles": torch.tensor([[torch.nan, torch.nan], [0.2, -0.2]]),
        "side_gaze_valid": torch.tensor([False, True]),
    }

    assert runtime(batch)["delta_y_side"].shape == (2, 1)

    batch["side_eye_angles"] = torch.tensor([[torch.nan, torch.nan], [1.2, 0.0]])
    with pytest.raises(ModelContractError, match=r"side_eye_angles.*outside declared range"):
        runtime(batch)
