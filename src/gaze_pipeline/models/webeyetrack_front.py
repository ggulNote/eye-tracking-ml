"""Safe PyTorch-backend wrapper for the official WebEyeTrack BlazeGaze model.

The public WebEyeTrack checkpoint is a native Keras v3 archive, not a
PyTorch state dictionary.  Keras v3 can materialize models made from built-in
layers as ``torch.nn.Module`` objects when its backend is selected before
import.  This module keeps that framework boundary explicit:

* verify the complete archive before deserialization;
* load only with Keras ``safe_mode=True`` and the PyTorch backend;
* map the pipeline's canonical CHW tensors to the checkpoint's named NHWC
  inputs; and
* return the canonical ``gaze_xy`` output expected by ``DefaultModelAdapter``.

Once loaded, the wrapped Keras parameters participate in ordinary PyTorch
optimizers and ``state_dict`` checkpoint/resume flows.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

OFFICIAL_BLAZEGAZE_MPIIFACEGAZE_SHA256 = (
    "5b011cfe82466896e27b1ac3e18130117cafbc02dbc964a1ad7315f62005cc05"
)
DEFAULT_BLAZEGAZE_PATH = Path("models/blazegaze_mpiifacegaze.keras")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KERAS_IMPORT_LOCK = threading.Lock()
_EXPECTED_INPUTS = {
    "image": (None, 128, 512, 3),
    "head_vector": (None, 3),
    "face_origin_3d": (None, 3),
}
_EXPECTED_OUTPUT = (None, 2)


class WebEyeTrackFrontError(RuntimeError):
    """Raised when the official BlazeGaze model cannot be loaded or called safely."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_keras_asset(
    weights_path: str | Path,
    *,
    expected_sha256: str = OFFICIAL_BLAZEGAZE_MPIIFACEGAZE_SHA256,
) -> tuple[Path, str]:
    """Resolve and SHA-256 verify a native Keras checkpoint before loading it."""

    path = Path(weights_path).expanduser().resolve(strict=False)
    if path.suffix.lower() != ".keras":
        raise WebEyeTrackFrontError(
            f"WebEyeTrack front checkpoint must use the native .keras format: {path}"
        )
    if not path.is_file():
        raise WebEyeTrackFrontError(f"WebEyeTrack front checkpoint does not exist: {path}")
    if not isinstance(expected_sha256, str):
        raise WebEyeTrackFrontError("expected_sha256 must be a 64-character hexadecimal digest")
    expected = expected_sha256.strip().lower()
    if not _SHA256_RE.fullmatch(expected):
        raise WebEyeTrackFrontError("expected_sha256 must be a 64-character hexadecimal digest")
    actual = _file_sha256(path)
    if not hmac.compare_digest(actual, expected):
        raise WebEyeTrackFrontError(
            f"WebEyeTrack front checkpoint SHA-256 mismatch for {path}: "
            f"expected {expected}, got {actual}"
        )
    return path, actual


def _keras_with_torch_backend() -> Any:
    """Import Keras only after forcing, then verifying, its PyTorch backend."""

    with _KERAS_IMPORT_LOCK:
        if "keras" not in sys.modules:
            os.environ["KERAS_BACKEND"] = "torch"
            # Keras otherwise prefers MPS automatically on Apple Silicon. The
            # generic trainer defaults to CPU when explicitly configured, and
            # Keras-created constants on MPS cannot interact with CPU batches.
            # Callers may override this before process start for an all-MPS run.
            os.environ.setdefault("KERAS_TORCH_DEVICE", "cpu")
        try:
            keras = importlib.import_module("keras")
        except ImportError as exc:
            raise WebEyeTrackFrontError(
                "Keras 3 is required for the official BlazeGaze .keras checkpoint; "
                "install a Keras version compatible with the project's PyTorch runtime"
            ) from exc
        try:
            backend = str(keras.config.backend()).lower()
        except Exception as exc:  # pragma: no cover - malformed third-party install
            raise WebEyeTrackFrontError("could not determine the active Keras backend") from exc
        if backend != "torch":
            raise WebEyeTrackFrontError(
                "Keras is already initialized with backend "
                f"{backend!r}; start a fresh process with KERAS_BACKEND=torch"
            )
        return keras


def _shape_tuple(value: Any) -> tuple[int | None, ...]:
    try:
        return tuple(None if item is None else int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise WebEyeTrackFrontError(f"invalid Keras tensor shape in checkpoint: {value!r}") from exc


def _validate_loaded_model(model: Any) -> None:
    if not isinstance(model, nn.Module):
        raise WebEyeTrackFrontError(
            "the loaded Keras model is not a torch.nn.Module; verify KERAS_BACKEND=torch"
        )
    inputs = getattr(model, "inputs", None)
    outputs = getattr(model, "outputs", None)
    if not isinstance(inputs, list | tuple) or not isinstance(outputs, list | tuple):
        raise WebEyeTrackFrontError("the loaded Keras object is not a Functional model")
    actual_inputs: dict[str, tuple[int | None, ...]] = {}
    for tensor in inputs:
        name = str(getattr(tensor, "name", "")).split(":", 1)[0]
        actual_inputs[name] = _shape_tuple(getattr(tensor, "shape", ()))
        if str(getattr(tensor, "dtype", "")) != "float32":
            raise WebEyeTrackFrontError(
                f"BlazeGaze input {name!r} must use float32, got {getattr(tensor, 'dtype', None)!r}"
            )
    if actual_inputs != _EXPECTED_INPUTS:
        raise WebEyeTrackFrontError(
            f"unexpected BlazeGaze input contract: {actual_inputs}; expected {_EXPECTED_INPUTS}"
        )
    if len(outputs) != 1 or _shape_tuple(getattr(outputs[0], "shape", ())) != _EXPECTED_OUTPUT:
        actual_outputs = [_shape_tuple(getattr(tensor, "shape", ())) for tensor in outputs]
        raise WebEyeTrackFrontError(
            f"unexpected BlazeGaze output contract: {actual_outputs}; expected [{_EXPECTED_OUTPUT}]"
        )
    if str(getattr(outputs[0], "dtype", "")) != "float32":
        raise WebEyeTrackFrontError(
            f"BlazeGaze output must use float32, got {getattr(outputs[0], 'dtype', None)!r}"
        )
    try:
        model.get_layer("cnn_encoder")
        model.get_layer("gaze_mlp")
    except (ValueError, AttributeError) as exc:
        raise WebEyeTrackFrontError(
            "BlazeGaze checkpoint must contain cnn_encoder and gaze_mlp submodels"
        ) from exc


def _load_safe_keras_model(path: Path) -> Any:
    keras = _keras_with_torch_backend()
    try:
        model = keras.saving.load_model(
            path,
            compile=False,
            safe_mode=True,
        )
    except Exception as exc:
        raise WebEyeTrackFrontError(
            f"could not safely load WebEyeTrack BlazeGaze checkpoint {path}: {exc}"
        ) from exc
    _validate_loaded_model(model)
    return model


def _canonical_inputs(
    front_image: Tensor,
    front_head_vector: Tensor,
    front_face_origin_3d: Tensor,
) -> dict[str, Tensor]:
    named = {
        "front_image": front_image,
        "front_head_vector": front_head_vector,
        "front_face_origin_3d": front_face_origin_3d,
    }
    for key, value in named.items():
        if not isinstance(value, Tensor):
            raise WebEyeTrackFrontError(f"{key} must be a torch.Tensor, got {type(value)!r}")
        if value.dtype != torch.float32:
            raise WebEyeTrackFrontError(f"{key} must use torch.float32, got {value.dtype}")
    if tuple(front_image.shape[1:]) != (3, 128, 512):
        raise WebEyeTrackFrontError(
            f"front_image must have shape [B,3,128,512], got {tuple(front_image.shape)}"
        )
    batch_size = int(front_image.shape[0])
    if batch_size <= 0:
        raise WebEyeTrackFrontError("front_image must have a non-empty batch dimension")
    for key, value in (
        ("front_head_vector", front_head_vector),
        ("front_face_origin_3d", front_face_origin_3d),
    ):
        if tuple(value.shape) != (batch_size, 3):
            raise WebEyeTrackFrontError(
                f"{key} must have shape [B,3] with B={batch_size}, got {tuple(value.shape)}"
            )
        if value.device != front_image.device:
            raise WebEyeTrackFrontError(
                f"{key} must be on {front_image.device}, got {value.device}"
            )
    return {
        "image": front_image.permute(0, 2, 3, 1).contiguous(),
        "head_vector": front_head_vector,
        "face_origin_3d": front_face_origin_3d,
    }


class WebEyeTrackFrontModel(nn.Module):
    """Expose the official Keras BlazeGaze checkpoint as a PyTorch module."""

    def __init__(
        self,
        keras_model: nn.Module,
        *,
        weights_path: Path,
        weights_sha256: str,
        unfreeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(unfreeze_encoder, bool):
            raise WebEyeTrackFrontError("unfreeze_encoder must be true or false")
        self.keras_model = keras_model
        self.weights_path = weights_path
        self.weights_sha256 = weights_sha256
        self.unfreeze_encoder = unfreeze_encoder
        encoder = self.keras_model.get_layer("cnn_encoder")  # type: ignore[attr-defined]
        encoder.trainable = unfreeze_encoder

    @property
    def encoder(self) -> nn.Module:
        """Return the nested encoder without registering a duplicate module path."""

        return self.keras_model.get_layer("cnn_encoder")  # type: ignore[attr-defined, no-any-return]

    @property
    def gaze_mlp(self) -> nn.Module:
        """Return the nested trainable gaze head."""

        return self.keras_model.get_layer("gaze_mlp")  # type: ignore[attr-defined, no-any-return]

    def forward(
        self,
        front_image: Tensor,
        front_head_vector: Tensor,
        front_face_origin_3d: Tensor,
    ) -> dict[str, Tensor]:
        keras_inputs = _canonical_inputs(
            front_image,
            front_head_vector,
            front_face_origin_3d,
        )
        # Keras creates some internal tensors through its backend rather than
        # from the input tensor. Match that device to the PyTorch batch so CPU,
        # MPS, and CUDA executions cannot silently mix devices.
        keras = _keras_with_torch_backend()
        with keras.device(str(front_image.device)):
            output = self.keras_model(keras_inputs, training=self.training)
        if not isinstance(output, Tensor) or tuple(output.shape) != (front_image.shape[0], 2):
            shape = getattr(output, "shape", None)
            raise WebEyeTrackFrontError(
                f"BlazeGaze must return a torch.Tensor [B,2], got {type(output)!r} shape={shape}"
            )
        return {"gaze_xy": output}


def create_model(
    weights_path: str | Path = DEFAULT_BLAZEGAZE_PATH,
    expected_sha256: str = OFFICIAL_BLAZEGAZE_MPIIFACEGAZE_SHA256,
    unfreeze_encoder: bool = False,
) -> WebEyeTrackFrontModel:
    """Generic-runtime factory for the official WebEyeTrack front model."""

    if not isinstance(unfreeze_encoder, bool):
        raise WebEyeTrackFrontError("unfreeze_encoder must be true or false")
    verified_path, actual_sha256 = verify_keras_asset(
        weights_path,
        expected_sha256=expected_sha256,
    )
    keras_model = _load_safe_keras_model(verified_path)
    return WebEyeTrackFrontModel(
        keras_model,
        weights_path=verified_path,
        weights_sha256=actual_sha256,
        unfreeze_encoder=unfreeze_encoder,
    )


__all__ = [
    "DEFAULT_BLAZEGAZE_PATH",
    "OFFICIAL_BLAZEGAZE_MPIIFACEGAZE_SHA256",
    "WebEyeTrackFrontError",
    "WebEyeTrackFrontModel",
    "create_model",
    "verify_keras_asset",
]
