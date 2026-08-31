"""Portable dual-camera live demo for the trained pair-wise gaze model."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gaze_pipeline.data.side_annotation import annotation_from_landmarks
from gaze_pipeline.data.webeyetrack_compat import (
    MediaPipeFaceLandmarkerDetector,
    estimate_metric_face_origin,
    head_vector_from_face_rt,
    webeyetrack_eye_patch,
)
from gaze_pipeline.model_runtime import build_model_runtime
from gaze_pipeline.models.webeyetrack_front import OFFICIAL_BLAZEGAZE_MPIIFACEGAZE_SHA256

PAIRWISE_CHECKPOINT_SHA256 = "127ca3014d9ef27cdabc327b0145e2c66a1741642bb278fb41a432156a24225f"
DEFAULT_ASSETS_DIR = Path("models/demo")
WINDOW_NAME = "Pairwise Eye Tracking Demo"


class LiveDemoError(RuntimeError):
    """Raised when a portable demo asset, camera, or inference contract fails."""


@dataclass(frozen=True, slots=True)
class DemoAssets:
    root: Path
    checkpoint: Path
    front_weights: Path
    face_landmarker: Path
    yunet_model: Path

    @classmethod
    def from_directory(cls, root: str | Path) -> DemoAssets:
        base = Path(root).expanduser().resolve(strict=False)
        return cls(
            root=base,
            checkpoint=base / "best_weights.pt",
            front_weights=base / "blazegaze_mpiifacegaze.keras",
            face_landmarker=base / "face_landmarker.task",
            yunet_model=base / "face_detection_yunet.onnx",
        )

    def validate(self) -> None:
        required = {
            "83.1% pipeline checkpoint": self.checkpoint,
            "BlazeGaze front model": self.front_weights,
            "MediaPipe face landmarker": self.face_landmarker,
            "YuNet side-face fallback": self.yunet_model,
        }
        missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
        if missing:
            details = "\n  ".join(missing)
            raise LiveDemoError(
                "시연 모델 파일이 없습니다. demo-models.zip을 models/demo에 풀어주세요:\n  "
                + details
            )
        _verify_sha256(
            self.checkpoint,
            PAIRWISE_CHECKPOINT_SHA256,
            label="83.1% pairwise checkpoint",
        )
        _verify_sha256(
            self.front_weights,
            OFFICIAL_BLAZEGAZE_MPIIFACEGAZE_SHA256,
            label="official BlazeGaze front model",
        )
        manifest = self.root / "manifest.json"
        if manifest.is_file():
            _validate_asset_manifest(manifest, self.root)


@dataclass(frozen=True, slots=True)
class LiveDemoOptions:
    assets_dir: Path = DEFAULT_ASSETS_DIR
    front_source: int | str = 0
    side_source: int | str = 1
    device: str = "auto"
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: float = 30.0
    screen_width: int | None = None
    screen_height: int | None = None
    windowed: bool = False
    mirror_front: bool = False
    mirror_side: bool = False
    side_rotate: int = 0
    smoothing: float = 0.22
    show_cameras: bool = False
    verify_only: bool = False


@dataclass(frozen=True, slots=True)
class PreparedInputs:
    front_image: torch.Tensor
    front_head_vector: torch.Tensor
    front_face_origin_3d: torch.Tensor
    side_image: torch.Tensor
    side_status: str


@dataclass(slots=True)
class AffineCalibration:
    """Optional screen affine correction collected from mouse target samples."""

    prediction_samples: list[np.ndarray] = field(default_factory=list)
    target_samples: list[np.ndarray] = field(default_factory=list)
    matrix: np.ndarray | None = None

    def add(self, prediction_xy: np.ndarray, target_xy: np.ndarray) -> bool:
        prediction = np.asarray(prediction_xy, dtype=np.float64).reshape(2)
        target = np.asarray(target_xy, dtype=np.float64).reshape(2)
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise LiveDemoError("보정 표본에는 유한한 x/y 값이 필요합니다.")
        self.prediction_samples.append(prediction)
        self.target_samples.append(target)
        return self.fit()

    def fit(self) -> bool:
        if len(self.prediction_samples) < 3:
            self.matrix = None
            return False
        inputs = np.column_stack(
            [np.asarray(self.prediction_samples), np.ones(len(self.prediction_samples))]
        )
        targets = np.asarray(self.target_samples)
        if np.linalg.matrix_rank(inputs) < 3:
            self.matrix = None
            return False
        solution, _, _, _ = np.linalg.lstsq(inputs, targets, rcond=None)
        self.matrix = solution.T
        return True

    def apply(self, prediction_xy: np.ndarray) -> np.ndarray:
        value = np.asarray(prediction_xy, dtype=np.float64).reshape(2)
        if self.matrix is None:
            return value.astype(np.float32)
        homogeneous = np.asarray([value[0], value[1], 1.0], dtype=np.float64)
        return (self.matrix @ homogeneous).astype(np.float32)

    def clear(self) -> None:
        self.prediction_samples.clear()
        self.target_samples.clear()
        self.matrix = None


class LiveFramePreprocessor:
    """Create the exact Front tensor and a close live Side ROI approximation."""

    def __init__(self, assets: DemoAssets) -> None:
        detector_config = {
            "model_asset_path": str(assets.face_landmarker),
            "num_faces": 1,
            "output_face_blendshapes": False,
            "mirror_retry": True,
        }
        self.front_detector = MediaPipeFaceLandmarkerDetector(detector_config)
        self.side_detector = MediaPipeFaceLandmarkerDetector(detector_config)
        self.yunet = _YuNetProfileEyeDetector(assets.yunet_model)

    def close(self) -> None:
        self.front_detector.close()
        self.side_detector.close()

    def prepare(self, front_bgr: np.ndarray, side_bgr: np.ndarray) -> PreparedInputs:
        import cv2  # type: ignore

        front_rgb = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB)
        side_rgb = cv2.cvtColor(side_bgr, cv2.COLOR_BGR2RGB)
        front = self.front_detector.detect(front_rgb, "front")
        front_patch, _, _ = webeyetrack_eye_patch(
            front_rgb,
            np.asarray(front["landmarks_xy"]),
            output_size_hw=(128, 512),
        )
        head_vector, _ = head_vector_from_face_rt(np.asarray(front["face_rt"]))
        face_origin, _, _, _ = estimate_metric_face_origin(
            np.asarray(front["landmarks_xyz_normalized"]),
            np.asarray(front["face_rt"]),
            front_rgb.shape[:2],
            iris_diameter_cm=1.2,
            iris_mode="both",
            initial_depth_cm=60.0,
            max_iterations=10,
            max_depth_step_cm=5.0,
        )
        side_roi, side_status = self._side_roi(side_rgb)
        side_canvas = np.zeros((128, 256, 3), dtype=np.uint8)
        side_canvas[:, 64:192] = side_roi
        return PreparedInputs(
            front_image=_image_tensor(front_patch),
            front_head_vector=torch.from_numpy(head_vector).float().unsqueeze(0),
            front_face_origin_3d=torch.from_numpy(face_origin).float().unsqueeze(0),
            side_image=_image_tensor(side_canvas),
            side_status=side_status,
        )

    def _side_roi(self, side_rgb: np.ndarray) -> tuple[np.ndarray, str]:
        import cv2  # type: ignore

        try:
            detected = self.side_detector.detect(side_rgb, "side")
            annotation = annotation_from_landmarks(
                np.asarray(detected["landmarks_xy"]),
                image_size_hw=side_rgb.shape[:2],
                presence=np.asarray(detected["landmark_presence"]),
                visibility=np.asarray(detected["landmark_visibility"]),
                min_auto_confidence=0.0,
                min_visibility_dominance=1.0,
                min_retained_bbox_ratio=0.70,
            )
            bbox = annotation.bbox_xyxy
            status = f"MediaPipe {annotation.visible_eye} eye"
        except Exception as mediapipe_error:
            try:
                bbox, direction = self.yunet.detect(side_rgb)
                status = f"YuNet {direction} eye fallback"
            except Exception as yunet_error:
                raise LiveDemoError(
                    "측면 눈을 찾지 못했습니다. 폰카메라에서 얼굴 옆면과 눈이 보이게 해주세요. "
                    f"MediaPipe={mediapipe_error}; YuNet={yunet_error}"
                ) from yunet_error

        height, width = side_rgb.shape[:2]
        x0 = max(0, min(width - 1, int(np.floor(bbox[0]))))
        y0 = max(0, min(height - 1, int(np.floor(bbox[1]))))
        x1 = max(x0 + 1, min(width, int(np.ceil(bbox[2]))))
        y1 = max(y0 + 1, min(height, int(np.ceil(bbox[3]))))
        crop = side_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            raise LiveDemoError("측면 눈 crop이 비어 있습니다.")
        roi = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(roi), status


class LiveGazeModel:
    """Load the verified pair-wise checkpoint without any training dataset."""

    def __init__(self, assets: DemoAssets, *, device: str = "auto") -> None:
        assets.validate()
        os.environ.setdefault("KERAS_BACKEND", "torch")
        self.device = _resolve_device(device)
        if self.device.type == "mps":
            os.environ.setdefault("KERAS_TORCH_DEVICE", "mps")
        self.front = build_model_runtime(
            "front",
            _front_model_config(assets.front_weights),
            device=self.device,
            forward_keys=("front_image", "front_head_vector", "front_face_origin_3d"),
        )
        self.side = build_model_runtime(
            "side",
            _side_model_config(),
            device=self.device,
            forward_keys=("side_image",),
        )
        payload = _safe_checkpoint(assets.checkpoint, self.device)
        components = payload.get("components")
        if not isinstance(components, dict):
            raise LiveDemoError("체크포인트에 components state_dict가 없습니다.")
        try:
            self.front.load_state_dict(components["front"], strict=True)
            self.side.load_state_dict(components["side"], strict=True)
            fusion_state = components["fusion"]
            residual_weight = fusion_state["residual_weight"]
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise LiveDemoError(f"83.1% 체크포인트 구조가 모델과 맞지 않습니다: {exc}") from exc
        self.residual_weight = torch.as_tensor(
            residual_weight,
            dtype=torch.float32,
            device=self.device,
        ).reshape(())
        self.front.eval()
        self.side.eval()

    @torch.inference_mode()
    def predict(self, inputs: PreparedInputs) -> np.ndarray:
        batch = {
            "front_image": inputs.front_image.to(self.device),
            "front_head_vector": inputs.front_head_vector.to(self.device),
            "front_face_origin_3d": inputs.front_face_origin_3d.to(self.device),
            "front_gaze_valid": torch.ones(1, dtype=torch.bool, device=self.device),
            "side_image": inputs.side_image.to(self.device),
            "side_gaze_valid": torch.ones(1, dtype=torch.bool, device=self.device),
        }
        front_output = self.front(batch)
        side_output = self.side(batch)
        front_xy = torch.as_tensor(front_output["gaze_xy"], device=self.device)
        side_y = torch.as_tensor(side_output["delta_y_side"], device=self.device)
        prediction = front_xy.clone()
        prediction[:, 1:2] = front_xy[:, 1:2] + self.residual_weight * side_y
        value = prediction[0].detach().float().cpu().numpy()
        if value.shape != (2,) or not np.isfinite(value).all():
            raise LiveDemoError(f"모델이 유효하지 않은 시선 좌표를 반환했습니다: {value}")
        return value

    def smoke_test(self) -> np.ndarray:
        """Run one label-free inference to validate every checkpoint component."""

        return self.predict(
            PreparedInputs(
                front_image=torch.zeros((1, 3, 128, 512), dtype=torch.float32),
                front_head_vector=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
                front_face_origin_3d=torch.tensor([[0.0, 0.0, 60.0]], dtype=torch.float32),
                side_image=torch.zeros((1, 3, 128, 256), dtype=torch.float32),
                side_status="smoke-test",
            )
        )


class DualCameraCapture:
    """Open and approximately synchronize two OpenCV camera sources."""

    def __init__(
        self,
        front_source: int | str,
        side_source: int | str,
        *,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        if front_source == side_source:
            raise LiveDemoError("정면과 측면 카메라는 서로 다른 source여야 합니다.")
        self.front = _open_camera(front_source, width=width, height=height, fps=fps)
        try:
            self.side = _open_camera(side_source, width=width, height=height, fps=fps)
        except Exception:
            self.front.release()
            raise

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.front.grab() or not self.side.grab():
            raise LiveDemoError("카메라 프레임 grab에 실패했습니다.")
        front_ok, front = self.front.retrieve()
        side_ok, side = self.side.retrieve()
        if not front_ok or front is None or not side_ok or side is None:
            raise LiveDemoError("카메라 프레임 retrieve에 실패했습니다.")
        return front, side

    def close(self) -> None:
        self.front.release()
        self.side.release()


def run_live_demo(options: LiveDemoOptions) -> None:
    """Validate assets, then run a fullscreen mouse-target/gaze-circle demo."""

    import cv2  # type: ignore

    assets = DemoAssets.from_directory(options.assets_dir)
    print("시연 모델을 불러오는 중입니다...")
    model = LiveGazeModel(assets, device=options.device)
    smoke_prediction = model.smoke_test()
    print(
        "모델 검증 완료: "
        f"device={model.device}, smoke_xy={smoke_prediction.tolist()}, "
        f"checkpoint={assets.checkpoint}"
    )
    if options.verify_only:
        print("verify-only 완료: 카메라는 열지 않았습니다.")
        return

    width, height = _screen_size(options.screen_width, options.screen_height)
    mouse_position = [width // 2, height // 2]
    smoothing = float(options.smoothing)
    if not 0.0 < smoothing <= 1.0:
        raise LiveDemoError("smoothing은 0보다 크고 1 이하여야 합니다.")
    calibration = AffineCalibration()
    recent_predictions: deque[np.ndarray] = deque(maxlen=15)
    smoothed: np.ndarray | None = None
    show_cameras = bool(options.show_cameras)
    status = "Warming up"
    last_frame_time = time.perf_counter()
    fps = 0.0

    preprocessor = LiveFramePreprocessor(assets)
    cameras: DualCameraCapture | None = None
    try:
        cameras = DualCameraCapture(
            options.front_source,
            options.side_source,
            width=options.camera_width,
            height=options.camera_height,
            fps=options.camera_fps,
        )
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if options.windowed:
            cv2.resizeWindow(WINDOW_NAME, width, height)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        def on_mouse(event: int, x: int, y: int, _flags: int, _data: Any) -> None:
            if event == cv2.EVENT_MOUSEMOVE:
                mouse_position[:] = [int(x), int(y)]

        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
        while True:
            front_frame, side_frame = cameras.read()
            front_frame = _orient_frame(front_frame, mirror=options.mirror_front, rotate=0)
            side_frame = _orient_frame(
                side_frame,
                mirror=options.mirror_side,
                rotate=options.side_rotate,
            )
            try:
                prepared = preprocessor.prepare(front_frame, side_frame)
                raw_prediction = model.predict(prepared)
                recent_predictions.append(raw_prediction)
                smoothed = (
                    raw_prediction.copy()
                    if smoothed is None
                    else smoothing * raw_prediction + (1.0 - smoothing) * smoothed
                )
                shown_prediction = calibration.apply(smoothed)
                gaze_pixel = normalized_to_pixel(shown_prediction, width, height, clamp=True)
                status = prepared.side_status
            except Exception as exc:
                gaze_pixel = None
                status = str(exc).splitlines()[0][:120]

            now = time.perf_counter()
            elapsed = max(now - last_frame_time, 1e-6)
            instant_fps = 1.0 / elapsed
            fps = instant_fps if fps == 0.0 else 0.12 * instant_fps + 0.88 * fps
            last_frame_time = now
            canvas = _render_canvas(
                width=width,
                height=height,
                mouse_position=tuple(mouse_position),
                gaze_position=gaze_pixel,
                status=status,
                fps=fps,
                calibration=calibration,
                front_frame=front_frame if show_cameras else None,
                side_frame=side_frame if show_cameras else None,
            )
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                break
            if key == ord("r"):
                smoothed = None
                recent_predictions.clear()
            elif key == ord("c"):
                calibration.clear()
            elif key == ord("v"):
                show_cameras = not show_cameras
            elif key == 32 and recent_predictions:
                averaged = np.mean(np.asarray(recent_predictions), axis=0)
                target = pixel_to_normalized(mouse_position, width, height)
                calibration.add(averaged, target)
    finally:
        if cameras is not None:
            cameras.close()
        preprocessor.close()
        cv2.destroyAllWindows()


def normalized_to_pixel(
    gaze_xy: np.ndarray,
    width: int,
    height: int,
    *,
    clamp: bool,
) -> tuple[int, int]:
    value = np.asarray(gaze_xy, dtype=np.float64).reshape(2)
    x = (float(value[0]) + 0.5) * width
    y = (float(value[1]) + 0.5) * height
    if clamp:
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
    return int(round(x)), int(round(y))


def pixel_to_normalized(pixel_xy: Any, width: int, height: int) -> np.ndarray:
    value = np.asarray(pixel_xy, dtype=np.float64).reshape(2)
    return np.asarray([value[0] / width - 0.5, value[1] / height - 0.5], dtype=np.float32)


def parse_camera_source(value: str | int) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    if not text:
        raise LiveDemoError("카메라 source는 장치 번호 또는 URL이어야 합니다.")
    return text


def _image_tensor(rgb: np.ndarray) -> torch.Tensor:
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0)


def _resolve_device(requested: str) -> torch.device:
    normalized = str(requested).strip().lower()
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if normalized == "auto":
        if torch.cuda.is_available():
            normalized = "cuda"
        elif mps_available:
            normalized = "mps"
        else:
            normalized = "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise LiveDemoError("CUDA를 요청했지만 사용 가능한 NVIDIA CUDA 장치가 없습니다.")
    if normalized == "mps" and not mps_available:
        raise LiveDemoError("MPS를 요청했지만 사용 가능한 Apple Silicon MPS 장치가 없습니다.")
    if normalized not in {"cpu", "cuda", "mps"}:
        raise LiveDemoError("device는 auto, cpu, cuda, mps 중 하나여야 합니다.")
    return torch.device(normalized)


def _front_model_config(weights_path: Path) -> dict[str, Any]:
    return {
        "enabled": True,
        "entrypoint": "gaze_pipeline.models.webeyetrack_front:create_model",
        "adapter_entrypoint": None,
        "init_args": {
            "weights_path": str(weights_path),
            "expected_sha256": OFFICIAL_BLAZEGAZE_MPIIFACEGAZE_SHA256,
            "unfreeze_encoder": False,
        },
        "pretrained": {"path": None, "strict": True},
        "input_contract": {
            "image_key": "front_image",
            "shape": ["B", 3, 128, 512],
            "dtype": "float32",
            "color_order": "RGB",
            "value_range": [0.0, 1.0],
            "validity_key": "front_gaze_valid",
            "auxiliary_keys": {
                "head_vector": {"key": "front_head_vector", "shape": ["B", 3]},
                "face_origin_3d": {"key": "front_face_origin_3d", "shape": ["B", 3]},
            },
        },
        "output_contract": {"gaze_key": "gaze_xy", "gaze_shape": ["B", 2]},
    }


def _side_model_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "entrypoint": "gaze_pipeline.models.side.blazegaze_transfer:create_model",
        "adapter_entrypoint": None,
        "init_args": {
            "encoder_weights_path": None,
            "image_feature_dim": 256,
            "embedding_dim": 256,
            "auxiliary_dim": 32,
            "dropout": 0.0,
        },
        "pretrained": {"path": None, "strict": True},
        "input_contract": {
            "image_key": "side_image",
            "shape": ["B", 3, 128, 256],
            "dtype": "float32",
            "color_order": "RGB",
            "value_range": [0.0, 1.0],
            "validity_key": "side_gaze_valid",
        },
        "output_contract": {
            "delta_y_key": "delta_y_side",
            "delta_y_shape": ["B", 1],
            "embedding_key": "side_embedding",
            "embedding_shape": ["B", 256],
            "quality_key": "quality",
            "quality_shape": ["B", 1],
        },
    }


def _safe_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise LiveDemoError(f"체크포인트를 안전하게 읽지 못했습니다: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LiveDemoError("체크포인트 최상위 값이 mapping이 아닙니다.")
    return payload


def _verify_sha256(path: Path, expected: str, *, label: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected.lower():
        raise LiveDemoError(
            f"{label} SHA-256 불일치: expected={expected.lower()}, actual={actual}, path={path}"
        )


def _validate_asset_manifest(manifest_path: Path, root: Path) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveDemoError(f"모델 manifest를 읽지 못했습니다: {manifest_path}: {exc}") from exc
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list):
        raise LiveDemoError(f"모델 manifest files가 list가 아닙니다: {manifest_path}")
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise LiveDemoError(f"모델 manifest 파일 항목이 잘못되었습니다: {item!r}")
        name = item["name"]
        if Path(name).name != name:
            raise LiveDemoError(f"모델 manifest에는 파일명만 허용됩니다: {name!r}")
        expected = item.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise LiveDemoError(f"모델 manifest SHA-256이 잘못되었습니다: {name}")
        candidate = root / name
        if not candidate.is_file():
            raise LiveDemoError(f"모델 manifest 파일이 없습니다: {candidate}")
        _verify_sha256(candidate, expected, label=f"asset {name}")


class _YuNetProfileEyeDetector:
    def __init__(self, model_path: Path) -> None:
        import cv2  # type: ignore

        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), 0.55, 0.3, 5000
        )

    def detect(self, rgb: np.ndarray) -> tuple[tuple[float, float, float, float], str]:
        import cv2  # type: ignore

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(bgr)
        if faces is None or len(faces) == 0:
            raise LiveDemoError("YuNet face detector found no face")
        face = max(faces, key=lambda row: float(row[2] * row[3]) * max(float(row[14]), 0.0))
        x, y, face_width, face_height = (float(value) for value in face[:4])
        if face_width <= 0 or face_height <= 0:
            raise LiveDemoError("YuNet face bbox is invalid")
        first_eye = np.asarray(face[4:6], dtype=np.float64)
        second_eye = np.asarray(face[6:8], dtype=np.float64)
        nose_x = float(face[8])
        looking_left = nose_x < float((first_eye[0] + second_eye[0]) / 2.0)
        eye = min((first_eye, second_eye), key=lambda point: point[0])
        visible_eye = "right"
        if not looking_left:
            eye = max((first_eye, second_eye), key=lambda point: point[0])
            visible_eye = "left"
        crop_width = max(12.0, face_width * 0.24)
        crop_height = crop_width / 2.0
        return (
            (
                float(eye[0] - crop_width / 2.0),
                float(eye[1] - crop_height / 2.0),
                float(eye[0] + crop_width / 2.0),
                float(eye[1] + crop_height / 2.0),
            ),
            visible_eye,
        )


def _open_camera(source: int | str, *, width: int, height: int, fps: float) -> Any:
    import cv2  # type: ignore

    if isinstance(source, int) and os.name == "nt":
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(source)
    else:
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise LiveDemoError(f"카메라를 열 수 없습니다: {source!r}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    capture.set(cv2.CAP_PROP_FPS, float(fps))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _orient_frame(frame: np.ndarray, *, mirror: bool, rotate: int) -> np.ndarray:
    import cv2  # type: ignore

    value = frame
    if mirror:
        value = cv2.flip(value, 1)
    rotations = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if rotate not in rotations:
        raise LiveDemoError("rotation은 0, 90, 180, 270 중 하나여야 합니다.")
    if rotations[rotate] is not None:
        value = cv2.rotate(value, rotations[rotate])
    return value


def _screen_size(width: int | None, height: int | None) -> tuple[int, int]:
    if width is not None or height is not None:
        if width is None or height is None or width <= 0 or height <= 0:
            raise LiveDemoError("screen-width와 screen-height는 양의 값으로 함께 지정해야 합니다.")
        return int(width), int(height)
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        result = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
        root.destroy()
        return result
    except Exception:
        return 1920, 1080


def _render_canvas(
    *,
    width: int,
    height: int,
    mouse_position: tuple[int, int],
    gaze_position: tuple[int, int] | None,
    status: str,
    fps: float,
    calibration: AffineCalibration,
    front_frame: np.ndarray | None,
    side_frame: np.ndarray | None,
) -> np.ndarray:
    import cv2  # type: ignore

    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    mx, my = mouse_position
    target_color = (255, 200, 40)
    cv2.circle(canvas, (mx, my), 12, target_color, 2, cv2.LINE_AA)
    cv2.line(canvas, (mx - 22, my), (mx + 22, my), target_color, 2, cv2.LINE_AA)
    cv2.line(canvas, (mx, my - 22), (mx, my + 22), target_color, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "MOUSE TARGET",
        (min(mx + 18, width - 190), max(28, my - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        target_color,
        2,
        cv2.LINE_AA,
    )
    if gaze_position is not None:
        gx, gy = gaze_position
        gaze_color = (80, 80, 255)
        cv2.circle(canvas, (gx, gy), 34, gaze_color, 4, cv2.LINE_AA)
        cv2.circle(canvas, (gx, gy), 5, gaze_color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "EYE GAZE",
            (min(gx + 42, width - 145), max(30, gy - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            gaze_color,
            2,
            cv2.LINE_AA,
        )
    calibrated = calibration.matrix is not None
    lines = [
        f"FPS {fps:.1f} | {'CALIBRATED' if calibrated else 'RAW MODEL'}",
        f"Status: {status}",
        "Move mouse = target | SPACE = add calibration point | C = clear calibration",
        "V = camera preview | R = reset smoothing | Q/ESC = exit",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (24, 34 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
    if calibration.prediction_samples:
        cv2.putText(
            canvas,
            f"Calibration samples: {len(calibration.prediction_samples)} (3+ non-collinear points)",
            (24, 34 + len(lines) * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (120, 220, 120),
            1,
            cv2.LINE_AA,
        )
    if front_frame is not None and side_frame is not None:
        thumb_width = min(320, max(160, width // 6))
        thumb_height = int(thumb_width * 9 / 16)
        front_thumb = cv2.resize(front_frame, (thumb_width, thumb_height))
        side_thumb = cv2.resize(side_frame, (thumb_width, thumb_height))
        x0 = width - thumb_width - 18
        y0 = 18
        canvas[y0 : y0 + thumb_height, x0 : x0 + thumb_width] = front_thumb
        y1 = y0 + thumb_height + 12
        canvas[y1 : y1 + thumb_height, x0 : x0 + thumb_width] = side_thumb
    return canvas


__all__ = [
    "AffineCalibration",
    "DEFAULT_ASSETS_DIR",
    "DemoAssets",
    "LiveDemoError",
    "LiveDemoOptions",
    "LiveGazeModel",
    "normalized_to_pixel",
    "parse_camera_source",
    "pixel_to_normalized",
    "run_live_demo",
]
