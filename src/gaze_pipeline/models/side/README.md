# Side Encoder models

이 package는 90° Side eye image에서 Front branch의 y 예측을 보정할 residual을 만드는 두
PyTorch model factory를 제공합니다. 두 모델은 같은 입력과 출력 contract를 사용하므로 마지막
model profile만 바꿔 runtime에서 교체할 수 있습니다.

더 자세한 설계, weight lineage와 converter 설명은
[Side Encoder 문서](../../../../docs/side-encoders.md)를 참고하세요.

## 제공 모델

| Model | Entrypoint | 특징 |
|---|---|---|
| BlazeGaze transfer | `gaze_pipeline.models.side.blazegaze_transfer:create_model` | WebEyeTrack BlazeGaze convolution encoder 구조를 PyTorch로 재현하고 encoder-only weight 전이를 지원 |
| MobileNetV4-Conv-S | `gaze_pipeline.models.side.mobilenet_v4:create_model` | `timm==1.0.28`의 `mobilenetv4_conv_small.e2400_r224_in1k` feature extractor 사용 |

두 factory는 `torch.nn.Module`을 반환합니다. 기본 config는 외부 다운로드가 발생하지 않도록
BlazeGaze는 random initialization, MobileNetV4는 `pretrained: false`를 사용합니다.

## Tensor contract

필수 입력:

```text
side_image: float32[B,3,128,256], RGB [0,1]
```

선택 입력:

```text
side_head_pose_2d: float32[B,2]
front_head_vector: float32[B,3]
side_eye_angles: float32[B,2]
side_iris_pose_2d: float32[B,2]
```

`side_head_pose_2d`와 `front_head_vector`는 서로 다른 head source이므로 동시에 전달할 수
없습니다. 선택 tensor는 `side_image`와 batch, dtype, device가 같아야 합니다. Auxiliary의
NaN/Inf는 model 내부에서 zero-fill되지만 validity 판단에는 사용되지 않습니다.

출력:

```text
delta_y_side:  float32[B,1]
side_embedding: float32[B,256]
quality:        float32[B,1], [0,1]
```

`delta_y_side`는 centered-normalized screen y 단위의 residual입니다. Runtime fusion은 Front의
x를 유지하고 다음과 같이 최종 y를 구성합니다.

```python
final_y = front_gaze_xy[:, 1:2] + delta_y_side
```

`quality`는 sigmoid 출력이지만 `side_gaze_valid`를 대체하지 않습니다. `side_gaze_valid`,
target과 Front prediction은 Side model에 전달하지 않습니다.

## Config로 모델 선택

Profile은 다음 순서로 적용합니다.

```text
configs/config.yaml
configs/profiles/blazegaze.yaml
configs/profiles/side_profile_90.yaml
configs/models/<selected-side-model>.yaml
```

BlazeGaze transfer:

```bash
python -m gaze_pipeline validate-config \
  --config configs/config.yaml \
  --profile configs/profiles/blazegaze.yaml \
  --profile configs/profiles/side_profile_90.yaml \
  --profile configs/models/side_blazegaze_transfer.yaml \
  --skip-path-checks
```

MobileNetV4:

```bash
python -m gaze_pipeline validate-config \
  --config configs/config.yaml \
  --profile configs/profiles/blazegaze.yaml \
  --profile configs/profiles/side_profile_90.yaml \
  --profile configs/models/side_mobilenet_v4.yaml \
  --skip-path-checks
```

`train`과 `evaluate`에도 같은 profile 순서를 전달하면 runtime이 resolve된
`model.side.entrypoint`와 `init_args`로 선택한 모델을 생성합니다.

## Factory 직접 실행

다음 예시는 image-only MobileNetV4를 network download 없이 실행합니다.

```python
import torch

from gaze_pipeline.models.side.mobilenet_v4 import create_model

model = create_model(pretrained=False).eval()
side_image = torch.rand(2, 3, 128, 256, dtype=torch.float32)

with torch.inference_mode():
    output = model(side_image)

assert output["delta_y_side"].shape == (2, 1)
assert output["side_embedding"].shape == (2, 256)
assert output["quality"].shape == (2, 1)
```

BlazeGaze transfer는 import만 바꾸면 같은 방식으로 실행할 수 있습니다.

```python
from gaze_pipeline.models.side.blazegaze_transfer import create_model

model = create_model(encoder_weights_path=None).eval()
```

Optional auxiliary 입력 예시:

```python
output = model(
    side_image,
    side_head_pose_2d=torch.zeros(2, 2),
    side_eye_angles=torch.zeros(2, 2),
    side_iris_pose_2d=torch.zeros(2, 2),
)
```

## Runtime loader로 실행

```python
import torch

from gaze_pipeline.config import load_and_validate_config, resolve_model_forward_keys
from gaze_pipeline.model_runtime import build_model_runtime

config = load_and_validate_config(
    "configs/config.yaml",
    profiles=(
        "configs/profiles/blazegaze.yaml",
        "configs/profiles/side_profile_90.yaml",
        "configs/models/side_mobilenet_v4.yaml",
    ),
)
runtime = build_model_runtime(
    "side",
    config["model"]["side"],
    forward_keys=resolve_model_forward_keys(config, "side"),
).eval()

batch = {
    "side_image": torch.rand(2, 3, 128, 256),
    "side_head_pose_2d": torch.zeros(2, 2),
    "side_eye_angles": torch.zeros(2, 2),
    "side_iris_pose_2d": torch.zeros(2, 2),
    "side_gaze_valid": torch.ones(2, dtype=torch.bool),
}

with torch.inference_mode():
    output = runtime(batch)

assert output["delta_y_side"].shape == (2, 1)
```

Runtime은 활성화된 `forward_keys`만 model에 전달하므로 `side_gaze_valid`는 loss, metric과 fusion
mask로만 유지됩니다.

## MobileNetV4 weight와 normalization

- `pretrained: false`: offline-safe random initialization
- `pretrained: true`: timm/Hugging Face cache에서 ImageNet weight를 찾고 cold cache이면 download
- 외부 입력은 항상 RGB `[0,1]`
- `normalize_imagenet: true`이면 registered mean/std buffer로 model 내부에서 정규화

Pretrained cache나 내려받은 binary는 repository에 commit하지 않습니다.

## BlazeGaze encoder weight 변환

공식 `.keras`에서 encoder tensor만 변환합니다.

```bash
python scripts/convert_blazegaze_encoder.py \
  --source /absolute/path/blazegaze.keras \
  --output /absolute/path/blazegaze_side_encoder.pt \
  --source-commit 14719ad861467c98890058f7c41a94638ae1db2b
```

변환된 encoder-only payload는 model-local argument로 전달합니다.

```python
model = create_model(
    encoder_weights_path="/absolute/path/blazegaze_side_encoder.pt"
)
```

이 payload를 runtime의 generic full-model pretrained checkpoint로 전달하면 안 됩니다. `.keras`,
변환된 `.pt`, dataset, 실제 얼굴 이미지와 credential은 commit하지 않습니다. TensorFlow는 변환
도구와 opt-in parity 검사에만 필요하며 main runtime dependency가 아닙니다.

## 검사

```bash
python -m pytest -q \
  tests/test_side_encoder_models.py \
  tests/test_mobilenet_v4_side_model.py \
  tests/test_blazegaze_transfer_side_model.py \
  tests/test_convert_blazegaze_encoder.py

ruff check src/gaze_pipeline/models/side tests/test_side_encoder_models.py
ruff format --check src/gaze_pipeline/models/side tests/test_side_encoder_models.py
```

전체 저장소 검사는 `make check`를 사용합니다. Windows에 `make`가 없으면 README의 개별
`pytest`, `ruff check`, `ruff format --check` 명령을 실행하세요.
