# Side Encoder 사용과 contract

이 문서는 strict 90° Side 입력용 PyTorch model factory 두 개와 BlazeGaze encoder weight
converter의 현재 사용법과 통합 경계를 설명합니다. 두 factory는 직접 import하여 실행할 수 있고
model profile도 config validation을 통과하지만, profile의 entrypoint를 읽어 model을 자동 생성하는
runtime loader/registry/adapter는 아직 없습니다.

## 구현 범위

이번 작업에 포함된 범위는 다음과 같습니다.

- MobileNetV4-Conv-S Side model factory
- BlazeGaze convolution transfer Side model factory
- 두 model에서 공유하는 auxiliary projection과 output head
- preprocessing profile을 복사하지 않는 model별 작은 config override
- 공식 BlazeGaze `.keras`의 encoder-only weight converter
- 위 기능의 단위 테스트와 config resolve 테스트

다음 항목은 구현하거나 변경하지 않았습니다.

- canonical Front/Side preprocessing과 Dataset contract
- runtime model loader, registry와 adapter
- Trainer, loss, metric, optimizer와 scheduler
- Front Encoder
- late fusion, residual target와 `delta_y_side`
- checkpoint 관리, MLflow 학습과 inference pipeline

## Side tensor contract

필수 입력은 RGB `[0,1]` 범위의 NCHW tensor입니다.

```text
side_image: float32[B,3,128,256]
```

선택 입력은 다음과 같습니다.

```text
side_head_pose_2d: float32[B,2]
front_head_vector: float32[B,3]
side_eye_angles: float32[B,2]
side_iris_pose_2d: float32[B,2]
```

공개 출력 contract는 다음과 같습니다.

| Key | Shape | 의미 |
|---|---|---|
| `gaze_xy` | `float32[B,2]` | `(x,y)` centered-normalized screen coordinate의 독립 Side 예측 |
| `side_embedding` | `float32[B,256]` | downstream 결합을 위한 Side representation |
| `quality` | optional `float32[B,1]` | sigmoid가 적용된 `[0,1]` 신호 |

현재 두 factory는 세 key를 모두 반환합니다. Downstream adapter는 `gaze_xy`와
`side_embedding`을 필수로 다루고 `quality`는 존재할 때 소비할 수 있습니다. `quality`는
`side_gaze_valid`를 대체하지 않습니다. `delta_y_side`나 다른 residual target은 public output이
아닙니다.

## Model entrypoint

| Model | Config entrypoint |
|---|---|
| BlazeGaze transfer | `gaze_pipeline.models.side.blazegaze_transfer:create_model` |
| MobileNetV4-Conv-S | `gaze_pipeline.models.side.mobilenet_v4:create_model` |

두 함수는 모두 `torch.nn.Module`을 반환합니다. Config의 entrypoint 문자열과 `init_args`는
검증할 수 있지만 이를 해석하여 factory를 호출하는 product loader는 아직 구현되지 않았습니다.

## Config 적용 순서

Config는 반드시 다음 순서로 합성합니다. 뒤의 model profile은 preprocessing이나 Dataset 설정을
복사하지 않고 `model.side`의 entrypoint, init args, initialization metadata와 output contract만
override합니다.

```text
1. configs/config.yaml                              # base
2. configs/profiles/blazegaze.yaml                 # Front preprocessing
3. configs/profiles/side_profile_90.yaml           # Side preprocessing
4. configs/models/<selected-side-model>.yaml       # model 선택
```

BlazeGaze transfer profile 검증:

```bash
make validate-config \
  PROFILES="configs/profiles/blazegaze.yaml configs/profiles/side_profile_90.yaml configs/models/side_blazegaze_transfer.yaml"
```

MobileNetV4 profile 검증:

```bash
make validate-config \
  PROFILES="configs/profiles/blazegaze.yaml configs/profiles/side_profile_90.yaml configs/models/side_mobilenet_v4.yaml"
```

이 명령은 config merge와 contract validation만 수행합니다. 선택한 profile이 training이나
inference에서 model을 자동 교체한다는 뜻은 아닙니다.

`make`가 없는 Windows 환경에서는 같은 순서로 CLI를 직접 실행할 수 있습니다.

```powershell
python -m gaze_pipeline validate-config `
  --config configs/config.yaml `
  --profile configs/profiles/blazegaze.yaml `
  --profile configs/profiles/side_profile_90.yaml `
  --profile configs/models/side_mobilenet_v4.yaml `
  --skip-path-checks
```

## Loader 없이 factory 직접 실행

다음 예시는 network download나 실제 이미지 없이 두 factory를 직접 import하고 synthetic
forward를 실행합니다.

```python
import torch

from gaze_pipeline.models.side.blazegaze_transfer import (
    create_model as create_blazegaze_side,
)
from gaze_pipeline.models.side.mobilenet_v4 import (
    create_model as create_mobilenet_side,
)

batch_size = 2
side_image = torch.rand(batch_size, 3, 128, 256, dtype=torch.float32)
auxiliary = {
    "side_head_pose_2d": torch.zeros(batch_size, 2, dtype=torch.float32),
    "side_eye_angles": torch.zeros(batch_size, 2, dtype=torch.float32),
    "side_iris_pose_2d": torch.zeros(batch_size, 2, dtype=torch.float32),
}

models = {
    "blazegaze": create_blazegaze_side(encoder_weights_path=None),
    "mobilenet_v4": create_mobilenet_side(pretrained=False),
}

for name, model in models.items():
    model.eval()
    with torch.inference_mode():
        output = model(side_image, **auxiliary)
    assert output["gaze_xy"].shape == (batch_size, 2), name
    assert output["side_embedding"].shape == (batch_size, 256), name
    if "quality" in output:
        assert output["quality"].shape == (batch_size, 1), name
```

## Optional auxiliary 규칙

- `side_head_pose_2d`와 `front_head_vector`는 서로 다른 head source이므로 동시에 전달할 수
  없습니다.
- 모든 auxiliary tensor는 `side_image`와 batch 크기, device와 dtype이 같아야 하며 위에
  정의된 정확한 shape를 사용해야 합니다.
- head, eye와 iris는 각각 별도 projection slot을 사용합니다. 누락된 slot은 0으로 유지하므로
  image-only forward도 가능합니다.
- auxiliary의 NaN과 양·음의 무한대는 projection 전에 0으로 바꾸지만 이를 유효성 판단으로
  사용하지 않습니다.
- `side_gaze_valid`, `target_gaze_xy`와 Front branch prediction은 model에 전달하지 않습니다.
  `side_gaze_valid`는 model 밖에서 loss, metric과 fusion mask로 보존해야 합니다.

## MobileNetV4-Conv-S

MobileNet factory는 `timm==1.0.28`의 다음 exact model identifier를 사용합니다.

```text
mobilenetv4_conv_small.e2400_r224_in1k
```

Classifier를 제거한 feature extractor와 adaptive global average pooling을 사용하므로
`[B,3,128,256]` 비정사각 입력을 처리합니다. 외부 input contract는 계속 RGB float32
`[0,1]`입니다.

`normalize_imagenet: true`이면 `image_mean`과 `image_std`를 registered buffer로 보관하고 model
내부에서 ImageNet normalization을 적용합니다. 이를 위해 canonical preprocessing이나 Dataset
output을 변경하면 안 됩니다.

기본 config의 `pretrained: false`는 network download 없이 random initialization으로 생성됩니다.
`pretrained: true`는 timm에 exact model ID의 pretrained weight를 요청합니다. Cold cache에서는
network download가 발생하고, 이후에는 timm/Hugging Face가 구성한 cache를 사용합니다. Offline
또는 재현성이 중요한 환경에서는 cache 위치와 내용을 미리 고정해야 하며 다운로드된 binary를
repository에 추가하면 안 됩니다.

## BlazeGaze convolution transfer

Encoder topology와 converter mapping은 다음 lineage로 고정합니다.

```text
repository: https://github.com/RedForestAI/WebEyeTrack
commit: 14719ad861467c98890058f7c41a94638ae1db2b
source: python/webeyetrack/blazegaze.py:get_cnn_encoder
license: MIT
```

PyTorch model은 공식 first Conv2D, single Blaze block 5개, double Blaze block 6개와 squeeze
Conv2D/BatchNorm 순서를 재현합니다. TensorFlow SAME의 동적 비대칭 padding도 유지합니다.
공식 width 512와 Side width 256의 공간 크기가 다르므로 convolution encoder tensor만 전이하고
Side flatten projection, auxiliary projection, embedding, gaze와 quality head는 새로 초기화합니다.

`encoder_weights_path=None`은 file I/O 없는 random initialization입니다. 공식 `.keras`의
encoder weight를 사용하려면 TensorFlow를 별도 converter 환경에만 설치하고 다음 명령을 한 번
실행합니다.

```bash
python scripts/convert_blazegaze_encoder.py \
  --source /absolute/path/blazegaze.keras \
  --output /absolute/path/blazegaze_side_encoder.pt \
  --source-commit 14719ad861467c98890058f7c41a94638ae1db2b
```

Converter는 명시적으로 mapping된 Conv2D, DepthwiseConv2D와 BatchNorm tensor만 변환합니다.
출력에는 model 전체 객체가 아니라 CPU encoder `state_dict`와 다음 lineage metadata가 들어갑니다.

- source repository와 commit
- source checkpoint SHA-256
- mapping version
- converted/skipped tensor 목록

`.keras`, 변환된 `.pt`, 실제 얼굴 이미지, dataset과 credential은 repository에 commit하지
않습니다. 두 binary 확장자는 `.gitignore` 대상입니다. 변환된 파일은 다음처럼 model-local
encoder loader에만 전달합니다.

```python
from gaze_pipeline.models.side.blazegaze_transfer import create_model

model = create_model(encoder_weights_path="/absolute/path/blazegaze_side_encoder.pt")
```

TensorFlow와 실제 checkpoint가 있을 때의 opt-in parity 검사:

```bash
export WEBEYETRACK_WEIGHTS="/absolute/path/blazegaze_mpiifacegaze.keras"
python -m pytest -q tests/test_convert_blazegaze_encoder.py
```

## 검사

Side model, 공통 contract, converter와 config resolve 단위 테스트:

```bash
python -m pytest -q \
  tests/test_side_encoder_models.py \
  tests/test_mobilenet_v4_side_model.py \
  tests/test_blazegaze_transfer_side_model.py \
  tests/test_convert_blazegaze_encoder.py
```

저장소의 config, 전체 test, lint와 format 검사:

```bash
make check
```

TensorFlow와 실제 `.keras`가 없는 기본 suite에서는 parity 검사 한 개만 skip됩니다. TensorFlow는
main `requirements.txt`의 runtime dependency가 아닙니다.

## Loader/adapter handoff contract

Downstream loader/adapter 담당자는 다음 경계를 유지해야 합니다.

1. Base → Front preprocessing → Side preprocessing → selected model 순서로 resolve된
   `model.side.entrypoint`와 `init_args`를 사용합니다.
2. Factory 반환값이 `torch.nn.Module`인지 확인하고 model과 input을 같은 device로 이동합니다.
3. `side_image`와 활성화된 auxiliary key만 forward에 전달합니다. 두 head source는 동시에
   전달하지 않습니다.
4. `side_gaze_valid`, target과 Front prediction은 forward에 넣지 않고 downstream mask로
   유지합니다.
5. `gaze_xy [B,2]`와 `side_embedding [B,256]`을 필수로 검증합니다. `quality`가 있으면
   `[B,1]`, finite, `[0,1]`인지 검증하되 validity mask로 대체하지 않습니다.
6. Side `gaze_xy`는 독립 예측으로 유지합니다. residual target, `delta_y_side` 또는 fusion
   의미를 model output에 추가하지 않습니다.
7. 일반 training checkpoint 처리와 BlazeGaze encoder-only transfer payload를 혼합하지
   않습니다.

이 handoff는 향후 통합 계약일 뿐이며 현재 저장소에 runtime loader/adapter가 구현되어 있다는
뜻은 아닙니다.
