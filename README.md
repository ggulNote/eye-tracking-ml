# eye-tracking-ml

웹캠·스마트폰 RGB 영상으로 화면의 시선 좌표 `(x, y)`를 예측하는 모델을 학습하기 위한 **로컬 우선 ML 파이프라인**입니다.

이 저장소는 향후 [WebEyeTrack](https://github.com/RedForestAI/WebEyeTrack)의 BlazeGaze 계열 모델이 들어와도 데이터 수집부터 전처리, 학습, 평가, MLflow 기록까지 같은 실행 방식으로 사용할 수 있도록 구성합니다.

> 현재는 파이프라인 구조와 데이터 계약을 검증하는 단계입니다. MediaPipe 눈 영역 추출과 BlazeGaze 학습 코드는 아직 연결되지 않았으며, `spatial_statistics + ridge` 모델이 전체 흐름을 확인하는 임시 기준 모델로 동작합니다.

## 현재 상태

| 구분 | 상태 | 설명 |
|---|---:|---|
| 로컬 데이터 파이프라인 | ✅ | webcam/phone 영상을 동일한 RGB 프레임 형식으로 변환 |
| 데이터·모델 입출력 계약 | ✅ | shape, dtype, 범위 검증 |
| 학습·평가·예측 흐름 | ✅ | 임시 ridge 모델로 실행 가능 |
| MLflow 연동 | ✅ | 설정, 메트릭, 모델을 로컬에 기록 |
| MediaPipe 눈 전처리 | 예정 | 얼굴 랜드마크, 양쪽 눈 crop, head pose 계산 |
| BlazeGaze 학습 | 예정 | 사전학습 가중치 또는 논문 모델 연결 |

## 아키텍처

```text
src/ggulnote_ml
├─ data
│  └─ Webcam/Phone Manifest → Participant Split → Processed Store
├─ preprocessing
│  └─ RGB/Gray Canonical Full-frame Adapter
├─ features
│  └─ 교체 가능한 Feature Extractor
├─ models
│  └─ Model Registry + 임시 Ridge Baseline
├─ pipelines
│  └─ Preprocess → Train → Evaluate → Predict
├─ tracking
│  └─ Local MLflow Metadata / Metrics / Model
└─ contracts.py
   └─ Raw / Canonical / WebEyeTrack Input-Output Contract
```

- 원본 데이터는 `webcam`, `phone` 구분과 관계없이 같은 canonical frame으로 변환됩니다.
- 학습·검증·테스트 분리는 프레임이 아닌 `participant_id` 기준으로 수행해 데이터 누수를 막습니다.
- 영상과 processed 데이터는 Git 프로젝트 밖의 `GGULNOTE_DATA_ROOT`에 저장합니다.
- MLflow에는 원본 영상 대신 설정, 데이터 버전, 해시, 메트릭, 모델만 기록합니다.
- `128×512`는 원본 영상 크기가 아니라 MediaPipe 정규화가 끝난 **양쪽 눈 영역**의 목표 크기입니다.

## 전체 파이프라인

```text
webcam / phone RGB 영상 + manifest.csv
                    │
                    ▼
              make preprocess
                    │
                    ▼
      processed/dataset-v001/*.npz
                    │
                    ▼
                make train
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    학습된 모델          Local MLflow UI
          │
          ▼
  make evaluate / make predict
```

## 모델 입출력

BlazeGaze가 연결되면 다음 규격을 사용합니다.

| 항목 | Shape | dtype / 범위 | 의미 |
|---|---|---|---|
| 원본 RGB 프레임 | `(T, H, W, 3)` | `uint8`, `[0, 255]` | webcam/phone 입력 영상 |
| `image` | `(B, 128, 512, 3)` | `float32`, `[0, 1]` | 정규화된 양쪽 눈 RGB 영역 |
| `head_vector` | `(B, 3)` | `float32` | 머리 방향 벡터 |
| `face_origin_3d` | `(B, 3)` | `float32`, cm | 카메라 좌표계의 얼굴 중심 |
| 출력 `norm_pog` | `(B, 2)` | `float32`, `[-0.5, 0.5]` | 화면 중심 기준 시선 좌표 |

현재 전처리는 다양한 원본 영상을 동일한 full-frame 형식으로 만드는 단계까지 구현되어 있습니다. 이후 MediaPipe와 homography 단계가 `image`, `head_vector`, `face_origin_3d`를 생성하게 됩니다.

자세한 규격은 [WebEyeTrack 데이터 계약](docs/webeyetrack-data-contract.md)을 참고하세요.

## 빠른 시작

Python 3.9 이상을 권장합니다.

```bash
git clone https://github.com/ggulNote/eye-tracking-ml.git
cd eye-tracking-ml

make setup
make setup-video
cp .env.example .env
```

`.env`에 이 컴퓨터에서 사용할 데이터 경로를 지정합니다.

```dotenv
GGULNOTE_DATA_ROOT=/Volumes/ggulnote-data
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Windows에서는 예를 들어 `GGULNOTE_DATA_ROOT=D:\\ggulnote-data`를 사용할 수 있습니다. `.env`와 실제 데이터는 Git에 올리지 않습니다.

## 데이터 준비

데이터 루트는 다음 구조를 권장합니다.

```text
ggulnote-data/
├─ raw/
│  ├─ webcam/
│  └─ phone/
├─ manifests/
└─ processed/
```

실제 파일 경로와 `participant_id`, 입력 장치, 정답 좌표는 manifest에 기록합니다. 시작용 형식은 [`configs/manifest-local.example.yaml`](configs/manifest-local.example.yaml)과 [`examples/manifest.csv`](examples/manifest.csv)를 참고하세요.

## 핵심 명령어

```bash
# 설정과 데이터 계약 확인
make validate CONFIG=configs/manifest-local.example.yaml

# raw 영상을 공통 형식으로 전처리
make preprocess CONFIG=configs/manifest-local.example.yaml

# 임시 기준 모델 학습
make train CONFIG=configs/manifest-local.example.yaml

# 테스트 및 synthetic 전체 흐름 검증
make test
make smoke

# MLflow UI 실행: http://127.0.0.1:5000
make mlflow
```

전체 검증은 한 명령으로 실행할 수 있습니다.

```bash
make check
```

## 문서

- [CONTRIBUTING.md](CONTRIBUTING.md): 브랜치, 커밋, PR, 코드 수정 규칙
- [docs/data-pipeline.md](docs/data-pipeline.md): raw → processed 데이터 흐름
- [docs/model-pipeline.md](docs/model-pipeline.md): 학습, 평가, MLflow 설계
- [docs/webeyetrack-data-contract.md](docs/webeyetrack-data-contract.md): BlazeGaze 입력·출력 규격

## 의존성 파일

- `requirements.txt`: 실행에 필요한 고정 버전
- `requirements-video.txt`: OpenCV 영상 전처리 기능
- `requirements-dev.txt`: 테스트와 개발 도구
- `pyproject.toml`: 패키지 메타데이터와 지원 버전 범위

## 저장 원칙

- Git에는 코드, 설정 예시, 문서만 저장합니다.
- 원본 영상, processed 배열, `.env`, MLflow artifact는 저장하지 않습니다.
- 데이터셋은 `dataset_version`, manifest hash, 전처리 버전으로 추적합니다.
