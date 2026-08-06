# ggulNote ML Pipeline

ggulNote의 시선 좌표 예측 모델을 위한 MLflow 기반 학습 파이프라인입니다.

아직 실제 webcam/phone 데이터와 운영 모델은 없습니다. 현재 버전은 synthetic 데이터로 전체 흐름을 검증하고, 실제 영상에는 프레임 정답 동기화와 WebEyeTrack 호환 입출력 계약을 제공합니다.

```text
local raw data -> participant split -> local processed dataset
-> feature extraction -> training -> evaluation -> MLflow metadata/model
```

> Synthetic 데이터의 성능 수치는 실제 시선 추정 성능을 의미하지 않습니다.

## 아키텍처

```text
src/ggulnote_ml
├── data/             webcam/phone manifest, participant split, processed store
├── preprocessing/    공통 full-frame 변환
├── features/         임시 feature adapter
├── models/           현재 Ridge placeholder와 model registry
├── pipelines/        preprocess/train/evaluate/predict orchestration
├── tracking/         로컬 MLflow metadata/model 기록
└── contracts.py      raw/canonical/WebEyeTrack 입출력 계약
```

- 실제 영상과 processed tensor는 `GGULNOTE_DATA_ROOT`에 저장합니다.
- Git에는 코드, config, 계약 문서와 테스트만 포함합니다.
- participant 단위로 train/validation/test를 분리합니다.
- `128×512×3`은 원본 frame이 아니라 향후 BlazeGaze eye patch 계약입니다.
- MediaPipe/BlazeGaze 구현 전까지 `spatial_statistics + ridge`는 실행 확인용 placeholder입니다.

상세 설계:

- [로컬 데이터 파이프라인](docs/data-pipeline.md)
- [모델 파이프라인](docs/model-pipeline.md)
- [WebEyeTrack 데이터 계약](docs/webeyetrack-data-contract.md)

## 1. 저장소 내려받기

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
```

Python 3.9 이상이 필요합니다. Conda 환경을 사용하고 있어도 괜찮지만, 프로젝트 명령은 충돌을 방지하기 위해 저장소 안의 `.venv`를 사용합니다.

## 2. 개발 환경 설치

가장 간단한 방법:

```bash
make setup
```

직접 설치하려면 다음 명령을 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

의존성 파일의 역할:

- `requirements.txt`: 학습과 MLflow 실행에 필요한 런타임 의존성
- `requirements-dev.txt`: 테스트 의존성 포함
- `requirements-video.txt`: 실제 mp4 영상 로딩을 위한 OpenCV 포함

## 3. 로컬 데이터 저장소 설정

실제 영상과 processed 데이터는 Git 프로젝트 밖에 둡니다. `.env.example`을 `.env`로 복사하고 절대 경로를 입력합니다.

```bash
cp .env.example .env
```

macOS:

```dotenv
GGULNOTE_DATA_ROOT=/Volumes/ggulnote-data
```

Windows:

```dotenv
GGULNOTE_DATA_ROOT=D:\ggulnote-data
```

```text
<GGULNOTE_DATA_ROOT>/
├── raw/
│   ├── webcam/
│   └── phone/
├── manifests/
│   └── manifest.csv
└── processed/
    └── <dataset_version>/
```

`.env`는 Git에 포함되지 않습니다. Shell 환경변수가 이미 설정돼 있으면 `.env`보다 우선합니다.

## 4. 설치 확인

설정 파일과 전체 synthetic 파이프라인을 확인합니다.

```bash
make validate
make smoke
make test
```

정상 결과:

```text
12 passed
```

`make smoke`는 MLflow 없이 synthetic 데이터 생성부터 모델 저장까지 실행합니다. 결과는 `outputs/local-<run-id>/`에 생성됩니다.

## 5. MLflow UI와 학습 실행

터미널을 두 개 사용합니다.

### 터미널 1: MLflow UI

```bash
make mlflow
```

아래 메시지가 나오면 정상입니다.

```text
Listening at: http://127.0.0.1:5000
```

브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:5000
```

### 터미널 2: 전처리와 학습

```bash
cd <REPOSITORY_DIRECTORY>
make preprocess CONFIG=configs/<actual-config>.yaml
make train CONFIG=configs/<actual-config>.yaml
```

학습이 완료되면 MLflow의 `ggulnote-gaze` Experiment에 새로운 Run이 추가되고, `ggulnote-gaze-ridge` 모델의 새 version이 등록됩니다.

MLflow 서버는 터미널 1에서 `Ctrl+C`로 종료합니다.

## 6. 팀 공통 명령

```bash
# 설정 검증
make validate CONFIG=configs/base.yaml

# MLflow 없이 전체 파이프라인 smoke test
make smoke

# raw -> processed 데이터셋 생성
make preprocess CONFIG=configs/<actual-config>.yaml

# MLflow에 기록하면서 학습
make train CONFIG=configs/<actual-config>.yaml

# 저장 모델을 test split에서 재평가
make evaluate \
  CONFIG=configs/base.yaml \
  MODEL=outputs/<run-id>/model.npz

# test split의 첫 번째 샘플 예측
make predict \
  CONFIG=configs/base.yaml \
  MODEL=outputs/<run-id>/model.npz

# 단위 및 E2E 테스트
make test
```

팀원은 `src/` 내부 모듈을 임의로 조합하기보다 config와 위 명령을 사용합니다. 각 Run에서 실제로 사용된 config, 데이터 분할, 입출력 schema 및 평가 결과가 함께 저장됩니다.

`dataset_version` 디렉터리가 이미 있으면 전처리는 중단됩니다. 일반적으로 `dataset-v002`처럼 새 버전을 만들고, 의도적으로 덮어쓸 때만 다음 명령을 사용합니다.

```bash
make preprocess CONFIG=configs/<actual-config>.yaml PREPROCESS_FLAGS=--force
```

## 7. 입출력 계약

| 파이프라인 경계 | 타입과 크기 |
|---|---|
| 원본 영상 샘플 | RGB `uint8[T,H,W,3]` |
| 공통 full-frame 출력 | `float32[B,T,C,H,W]`, RGB `C=3` / gray `C=1` |
| 특징 추출 출력 | `float32[B,T,F]` |
| 모델 출력 | 정규화 시선 좌표 `float32[B,2]`, 범위 `0~1` |
| 추후 WebEyeTrack 입력 | eye patch `[B,128,512,3]` + head `[B,3]` + face origin `[B,3]` |
| 추후 WebEyeTrack 출력 | 화면 중심 기준 `float32[B,2]`, 범위 `-0.5~0.5` |

- `B`: batch size
- `T`: 한 샘플의 프레임 수
- `C`: config의 `preprocessing.color_mode`에 따른 channel 수. `webeyetrack_v1`은 RGB만 허용
- `H`, `W`: 공통 full-frame 높이와 너비 (`128×512` eye patch와 다른 값)
- `F`: 프레임별 feature 수

모든 경계에서 dtype, shape, NaN/무한대 및 좌표 범위를 검사합니다. 규격이 맞지 않으면 학습 초기에 명확한 오류를 발생시킵니다.

## 8. 설정 변경

기본 설정은 `configs/base.yaml`입니다.

```yaml
data:
  source: manifest
  manifest_path: manifests/manifest.csv
  dataset_version: dataset-v001
  processed_dir: processed
  train_from_processed: true

feature_extraction:
  name: identity
  version: v1

model:
  name: ridge
  version: v1
  alpha: 1.0
```

현재 feature extractor:

- `identity`: 각 RGB 프레임을 펼쳐 `[B,T,F]`로 변환
- `spatial_statistics`: RGB 통계와 밝기 중심점으로 `[B,T,8]` 생성

현재 모델:

- `ridge`: 정규화된 `(x,y)` 좌표를 예측하는 다중 출력 Ridge 회귀 모델

새 모델은 `GazeRegressor` 계약을 구현한 뒤 `src/ggulnote_ml/models/registry.py`에 등록합니다. 이후 코드를 직접 분기하지 않고 config의 `model.name`과 `model.version`만 변경합니다.

## 9. 실제 webcam/phone 영상 사용

OpenCV 의존성을 추가로 설치합니다.

```bash
make setup-video
```

`examples/manifest.example.csv`를 복사해 실제 manifest를 작성하고 config를 변경합니다.

시작용 설정은 `configs/manifest-local.example.yaml`입니다.

```yaml
data:
  source: manifest
  manifest_path: manifests/manifest.csv
  dataset_version: dataset-v001
  processed_dir: processed
  train_from_processed: true
```

`manifest_path`와 `processed_dir`의 상대 경로는 Git 프로젝트가 아니라 `GGULNOTE_DATA_ROOT` 기준입니다.

필수 manifest 열:

| 열 | 설명 |
|---|---|
| `video_path` | manifest 기준 상대 경로 또는 절대 경로 |
| `source` | `webcam` 또는 `phone` |
| `participant_id` | 익명화된 참여자 ID |
| `session_id` | 촬영 세션 ID |
| `target_x` | `target_coordinate_space`에 따른 정답 x 좌표 |
| `target_y` | `target_coordinate_space`에 따른 정답 y 좌표 |

실제 데이터에 권장하거나 좌표 형식에 따라 필요한 열:

| 열 | 설명 |
|---|---|
| `label_timestamp_ms` / `label_frame_index` | 권장, 정답과 영상 프레임을 동기화하는 시점 |
| `target_coordinate_space` | `top_left_normalized`, `screen_centered_normalized`, `screen_pixels` |
| `screen_width_px`, `screen_height_px` | pixel 좌표 변환 및 평가용 화면 크기 |
| `rotation_degrees` | 선택, `0/90/180/270` |
| `camera_intrinsics_path` | 선택, 장치별 카메라 보정 JSON 경로 |

같은 영상에 정답 시점이 여러 개라면 `video_path`를 반복하여 시점별로 한 행씩 기록합니다. 모든 선택 열과 좌표 변환 규칙은 [WebEyeTrack 데이터 계약](docs/webeyetrack-data-contract.md)을 참고하세요.

## 10. 데이터 분할 원칙

train/validation/test는 영상이나 프레임 단위가 아니라 `participant_id` 단위로 분리합니다. 동일 사용자의 영상이 여러 split에 들어가 성능이 과대평가되는 것을 방지합니다.

각 split의 참여자와 샘플 수는 다음 위치에 기록됩니다.

- 로컬 output의 `split_manifest.json`
- MLflow Run의 `data/split_manifest.json` artifact

## 11. MLflow에서 확인할 내용

- 전체 config와 Git commit SHA
- 데이터 소스와 split manifest
- dataset version, dataset hash, manifest hash
- participant/sample count, preprocessor version
- 전처리, feature extractor, 모델 버전
- 입출력 shape와 dtype
- train/validation/test MAE와 RMSE
- 정규화 좌표 거리 및 픽셀 거리 오차
- MLflow model signature
- Model Registry version

로컬 환경에서는 SQLite `mlflow.db`와 `mlruns/`를 사용합니다. MLflow에는 config, 데이터 통계·해시, metric, schema, 모델만 기록하며 원본 영상과 processed tensor는 올리지 않습니다.
`test_predictions.csv`는 로컬 `outputs/<run-id>/`에만 저장합니다.

현재 구성에는 S3/MinIO 업로드·다운로드나 object storage 인증 코드가 없습니다. 나중에 중앙 서버 학습이 필요해질 때 NAS, 외장 디스크 복사 또는 object storage를 별도로 검토합니다.

## 12. 프로젝트 구조

```text
configs/                    실행 설정
examples/                   manifest 예시
docs/                       WebEyeTrack 데이터/모델 경계 문서
src/ggulnote_ml/data/       synthetic 및 영상 adapter
src/ggulnote_ml/preprocessing/
src/ggulnote_ml/features/
src/ggulnote_ml/models/
src/ggulnote_ml/evaluation/
src/ggulnote_ml/tracking/
src/ggulnote_ml/pipelines/
tests/                      단위 및 E2E 테스트
outputs/                    로컬 실행 결과

외부 GGULNOTE_DATA_ROOT/
  raw/                      실제 webcam/phone 영상
  manifests/                로컬 manifest
  processed/<version>/      NPZ, metadata, dataset hash
```

## 13. Git에 포함하지 않는 파일

`.gitignore`에 따라 다음 파일은 Git에 올라가지 않습니다.

- `.venv/`
- `.env`
- 실제 `data/raw`, `data/interim`, `data/processed` 내용
- `outputs/`
- `mlflow.db`
- `mlruns/`, `mlartifacts/`
- Python cache와 테스트 cache
- `*.mp4`, `*.mov`, `*.avi`, `*.mkv`

특히 실제 얼굴 영상, 개인정보가 포함된 metadata, 로컬 MLflow DB를 Git에 커밋하지 마세요.
