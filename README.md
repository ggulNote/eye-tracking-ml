# ggulNote ML Pipeline

ggulNote의 시선 좌표 예측 모델을 위한 MLflow 기반 학습 파이프라인입니다.

아직 실제 webcam/phone 데이터와 운영 모델은 없습니다. 현재 버전은 synthetic 데이터로 아래 전체 흐름이 정상적으로 연결되는지 검증하는 베이스라인입니다.

```text
data source -> participant split -> preprocessing -> feature extraction
-> training -> evaluation -> MLflow tracking/registry -> model inference
```

> Synthetic 데이터의 성능 수치는 실제 시선 추정 성능을 의미하지 않습니다.

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

## 3. 설치 확인

설정 파일과 전체 synthetic 파이프라인을 확인합니다.

```bash
make validate
make smoke
make test
```

정상 결과:

```text
5 passed
```

`make smoke`는 MLflow 없이 synthetic 데이터 생성부터 모델 저장까지 실행합니다. 결과는 `outputs/local-<run-id>/`에 생성됩니다.

## 4. MLflow UI와 학습 실행

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

### 터미널 2: 학습

```bash
cd <REPOSITORY_DIRECTORY>
make train
```

학습이 완료되면 MLflow의 `ggulnote-gaze` Experiment에 새로운 Run이 추가되고, `ggulnote-gaze-ridge` 모델의 새 version이 등록됩니다.

MLflow 서버는 터미널 1에서 `Ctrl+C`로 종료합니다.

## 5. 팀 공통 명령

```bash
# 설정 검증
make validate CONFIG=configs/base.yaml

# MLflow 없이 전체 파이프라인 smoke test
make smoke

# MLflow에 기록하면서 학습
make train CONFIG=configs/base.yaml

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

## 6. 입출력 계약

| 파이프라인 경계 | 타입과 크기 |
|---|---|
| 원본 영상 샘플 | RGB `uint8[T,H,W,3]` |
| 전처리 출력 | `float32[B,T,3,H,W]` |
| 특징 추출 출력 | `float32[B,T,F]` |
| 모델 출력 | 정규화 시선 좌표 `float32[B,2]`, 범위 `0~1` |

- `B`: batch size
- `T`: 한 샘플의 프레임 수
- `H`, `W`: 전처리된 영상 높이와 너비
- `F`: 프레임별 feature 수

모든 경계에서 dtype, shape, NaN/무한대 및 좌표 범위를 검사합니다. 규격이 맞지 않으면 학습 초기에 명확한 오류를 발생시킵니다.

## 7. 설정 변경

기본 설정은 `configs/base.yaml`입니다.

```yaml
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

## 8. 실제 webcam/phone 영상 사용

OpenCV 의존성을 추가로 설치합니다.

```bash
make setup-video
```

`examples/manifest.example.csv`를 복사해 실제 manifest를 작성하고 config를 변경합니다.

```yaml
data:
  source: manifest
  manifest_path: data/manifest.csv
```

필수 manifest 열:

| 열 | 설명 |
|---|---|
| `video_path` | manifest 기준 상대 경로 또는 절대 경로 |
| `source` | `webcam` 또는 `phone` |
| `participant_id` | 익명화된 참여자 ID |
| `session_id` | 촬영 세션 ID |
| `target_x` | `0~1`로 정규화한 정답 x 좌표 |
| `target_y` | `0~1`로 정규화한 정답 y 좌표 |
| `rotation_degrees` | 선택 항목, `0/90/180/270` |

현재 manifest 계약은 한 영상 clip이 하나의 정답 시선 좌표를 갖는 calibration 데이터를 전제로 합니다. 프레임마다 정답 좌표가 다른 연속 영상은 별도의 frame-label adapter가 필요합니다.

## 9. 데이터 분할 원칙

train/validation/test는 영상이나 프레임 단위가 아니라 `participant_id` 단위로 분리합니다. 동일 사용자의 영상이 여러 split에 들어가 성능이 과대평가되는 것을 방지합니다.

각 split의 참여자와 샘플 수는 다음 위치에 기록됩니다.

- 로컬 output의 `split_manifest.json`
- MLflow Run의 `data/split_manifest.json` artifact

## 10. MLflow에서 확인할 내용

- 전체 config와 Git commit SHA
- 데이터 소스와 split manifest
- 전처리, feature extractor, 모델 버전
- 입출력 shape와 dtype
- train/validation/test MAE와 RMSE
- 정규화 좌표 거리 및 픽셀 거리 오차
- test 예측 CSV
- MLflow model signature
- Model Registry version

로컬 환경에서는 SQLite `mlflow.db`와 `mlruns/`를 사용합니다. 이 파일들은 각 팀원이 직접 실행해서 생성하는 로컬 결과이며 Git에는 포함하지 않습니다.

모든 팀원이 같은 MLflow Run을 공유해야 한다면 추후 공용 MLflow 서버, PostgreSQL, S3 또는 MinIO 구성이 필요합니다.

## 11. 프로젝트 구조

```text
configs/                    실행 설정
data/raw/                   원본 영상
data/interim/               중간 처리 결과
data/processed/             학습 입력 산출물
examples/                   manifest 예시
src/ggulnote_ml/data/       synthetic 및 영상 adapter
src/ggulnote_ml/preprocessing/
src/ggulnote_ml/features/
src/ggulnote_ml/models/
src/ggulnote_ml/evaluation/
src/ggulnote_ml/tracking/
src/ggulnote_ml/pipelines/
tests/                      단위 및 E2E 테스트
outputs/                    로컬 실행 결과
```

## 12. Git에 포함하지 않는 파일

`.gitignore`에 따라 다음 파일은 Git에 올라가지 않습니다.

- `.venv/`
- 실제 `data/raw`, `data/interim`, `data/processed` 내용
- `outputs/`
- `mlflow.db`
- `mlruns/`, `mlartifacts/`
- Python cache와 테스트 cache

특히 실제 얼굴 영상, 개인정보가 포함된 metadata, 로컬 MLflow DB를 Git에 커밋하지 마세요.

