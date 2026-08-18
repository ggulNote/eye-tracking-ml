# eye-tracking-ml

웹캠 또는 스마트폰 RGB 영상을 동일한 형식으로 전처리하고, 시선 좌표 `(x, y)`를 예측하는 모델을 학습·평가하기 위한 로컬 ML 파이프라인입니다.

## 아키텍처

```text
RGB 영상
├─ Webcam
└─ Phone
    ↓
전처리
├─ 영상 로딩
├─ RGB 형식 통일
├─ Participant 기준 데이터 분리
└─ Processed Dataset 생성
    ↓
Feature Extraction
└─ 모델에 따라 교체 가능한 특징 추출 단계
    ↓
Model
├─ 현재: 파이프라인 검증용 Ridge Baseline
└─ 예정: WebEyeTrack / BlazeGaze
    ↓
Train / Evaluate / Predict
    ↓
MLflow
└─ 설정, 데이터 버전, 메트릭, 모델 기록
```

- `webcam`과 `phone` 영상은 전처리 후 동일한 RGB 데이터 형식으로 변환됩니다.
- 원본 영상은 `(T, H, W, 3)` 형태로 처리합니다.
- BlazeGaze를 연결하면 눈 영역은 `(B, 128, 512, 3)` 형태로 모델에 입력됩니다.
- 모델은 눈 영상, 머리 방향, 얼굴 위치를 받아 화면의 시선 좌표 `(x, y)`를 예측합니다.
- `128×512`는 원본 영상 크기가 아니라 전처리가 끝난 양쪽 눈 영역의 크기입니다.
- 학습·검증·테스트 데이터는 `participant_id` 기준으로 분리합니다.
- 데이터는 Git 저장소 밖의 `GGULNOTE_DATA_ROOT`에 저장합니다.
- MLflow에는 원본 영상이 아닌 설정, 메트릭, 데이터 버전, 모델만 기록합니다.
- `ggulnote-video-preprocess`는 정면 webcam에만 MediaPipe 얼굴·홍채, EAR, 8차원 2D 특징을 생성합니다. 측면 phonecam은 동기화된 이미지와 정답 좌표를 이미지 모델에 전달합니다.

## 모델 입출력

```text
입력
├─ image:          (B, 128, 512, 3) RGB float32
├─ head_vector:    (B, 3)           float32
└─ face_origin_3d: (B, 3)           float32

출력
└─ norm_pog:       (B, 2)           화면 중심 기준 시선 좌표
```

## 핵심 명령어

```bash
make setup
make setup-video
make setup-capture

make validate
make preprocess
make train
make evaluate
make predict

make mlflow
make test
make smoke
```

## 논문 참고 2카메라 시선 데이터 수집

GUI 지원 OpenCV를 별도로 설치한 뒤 연결 가능한 카메라 번호를 확인합니다.

```bash
make setup-capture
make list-cameras
make check-display
make check-cameras
```

`check-display`는 카메라나 참가자 데이터를 만들지 않고 Dot Test 화면만 엽니다.
초록 테두리가 화면 네 면에 모두 닿는지 확인한 뒤 `Q` 또는 `Esc`로 닫습니다.

`check-cameras`는 데이터를 만들지 않고 두 카메라를 엽니다. 정면 webcam은
MediaPipe face+iris 검출이 연속 5프레임 성공해야 통과합니다. phonecam은
MediaPipe를 적용하지 않고 측면 이미지 모델 입력으로 보존하므로, 한쪽 눈의
초점·노출·구도를 육안으로 확인합니다.

이름 기반 폴더로 simulation 저장 계약을 먼저 검증합니다.

```bash
make simulate PARTICIPANT=테스트참가자 HEAD_POSE=neutral PROTOCOL=all DATASET_ROOT=/private/tmp/gaze-simulation
```

실제 전체 프로토콜을 실행합니다. `PARTICIPANT`와 `HEAD_POSE`를 생략하면 이름과
head pose를 차례로 한 번만 입력받습니다. 이후 해당 참가자·자세 전용 레이턴시를
먼저 측정하고, 측정이 성공한 경우에만 Dot Test를 시작합니다.

```bash
make collect
# 이름 적어주세요: 안은제
# headpose를 입력해주세요 [neutral/head_up/head_down]: neutral
```

A+B 전체 새 데이터 흐름은 같은 참가자 이름으로 아래 순서를 지킵니다. 카메라 검사
후에는 위치·해상도·연결 방식을 바꾸지 않습니다. `make collect` 한 번으로 레이턴시,
Dot Test, 프레임 동기화, 정면 MediaPipe 특징 추출까지 연속 실행합니다.

```bash
make check-cameras
make collect PARTICIPANT=안은제 HEAD_POSE=neutral
```

촬영은 완료됐지만 자동 후처리만 실패한 경우에는 원본을 다시 찍지 않습니다.
`feature_maps/synchronized.csv`가 없으면 `make prepare-participant`, 이미 있으면
`make video-features`로 MediaPipe 단계만 재개합니다. 정면 특징까지 있고
`feature_maps/webeyetrack/`만 없으면 `make webeyetrack-inputs PARTICIPANT=이름
HEAD_POSE=neutral`로 마지막 단계만 재개합니다.

상대방이 공유한 iCloud Drive 폴더로 완료 데이터를 자동 백업하려면 기존 폴더의
절대 경로를 지정합니다. 수집·후처리가 정상 완료된 뒤 모든 파일을 SHA-256으로
검증하고 로컬 삭제 여부를 묻습니다. `네`, `예`, `y`, `yes` 중 하나를 입력한
경우에만 해당 head pose의 로컬 폴더를 삭제합니다.

```bash
make collect ICLOUD_BACKUP_ROOT="/Users/ahn-eunje/Library/Mobile Documents/com~apple~CloudDocs/공유폴더이름"
```

공유 폴더가 없거나 보기 전용이거나 같은 참가자·자세 백업이 이미 있으면 덮어쓰지
않고 중단하며 로컬 원본을 유지합니다. iCloud 서버 업로드 상태는 Finder의 상태
아이콘에서도 확인합니다.

레이턴시를 이미 성공적으로 측정했고 Dot Test만 다시 시작해야 하는 예외 상황에서는
기존 `calibration/latency.json`을 확인한 뒤 다음처럼 명시합니다.

```bash
make collect PARTICIPANT=안은제 HEAD_POSE=neutral CAPTURE_ONLY=1
```

기본 `intrinsics_2d` 모드는 실제 웹캠·폰캠 `Camera.mat`을 참가자 폴더에 복사하고,
B 전처리에서 OpenCV 렌즈 왜곡 보정 후 MediaPipe 좌표를 추출합니다.
`monitorPose.mat`과 stereo 보정은 요구하지 않으며, 보정된 2D 영상 좌표에서 화면
좌표를 직접 학습합니다.
카메라 위치·줌·해상도가 바뀌면 같은 모델의 입력 분포가 달라지므로 다시 수집하거나
별도의 장비 설정으로 관리해야 합니다.

레이턴시 측정만 들어 있는 `안은제/neutral/calibration` 폴더에는 이어지는 수집 결과를
같은 이름으로 저장할 수 있습니다.

데이터는 `data/raw/participants/안은제/{neutral,head_up,head_down}/` 아래에 자세별로
생성됩니다. 자세마다 한 번만 수집하며 정면 영상은 `video/web/capture.mp4`, 참가자 좌측 30–45° iPhone 영상은
`video/phone/capture.mp4`에 저장합니다. 각 클릭의 가장 좋은 프레임 한 쌍은
`images/web`, `images/phone`에 같은 sample ID로 저장하고, 최종 정답과 품질 정보는
단일 `labels/labels.csv`에 기록합니다. 동기화·MediaPipe 결과는 같은 참가자의
`feature_maps/` 안에 생성됩니다.

실제 수집이 끝나면 동기화, 정면 MediaPipe 특징, WebEyeTrack용 `512x128` 눈 ROI,
`head_vector[3]`, `face_origin_3d[3]`까지 자동 생성됩니다. 기존 완료 촬영본만 다시
처리할 때는 `make webeyetrack-batch`를 사용합니다.

## 모델 학습 담당자용 데이터 인수인계

참가자 한 명의 데이터는 이름 아래 `neutral`, `head_up`, `head_down` 촬영으로
분리됩니다. 각 자세 폴더는 독립된 촬영이지만 세 폴더의 `participant`는 같은
사람이므로, train/validation/test 분할은 **head pose가 아니라 참가자 단위**로
수행해야 합니다.

```text
participants/<participant>/<head_pose>/
├── calibration/              # 카메라 내부 보정과 레이턴시 측정
├── video/{web,phone}/        # 수정하지 않는 원본 MP4·timestamp
├── images/{web,phone}/       # 수집 시 고른 레이턴시 보정 전 JPEG
├── labels/labels.csv         # 점당 선택본·정답·품질의 수집 라벨
├── metadata/                 # 전체 프레임 기록·점 계획·설정
├── feature_maps/             # 동기화·렌즈 보정·모델 입력
└── participant.json          # 촬영 상태와 스키마의 최상위 색인
```

정상 완료한 자세 하나는 학습점 63개(정적 27개+세로 왕복 36개)와 평가점
18개로 구성됩니다. 따라서 web/phone 쌍은 81개이며, 이미지 모델 manifest는
학습 126행(63쌍×2 view), 평가 36행(18쌍×2 view)입니다.

### 모델별 시작 파일

| 목적 | 입력 파일 | 사용 방법 |
|---|---|---|
| 정면 WebEyeTrack | `feature_maps/webeyetrack/training.csv` | `eye_patch_path`, `head_vector[3]`, `face_origin_3d[3]`를 입력으로 사용 |
| 정면 MediaPipe 8차원 모델 | `feature_maps/training_features.csv` | `feature_valid=1`인 정면 눈·홍채 특징 사용 |
| 측면 이미지 모델 | `feature_maps/training.csv` | `view=phonecam`, `collection_split=training`으로 필터 |
| 정면 이미지 모델 | `feature_maps/training.csv` | `view=webcam`, `collection_split=training`으로 필터 |
| 정면+측면 fusion | `feature_maps/training.csv` | 같은 `pair_id`의 webcam/phonecam 행을 결합 |
| 최종 평가 | 각 `evaluation.csv` | 학습에 포함하지 않고 동일한 계약으로 평가 |

### 같은 이미지를 여러 폴더에 두는 이유

| 위치 | 의미 | 모델 학습 권장 여부 |
|---|---|---|
| `video/*/capture.mp4` | Dot Test 전체 원본 영상 | 재처리용 원본 |
| `images/*/sNNNNNN.jpg` | 클릭 후 품질 기준으로 수집 당시에 고른 JPEG; 레이턴시·렌즈 보정 전 | 육안 품질 확인용 |
| `feature_maps/*/frames/*.png` | 레이턴시 동기화 후 MP4에서 다시 추출하고 각 `Camera.mat`으로 렌즈 왜곡을 제거한 PNG | 이미지 모델 입력 |
| `feature_maps/webeyetrack/eye_roi/*.png` | 정면 PNG에서 정규화한 `512x128` 양쪽 눈 패치 | WebEyeTrack 입력 |

같은 `sample`은 같은 표적을 뜻하지만 `images/`와 `feature_maps/*/frames/`의
원본 프레임 번호와 timestamp가 항상 같지는 않습니다. 후처리는 같은
프로토콜·segment·target 안에서 가장 가까운 유효 동기화 프레임을 다시 선택합니다.
모델은 `feature_maps` manifest의 `source_frame`, `source_timestamp`,
`corrected_timestamp`를 기준으로 추적하십시오.

### 이름이 겹치는 파일

| 이름 | 위치별 차이 |
|---|---|
| `training.csv` | `feature_maps/training.csv`는 두 카메라 이미지 manifest, `feature_maps/webeyetrack/training.csv`는 정면 눈 ROI·3D 입력 전용 |
| `evaluation.csv` | 위 두 입력 계약의 평가 전용 파일이며 어느 것도 학습에 넣지 않음 |
| `features.csv` | `web/features.csv`에는 정면 MediaPipe 행이 있고, `phone/features.csv`는 현재 phone MediaPipe 미사용으로 헤더만 존재 |
| `summary.json` | `feature_maps/summary.json`은 영상 전처리 요약, `webeyetrack/summary.json`은 눈 ROI·head pose 품질 요약 |
| `timestamps.csv` | `video/web`과 `video/phone` 원본 프레임의 카메라별 Unix ns 시각 |
| `Camera.mat` | web과 phone의 서로 다른 렌즈 보정값; 두 파일을 교환하거나 재사용하면 안 됨 |

### 학습 전 필수 확인

- `participant.json`의 `status=completed`와 `calibration/latency.json`의
  `status=valid`를 확인합니다.
- 원본 `labels.csv`와 `frame_log.csv`의 timestamp는 레이턴시 보정 전입니다.
  학습 프레임은 `feature_maps`의 corrected timestamp와 manifest를 사용합니다.
- `feature_maps/*/frames`에는 렌즈 왜곡 보정이 이미 적용되어 있으므로 다시
  `undistort`하지 않습니다.
- `split=evaluation` 또는 `collection_split=evaluation`은 학습에 넣지 않습니다.
- 같은 사람의 `neutral`, `head_up`, `head_down`을 서로 다른 데이터 split에 두지
  않습니다. 새 참가자 일반화 평가는 참가자 이름을 그룹 키로 분리합니다.
- 현재 MediaPipe는 정면 web에만 적용됩니다. `phone/features.csv`가 헤더만 있는 것은
  정상이며 측면 branch는 `phone/frames` 이미지를 사용합니다.
- 정답 좌표의 기준은 일반 모델에서는 `x_norm`, `y_norm`, WebEyeTrack에서는
  `target_x_centered`, `target_y_centered`로 통일합니다.
- 평가에서는 유효 샘플 MAE뿐 아니라 얼굴·홍채 미검출률과 `invalid_reason`도 함께
  보고합니다.
- 얼굴 영상과 참가자 이름은 직접 식별 정보입니다. 데이터 원본은 Git/MLflow에
  올리지 않고 외부 전달본은 연구용 ID로 익명화합니다.

폴더별 역할, 모든 주요 CSV 열, timestamp·좌표·유효성 규칙은
[참가자 데이터 폴더와 CSV 설명](docs/participant-data-layout.md)을 먼저 읽고,
코드 수준의 고정 열 목록은 [CSV 스키마 전체 목록](docs/csv-schemas.md)을 참고합니다.

실명과 얼굴 영상을 함께 보관하면 직접 식별 가능한 연구 데이터가 됩니다. 실제
수집에서는 가능하면 동의서의 대응표에만 실명을 두고, 입력 이름에는 연구용 별칭을
사용하십시오.

macOS는 `avfoundation`, Linux는 `v4l2`, Windows는 `dshow` backend를 사용합니다. iPhone 연속성 카메라는 후면 카메라만 제공합니다.

자세한 설정과 입출력 규격은 [다중 카메라 캡처 문서](docs/multi-camera-capture.md)를 참고합니다.

전체 검증은 다음 명령으로 실행할 수 있습니다.

```bash
make check
```

자세한 내용은 다음 문서를 참고합니다.

- [데이터 파이프라인](docs/data-pipeline.md)
- [모델 파이프라인](docs/model-pipeline.md)
- [WebEyeTrack 입출력 규격](docs/webeyetrack-data-contract.md)
- [다중 카메라 캡처](docs/multi-camera-capture.md)
- [참가자 데이터 폴더와 CSV 설명](docs/participant-data-layout.md)
- [A+B 영상 전처리 연결](docs/video-preprocessing-handoff.md)
- [CSV 스키마 전체 목록](docs/csv-schemas.md)
- [협업 규칙](CONTRIBUTING.md)
