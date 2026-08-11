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
- `ggulnote-video-preprocess`는 동기화 영상 프레임의 MediaPipe 얼굴·홍채, EAR, 8차원 2D 특징을 생성합니다. WebEyeTrack용 3D head/eye-patch processor와 BlazeGaze 모델은 별도 후속 범위입니다.

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
make check-cameras
```

`check-cameras`는 데이터를 만들지 않고 두 카메라를 엽니다. 웹캠은 정면 얼굴,
phonecam은 한쪽 눈이 보이는 측면 얼굴을 허용하지만 두 영상 모두 MediaPipe
face+iris 검출이 연속 5프레임 성공해야 통과합니다. phonecam에서도 얼굴 윤곽은
프레임 안에 남겨야 합니다.

다음 참가자 번호를 확인하고 simulation으로 저장 계약을 먼저 검증합니다.

```bash
make suggest-participant
make simulate PARTICIPANT=p00 PROTOCOL=all DATASET_ROOT=/private/tmp/gaze-simulation
```

실제 전체 프로토콜을 실행합니다. `PARTICIPANT`를 생략하면 기존 폴더를 확인한 추천 번호를 보여주고 입력받습니다.

```bash
make collect PARTICIPANT=p00
```

A+B 전체 새 데이터 흐름은 같은 참가자 번호로 아래 순서를 지킵니다. 카메라 검사
후에는 위치·해상도·연결 방식을 바꾸지 않습니다.

```bash
make check-cameras
make measure-latency PARTICIPANT=p00
make collect PARTICIPANT=p00 ALLOW_MISSING_CALIBRATION=1
make sync-participant PARTICIPANT=p00
make video-features PARTICIPANT=p00
```

레이턴시 측정만 들어 있는 `p00/Calibration` 폴더는 이미 수집된 참가자로 세지지
않으므로, 이어지는 수집에서도 `p00`을 그대로 사용할 수 있습니다.

데이터는 `data/raw/participants/p00/` 아래에 생성됩니다. 참가자마다 한 번만 수집하며 정면 카메라는 `webcam/capture.mp4`, 참가자 좌측 30–45° iPhone은 `phonecam/capture.mp4`에 저장합니다. 정적 학습 3×9, 세로 왕복 3열×6점, 정적 평가 3×6은 참가자가 점을 제대로 본 뒤 마우스 왼쪽 버튼 또는 Space로 확정합니다. 각 클릭 후 안정 구간에서 눈 검출·얼굴 검출·선명도·노출을 기준으로 가장 좋은 프레임 한 쌍만 `images/webcam`, `images/phonecam`에 같은 sample ID로 저장하고 `labels/image_samples.csv`에서 원본 영상 프레임·좌표와 연결합니다.

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
- [A+B 영상 전처리 연결](docs/video-preprocessing-handoff.md)
- [CSV 스키마 전체 목록](docs/csv-schemas.md)
- [협업 규칙](CONTRIBUTING.md)
