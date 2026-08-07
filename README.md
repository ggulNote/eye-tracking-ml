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
- 현재는 전체 파이프라인 검증용 모델이 연결되어 있으며, MediaPipe와 BlazeGaze는 추후 추가할 예정입니다.

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

## MacBook 웹캠 + iPhone 동시 캡처

GUI 지원 OpenCV를 별도로 설치한 뒤 연결 가능한 카메라 번호를 확인합니다.

```bash
make setup-capture
make list-cameras
```

`configs/capture.yaml`에서 `device_index`를 확인하고 동시 캡처를 실행합니다.

```bash
make capture
```

기본 설정은 `0=MacBook`, `1=iPhone 연속성 카메라`입니다. 두 미리보기가 반대로 나오면 두 번호를 교환합니다. `Q` 또는 `Esc`로 종료하면 카메라별 MP4와 공통 세션 시계 기준 timestamp CSV가 `data/raw/captures/`에 저장됩니다. 원본 영상은 `.gitignore` 대상입니다.

macOS는 `avfoundation`, Linux는 `v4l2`, Windows는 `dshow` backend를 사용합니다. `mirror`는 미리보기에만 적용되며 학습용 원본 파일은 반전하지 않습니다. iPhone 연속성 카메라는 후면 카메라만 제공합니다.

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
- [협업 규칙](CONTRIBUTING.md)
