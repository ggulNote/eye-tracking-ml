# 영상 전처리와 후속 파이프라인 연결

이 브랜치의 책임 범위는 A의 latency 동기화 결과를 읽어 동기화 프레임을 추출하고, 영상용 MediaPipe 특징을 생성하는 데까지다. 별도 작업자가 구현한 정적 이미지 전처리와는 독립된 처리이며, 추출된 PNG와 image manifest만 그 파이프라인에도 전달한다.

## 전체 경계

```text
feat/latency-sync
  synchronized_frames.csv
feat/click-confirmed-dot-capture
  image_samples.csv (점당 품질 최적 프레임 1쌍)
        |
        v
영상 전처리(현재 브랜치)
  같은 protocol/segment에서 가장 가까운 유효 동기화 pair 선택
  MP4 frame decode
  MediaPipe face + refined iris landmarks
  left/right EAR + eye-closed state
  normalized 2D feature vector (8 dimensions)
        |
        +--> camera processed CSV / video training CSV
        |
        +--> PNG + image manifest --> 별도 정적 이미지 전처리
```

원본 `labels.csv`, `image_samples.csv`, `synchronized_frames.csv`, `capture.mp4`는 읽기만 하며 수정하거나 덮어쓰지 않는다.

## 입력

```text
data/raw/participants/p00/
├── participant.json
├── labels/image_samples.csv
├── synchronized/synchronized_frames.csv
├── webcam/capture.mp4
└── phonecam/capture.mp4
```

`image_samples.csv`는 클릭 한 번당 품질이 가장 좋았던 원본 pair를 지정한다. 전처리기는 같은 `protocol`, `segment`, `target`, `direction` 안에서 이 원본 pair와 가장 가까운 `valid=1`, `usable=1` 동기화 pair를 하나 선택한다. 따라서 전체 프로토콜을 완료하면 학습 63쌍과 평가 18쌍, 총 81쌍만 처리한다.

실제 디코딩에는 선택된 동기화 CSV의 `webcam_frame`, `phonecam_frame`을 그대로 사용한다. MP4 FPS나 근사 timestamp로 프레임을 다시 추측하지 않는다.

## 실행

```bash
pip install -e ".[video,landmarks]"
ggulnote-video-preprocess \
  --participant p00 \
  --dataset-root data/raw/participants \
  --output-root data/interim/dual_view \
  --ear-threshold 0.20
```

이 프로젝트는 `mp.solutions.face_mesh`와 Apple Silicon용 universal2 wheel을 함께 제공하는 `mediapipe==0.10.21`을 사용합니다. `make setup-capture`가 카메라 GUI와 MediaPipe 의존성을 함께 설치합니다.

같은 명령은 `python -m ggulnote_ml.video_preprocessing`으로도 실행할 수 있다.

## 영상 8차원 특징 순서

모든 좌표는 MediaPipe가 반환하는 frame 기준 normalized 2D 좌표다.

1. `left_eye_center_x`
2. `left_eye_center_y`
3. `left_iris_center_x`
4. `left_iris_center_y`
5. `right_eye_center_x`
6. `right_eye_center_y`
7. `right_iris_center_x`
8. `right_iris_center_y`

카메라 기하 보정은 적용하지 않는다. 이후 geometry 브랜치를 병합할 때 이 8개 원본 2D 좌표를 변환 입력으로 사용할 수 있다.

좌우 EAR은 별도 `left_ear`, `right_ear` 열이다. 각 EAR이 threshold 미만인지 좌우 눈 감김 열에 기록하고, 두 눈이 모두 감긴 경우에만 `eye_closed=1` 및 `feature_valid=0`으로 처리한다. 따라서 phonecam 측면 영상에서 한쪽 눈만 보이거나 닫힌 것으로 추정되어도 다른 한쪽이 열려 있으면 8차원 계약을 유지한다. 다만 MediaPipe FaceMesh가 시작되려면 한쪽 눈 단독 crop이 아니라 얼굴 윤곽이 포함되어야 하며, 보이지 않는 쪽 좌표는 모델의 추정값이라는 점을 해석 시 고려해야 한다.

## 출력

```text
data/interim/dual_view/
├── p00/
│   ├── webcam/
│   │   ├── *.png
│   │   └── processed_features.csv
│   └── phonecam/
│       ├── *.png
│       └── processed_features.csv
└── manifests/
    ├── p00_training.csv
    ├── p00_evaluation.csv
    ├── p00_video_training.csv
    ├── p00_video_evaluation.csv
    └── p00_summary.json
```

카메라별 `processed_features.csv`에는 training과 evaluation을 모두 보존한다. 각 행은 참가자, 카메라, A의 sample/pair, 원본 frame 번호, 원본 timestamp, latency 보정 timestamp, target, 얼굴·홍채 검출 여부, EAR, 눈 감김, 8차원 특징과 무효 사유를 포함한다.

`p00_video_training.csv`에는 다음 조건을 모두 만족한 pair의 두 카메라 행만 들어간다.

- A 결과의 `valid=1`, `usable=1`, `split=train`
- webcam과 phonecam 모두 얼굴·홍채 검출 성공
- 두 카메라 모두 `feature_valid=1`
- evaluation 행이 아님

`p00_video_evaluation.csv`에는 evaluation pair를 별도로 저장한다. 검출 실패 행도 평가 coverage 확인을 위해 `feature_valid=0`과 사유를 유지하며 학습 CSV에는 절대 들어가지 않는다.

## 정적 이미지 전처리로 전달

`p00_training.csv`와 `p00_evaluation.csv`는 GitHub `main`의 정적 이미지 파이프라인 입력 규격이다. 이 파일은 `sample_id`, `subject_id`, `view`, `image_path`, `pair_id`, 화면 pixel target과 화면 크기를 포함한다. 영상 MediaPipe 결과는 해당 정적 이미지 파이프라인이 재사용하지 않으며 두 처리 경로는 독립적이다.

```bash
DUAL_VIEW_MANIFEST="/absolute/path/p00_training.csv" \
make prepare GAZE_DATA_ROOT="/absolute/path/data/interim/dual_view"
```
