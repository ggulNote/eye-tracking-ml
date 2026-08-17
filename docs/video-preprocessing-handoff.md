# 영상 전처리와 후속 파이프라인 연결

## 입력과 출력

전처리는 참가자 폴더 밖에 결과를 흩어놓지 않습니다. 한 참가자의 수집,
동기화, 특징 결과는 같은 참가자 이름과 head pose 폴더에서 관리합니다.

```text
안은제/
└── neutral/
    ├── calibration/{web,phone}/Camera.mat
    ├── video/{web,phone}/capture.mp4
    ├── labels/labels.csv
    ├── metadata/frame_log.csv
    └── feature_maps/
        ├── synchronized.csv
        ├── web/{frames,features.csv}
        ├── phone/{frames,features.csv}
        ├── training.csv
        ├── evaluation.csv
        ├── training_features.csv
        ├── evaluation_features.csv
        ├── webeyetrack/{eye_roi,inputs.csv,training.csv,evaluation.csv,summary.json}
        └── summary.json
```

## 처리 경계

1. `sync-participant`가 `metadata/frame_log.csv`에 카메라별 latency를 적용합니다.
2. 보정 timestamp가 가까운 web/phone 프레임을 `feature_maps/synchronized.csv`에
   일대일로 연결합니다.
3. `labels/labels.csv`의 점당 품질 최적 샘플을 같은 프로토콜·segment·target의
   유효 동기화 pair에 연결합니다.
4. 두 MP4에서 지정된 프레임 번호를 직접 디코딩합니다. FPS로 프레임을 추측하지
   않습니다.
5. 각 카메라의 `Camera.mat`으로 렌즈 왜곡을 보정합니다.
6. web에는 MediaPipe 얼굴·홍채, EAR, 8차원 특징을 적용합니다.
7. phone은 측면 이미지 모델용 동기화 이미지로 보존하며 MediaPipe를 강제하지 않습니다.
8. web의 양쪽 눈을 `512x128`로 정규화하고, `Camera.mat` 기반 `head_vector[3]`,
   `face_origin_3d[3]`를 생성합니다. phone에는 이 정면용 MediaPipe 단계를 적용하지
   않습니다.

원본 MP4, `metadata/frame_log.csv`, `labels/labels.csv`는 덮어쓰지 않습니다.

## 실행

새 참가자 수집에서는 `make collect`가 촬영 직후 이 전처리를 자동 실행합니다.
아래 명령은 기존 촬영본을 처리하거나 `feature_maps/synchronized.csv`가 생성되기 전
자동 후처리 실패를 재개할 때 사용합니다. 동기화는 끝났고 MediaPipe 단계만 실패한
경우에는 `make video-features`만 실행합니다.

동기화와 특징 추출을 한 번에 실행합니다.

```bash
make prepare-participant PARTICIPANT=안은제 HEAD_POSE=neutral
```

나누어 실행하려면 다음과 같습니다.

```bash
make sync-participant PARTICIPANT=안은제 HEAD_POSE=neutral
make video-features PARTICIPANT=안은제 HEAD_POSE=neutral
make webeyetrack-inputs PARTICIPANT=안은제 HEAD_POSE=neutral
```

완료된 모든 참가자·자세를 일괄 처리할 때는 다음 명령을 사용합니다. 이미 결과가 있는
촬영은 건너뜁니다. 의도적으로 다시 만들 때만 `FORCE=1`을 붙입니다.

```bash
make webeyetrack-batch
make webeyetrack-batch FORCE=1
```

직접 CLI를 사용할 수도 있습니다.

```bash
python -m ggulnote_ml.video_preprocessing \
  --participant 안은제 \
  --head-pose neutral \
  --dataset-root data/raw/participants \
  --ear-threshold 0.20 \
  --feature-cameras webcam
```

`--output-root`를 생략하는 것이 기본이며 `안은제/neutral/feature_maps/`에 저장합니다.
외부 실험 경로가 꼭 필요한 경우에만 `--output-root`를 명시합니다.

## 정면 8차원 특징 순서

1. `left_eye_center_x`
2. `left_eye_center_y`
3. `left_iris_center_x`
4. `left_iris_center_y`
5. `right_eye_center_x`
6. `right_eye_center_y`
7. `right_iris_center_x`
8. `right_iris_center_y`

좌우 EAR과 눈 감김 여부는 별도 열입니다. 얼굴·홍채 미검출 또는 양쪽 눈 감김은
`feature_valid=0`과 `invalid_reason`으로 남기며 학습 특징에서 제외합니다.

## 학습과 평가 분리

- `training.csv`: train 정적 27점 + 세로 왕복 클릭 36점의 두 카메라 이미지
- `evaluation.csv`: 학습하지 않는 정적 평가 18점의 두 카메라 이미지
- `training_features.csv`: 유효한 정면 MediaPipe 학습 특징
- `evaluation_features.csv`: 평가 coverage를 보기 위해 실패 행까지 보존한 특징

phone 학습은 `training.csv`의 `view=phonecam`, 정면 학습은 `view=webcam` 행을
사용합니다. 같은 `pair_id`로 두 뷰를 fusion할 수 있습니다.

전체 폴더와 열 설명은 [참가자 데이터 폴더와 CSV 설명](participant-data-layout.md)을
참고합니다.
