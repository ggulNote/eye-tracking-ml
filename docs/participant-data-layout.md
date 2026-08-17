# 참가자 데이터 폴더와 CSV 설명

## 한 줄 요약

참가자 한 명의 결과는 입력한 이름 아래 `neutral`, `head_up`, `head_down` 폴더로
나누어 저장합니다. 각 자세 폴더에서 모델 학습에 직접 전달하는 정답표는
`labels/labels.csv` **하나**입니다. 동기화 계산에 필요한 전체 프레임 기록은
라벨과 혼동되지 않도록 `metadata/frame_log.csv`에 따로 보존합니다.

## 최종 폴더 구조

```text
data/raw/participants/안은제/
├── neutral/
│   ├── calibration/{latency.json,latency_runs/,web/Camera.mat,phone/Camera.mat}
│   ├── video/{web,phone}/
│   ├── images/{web,phone}/
│   ├── labels/labels.csv
│   ├── metadata/{frame_log.csv,protocol.csv,capture_config.yaml}
│   ├── feature_maps/{synchronized.csv,synchronization.json,web/,phone/,webeyetrack/}
│   └── participant.json
├── head_up/
│   └── <neutral과 같은 전체 구조>
└── head_down/
    └── <neutral과 같은 전체 구조>
```

자세 하나를 완료하면 web 81장과 phone 81장이 생성됩니다. 세 자세를 모두 완료하면
참가자 한 명당 243개의 카메라 쌍, 즉 이미지 486장이 만들어집니다.

폴더 이름에는 공백을 넣지 않습니다. `feature map` 대신 `feature_maps`를 쓰는
이유는 macOS·Windows·Linux 터미널과 학습 코드에서 경로를 안전하게 다루기
위해서입니다.

## 폴더별 의미

### `calibration/`

카메라 영상과 화면 표시 사이의 오차를 보정하기 위한 장비 설정입니다.
참가자의 시선 정답 자체가 아니며, 같은 고정 장비를 정확히 해석하기 위한 값입니다.

- `latency.json`: 화면에 점이 표시된 시점과 그 변화가 카메라 프레임에 보인 시점의
  차이입니다. 이 값을 빼야 서로 같은 실제 순간의 web/phone 프레임과 표적 좌표를
  고를 수 있습니다.
- `latency_runs/<run_id>/`: 검정·흰색 전환 실험의 원본 영상, 밝기 값, 검출 결과입니다.
  최종 중앙값만 보지 않고 실패 원인을 다시 확인할 수 있게 보존합니다.
- `web/Camera.mat`: 정면 웹캠 렌즈의 초점·중심·왜곡 계수입니다.
- `phone/Camera.mat`: 측면 폰카메라 렌즈의 별도 보정값입니다. 두 렌즈가 다르므로
  web 값을 phone에 복사하면 안 됩니다.

현재 기본 `intrinsics_2d` 모드는 두 `Camera.mat`으로 렌즈 왜곡을 펴고, 화면 좌표는
수집된 `x_norm`, `y_norm`을 직접 학습합니다. `monitorPose.mat`과 stereo 보정은
3D 기하를 사용할 때만 필요하며 현재 2D 학습에는 필수가 아닙니다.

### `video/`

Dot Test 전체 구간을 촬영한 원본 영상입니다.

- `video/web/capture.mp4`: 노트북 정면 웹캠 영상
- `video/phone/capture.mp4`: 참가자 왼쪽 30–45° 측면 폰카메라 영상
- 각 `timestamps.csv`: 영상 프레임 번호와 실제 Unix 나노초 시각의 대응표
- 각 `recording.json`: 저장 FPS·해상도·프레임 수 등 영상 기록 정보

원본 영상은 재동기화하거나 다른 전처리를 시험할 수 있도록 수정하지 않습니다.

### `images/`

참가자가 점을 확인한 뒤 클릭했을 때, 안정 구간 후보 중 품질 점수가 가장 높은
프레임을 카메라별로 한 장씩 저장합니다.

- `images/web/s000000.jpg`
- `images/phone/s000000.jpg`

같은 `s000000`은 같은 표적을 본 한 쌍입니다. 전체 프로토콜을 완료하면 카메라별
81장입니다.

- 정적 학습 27점
- 세로 왕복 클릭 36점
- 정적 평가 18점

이 폴더의 이미지는 수집 당시 선택본입니다. MediaPipe 결과를 이미지 폴더 안에
섞지 않습니다. 원본 선택 이미지와 전처리 산출물을 구분하기 위해 MediaPipe와
동기화 결과는 `feature_maps/`에 둡니다.

### `labels/`

`labels/labels.csv` 하나만 존재합니다. web 이미지, phone 이미지, 두 카메라의
프레임 번호·시각, 표적 정답 좌표, 품질 정보를 한 행에 합친 최종 샘플 표입니다.
기존 `image_samples.csv`의 역할을 이 파일로 통합했습니다.

### `metadata/`

학습 라벨이 아니라 수집을 재현하고 동기화하기 위한 기록입니다.

- `frame_log.csv`: 촬영 중 읽은 모든 카메라 프레임 쌍과 당시 화면 표적
- `protocol.csv`: 점이 어떤 순서·좌표·용도로 제시되도록 계획됐는지 기록
- `capture_config.yaml`: 해당 참가자를 촬영할 때 실제 사용한 설정 스냅샷

기존 `events/` 폴더는 이름만으로 의미가 불명확하고 프로토콜마다 CSV가 나뉘어
있었습니다. 이를 `metadata/protocol.csv` 한 파일로 합쳤습니다.

### `feature_maps/`

레이턴시 보정과 모델 입력 전처리가 끝난 산출물입니다.

- `synchronized.csv`: web/phone 프레임의 레이턴시를 빼고 같은 실제 순간끼리 매칭한 표
- `synchronization.json`: 전체·유효·무효 동기화 쌍 수와 사용한 latency 요약
- `web/frames/`: 렌즈 왜곡 보정 후 동기화된 정면 이미지
- `web/features.csv`: 정면 이미지의 MediaPipe 8차원 특징, EAR, 검출 유효성
- `phone/frames/`: 같은 순간의 렌즈 왜곡 보정 측면 이미지
- `phone/features.csv`: 측면 특징 계약 파일
- `training.csv`, `evaluation.csv`: 이미지 모델에 전달할 카메라별 manifest
- `training_features.csv`: 학습에 사용할 유효한 MediaPipe 행
- `evaluation_features.csv`: 평가 샘플의 MediaPipe 결과
- `summary.json`: 입력 경로, 보정 사용 여부, 행 수, 실패 사유 집계
- `webeyetrack/eye_roi/`: 정면 이미지에서 정규화한 `512x128` 양쪽 눈 패치
- `webeyetrack/inputs.csv`: 눈 ROI·머리 방향·3D 얼굴 위치·정답을 합친 전체 입력 표
- `webeyetrack/training.csv`: 학습 split의 유효 입력만 포함
- `webeyetrack/evaluation.csv`: 최종 평가 split의 유효 입력만 포함
- `webeyetrack/summary.json`: split별 유효율, 얼굴 거리, pose 재투영 오차 요약

현재 정책은 **MediaPipe를 web에만 적용**합니다. 측면 phone 영상은 한쪽 눈이 주로
보이기 때문에 정면 FaceMesh 검출을 강제하지 않습니다. 따라서
`feature_maps/phone/features.csv`는 경로 계약을 유지하는 헤더 파일이고,
측면 모델은 `feature_maps/phone/frames/`를 입력으로 사용합니다. 나중에 phone 전용
ResNet 특징을 만들면 같은 phone 폴더에 별도 파일로 추가할 수 있습니다.

WebEyeTrack 입력 생성은 정면 web에만 적용합니다. `face_origin_z_cm`가 카메라와 얼굴의
추정 거리이며, 별도 줄자 값을 CSV에 넣는 대신 보정된 정면 영상에서 프레임마다
계산합니다. 이 값은 같은 장비에서 비정상 거리나 보정 오류를 찾는 품질 검사에도
사용합니다.

### `participant.json`

참가자 이름, `head_pose`, 완료/중단 여부, 카메라 역할·설정, 화면 크기, 사용 프로토콜,
보정 상태와 위 폴더 경로를 기록하는 최상위 색인입니다. 데이터를 전달받은 코드는
이 파일로 스키마 버전과 장비 조건을 먼저 검사합니다.

## `labels/labels.csv` 열 설명

| 열 | 의미 |
|---|---|
| `sample` | 점 하나에 대응하는 고유 ID (`s000000` 형식) |
| `participant` | 실행 시작 시 입력한 참가자 이름 |
| `head_pose` | `neutral`, `head_up`, `head_down` 중 현재 촬영 자세 |
| `protocol` | `train_static_3x9`, `vertical_click_3col_6row`, `evaluation_static_3x6` 중 해당 프로토콜 |
| `split` | `train`은 학습용, `evaluation`은 최종 평가 전용 |
| `pair` | 품질 최적 프레임이 나온 수집 프레임 쌍 번호 |
| `display_timestamp` | 선택 프레임을 읽을 때 화면에 점이 표시된 Unix ns 시각 |
| `confirmation_timestamp` | 참가자가 해당 점을 확인하고 클릭한 Unix ns 시각 |
| `confirmation_offset_ms` | 점이 나타난 뒤 클릭까지 걸린 시간(ms) |
| `candidate_count` | 클릭 후 안정 구간에서 품질 비교에 사용한 후보 프레임 수 |
| `pair_quality_score` | web과 phone 후보 품질을 합한 선택 점수 |
| `web_image` | 선택된 정면 이미지의 참가자 폴더 기준 경로 |
| `web_frame` | 정면 원본 MP4의 프레임 번호 |
| `web_timestamp` | 정면 프레임을 받은 Unix ns 시각 |
| `web_face_detected` | 수집 시 OpenCV 얼굴 휴리스틱 성공 여부(0/1) |
| `web_eyes_detected` | 수집 시 OpenCV 눈 휴리스틱 검출 수 |
| `web_mediapipe_face_detected` | 정면 MediaPipe 얼굴 검출 여부(0/1) |
| `web_mediapipe_iris_detected` | 정면 MediaPipe 홍채 검출 여부(0/1) |
| `web_sharpness` | 정면 이미지 Laplacian 분산 기반 선명도 |
| `web_brightness` | 정면 얼굴 ROI 또는 프레임의 평균 밝기 |
| `phone_image` | 선택된 측면 이미지의 참가자 폴더 기준 경로 |
| `phone_frame` | 측면 원본 MP4의 프레임 번호 |
| `phone_timestamp` | 측면 프레임을 받은 Unix ns 시각 |
| `phone_face_detected` | 측면 OpenCV 얼굴 휴리스틱 성공 여부(참고값) |
| `phone_eyes_detected` | 측면 OpenCV 눈 휴리스틱 검출 수(참고값) |
| `phone_mediapipe_face_detected` | 현재 phone MediaPipe 미사용이므로 보통 0 |
| `phone_mediapipe_iris_detected` | 현재 phone MediaPipe 미사용이므로 보통 0 |
| `phone_sharpness` | 측면 이미지 선명도 |
| `phone_brightness` | 측면 이미지 평균 밝기 |
| `x_px`, `y_px` | Dot Test 캔버스의 표적 픽셀 좌표 |
| `x_norm`, `y_norm` | 화면 좌상단 `(0,0)`, 우하단 `(1,1)`의 정규화 표적 좌표 |
| `x_centered`, `y_centered` | 화면 중심 `(0,0)`, 범위 `[-0.5,0.5]`의 표적 좌표 |
| `segment` | 해당 프로토콜 안에서 실제 제시된 순서 번호 |
| `target` | 원래 격자에서의 표적 위치 ID |
| `direction` | `static`, `top_to_bottom`, `bottom_to_top` |

수집 시점의 `labels.csv` 프레임 번호는 아직 레이턴시 보정 전입니다. 학습 전에는
`feature_maps/synchronized.csv`와 매칭하여 보정된 프레임을 사용합니다.

## `metadata/frame_log.csv` 열 설명

이 파일은 모든 프레임 쌍을 기록하므로 보통 81행보다 훨씬 많습니다.

| 열 묶음 | 의미 |
|---|---|
| `participant`, `head_pose`, `protocol`, `split`, `pair` | 참가자·자세·프로토콜·용도·전체 프레임 쌍 번호 |
| `display_timestamp` | 해당 반복에서 화면을 갱신한 Unix ns 시각 |
| `webcam_frame`, `webcam_timestamp` | 정면 원본 프레임 번호와 Unix ns 시각 |
| `phonecam_frame`, `phonecam_timestamp` | 측면 원본 프레임 번호와 Unix ns 시각 |
| `x_px`, `y_px` | 당시 표시 표적의 픽셀 좌표; 안내/종료 화면은 비어 있음 |
| `x_norm`, `y_norm` | 당시 표시 표적의 `[0,1]` 좌표 |
| `x_centered`, `y_centered` | 당시 표시 표적의 중심 원점 좌표 |
| `segment`, `target`, `direction` | 프로토콜 구간·격자 위치·진행 방향 |
| `confirmation_timestamp` | 해당 점 클릭 시각; 클릭 전이면 비어 있음 |
| `confirmation_offset_ms` | 점 표시부터 클릭까지의 시간 |
| `usable` | 클릭 후 안정 구간으로 사용할 수 있으면 1 |

내부 열 이름에 `webcam`·`phonecam`이 남은 것은 동기화 수식과 기존 코드 계약을
유지하기 위한 카메라 역할명입니다. 실제 디렉터리 이름은 `web`·`phone`입니다.

## `metadata/protocol.csv` 열 설명

| 열 | 의미 |
|---|---|
| `participant`, `head_pose` | 참가자 이름과 현재 촬영 자세 |
| `protocol` | 프로토콜 이름 |
| `split` | 학습·평가·중앙 응시 구분 |
| `segment` | 실제 실행 순서 |
| `repeat` | 반복 번호 |
| `target` | 격자 위치 ID |
| `x_norm`, `y_norm` | 계획된 점 좌표 |
| `direction` | 정적 또는 위아래 진행 방향 |
| `confirmation_required` | 참가자 클릭이 필요하면 1 |

## `video/*/timestamps.csv` 열 설명

| 열 | 의미 |
|---|---|
| `frame` | 해당 MP4 안의 0부터 시작하는 프레임 번호 |
| `elapsed_ms` | 이 촬영 시작 이후 경과 시간 |
| `timestamp` | 프레임을 받은 Unix ns 시각 |

## `feature_maps/synchronized.csv` 열 설명

| 열 묶음 | 의미 |
|---|---|
| `participant`, `head_pose`, `pair` | 참가자·자세와 동기화 결과 쌍 번호 |
| `webcam_frame`, `phonecam_frame` | 같은 실제 순간으로 선택된 두 원본 프레임 |
| `webcam_timestamp`, `phonecam_timestamp` | 보정 전 카메라 수신 시각 |
| `webcam_latency_ms`, `phonecam_latency_ms` | 적용한 카메라별 레이턴시 중앙값 |
| `webcam_corrected_timestamp`, `phonecam_corrected_timestamp` | 수신 시각에서 레이턴시를 뺀 시각 |
| `corrected_time_diff_ms` | 보정 후 두 카메라 시각 차이 |
| `reference_timestamp` | 두 보정 시각의 대표 기준 시각 |
| `target_timestamp` | 기준 시각에 대응시킨 화면 표적 시각 |
| `x_norm`, `y_norm`, `x_centered`, `y_centered` | 보정 후 프레임에 붙는 정답 좌표 |
| `protocol`, `split`, `segment`, `target`, `direction` | 해당 정답의 프로토콜 정보 |
| `usable` | 안정 구간인지 여부 |
| `target_interpolated` | 동적 표적 보간을 사용했는지 여부 |
| `valid_sync` | 두 카메라 시간차가 허용 범위인지 여부 |
| `valid_target` | 기준 시각과 유효한 표적을 연결했는지 여부 |
| `valid` | `valid_sync`와 `valid_target`이 모두 참인지 여부 |
| `invalid_reason` | 무효일 때 구체적인 사유 |

## `feature_maps/*/features.csv` 열 설명

앞부분의 `sample_id`·`participant`·`head_pose`·`camera`·`pair_id`·프레임/타임스탬프·프로토콜·
정답 좌표는 특징을 원본과 다시 연결하는 키입니다. `intrinsics_applied`,
`intrinsics_rms_px`, `camera_matrix_path`는 렌즈 보정 적용 상태입니다.

검출·전처리 열:

- `face_detected`, `iris_detected`, `landmark_count`: MediaPipe 검출 상태
- `left_ear`, `right_ear`: 눈 벌어짐 정도(Eye Aspect Ratio)
- `left_eye_closed`, `right_eye_closed`, `eye_closed`: 눈 감김 판정
- `feature_valid`: 얼굴·홍채가 검출되고 눈을 뜬 유효 행이면 1
- `invalid_reason`: `face_not_detected`, `iris_not_detected`, `eye_closed` 등

8차원 특징의 고정 순서:

1. `left_eye_center_x`
2. `left_eye_center_y`
3. `left_iris_center_x`
4. `left_iris_center_y`
5. `right_eye_center_x`
6. `right_eye_center_y`
7. `right_iris_center_x`
8. `right_iris_center_y`

마지막 `image_path`는 해당 특징을 만든 `feature_maps/web/frames` 또는
`feature_maps/phone/frames` 이미지 경로입니다.

현재 MediaPipe 특징은 정면 `web`에만 생성합니다. 따라서
`feature_maps/phone/features.csv`는 같은 입출력 계약을 보존하는 헤더 파일이며,
행이 없는 것이 정상입니다. 측면 branch는 `feature_maps/phone/frames/`의 이미지를
사용합니다.

## `feature_maps/training.csv`, `evaluation.csv` 열 설명

두 파일은 렌즈 보정된 정면·측면 PNG를 이미지 모델에 전달하는 manifest입니다.
`training.csv`에는 학습점, `evaluation.csv`에는 학습에 사용하지 않는 평가점만
들어갑니다. 한 프레임 쌍이 `view=webcam`, `view=phonecam` 두 행으로 표현되며 같은
`pair_id`로 fusion합니다.

| 열 | 의미 |
|---|---|
| `sample_id` | 카메라 view까지 포함한 행 고유 ID |
| `subject_id` | 참가자 ID |
| `head_pose` | `neutral`, `head_up`, `head_down` |
| `view` | `webcam` 또는 `phonecam` |
| `image_path` | `feature_maps` 기준 렌즈 보정 PNG 상대 경로 |
| `pair_id` | 같은 순간의 webcam·phonecam 행이 공유하는 결합 키 |
| `target_x_px`, `target_y_px` | 정규화 정답을 화면 크기로 환산한 모델용 픽셀 좌표 |
| `screen_width_px`, `screen_height_px` | Dot Test 캔버스 크기 |
| `collection_split` | `training` 또는 `evaluation` |
| `protocol` | 원본 수집 프로토콜 |
| `source_frame` | 원본 MP4에서 추출한 0-based 프레임 번호 |
| `source_timestamp` | 해당 원본 프레임의 Unix ns 수신 시각 |
| `corrected_timestamp` | 카메라별 레이턴시를 뺀 Unix ns 시각 |
| `reference_timestamp` | 동기화된 두 corrected timestamp의 대표 시각 |
| `target_timestamp` | 해당 프레임에 연결된 화면 표적의 Unix ns 시각 |

정상 자세 하나에서 `training.csv`는 63쌍×2 view=126행,
`evaluation.csv`는 18쌍×2 view=36행입니다.

## `feature_maps/webeyetrack/*.csv` 열 설명

`inputs.csv`는 정면 샘플 전체와 실패 행을 보존합니다. `training.csv`와
`evaluation.csv`는 각각 해당 split에서 `valid=1`인 행만 포함합니다.

| 열 | 의미 |
|---|---|
| `schema_version` | WebEyeTrack 입력 계약 버전 |
| `sample_id` | 정면 카메라 샘플 고유 ID |
| `participant`, `head_pose` | 참가자와 촬영 자세 |
| `pair_id` | 멀티뷰 프레임 쌍 결합 키 |
| `collection_split` | `training` 또는 `evaluation` |
| `protocol` | 원본 수집 프로토콜 |
| `source_image_path` | 렌즈 보정된 정면 원본 PNG 경로 |
| `eye_patch_path` | 정규화한 `512x128` 양쪽 눈 패치 경로 |
| `target_x_centered`, `target_y_centered` | 화면 중심 기준 `[-0.5,0.5]` 정답 |
| `head_vector_x`, `head_vector_y`, `head_vector_z` | 카메라 좌표계의 머리 방향 단위 벡터 |
| `face_origin_x_cm`, `face_origin_y_cm`, `face_origin_z_cm` | 카메라 좌표계의 얼굴 기준점 위치(cm); z는 추정 얼굴 거리 |
| `pose_reprojection_error_px` | solvePnP head pose의 평균 재투영 오차 |
| `quality_score` | 재투영 오차를 0~1로 변환한 품질 점수 |
| `left_ear`, `right_ear` | 양쪽 눈 Eye Aspect Ratio |
| `valid` | 모델 입력 계약과 품질 조건을 통과하면 1 |
| `invalid_reason` | 이미지 읽기·얼굴/홍채·눈 감김·head pose 실패 사유 |

`labels.csv`의 `pair_quality_score`는 수집 후보 이미지 선택용이고,
WebEyeTrack의 `quality_score`는 head pose 재투영 품질이므로 같은 confidence로
해석하거나 합치지 않습니다.

## 모델 학습 시 파일 선택

| 모델 | 학습 파일 | 평가 파일 |
|---|---|---|
| 정면 WebEyeTrack | `feature_maps/webeyetrack/training.csv` | `feature_maps/webeyetrack/evaluation.csv` |
| 정면 MediaPipe 8차원 | `feature_maps/training_features.csv` | `feature_maps/evaluation_features.csv` |
| 정면 이미지 branch | `feature_maps/training.csv`에서 `view=webcam` | `feature_maps/evaluation.csv`에서 `view=webcam` |
| 측면 이미지 branch | `feature_maps/training.csv`에서 `view=phonecam` | `feature_maps/evaluation.csv`에서 `view=phonecam` |
| 멀티뷰 fusion | 두 view를 `pair_id`로 결합 | 같은 방식으로 evaluation 결합 |

`images/`는 레이턴시 보정 전 수집 선택본이므로 최종 이미지 모델에는
`feature_maps/{web,phone}/frames/`와 위 manifest를 사용합니다. 같은 sample ID라도
동기화 과정에서 인접한 유효 프레임을 다시 선택할 수 있어 두 폴더의 원본 frame과
timestamp가 항상 같지는 않습니다.

## 학습·평가 주의사항

- 동일 참가자의 세 head pose를 서로 다른 train/test에 나누지 않습니다. 참가자 ID를
  group key로 사용하여 사람 단위로 분리합니다.
- 기존 `evaluation`은 같은 참가자의 학습점 사이에 놓인 미사용 위치를 평가합니다.
  새 사람 일반화 평가는 별도의 참가자 hold-out으로 수행합니다.
- 일반 모델의 정답은 `x_norm`, `y_norm`, WebEyeTrack 정답은 centered 좌표로
  통일합니다. 픽셀 좌표는 렌더링 반올림 규칙에 따라 1px 정도 차이날 수 있습니다.
- `source_timestamp`는 보정 전, `corrected_timestamp`는 레이턴시 보정 후입니다.
  일반 timestamp는 Unix ns이고 `*_ms` 열만 millisecond입니다.
- `feature_maps` PNG에는 렌즈 왜곡 보정이 이미 적용되어 있으므로 다시
  undistort하지 않습니다.
- 평가 실패 행을 조용히 제거하지 말고 유효 샘플 오차와 얼굴·홍채 미검출률을 함께
  보고합니다. 전체 실패율은 `webeyetrack/inputs.csv`와 각 `summary.json`으로
  확인합니다.
- 일부 JSON은 수집 컴퓨터의 절대 경로를 기록합니다. 다른 컴퓨터에서는 데이터셋
  root와 CSV 상대 경로를 기준으로 파일을 찾습니다.
- 실제 이름과 얼굴 영상은 직접 식별 정보입니다. 외부 전달본은 연구용 ID로
  익명화하고 원본 데이터는 Git 또는 MLflow artifact에 업로드하지 않습니다.

## 참가자 한 명 처리 순서

카메라 위치·줌·해상도·연결 방식을 중간에 바꾸지 않습니다.

```bash
make check-cameras
make collect PARTICIPANT=안은제 HEAD_POSE=neutral
```

`make collect`가 해당 참가자·head pose의 레이턴시를 먼저 측정하고, 성공한 경우에만
Dot Test를 이어서 실행합니다. 촬영이 정상 완료되면 프레임 동기화와 정면 MediaPipe
특징 추출까지 자동으로 실행하여 `feature_maps/`를 채웁니다.

같은 순서를 `HEAD_POSE=head_up`, `HEAD_POSE=head_down`으로 반복합니다. 세 자세에서
CSV의 `participant`는 모두 `안은제`이고 `head_pose`만 달라야 합니다. 모델의
train/validation/test 분할은 자세 폴더가 아니라 참가자 이름 단위로 수행합니다.

자동 후처리만 실패했거나 기존 촬영본을 처리할 때 사용하는 `prepare-participant`는
다음 두 작업을 순서대로 수행합니다.

1. `sync-participant`: `metadata/frame_log.csv`에 latency를 적용하여
   `feature_maps/synchronized.csv` 생성
2. `video-features`: 같은 순간의 web/phone 프레임을 뽑고 정면 MediaPipe 특징 생성

정상 완료 후에는 해당 이름 폴더 하나를 압축하거나 공유 저장소로 전달하면 됩니다.

실명을 폴더명으로 사용하면 얼굴 영상과 바로 연결됩니다. 연구 동의·접근 권한·보관
정책을 적용하고, 외부 전달본에는 실명 대신 연구용 별칭을 사용하는 편이 안전합니다.

## 기존 참가자 폴더 변환

먼저 파일을 바꾸지 않는 dry run으로 대상을 확인합니다.

```bash
make migrate-layout
```

실제 변환은 명시적으로 승인할 때만 실행합니다.

```bash
make migrate-layout APPLY=1
```

한 명만 변환할 수도 있습니다.

```bash
make migrate-layout PARTICIPANT=p03 APPLY=1
```

변환기는 기존 영상과 이미지를 삭제하지 않고 새 위치로 이동합니다. 기존 전체
프레임 라벨은 `metadata/frame_log.csv`, 기존 점당 이미지 표는 단일
`labels/labels.csv`, 기존 events는 `metadata/legacy_events/`에 원본 그대로 보존하며
합친 표를 `metadata/protocol.csv`에 만듭니다. `participant.json` 원본도
`metadata/participant_legacy.json`에 백업합니다.
