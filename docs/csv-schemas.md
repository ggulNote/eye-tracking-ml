# CSV 스키마 전체 목록

이 문서는 프로젝트가 읽거나 생성하는 모든 CSV 열(column)을 정의합니다. 여기서 `timestamp`는 별도 표기가 없으면 Unix nanosecond 정수입니다. 기존 참가자의 원본 CSV는 수정하지 않으며, `csv_schema_version=2`부터 아래 규격을 사용합니다.

## 1. 원본 수집

### `labels/labels.csv`

카메라 프레임 한 쌍마다 한 행을 기록하는 원본 라벨입니다.

| 열 | 의미 |
|---|---|
| `participant` | `p00` 형식 참가자 ID |
| `protocol` | 현재 프로토콜 ID |
| `split` | `train`, `evaluation`, `reference` |
| `pair` | 수집 전체에서 증가하는 카메라 프레임 쌍 번호 |
| `display_timestamp` | 해당 표적 상태를 화면에 표시한 기준 시각 |
| `webcam_frame` | 웹캠 MP4 프레임 번호 |
| `webcam_timestamp` | 웹캠 프레임 수신 시각 |
| `phonecam_frame` | 폰캠 MP4 프레임 번호 |
| `phonecam_timestamp` | 폰캠 프레임 수신 시각 |
| `x_px`, `y_px` | 표적 화면 픽셀 좌표 |
| `x_norm`, `y_norm` | 좌상단 원점 `[0,1]` 표적 좌표 |
| `x_centered`, `y_centered` | 화면 중심 원점 `[-0.5,0.5]` 표적 좌표 |
| `segment` | 프로토콜 내부 표시 구간 번호; 표적이 없으면 빈 값 |
| `target` | 중복 표시와 무관한 표적 위치 ID |
| `direction` | `static`, `top_to_bottom`, `bottom_to_top` 등 |
| `confirmation_timestamp` | 참가자가 현재 점을 확정한 시각; 클릭 전에는 빈 값 |
| `confirmation_offset_ms` | 점 표시부터 확정까지 걸린 시간(ms) |
| `usable` | 안정 구간이면 `1`, 아니면 `0` |

`time_diff_ms`, `confirmed`, `settling`, `training` 등은 남은 열로 계산할 수 있으므로 저장하지 않습니다.

### `labels/image_samples.csv`

클릭 한 번당 품질이 가장 좋은 이미지 한 쌍을 기록합니다.

| 열 | 의미 |
|---|---|
| `sample` | `s000000` 형식의 참가자 내부 고유 이미지 쌍 ID |
| `participant` | 참가자 ID |
| `protocol` | 프로토콜 ID |
| `split` | 데이터 용도 |
| `pair` | 선택 이미지가 나온 `labels.csv`의 pair 번호 |
| `display_timestamp` | 선택 프레임의 화면 기준 시각 |
| `confirmation_timestamp` | 클릭 확정 시각 |
| `confirmation_offset_ms` | 점 표시부터 확정까지 걸린 시간(ms) |
| `candidate_count` | 대표 프레임 선정에 사용한 후보 쌍 수 |
| `pair_quality_score` | 두 카메라 품질 점수 합계 |
| `webcam_image` | 참가자 폴더 기준 웹캠 이미지 상대 경로 |
| `webcam_frame` | 원본 웹캠 MP4 프레임 번호 |
| `webcam_timestamp` | 웹캠 프레임 수신 시각 |
| `webcam_face_detected` | OpenCV 얼굴 검출 여부 |
| `webcam_eyes_detected` | 검출된 눈 수, 최대 2 |
| `webcam_mediapipe_face_detected` | 선택 후보의 MediaPipe 얼굴 검출 여부 |
| `webcam_mediapipe_iris_detected` | 선택 후보의 MediaPipe refined iris 검출 여부 |
| `webcam_sharpness` | 얼굴 ROI 또는 전체 프레임의 Laplacian 분산 |
| `webcam_brightness` | 품질 평가 ROI 평균 밝기 |
| `phonecam_image` | 참가자 폴더 기준 폰캠 이미지 상대 경로 |
| `phonecam_frame` | 원본 폰캠 MP4 프레임 번호 |
| `phonecam_timestamp` | 폰캠 프레임 수신 시각 |
| `phonecam_face_detected` | OpenCV 얼굴 검출 여부 |
| `phonecam_eyes_detected` | 검출된 눈 수, 최대 2 |
| `phonecam_mediapipe_face_detected` | 선택 후보의 MediaPipe 얼굴 검출 여부 |
| `phonecam_mediapipe_iris_detected` | 선택 후보의 MediaPipe refined iris 검출 여부 |
| `phonecam_sharpness` | 얼굴 ROI 또는 전체 프레임의 Laplacian 분산 |
| `phonecam_brightness` | 품질 평가 ROI 평균 밝기 |
| `x_px`, `y_px` | 표적 화면 픽셀 좌표 |
| `x_norm`, `y_norm` | 좌상단 원점 `[0,1]` 좌표 |
| `x_centered`, `y_centered` | 화면 중심 원점 `[-0.5,0.5]` 좌표 |
| `segment` | 프로토콜 표시 구간 번호 |
| `target` | 표적 위치 ID |
| `direction` | 표적 진행 방향 |

실제 촬영에서는 MediaPipe 얼굴+홍채 검출 성공을 가장 큰 우선순위로 두고,
OpenCV 눈 검출·선명도·노출을 보조 점수로 사용합니다. 최종 눈 감김 판정은 B
MediaPipe 전처리가 수행합니다.

### `events/<protocol>.csv`

프로토콜의 계획된 표적 순서를 한 행씩 기록합니다.

| 열 | 의미 |
|---|---|
| `protocol` | 프로토콜 ID |
| `split` | 데이터 용도 |
| `segment` | 표시 순서 번호 |
| `repeat` | 프로토콜 반복/왕복 번호 |
| `target` | 표적 위치 ID |
| `x`, `y` | `[0,1]` 표적 위치 |
| `direction` | 표적 진행 방향 |
| `confirmation_required` | 클릭 확정 필요 여부 |

모든 현재 표적은 정지 점이므로 중복이던 `end_x`, `end_y`는 제거했습니다. 실제 클릭 시간과 다른 `duration_ms`도 저장하지 않습니다.

### `webcam/timestamps.csv`, `phonecam/timestamps.csv`

| 열 | 의미 |
|---|---|
| `frame` | 해당 MP4 프레임 번호 |
| `elapsed_ms` | 공통 수집 시작 이후 monotonic 경과 시간(ms) |
| `timestamp` | 프레임 수신 Unix nanosecond |

## 2. 레이턴시 측정

### `Calibration/latency_runs/<run>/display_events.csv`

| 열 | 의미 |
|---|---|
| `event` | 검정/흰색 전환 번호 |
| `display_timestamp` | 화면 전환 기준 시각 |
| `light_state` | 흰색 `1`, 검정 `0` |

### `<camera>_brightness.csv`

| 열 | 의미 |
|---|---|
| `frame` | 레이턴시 영상 프레임 번호 |
| `timestamp` | 프레임 수신 시각 |
| `brightness` | 설정된 카메라 ROI 평균 밝기 |

### `<camera>_timestamps.csv`

레이턴시 MP4의 프레임 매핑이며 일반 촬영 `timestamps.csv`와 동일한 `frame,elapsed_ms,timestamp` 규격입니다.

### `detections.csv`

| 열 | 의미 |
|---|---|
| `camera` | `webcam` 또는 `phonecam` |
| `event` | 화면 전환 번호 |
| `light_state` | 전환 후 밝기 상태 |
| `display_timestamp` | 화면 전환 기준 시각 |
| `detected_frame` | 밝기 변화가 검출된 카메라 프레임 |
| `detected_timestamp` | 검출 프레임 수신 시각 |
| `latency_ms` | 표시부터 검출까지 지연(ms) |
| `brightness_change` | 검출에 사용한 밝기 변화량 |
| `valid` | 유효 검출 여부 |
| `invalid_reason` | 무효 사유; 유효하면 빈 값 |

## 3. 레이턴시 보정·카메라 동기화

### `synchronized/synchronized_frames.csv`

| 열 | 의미 |
|---|---|
| `participant` | 참가자 ID |
| `pair` | 동기화 결과 쌍 번호 |
| `webcam_frame`, `phonecam_frame` | 선택된 원본 영상 프레임 번호 |
| `webcam_timestamp`, `phonecam_timestamp` | 보정 전 프레임 시각 |
| `webcam_latency_ms`, `phonecam_latency_ms` | 적용한 카메라별 median 레이턴시 |
| `webcam_corrected_timestamp`, `phonecam_corrected_timestamp` | 레이턴시를 뺀 프레임 시각 |
| `corrected_time_diff_ms` | 두 보정 시각 차이(ms) |
| `reference_timestamp` | 두 보정 시각의 중앙값 |
| `target_timestamp` | 연결된 화면 표적 시각 |
| `x_norm`, `y_norm` | `[0,1]` 표적 좌표 |
| `x_centered`, `y_centered` | `[-0.5,0.5]` 표적 좌표 |
| `protocol` | 연결된 프로토콜 ID |
| `split` | 데이터 용도 |
| `segment` | 프로토콜 표시 구간 |
| `target` | 표적 위치 ID |
| `direction` | 표적 진행 방향 |
| `usable` | 안정 구간 여부 |
| `target_interpolated` | 과거 `dynamic_*` 데이터 좌표 보간 여부 |
| `valid_sync` | 카메라 시간차 조건 통과 여부 |
| `valid_target` | 유효 표적 연결 여부 |
| `valid` | 최종 유효 여부 |
| `invalid_reason` | 최종 무효 사유 |

이 파일은 계산 결과 감사용이므로 레이턴시와 보정 전·후 timestamp를 모두 유지합니다.

## 4. 영상 MediaPipe 전처리

### `manifests/p00_training.csv`, `p00_evaluation.csv`

점당 하나의 레이턴시 보정 이미지와 정답 좌표를 카메라별 한 행으로 연결합니다.

```text
sample_id,subject_id,view,image_path,pair_id,
target_x_px,target_y_px,screen_width_px,screen_height_px,
collection_split,protocol,source_frame,source_timestamp,
corrected_timestamp,reference_timestamp,target_timestamp
```

`collection_split`은 B 출력에서 `training` 또는 `evaluation`입니다. 입력 A CSV에서는 `split=train` 또는 `split=evaluation`을 사용합니다.

### 카메라별 `processed_features.csv`

### `p00_video_training.csv`, `p00_video_evaluation.csv`

두 종류의 CSV는 같은 열 규격을 사용합니다.

```text
schema_version,sample_id,participant,camera,pair_id,pair,
source_frame,source_timestamp,corrected_timestamp,
reference_timestamp,target_timestamp,protocol,collection_split,
x_norm,y_norm,sync_valid,usable,
face_detected,iris_detected,landmark_count,
left_ear,right_ear,left_eye_closed,right_eye_closed,eye_closed,
feature_valid,invalid_reason,
left_eye_center_x,left_eye_center_y,
left_iris_center_x,left_iris_center_y,
right_eye_center_x,right_eye_center_y,
right_iris_center_x,right_iris_center_y,
image_path
```

8차원 특징은 두 눈 중심과 두 홍채 중심의 정규화된 2D `(x,y)`입니다. `training` 열은 `collection_split`과 중복되므로 저장하지 않습니다.

## 5. 학습 파이프라인 CSV 인터페이스

### 입력 manifest CSV

필수 열:

```text
video_path,source,participant_id,session_id,target_x,target_y
```

지원하는 선택 열:

```text
sample_id,label_frame_index,label_timestamp_ms,rotation_degrees,
target_coordinate_space,screen_width_px,screen_height_px,
screen_width_cm,screen_height_cm,device_id,camera_intrinsics_path,
calibration_point_id,sample_weight,valid
```

`label_frame_index`와 `label_timestamp_ms`는 동시에 사용하지 않습니다.

### `test_predictions.csv`

| 열 | 의미 |
|---|---|
| `participant_id` | 참가자 ID |
| `session_id` | 입력 manifest의 세션 ID |
| `target_x`, `target_y` | 실제 `[0,1]` 좌표 |
| `predicted_x`, `predicted_y` | 모델 예측 `[0,1]` 좌표 |
