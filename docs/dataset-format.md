# 새 Dual-view DB 형식

기본 파이프라인은 새로 수집한 `webcam` 정면 이미지와 `phonecam` 90도 측면 이미지를 CSV manifest로 읽습니다. 이전 실험 데이터는 기본 입력으로 사용하지 않습니다.

## 1. 폴더 구조

권장 구조는 다음과 같습니다. 실제 연결은 폴더명이나 파일 순서가 아니라 manifest의 `view`와 `pair_id`로 결정합니다.

```text
dual_view/
└── <subject_id>/
    ├── webcam/
    │   └── *.jpg
    └── phonecam/
        └── *.jpg
```

`image_path`는 `GAZE_DATA_ROOT` 아래의 상대경로여야 합니다. 파이프라인은 원본 이미지를 복사하지 않고 경로와 metadata만 manifest에 기록합니다.

## 2. 필수 CSV 열

```csv
sample_id,subject_id,view,image_path,pair_id,target_x_px,target_y_px,screen_width_px,screen_height_px
p001_001_front,p001,webcam,p001/webcam/001.jpg,p001_001,960,540,1920,1080
p001_001_side,p001,phonecam,p001/phonecam/001.jpg,p001_001,960,540,1920,1080
```

| 열 | 의미 |
|---|---|
| `sample_id` | 이미지 한 장의 고유 ID |
| `subject_id` | 사람 ID이며 train/validation/test 분할 기준 |
| `view` | `webcam`/`front` 또는 `phonecam`/`side` |
| `image_path` | DB root 기준 이미지 상대경로 |
| `pair_id` | 같은 시점에 촬영한 정면·측면 이미지의 공통 ID |
| `target_x_px`, `target_y_px` | 화면에서 사용자가 본 위치의 pixel 좌표 |
| `screen_width_px`, `screen_height_px` | label을 정규화할 화면 해상도 |

기본 config는 pairing을 켜므로 `pair_id`가 필요합니다. 짝이 없는 이미지는 branch 단독 데이터에는 남지만 fusion 대상에서는 제외됩니다.

## 3. 선택 열

공통으로 다음 열을 추가할 수 있습니다.

| 열 | 형식 | 용도 |
|---|---|---|
| `session_id` | 문자열 | 촬영 회차 구분 |
| `screen_width_mm`, `screen_height_mm` | 양수 | cm 단위 metric 계산 |
| `facial_landmarks_xy` | `[[x,y], ...]` | annotation 기반 얼굴 전처리 |
| `head_rotation_3d` | `[x,y,z]` | 제공된 3D head rotation |
| `head_translation_3d` | `[x,y,z]` | 제공된 3D head translation |
| `face_center_3d` | `[x,y,z]` | 3D 얼굴 중심 |
| `gaze_target_3d` | `[x,y,z]` | 3D gaze target |

90도 Side 전처리를 annotation 기반으로 실행하려면 다음 열을 함께 기록합니다.

| 열 | 형식 | 용도 |
|---|---|---|
| `visible_eye` | `left` 또는 `right` | 보이는 한쪽 눈 |
| `visible_eye_bbox_xyxy` | `[x1,y1,x2,y2]` | 눈 ROI |
| `visible_eye_keypoints_xy` | `[[x,y], ...]` 6점 | 눈꺼풀 방향각과 EAR |
| `iris_center_xy` | `[x,y]` | iris 기반 세로 eye pose |
| `profile_head_origin_xy` | `[x,y]` | 측면 head vector 시작점 |
| `profile_head_forward_xy` | `[x,y]` | 측면 head vector 끝점 |
| `eye_annotation_valid` | `true` 또는 `false` | annotation 유효성 |

배열은 JSON 문법으로 쓰고 CSV에서는 해당 cell을 따옴표로 감쌉니다.

## 4. Pair와 split 규칙

- 같은 촬영 순간의 Front/Side만 동일한 `pair_id`를 사용합니다.
- 두 row의 `subject_id`와 화면 target이 같아야 합니다.
- 파일명, 정렬 순서, 근사 촬영시간으로 pair를 추측하지 않습니다.
- split은 `subject_id` 단위이므로 같은 사람의 모든 이미지와 pair는 한 split에만 속합니다.
- 기본 비율은 train 70%, validation 15%, test 15%입니다.

사람 수가 적으면 실제 비율은 정확히 70/15/15가 아닐 수 있습니다. 세 split에 최소 한 사람씩 배정할 수 없으면 파이프라인은 오류로 중단합니다.

## 5. 화면 좌표

`target_x_px`, `target_y_px`는 이미지 속 눈 위치가 아니라 사용자가 바라본 **화면 위치**입니다.

```text
x_norm = target_x_px / screen_width_px  - 0.5
y_norm = target_y_px / screen_height_px - 0.5
```

따라서 모델 target은 화면 중심이 `(0, 0)`인 약 `[-0.5, 0.5]` 범위입니다. 카메라 이미지 크기를 정규화 분모로 사용하면 안 됩니다.

## 6. 준비 명령

```bash
DUAL_VIEW_MANIFEST="/absolute/path/to/dual_view_manifest.csv" \
make prepare GAZE_DATA_ROOT="/absolute/path/to/dual_view"
```

실행하면 원본 이미지 대신 검증된 canonical manifest, split manifest, config와 hash가 `outputs/`에 저장됩니다. `mlflow.db`에는 실행 metadata만 기록되며 이미지 학습 DB로 사용되지 않습니다.

현재 source tree에 과거 형식 parser가 호환용으로 남아 있지만, 기본 `configs/config.yaml`은 `generic_csv`만 선택합니다.
