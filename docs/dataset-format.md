# Dual-view DB 형식

파이프라인은 정면 `webcam` frame과 측면 `phonecam` frame을 canonical CSV manifest로 읽습니다. 현재 측정 DB에서는 `head_down`과 `neutral` session만 사용하며, 기존에 저장된 eye ROI가 아니라 보정된 원본 frame에서 ROI를 다시 만듭니다.

## 1. 측정 DB 원본

```text
project_data/data/
└── <subject_name>/
    └── <session>/
        └── feature_maps/
            ├── web/
            │   ├── frames/
            │   └── webeyetrack/inputs.csv
            └── phone/
                └── frames/
```

- `<subject_name>`은 숫자 ID가 아니어도 됩니다. builder가 Unicode를 NFC로 정규화합니다.
- `feature_maps/web|phone/frames`의 렌즈 보정 frame만 image source로 사용합니다.
- `inputs.csv.source_image_path`가 과거 컴퓨터의 절대경로여도 basename으로 현재 frame과 연결합니다.
- 기존 `eye_patch_path`와 precomputed eye ROI는 모델 입력으로 사용하지 않습니다.

## 2. Canonical manifest

일반 DB를 직접 연결할 때의 최소 열은 다음과 같습니다.

```csv
sample_id,subject_id,session_id,view,image_path,pair_id,target_x_px,target_y_px,screen_width_px,screen_height_px
kim_neutral_001_front,kim,neutral,webcam,kim/neutral/feature_maps/web/frames/001.png,kim_neutral_001,960,540,1920,1080
kim_neutral_001_side,kim,neutral,phonecam,kim/neutral/feature_maps/phone/frames/001.png,kim_neutral_001,960,540,1920,1080
```

| 열 | 의미 |
|---|---|
| `sample_id` | 이미지 한 장의 고유 ID |
| `subject_id` | 피험자 ID이자 split 그룹 |
| `session_id` | 촬영 session. 현재 허용값은 `head_down`, `neutral` |
| `view` | `webcam`/`front` 또는 `phonecam`/`side` |
| `image_path` | `GAZE_DATA_ROOT` 기준 상대경로 |
| `pair_id` | 같은 순간의 Front와 Side가 공유하는 ID |
| `target_x_px`, `target_y_px` | 사용자가 바라본 화면 pixel 좌표 |
| `screen_width_px`, `screen_height_px` | label 정규화에 사용하는 화면 해상도 |

실제 화면 크기가 있으면 `screen_width_mm`, `screen_height_mm`를 추가해 cm metric을 계산할 수 있습니다.

## 3. `inputs.csv`의 upstream 정보

측정 DB builder는 Front `inputs.csv`에서 다음 값을 읽습니다.

| 열 | 의미 |
|---|---|
| `head_vector` | MediaPipe 기반 3D 머리 방향 `[x,y,z]` |
| `face_origin` | 카메라 좌표계의 3D 얼굴 위치. manifest에는 `face_origin_3d`로 기록 |
| `valid` | upstream producer가 승인한 frame 여부. `1`이면 사용, `0`이면 pair 전체 제외 |

`valid`는 이미 생성된 upstream 품질 판정입니다. 이 저장소는 로컬에서 눈 감김 지표를 다시 계산하지 않고, `valid=0`인 Front와 paired Side를 함께 제외합니다. `head_vector`와 `face_origin`은 `valid=1`이고 값이 유한할 때만 Front 보조 입력으로 사용합니다.

## 4. Side 눈 bbox annotation

Side ROI는 별도 CSV에서 `sample_id`로 연결합니다. 최소 열은 다음 네 개입니다.

```csv
sample_id,visible_eye,visible_eye_bbox_xyxy,eye_annotation_valid
kim_neutral_001_side,left,"[820,410,1170,590]",true
```

| 열 | 의미 |
|---|---|
| `sample_id` | Side manifest row와 같은 ID |
| `visible_eye` | 보이는 눈: `left` 또는 `right` |
| `visible_eye_bbox_xyxy` | 원본 phone frame 기준 `[x1,y1,x2,y2]` |
| `eye_annotation_valid` | 실제 frame에서 검수된 bbox인지 여부 |

현재 production 설정은 bbox를 frame 경계에서 자른 뒤 `128×256`으로 직접 resize하는 `stretch` 방식입니다. 사람마다 bbox 크기가 달라도 최종 tensor는 `[3,128,256]`이며, 검은 padding을 넣지 않고 종횡비도 유지하지 않습니다.

선택 feature를 사용할 때만 다음 열을 추가합니다.

| 열 | model feature |
|---|---|
| `profile_head_origin_xy`, `profile_head_forward_xy` | `side_head_pose_2d` |
| `visible_eye_keypoints_xy` 6점 | `side_eye_angles` |
| `iris_center_xy` | `side_iris_pose_2d` |

고정 bbox나 더미 좌표를 여러 사람에게 복제한 row에는 `eye_annotation_valid=true`를 사용하면 안 됩니다. 현재 파이프라인은 annotation이 없는 phone frame에서 눈을 자동 검출하지 않습니다.

## 5. Pair와 subject-wise split

- 같은 촬영 순간의 Front와 Side만 동일한 `pair_id`를 사용합니다.
- pair의 `subject_id`와 target은 같아야 합니다.
- 파일명 순서나 근사 timestamp로 pair를 추측하지 않습니다.
- `subject_id` 단위로 seed를 고정해 무작위 분할합니다.
- 기본 비율은 train 70%, validation 15%, test 15%입니다.

한 피험자는 한 split에만 들어갑니다. 세 split에 최소 한 명씩 배정할 수 없으면 준비 단계에서 중단합니다.

## 6. Target 좌표

`target_x_px`, `target_y_px`는 이미지 속 눈 위치가 아니라 화면에서 본 위치입니다.

```text
x_norm = target_x_px / screen_width_px  - 0.5
y_norm = target_y_px / screen_height_px - 0.5
```

화면 중심은 `(0,0)`이고 target 범위는 보통 `[-0.5,0.5]`입니다. 카메라 이미지 크기를 분모로 사용하지 않습니다.

## 7. 측정 DB 준비와 확인

```bash
make measured-manifest \
  MEASURED_DATA_ROOT="/absolute/path/to/project_data/data" \
  SIDE_ANNOTATIONS="/absolute/path/to/side_annotations.csv"

make measured-prepare \
  MEASURED_DATA_ROOT="/absolute/path/to/project_data/data" \
  SIDE_ANNOTATIONS="/absolute/path/to/side_annotations.csv"

make measured-preview \
  MEASURED_DATA_ROOT="/absolute/path/to/project_data/data" \
  SIDE_ANNOTATIONS="/absolute/path/to/side_annotations.csv"
```

`measured-manifest`는 `head_down`·`neutral`, Front pose, Side bbox를 모두 검증합니다. `measured-preview`는 실제 Front/Side 전처리 결과를 이미지로 저장합니다.

로컬 manifest에는 피험자 이름과 경로가 남으므로 측정 profile은 manifest와 prediction을 MLflow artifact로 올리지 않습니다. 전처리 `.pkl` cache도 신뢰하는 로컬 경로에서만 사용해야 합니다.
