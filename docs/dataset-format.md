# Dual-view DB 형식

파이프라인은 정면 `webcam` frame과 측면 `phonecam` frame을 canonical CSV manifest로 읽습니다. 현재 측정 DB에서는 `head_down`과 `neutral` session만 사용하며, 기존에 저장된 eye ROI가 아니라 보정된 원본 frame에서 ROI를 다시 만듭니다.

## 1. 측정 DB 원본

```text
Participants/
└── <subject_name>/
    └── <session>/
        └── feature_maps/
            ├── web/frames/
            ├── phone/frames/
            ├── webeyetrack/inputs.csv
            ├── training.csv
            └── evaluation.csv
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

`valid`는 이미 생성된 upstream 눈 상태 판정입니다. 정확히 `1`이면 눈 뜸(open), `0`이면 눈 감음(close)으로 해석합니다. 이 저장소는 로컬에서 눈 감김 지표를 다시 계산하지 않고, `valid=0`인 Front와 paired Side를 Side 검출·검수 및 학습에서 함께 제외합니다. 다른 값은 데이터 오류입니다. `head_vector`와 `face_origin`은 `valid=1`이고 값이 유한할 때만 Front 보조 입력으로 사용합니다.

## 4. 외부 Side 눈 ROI

현재 production은 다음처럼 Front/Side ROI와 `inputs.csv`를 한 session 폴더에서 읽습니다.

```text
process_data/<subject>/<head_down|neutral>/
├── eye_roi/<pair_id>.png
├── side_eye_roi/<원래-phone-frame-파일명>.png
└── inputs.csv
```

Front는 `inputs.csv.eye_patch_path`, Side는 `pair_id`가 포함된 원본 phone frame basename으로
연결합니다. Front `128×512`는 resize 없이 그대로 사용합니다. Side는 `128×128`보다 큰 축은 중앙
crop, 작은 축은 검정 padding한 뒤 `128×256` 검정 canvas 중앙에 배치합니다. `inputs.csv`가 없는
session, `valid=0` pair, 완전하지 않은 pair와 품질 제외 session은 manifest에 들어가지 않습니다.

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

## 7. Legacy Side bbox 자동 생성·검수

아래 절차는 외부 `side_eye_roi` 이미지가 없을 때만 사용하는 이전 방식입니다. 현재 production
학습 경로에서는 실행하지 않습니다.

```bash
make measured-side-annotations \
  MEASURED_DATA_ROOT="/absolute/path/to/Participants"
```

MediaPipe Face Landmarker가 보이는 눈을 선택하고 눈꺼풀·iris landmark로 bbox를 생성합니다. 측면
얼굴에서 MediaPipe가 실패하면 OpenCV Zoo YuNet의 얼굴·눈 후보를 사용하고, 같은 촬영 session의
신뢰 가능한 앞뒤 bbox가 있으면 시간축 보간으로 보완합니다. 눈 크기, 눈 뜬 높이, 좌우 눈 가시성
차이, landmark 신뢰도, bbox 경계, 선명도와 종합 자동검출 신뢰도를 검사하며 자동 승인되지 않은
row는 `side_annotations_review_queue.csv`와 `review_queue_*.jpg`로 보냅니다. Haar fallback은 좌표
후보만 만들고 기본적으로 자동 승인하지 않습니다.

contact sheet의 초록색 `ACCEPT` 박스만 자동 승인된 학습 후보입니다. 빨간색
`REVIEW-NOT-USED` 박스는 귀나 머리카락에 잘못 놓일 수 있는 저신뢰 후보이며, 검수 전에는 manifest와
학습에 들어가지 않습니다.

검수 CSV에서 bbox와 `visible_eye`를 확인하고 `eye_annotation_valid=true`로 바꾼 뒤 병합합니다.

```bash
make measured-side-annotations \
  MEASURED_DATA_ROOT="/absolute/path/to/Participants" \
  SIDE_REVIEW_OVERRIDES="/absolute/path/to/edited_review_queue.csv"
```

빈 bbox나 더미 좌표를 강제로 유효 처리하지 않으며, 최종적으로 모든 row가 승인되어야
`measured-manifest`가 통과합니다.

## 8. 측정 DB 준비와 확인

```bash
make measured-manifest \
  MEASURED_DATA_ROOT="/absolute/path/to/Participants" \
  SIDE_ANNOTATIONS="/absolute/path/to/side_annotations.csv"

make measured-prepare \
  MEASURED_DATA_ROOT="/absolute/path/to/Participants" \
  SIDE_ANNOTATIONS="/absolute/path/to/side_annotations.csv"

make measured-preview \
  MEASURED_DATA_ROOT="/absolute/path/to/Participants" \
  SIDE_ANNOTATIONS="/absolute/path/to/side_annotations.csv"
```

`measured-manifest`는 `head_down`·`neutral`, Front pose, Side bbox를 모두 검증합니다. `measured-preview`는 실제 Front/Side 전처리 결과를 이미지로 저장합니다.

로컬 manifest에는 피험자 이름과 경로가 남으므로 측정 profile은 manifest와 prediction을 MLflow artifact로 올리지 않습니다. 전처리 `.pkl` cache도 신뢰하는 로컬 경로에서만 사용해야 합니다.
