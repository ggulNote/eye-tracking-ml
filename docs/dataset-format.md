# Static-image dataset contract

## 1. MPIIFaceGaze의 sample 단위

`p01.txt`는 영상 timestamp 목록이 아니라 **정지 이미지 index + annotation**입니다. 확인한 p01은 header 없는 whitespace 구분 28열, 2,904행이며, 2,904개의 상대 경로가 실제 JPG 2,904장과 1:1로 대응하고 누락 경로가 없습니다.

경로는 다음처럼 해석합니다.

```text
dataset_root/
└── p01/
    ├── p01.txt
    ├── Calibration/
    └── day01/
        └── 0005.jpg

annotation column 1 = day01/0005.jpg
resolved image path = dataset_root/p01/day01/0005.jpg
```

따라서 DataLoader의 한 sample은 `p01.txt`의 한 행과 그 행이 가리키는 이미지 한 장입니다. 앞뒤 행을 영상 frame처럼 묶지 않습니다.

## 2. 28개 열

아래 번호는 사람이 읽는 1-based 번호입니다. `configs/config.yaml`의 index는 0-based입니다.

| 1-based 열 | 개수 | 의미 |
|---|---:|---|
| 1 | 1 | subject 폴더 기준 상대 이미지 경로 |
| 2–3 | 2 | 노트북 **화면**의 gaze point `(x_px, y_px)` |
| 4–15 | 12 | 6개 facial landmark의 `(x,y)`: 네 eye corner와 두 mouth corner, 입력 이미지 pixel 좌표 |
| 16–18 | 3 | camera coordinate의 3D head rotation vector |
| 19–21 | 3 | camera coordinate의 3D head translation |
| 22–24 | 3 | `fc`: 3D face center |
| 25–27 | 3 | `gt`: camera coordinate의 3D gaze target |
| 28 | 1 | 표준 평가 subset에서 사용할 눈 `left` 또는 `right` |

마지막 `left/right`는 webcam/phonecam view가 아니며, 두 파일을 연결하는 pair ID도 아닙니다. head rotation vector를 별도 변환 없이 yaw/pitch/roll 세 숫자로 해석해서도 안 됩니다.

표준 canonical sample은 다음과 같은 key를 갖도록 설계합니다.

```text
sample_id:              str
subject_id:             str
session_id:             str       # dayXX
image_path:             Path
image:                  uint8[H,W,3] RGB
gaze_screen_xy_px:      float32[2]
facial_landmarks_xy:    float32[6,2]
head_rotation_3d:       float32[3]
head_translation_3d:    float32[3]
face_center_3d:         float32[3]
gaze_target_3d:         float32[3]
evaluation_eye:         str
view:                   "front"
pair_id:                null
```

## 3. 화면 label과 image pixel을 구분하기

열 2–3은 사진 안의 얼굴/눈 위치가 아니라 participant의 노트북 **화면 위 응시점**입니다. 다음 값을 섞으면 안 됩니다.

- image width/height: 얼굴 사진을 crop/resize할 때 사용
- screen width/height pixel: screen gaze target normalize와 pixel metric에 사용
- screen width/height millimeter: physical cm metric에 사용

p01 calibration을 직접 확인한 값은 다음과 같습니다.

```text
screen: 1440 × 900 px
physical: 286.4708 × 179.0442 mm
```

centered-normalized 2D label은 다음처럼 만듭니다.

```text
x_norm = x_screen_px / screen_width_px  - 0.5
y_norm = y_screen_px / screen_height_px - 0.5
```

역변환도 같은 분모 convention을 사용합니다. `W` 대신 `W-1`을 사용하고 싶다면 target 생성, model adapter, metric, visualization을 모두 함께 바꿔야 합니다.

3D gaze direction을 사용할 때는 다음처럼 만듭니다.

```text
gaze_vector = gaze_target_3d - face_center_3d
gaze_unit = gaze_vector / norm(gaze_vector)
```

`gaze_target_3d` 자체나 head rotation을 gaze direction으로 사용하지 않습니다.

## 4. 이미지 전처리 주의사항

MPIIFaceGaze 제공 이미지는 privacy 처리가 되어 얼굴 밖이 이미 검은색입니다. raw webcam 배경이 아니므로 같은 background mask를 다시 적용할 필요가 없습니다.

p01 실측에서는 대부분 1280×720이지만 7장은 320×240입니다. 따라서 다음 규칙을 지킵니다.

- parser에서 원본 resolution을 하나로 hard-code하지 않습니다.
- landmark는 각 이미지의 원본 좌표이므로 crop/resize와 동일한 affine transform을 적용합니다.
- 저해상도 또는 먼 얼굴은 자동 삭제하지 않고 quality flag와 metric으로 먼저 기록합니다.
- 제거가 필요하면 config로 켠 filter와 제외 manifest를 MLflow에 저장합니다.

## 5. 왜 MPIIFaceGaze에서는 pairing을 끄는가

MPIIFaceGaze에는 동기화된 정면 webcam과 측면 phonecam 이미지 쌍이 없습니다. `dayXX`와 파일 번호에는 정확한 timestamp가 없고 annotated frame이 연속적이지 않으므로 인접 파일을 pair 또는 짧은 sequence로 취급하지 않습니다.

한 이미지에서 full face와 두 eye crop을 만들 수는 있지만, 이는 서로 다른 camera view pairing이 아닙니다. 모두 같은 row와 같은 label에서 파생된 multi-crop input입니다.

따라서 기준값은 다음과 같습니다.

```yaml
data:
  mode: image
  views:
    available: [front]
  pairing:
    enabled: false
model:
  front:
    enabled: true
  side:
    enabled: false
fusion:
  enabled: false
```

이 상태로 front model을 먼저 학습·평가할 수는 있지만, 이 dataset만으로 webcam+phonecam late fusion을 학습하거나 공정하게 평가할 수는 없습니다.

## 6. 향후 webcam/phonecam 정지 이미지 형식

사용자가 예상한 디렉터리는 adapter로 지원할 수 있습니다.

```text
dataset_root/
└── <subject_id>/
    ├── webcam/
    │   └── <image>.jpg
    └── phonecam/
        └── <image>.jpg
```

그러나 폴더와 파일명만으로 동시 촬영 여부를 안전하게 알 수 없습니다. 수집 시 다음 manifest를 함께 만드는 것을 권장합니다.

```yaml
data:
  views:
    source_to_branch: {webcam: front, phonecam: side}
  pairing:
    enabled: true
    pair_id_key: capture_group
```

```csv
sample_id,subject_id,view,image_path,capture_group,target_x_px,target_y_px,screen_width_px,screen_height_px
s001_f,p001,webcam,p001/webcam/000123.jpg,g000123,720,450,1440,900
s001_s,p001,phonecam,p001/phonecam/004821.jpg,g000123,720,450,1440,900
```

reader는 `view`의 source 값을 canonical `front`/`side`로 변환하고 configured `capture_group` 값을 canonical `pair_id`로 옮깁니다. pairing이 켜졌을 때만 configured pair header가 필수입니다. join 후에도 `subject_id`와 target이 같은지 검증하며 파일명 숫자, 정렬 순서, 촬영 시각의 근사치만으로 pair를 추측하지 않습니다.

generic CSV는 다음 optional column도 보존합니다. 배열은 JSON 문법으로 CSV quoting하여 기록합니다.

- `session_id`, `evaluation_eye`
- `facial_landmarks_xy`: `[[x,y], ...]`; generic detector 계약은 4점 이상, MPIIFaceGaze parser는 정확히 6점
- `head_rotation_3d`, `head_translation_3d`, `face_center_3d`, `gaze_target_3d`: 각각 `[x,y,z]`

pair 없는 이미지 정책은 다음 중 하나를 config로 선택할 수 있습니다.

- `branch_only`: 해당 branch의 단독 loss에는 사용하고 fusion에는 제외 — 기본 권장
- `drop`: 모든 학습과 평가에서 제외
- `error`: manifest가 완전해야 하는 dataset에서 즉시 실패

## 7. Split 계약

기본 split key는 `subject_id`입니다. 한 subject의 모든 day와, 한 row에서 파생된 full-face/eye crop, 같은 `pair_id`의 두 view는 반드시 같은 split에 있어야 합니다.

- unseen-person 일반화: subject group split 또는 leave-one-person-out
- 같은 사람의 새로운 session 일반화: `dayXX` group split을 별도 profile로 사용
- 금지: 전체 image를 무작위로 섞은 뒤 row 단위 split

70/15/15는 subject 수가 적을 때 정확한 sample 비율과 다를 수 있으므로 실제 subject 목록과 image 수를 split manifest에 기록합니다. positive ratio가 3개인데 subject group이 2개뿐인 경우처럼 모든 split을 채울 수 없으면 `SplitError`로 실패합니다.

### 원본 screen 범위 밖 label

전체 원본을 검증하면 일부 annotation의 screen target이 해당 subject의 `screenSize.mat` 범위를 벗어납니다. 파이프라인은 이 값을 clamp하거나 다른 값으로 추정하지 않습니다.

```text
target_bounds_policy: keep_flagged
manifest column: target_in_screen_bounds = false
```

이 sample은 원본 보존과 품질 분석을 위해 manifest/split에 남습니다. 화면 내부 PoG metric을 계산할 때는 flag로 포함·제외 정책을 명시하고, summary의 전체 및 subject별 out-of-bounds count를 함께 기록합니다. Generic CSV는 잘못된 label을 조기에 발견하도록 기본 정책이 `error`입니다.

## 8. Model I/O 요약

### Front-only 기준

```text
input:
  front_image: float32[B,3,224,224], RGB, [0,1]

output:
  gaze_xy: float32[B,2], centered-normalized screen (x,y)
  front_embedding: float32[B,D_front]  # fusion을 쓸 때만 필수
```

### Dual-view fusion 기준

```text
input:
  front_image: float32[B,3,Hf,Wf]
  side_image:  float32[B,3,Hs,Ws]
  pair_mask:   bool[B]

branch output:
  front.gaze_xy, front.embedding
  side.gaze_xy,  side.embedding

fusion output:
  gaze_xy: float32[B,2]
```

모든 shape, dtype, color order, normalization, 좌표 단위는 model adapter가 config contract와 일치하도록 검증해야 합니다.
