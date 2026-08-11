# Configuration reference

이 문서는 [`../configs/config.yaml`](../configs/config.yaml)의 의미를 설명합니다. YAML의 값은 실행 직전 환경 변수와 CLI override를 반영해 resolve합니다. `prepare`는 그 결과에서 secret을 제거한 `resolved_config.yaml`과 SHA-256 sidecar를 run 폴더에 atomic write합니다. MLflow에는 credential과 machine-local path를 한 번 더 제거한 JSON 사본과 local snapshot의 SHA-256을 기록합니다.

현재 실행 가능한 범위는 config 검증, data preparation, Dataset/DataLoader, front/side 전처리와
MLflow **data-preparation** 기록입니다. `model` 이후의 adapter, 학습, loss/metric, checkpoint,
late fusion과 MLflow training 기록은 향후 runner를 위한 선언이며 아직 실행되지 않습니다.

## 1. 전체 구조

| 그룹 | 현재 상태 | 역할 |
|---|---|---|
| `schema_version`, `experiment` | 구현 | config 버전, run 이름, seed, tag |
| `paths` | 부분 구현 | data/preparation 경로는 사용; 학습 산출물 경로는 예약 |
| `task` | 구현 | target shape와 화면 좌표 계약 |
| `data` | 구현 | 정지 이미지 reader, view mapping, pairing, split, DataLoader |
| `preprocessing` | 구현 | 실행 순서가 고정된 전처리 stage와 branch별 override |
| `model` | 계약만 부분 구현 | I/O와 feature key 검증은 구현; import/forward adapter는 미구현 |
| `fusion` | 검증만 구현 | pairing/branch 의존성 검사; fusion module은 미구현 |
| `training`, `optimizer`, `scheduler` | 미구현 | 향후 trainer 설정 |
| `loss`, `metrics` | 미구현 | 향후 loss/metric executor 설정 |
| `checkpoint`, `model_export` | 미구현 | 향후 resume/export 정책 |
| `mlflow` | preparation만 구현 | preparation params/artifact/environment 기록; training 기록은 미구현 |

## 2. `experiment`와 `paths`

### `experiment`

- `name`: 같은 목적의 run을 묶는 MLflow experiment 및 output 상위 폴더 이름입니다.
- `run_name`: 개별 실행 이름입니다. 기본값은 experiment 이름과 실행 시각의 조합입니다.
- `description`: 실험의 가설과 변경점을 사람이 읽을 수 있게 적습니다.
- `seed`: 현재 subject split, DataLoader shuffle과 augmentation에 전달합니다. 향후 trainer는
  model 초기화에도 같은 seed를 사용해야 합니다.
- `deterministic`: 향후 trainer가 결정적 연산을 요청하기 위한 선언입니다. 현재 data
  split/augmentation 재현성은 `seed`로 보장하며, 이 flag가 별도 학습 backend를 켜지는 않습니다.
- `tags`: 실험 설명용 metadata입니다. 현재 preparation tracker의 검색 tag는 별도
  `mlflow.tags`에서 읽으며, 이 값을 학습 로직 변경에 사용하지 않습니다.

### `paths`

- `data_root`: dataset 최상위 경로입니다. tracked YAML에 개인 절대 경로를 넣지 않고 `GAZE_DATA_ROOT`로 override합니다.
- `output_root`: 모든 run 산출물의 최상위입니다.
- `run_dir`: 현재 manifest와 resolved config를 저장하는 preparation run 경로입니다.
- `checkpoint_dir`, `model_dir`: 향후 trainer/checkpoint exporter가 사용할 예약 경로입니다.
- `metric_dir`, `prediction_dir`, `report_dir`: 향후 평가 runner가 사용할 예약 경로입니다.

## 3. `task`: `[-0.5, 0.5]`와 ReLU의 관계

### 좌표 변환

`centered_normalized_screen`은 화면의 왼쪽 위를 중심 기준 좌표로 바꿉니다.

```text
x_norm = x_screen_px / screen_width_px  - 0.5
y_norm = y_screen_px / screen_height_px - 0.5
```

중요한 점은 분모가 **카메라 이미지 크기**가 아니라 manifest의 **화면 크기**라는 것입니다. 두 값을 바꾸어 쓰면 label 자체가 틀립니다.

config는 `normalization_denominator: screen_size`를 사용합니다. 실제 pixel index가 `0 ... W-1`이므로 위 수식의 실제 범위는 정확히 `[-0.5, 0.5)`입니다. 문서와 config의 `target_range: [-0.5, 0.5]`는 모델 계약을 읽기 쉽게 나타낸 nominal bound입니다. 만약 `W-1`, `H-1`을 분모로 선택하면 양 끝이 정확히 `-0.5`, `0.5`가 되지만, 학습과 역변환에서 같은 규칙을 반드시 써야 합니다.

예시는 다음과 같습니다.

| 화면 위치 | centered-normalized 값 |
|---|---:|
| 왼쪽 위 | 약 `(-0.5, -0.5)` |
| 화면 중심 | `(0.0, 0.0)` |
| 오른쪽 아래 | 약 `(0.5, 0.5)` |

`[0, 1]`도 가능한 좌표 표현이지만 화면 중심이 `(0.5, 0.5)`가 됩니다. 중심 기준 오차와 양·음 방향을 직접 다루기 쉬워 이 설계에서는 `[-0.5, 0.5]`를 사용합니다.

### 왜 ReLU가 아닌가

이 범위는 activation의 범위가 아니라 **정답 label의 좌표 convention**입니다. 좌표 변환은
현재 data pipeline에 구현되어 있지만 output activation은 향후 model runner 계약입니다.

- ReLU는 음수를 모두 0으로 만들므로 화면 중심보다 왼쪽/위쪽을 표현하지 못합니다.
- ReLU는 위쪽 상한도 없어서 `[0, 1]` 출력 보장에도 적합하지 않습니다.
- 향후 기본 회귀 head는 `output_activation: identity`인 linear output을 사용합니다. loss가 예측을 label 범위 근처로 학습시키도록 설계합니다.
- 반드시 bound가 필요하면 `0.5 * tanh(raw)`인 `scaled_tanh`를 사용할 수 있지만 경계에서 gradient가 작아질 수 있습니다.
- `[0, 1]` label을 쓰는 설계라면 sigmoid가 수학적으로 맞지만, 이 경우에도 ReLU는 아닙니다.

평가 전에 예측을 강제로 clamp하면 큰 오류가 숨겨지므로 `clamp_for_metrics: false`입니다. 화면 위에 점을 그릴 때만 `clamp_for_visualization: true`를 사용할 수 있습니다.

### 나머지 `task` 필드

- `output_key`, `output_dim`, `output_order`, `output_dtype`: 표준 model output은 `gaze_xy: FloatTensor[B,2]`이고 순서는 `(x,y)`입니다.
- `origin`, `x_positive_direction`, `y_positive_direction`: 좌표 원점과 축 방향을 명시합니다. 여기서는 중심, 오른쪽 +x, 아래쪽 +y입니다.
- `uncertainty.enabled`: 모델이 좌표뿐 아니라 불확실성을 학습할지 정합니다. 좌표에 ReLU를 쓰는 것과 무관하며, 양수 표준편차는 softplus, log-variance는 identity로 따로 parameterize합니다.

## 4. `data`: 정지 이미지 reader와 pairing

### 공통/reader

- `mode: image`: sample 단위가 frame sequence가 아니라 정지 이미지 한 장임을 뜻합니다.
- `dataset_root`: `paths.data_root`를 재사용합니다.
- `image_extensions`: 허용할 파일 확장자입니다.
- `reader.type: generic_csv`: 새 DB manifest reader를 선택합니다.
- `manifest_path`: 이미지와 label을 연결한 CSV 경로입니다. 기본값은 `DUAL_VIEW_MANIFEST` 환경 변수에서 받습니다.
- `verify_image_exists`, `verify_image_shape`: manifest의 이미지 경로와 실제 이미지 크기를 확인합니다.
- `fail_on_bad_row`: 잘못된 row를 건너뛰지 않고 즉시 오류로 처리합니다.
- `target_bounds_policy: error`: 화면 밖 label을 데이터 오류로 처리하며 clamp하지 않습니다.

필수/선택 CSV 열은 [`dataset-format.md`](dataset-format.md)에 정리했습니다.

### `views`

- `available`: 현재 DB에 존재하는 branch입니다. 기본값은 `[front, side]`입니다.
- `source_to_branch`: generic CSV의 `view` 값(예: `webcam`, `phonecam`)을 canonical `front` 또는 `side`로 실제 변환합니다. 이미 `front`/`side`인 값은 그대로 사용합니다.
- `directory_to_branch`: `<subject>/webcam`, `<subject>/phonecam` 구조를 각각 front/side에 매핑하는 보조 규칙입니다. 폴더 이름 자체를 모델 코드에 hard-code하지 않습니다.

### `pairing`

새 dual-view DB에서는 pairing을 기본으로 켜고 다음 규칙을 사용합니다.

- `unit: image`: 두 정지 이미지를 한 fusion sample로 묶습니다.
- `strategy: explicit_pair_id`: 수집 시 같은 gaze event에 부여한 ID로만 join합니다.
- `pair_id_key`: manifest에서 ID가 들어 있는 column 이름입니다. 예를 들어 `capture_group`으로 바꾸면 generic reader는 그 header를 요구하고 canonical `pair_id`로 옮깁니다. pairing이 꺼져 있으면 pair column은 없어도 됩니다.
- `require_same_subject`, `require_same_target`: 잘못된 사람/응시점 간 join을 차단합니다.
- `max_target_distance_normalized`: target의 수치 허용오차입니다. 동일 label이면 0을 사용합니다.
- `unpaired_policy: branch_only`: pair 없는 이미지는 encoder 학습에는 쓰되 fusion loss에는 넣지 않습니다.
- `exclude_unpaired_from_fusion_metrics`: 두 입력이 없는 결과를 fusion 성능으로 계산하지 않습니다.

### `split`

- `strategy: grouped_ratio`: image가 아니라 `group_key`를 먼저 나눈 뒤 해당 group의 모든 이미지를 따라 보냅니다.
- `group_key: subject_id`: 같은 사람이 train과 validation/test에 동시에 들어가는 identity leakage를 막습니다.
- `ratios`: train/validation/test 목표 비율입니다. 합이 1이어야 합니다.
- `seed`, `shuffle_groups`: 같은 manifest를 결정적으로 다시 만듭니다.
- `stratify_by`: 현재 구현값은 `null`뿐입니다. 다른 값을 주면 stratification을 한 것처럼 오해하지 않도록 unsupported error로 중단합니다.
- `prevent_group_leakage`: overlap이 발견되면 즉시 실패합니다.
- `manifest_dir`: 실제 배정 목록을 저장합니다. `reuse_existing_manifest`는 현재 `false`만 지원하며, true는 기존 manifest를 읽지 않고 새로 쓴 것처럼 진행하지 않고 unsupported error로 중단합니다.

70/15/15는 **sample 수가 아닌 목표 사람 비율**이며 정수 인원 때문에 정확히 일치하지 않을 수 있습니다. 모든 positive-ratio split에 최소 한 사람을 줄 수 없으면 빈 validation/test를 만들지 않고 실패합니다.

### `dataloader`

`batch_size`는 한 optimization step 전 GPU/CPU에 올리는 sample 수, `num_workers`는 image loader process 수입니다. `pin_memory`는 CUDA 전송 최적화, `persistent_workers`와 `prefetch_factor`는 worker 재사용/선읽기입니다. train만 shuffle하고 validation/test는 고정 순서를 사용합니다. `drop_last_train`은 마지막 작은 batch를 버릴지 정합니다.

## 5. `preprocessing`

`stage_order`가 실제 실행 순서의 유일한 기준입니다. 각 stage의 `enabled`를 바꾸어 ablation할 수 있지만 입력 dependency가 깨지면 config validation이 실패해야 합니다.

| stage | 핵심 필드와 의미 |
|---|---|
| `decode` | OpenCV BGR decode 후 RGB, `uint8`로 표준화 |
| `exif_orientation` | phone image의 회전 metadata를 pixel에 반영 |
| `validate` | 손상 및 최소 해상도 확인; 원본 해상도는 hard-code하지 않음 |
| `face_landmarks` | base에서는 annotation/OpenCV Haar를 사용합니다. WebEyeTrack profile은 MediaPipe Face Landmarker로 478개 face/iris landmark, presence/visibility, facial transform을 생성합니다. |
| `eye_selection` | `mode: both`는 front 양쪽 눈, `fixed`는 지정 눈, `best_visible`은 side에서 더 잘 보이는 눈을 선택합니다. |
| `eye_state` | MediaPipe 경로의 양쪽 EAR을 계산합니다. `threshold: 0.20`, `required_eye_policy`, `on_closed`로 사용할 sample을 결정합니다. strict-profile annotation 경로는 `eye_region_warp`에서 single-eye EAR을 계산합니다. |
| `metric_head_pose` | facial transform과 metric reconstruction으로 `head_vector [3]`, `head_euler_degrees [3]`, `face_origin_3d [3]` cm를 생성합니다. |
| `eye_region_warp` | front의 공식 양쪽 눈 homography, MediaPipe side의 선택 눈 crop, 또는 strict-profile annotation 기반 한쪽 눈 crop을 실행합니다. |
| `face_roi` | landmark bbox와 margin으로 ROI를 계산. `mode: crop`은 잘라내고, `preserve_canvas`는 원본 크기·위치를 유지하며 ROI 좌표만 저장 |
| `background_mask` | 얼굴 밖을 `fill_rgb`로 채울지 선택. `face_roi_bbox`는 계산된 사각형 ROI 내부만 남기고 원본 캔버스의 나머지를 검정색으로 처리 |
| `resize` | `size_hw`로 letterbox resize, 종횡비 유지, 검은 padding, interpolation 지정 |
| `normalize` | `zero_one`이면 `/255`; pretrained model이 mean/std를 요구하면 contract와 함께 변경 |
| `augment` | train에만 적용. 좌우 flip은 gaze x와 landmark를 함께 변환해야 하므로 기본 0 |

각 품질 stage의 `on_failure`는 실패 시 `mark_invalid`, `drop`, `error` 중 무엇을 할지
정합니다. `eye_state.on_closed`는 눈 감김만 따로 `mark_invalid`, `drop`, `error`, `keep` 중
선택합니다. 기본 WebEyeTrack profile은 닫힌 눈이나 pose 실패의 정답을 `(0,0)`으로 바꾸지
않고 `front_gaze_valid=false` 또는 `side_gaze_valid=false`로 표시합니다. `(0,0)`도 정상적인
화면 중심 label이므로 향후 loss와 metric에서 validity mask로 제외해야 합니다. `NaN` pose와
`False` validity sentinel은 batch shape를 일정하게 유지하기 위한 값이지 유효한 입력이
아닙니다.

`branch_overrides.front`와 `side`는 같은 stage의 `enabled`와 parameter를 view별로
덮어씁니다. 기본 dual-view 설정은 양쪽에서 `face_roi.mode=preserve_canvas`와
`background_mask.method=face_roi_bbox`를 사용하여 원본 위치의 사각형 얼굴 ROI만 남기고,
측면 ROI margin을 더 크게 둡니다. 타원형 `face_hull`은 선택 기능일 뿐 dual-view 기본값이
아닙니다. image shape는 parser에 고정하지 않고 각 파일에서 읽으며, crop/resize 시 landmark에도 같은 좌표 변환을 적용합니다.

### WebEyeTrack/BlazeGaze profile과의 차이

기준 `config.yaml`의 224×224 full-face 입력은 **모델 교체가 가능한 generic contract
예시**입니다. source image 자체는 BlazeGaze model input이 아닙니다.
[`blazegaze.yaml`](../configs/profiles/blazegaze.yaml)은
다음 exact front 전처리를 선택합니다.

```text
source RGB image
→ MediaPipe 478 landmark와 facial transform
→ landmark [103,150,379,332]를 nose [4] 기준으로 x=0.40, y=0.20 padding
→ 512×512 face homography
→ 변환된 landmark [151]과 [195] 사이를 full-width crop
→ 128×512 양쪽 눈 strip
→ float32 / 255
```

`face_hull` 타원 mask, Haar bbox, black-canvas 재생성, ImageNet mean/std는 이 exact front
경로에 포함되지 않습니다. 공식 공개 Keras checkpoint의 실제 입력과 프로젝트 batch key는
다음처럼 대응합니다.

| 의미 | 공식 `.keras` 입력 | 이 프로젝트의 PyTorch batch |
|---|---|---|
| 양쪽 눈 image | `image [B,128,512,3]`, NHWC | `front_image [B,3,128,512]`, CHW |
| 머리 방향 | `head_vector [B,3]` | `front_head_vector [B,3]` |
| metric 얼굴 중심 | `face_origin_3d [B,3]`, cm | `front_face_origin_3d [B,3]`, cm |
| 유효성 | checkpoint 입력 아님 | `front_gaze_valid [B]` |

논문은 metric pose를 회전행렬과 이동벡터로 설명하지만 공개 checkpoint는 그 pose matrix를
펼친 tensor로 받지 않습니다. 실제 보조 입력은 위 표의 두 3차원 vector입니다. 향후 adapter는
CHW→NHWC와 batch key mapping을 담당하고, 공식 TensorFlow/Keras `.keras`를 현재 PyTorch
`.pt`처럼 직접 load해서는 안 됩니다. 공식
근거는 [논문](https://arxiv.org/html/2508.19544v1)과
[eye-patch 구현](https://github.com/RedForestAI/WebEyeTrack/blob/14719ad861467c98890058f7c41a94638ae1db2b/python/webeyetrack/model_based.py#L31-L84),
[model loader](https://github.com/RedForestAI/WebEyeTrack/blob/14719ad861467c98890058f7c41a94638ae1db2b/python/webeyetrack/blazegaze.py#L255-L351)에서 확인할 수 있습니다.

front는 양쪽 눈 strip이므로 `required_eye_policy: all_open`을 사용합니다. 한쪽이라도
EAR `< 0.20`이면 기본 `on_closed: mark_invalid`가 적용됩니다. `metric_head_pose`는 annotation
face center가 있으면 단위를 cm로 바꾸어 사용하고, 없으면 iris diameter `1.20 cm`를 기준으로
metric face origin을 복원합니다. stage-1 profile은 batch 8, 20 epoch, Adam `1e-3`,
exponential decay `0.95`, weighted L2 PoG/reconstruction/embedding-consistency loss도 선언하지만
현재 `prepare` 명령이 이 학습 loop까지 실행한다는 뜻은 아닙니다.

### Side phonecam profile

side는 WebEyeTrack 논문에 없는 프로젝트 확장이며 다음 중 하나를 마지막 profile로
선택합니다.

| profile | 검출 계약 / image contract | EAR 정책 | 권장 용도 |
|---|---|---|---|
| [`side_profile_90.yaml`](../configs/profiles/side_profile_90.yaml) | annotation-first / 한쪽 눈 `side_image [B,3,128,256]` | annotation의 한쪽 눈 EAR | 정면 mesh가 실패하는 strict 90° profile |
| [`side_one_eye.yaml`](../configs/profiles/side_one_eye.yaml) | MediaPipe mesh / 선택 눈 `side_image [B,3,128,256]` | `selected_eye_open` | mesh가 검출되는 큰 yaw·3/4에서 한쪽 눈이 안정적일 때 |
| [`side_full_face.yaml`](../configs/profiles/side_full_face.yaml) | MediaPipe mesh / 얼굴 ROI `side_image [B,3,224,224]` | `selected_eye_open` | mesh가 검출되고 얼굴·머리 문맥을 새 backbone에 학습할 때 |

MediaPipe 기반 두 profile은 `side_head_vector`, `side_face_origin_3d`, `side_gaze_valid`를
만들며 `side_one_eye`는 선택 눈도 반환합니다. front와 RGB `/255`, landmark 좌표계, EAR와
metric pose 계약은 공유하지만 image crop까지 같게 만들 필요는 없습니다. 두 표현 모두 공식
양쪽 눈 checkpoint와 직접 호환되지 않으므로 custom backbone이 기본입니다. encoder transfer는
명시적 importer로 shape이 맞는 layer만 초기화하고 side data로 fine-tuning하는 별도
실험입니다.

극단적인 profile에서는 가려짐과 큰 yaw 때문에 MediaPipe가 478개 landmark나 facial
transform을 찾지 못할 수 있습니다. `mirror_retry: true`는 좌우 반전본으로 검출을 한 번 더
시도할 뿐 가려진 눈이나 landmark를 만들어내지 않습니다. 실패 sample은 invalid/drop
정책으로 처리해야 합니다. `side_full_face`도 face ROI와 3D pose에 MediaPipe landmark가
필요하므로 detector-free fallback은 아닙니다.

#### `side_profile_90.yaml`: annotation-first visible eye

완전한 측면은 반대쪽 눈과 얼굴 절반이 보이지 않아 정면용 478-point mesh를 전제로 하지
않습니다. 이 profile은 `face_landmarks`, `eye_selection`, `eye_state`, `metric_head_pose`를
side branch에서 끄고 `eye_region_warp.method: profile90_annotation`이 canonical annotation을
직접 읽도록 합니다. `preprocessing.branch_overrides.side.representation`과
`model.side.compatibility.preprocessing_id`는 모두 `profile90_selectable_features_v3`입니다.

| config | 의미 |
|---|---|
| `bbox_key` | 눈꺼풀과 양쪽 corner 문맥을 포함한 source-pixel eye bbox field |
| `eyelid_keypoints_key` | EAR 순서 `p1 corner, p2 upper-1, p3 upper-2, p4 corner, p5 lower-2, p6 lower-1`의 6점 field |
| `iris_center_key` | source pixel의 보이는 iris 중심 field |
| `head_origin_key` | 귀에 가까운 profile 기준점 field |
| `head_forward_key` | 코끝 field; origin에서 이 점으로 향하는 vector를 unit length로 만듦 |
| `size_hw: [128,256]` | 한쪽 눈 model image의 `(height,width)` |
| `crop_mode` | `affine`: 눈 corner 축을 수평 정렬, `letterbox`: bbox 비율 보존 |
| `landmark_crop_scale_xy` | `affine` crop의 수평·수직 크기. 보이는 눈 너비의 배수 |
| `bbox_scale_xy` | `letterbox` 또는 bbox-only fallback의 수평·수직 문맥 배율 |
| `feature_extraction.side_headpose` | head feature 사용 여부와 `side_2d`/`front_3d` source 선택 |
| `feature_extraction.side_eyeangle` | a0→a1, a0→a2 두 방향각 feature 사용 여부 |
| `feature_extraction.side_eyelidangle` | 사용자용 이름을 유지한 iris-relative pose feature 사용 여부 |
| `feature_extraction.side_eyelidangle.vertical_only` | iris-눈꺼풀 중심 offset에서 수직 성분만 유지할지 선택 |
| `eyelid_tail_indices: [3,2,4]` | `a0=p4` temporal corner, `a1=p3` upper, `a2=p5` lower 지정 |
| `ear_threshold` | 한쪽 눈 open/closed 초기 경계. phonecam validation split에서 재보정 필요 |
| `on_closed` | 닫힌 눈의 `mark_invalid`, `drop`, `error`, `keep` 정책 |
| `on_failure` | annotation 누락/퇴화 geometry의 `mark_invalid`, `drop`, `error` 정책 |

`side_headpose.source: front_3d`의 추가 안전 계약은 다음과 같습니다.

| config | 의미 |
|---|---|
| `front_3d_key: front_head_vector` | paired front에서 승격되는 canonical orientation tensor |
| `front_3d_validity_key: front_head_orientation_valid` | face origin과 무관한 orientation 계산 성공 여부 |
| `front_3d_invalid_policy: zero_fill_and_mask` | invalid/NaN을 0 vector로 바꾸고 `side_gaze_valid=false`로 전파 |
| `front_3d_invalid_policy: error` | invalid pose를 만나면 sample 처리를 즉시 실패 |

annotation 예제는
[`profile90_annotations.yaml`](../configs/examples/profile90_annotations.yaml)에 있습니다.
좌표는 원본 image의 top-left origin pixel이며 `+x`는 오른쪽, `+y`는 아래쪽입니다. 이 예제는
현재 예제의 `side_example1.jpeg`부터 `side_example5.jpeg`까지 같은 strict-profile 계약으로
등록되어 있습니다.
수동 좌표는 형식을 보여주기 위한 것이므로 모든 학습 image를 별도로 label해야 합니다.
검수된 행만 `eye_annotation_valid: true`로 표시하며, `false`인 행은 고정 크기 invalid patch와
`side_gaze_valid=false`로 처리합니다.

Side image 계약은 feature toggle과 독립적입니다. `eye_region_warp.enabled: true`이면
`side_image [B,3,128,256]` RGB `[0,1]` 한쪽 눈 patch를 항상 만듭니다. 선택 기능은 정확히
다음 위치에 있습니다.

```yaml
preprocessing:
  branch_overrides:
    side:
      eye_region_warp:
        feature_extraction:
          side_headpose:
            enabled: true
            source: side_2d  # side_2d | front_3d
          side_eyeangle:
            enabled: true
          side_eyelidangle:
            enabled: true
            vertical_only: true
```

`model.side.input_contract.forward_keys: auto`는 고정된 입력 목록이 아닙니다.
`resolve_model_forward_keys()`가 `side_image`로 시작한 뒤 enabled feature의 실제 key만
추가합니다. 기본 설정의 resolved 순서는 다음과 같습니다.

```text
side_image [B,3,128,256]
side_head_pose_2d [B,2]
side_eye_angles [B,2]
side_iris_pose_2d [B,2]
```

현재 구현된 `select_model_forward_inputs(batch, config, "side")`는 이 key만 선택합니다.
따라서 toggle을 바꿀 때 별도의 hard-coded input list를 수정하지 않습니다. 선택된 tensor를
외부 모델에 전달하는 runner/adapter는 아직 구현되지 않았습니다.

`side_headpose.source: side_2d`는 `profile_head_origin_xy`에서
`profile_head_forward_xy`(코끝)로 향하는 image-plane unit vector
`side_head_pose_2d [B,2]`를 선택합니다. `front_3d`는 같은 시점의 paired front가 만든
`front_head_vector [B,3]`로 이 key를 교체합니다. 따라서 `data.pairing.enabled: true`, front
branch, front metric-head-pose stage가 모두 필요합니다. 이 3D vector의 좌표계는
`front_camera`입니다. 두 카메라의 extrinsic 변환이 없으면 side-camera frame vector로
해석하거나 두 좌표계의 성분을 직접 비교하면 안 됩니다.
`front_head_orientation_valid=false`이거나 vector가 non-finite이면 기본 정책은 model에
NaN을 넘기지 않고 0 vector로 치환한 뒤 `side_gaze_valid=false`로 표시하는 것입니다. 향후
loss·metric·fusion은 이 sample을 제외해야 합니다. 이 feature에는 face origin이 필요하지 않으므로
`front_head_pose_valid`가 아니라 orientation validity만 사용합니다.

`side_eyeangle.enabled: true`이면 temporal corner `a0=p4`, upper neighbor `a1=p3`, lower
neighbor `a2=p5`에서 `v1=a1-a0`, `v2=a2-a0`를 만듭니다. roll과 좌우 눈에 일관된 semantic
eye-local 축은 `+x=temporal→nasal/inward`, `+y=upper→lower`입니다. model input은

```text
side_eye_angles = [alpha_upper/π, alpha_lower/π]
alpha_upper = atan2(v1_y, v1_x)
alpha_lower = atan2(v2_y, v2_x)
shape [B,2], range [-1,1]
```

입니다. raw `v1`, `v2`는 plot/debug용이며 forward에 전달하지 않습니다. 두 vector 사이의
기존 포함각 `θ/π`, `θ=acos((v1·v2)/(∥v1∥∥v2∥))`도 aperture 진단값일 뿐 v3 model
feature가 아닙니다.

`side_eyelidangle.enabled: true`는 이름과 달리 canonical
`side_iris_pose_2d [B,2]`를 선택합니다. 이는 iris 중심에서 eyelid 중심을 뺀 offset을 눈
너비와 opening으로 정규화한 기존 iris vertical cue입니다.
`feature_extraction.side_eyelidangle.vertical_only: true`이면
명시적으로 `[0, normalized_vertical_offset]`으로 매핑하여 x 성분을 0으로 둡니다.
eyelid-only included-angle scalar와 혼동하지 않습니다.

세 auxiliary feature를 모두 끄는 config 확인 명령은 다음과 같습니다. 이 경우에도
`side_image`는 남습니다.

```bash
python -m gaze_pipeline validate-config \
  --config configs/config.yaml \
  --profile configs/profiles/blazegaze.yaml \
  --profile configs/profiles/side_profile_90.yaml \
  --skip-path-checks \
  preprocessing.branch_overrides.side.eye_region_warp.feature_extraction.side_headpose.enabled=false \
  preprocessing.branch_overrides.side.eye_region_warp.feature_extraction.side_eyeangle.enabled=false \
  preprocessing.branch_overrides.side.eye_region_warp.feature_extraction.side_eyelidangle.enabled=false
```

paired front의 3D head vector를 선택하는 명령은 다음과 같습니다.

```bash
python -m gaze_pipeline validate-config \
  --config configs/config.yaml \
  --profile configs/profiles/blazegaze.yaml \
  --profile configs/profiles/side_profile_90.yaml \
  --skip-path-checks \
  preprocessing.branch_overrides.side.eye_region_warp.feature_extraction.side_headpose.source=front_3d \
  data.pairing.enabled=true
```

`side_ear`, `side_selected_eye_index`, `side_gaze_valid`는 forward input이 아니라 diagnostic 및
quality 정보입니다. 특히 EAR는 선택 눈의 open/closed 판정에만 사용하고,
`side_gaze_valid`는 annotation/EAR 품질을 반영하며, 향후 loss·metric·fusion의 mask로
사용해야 합니다.
config의 `input_contract.forward_keys: auto`와 `diagnostics.*.passed_to_model: false`가 이
경계를 명시합니다.

`side_head_pose_2d`는 3D rotation/euler angle이 아니며 `side_eye_angles`와
`side_iris_pose_2d`는 calibrated screen gaze가 아닙니다. 따라서 이 feature를
`target_gaze_xy` 대신 사용하거나 gaze metric으로 평가하지 않습니다. 실제 시선 target은
같은 촬영 시점의 screen target 및 camera/screen calibration에서 별도로 만듭니다.

`model.side.initialization`의 `webeyetrack_encoder_transfer`는 공식 양안 encoder 전체가 한쪽
눈 입력과 호환된다는 뜻이 아닙니다. `load_scope`에 적힌 convolution/depthwise/batch-norm 중
shape이 일치하는 layer만 명시적 importer로 옮깁니다. 한쪽 눈 `128×256`의 공간 구조는 공식
BlazeGaze 양안 `128×512` checkpoint와 정확히 같지 않으므로 `spatial_projection`,
`gaze_mlp`, `gaze_output`은 다시 초기화합니다. 이는 partial transfer 실험이지 공식 model의
exact 재현이 아닙니다. 현재 `importer_entrypoint: null`이면 실제 weight import가 실행되는
상태가 아니며, importer와 공식 sample parity test를 구현한 뒤 켜야 합니다.

논문의 stage 2 first-order MAML은 subject별 support/query episode와 별도 optimizer loop가
필요하므로 이 stage-1 profile이 자동으로 재현한다고 보지 않습니다. side phonecam과 late
fusion 역시 WebEyeTrack이 검증한 기능이 아닙니다.

## 6. `model`: 어떤 PyTorch 모델도 연결하는 계약

> **구현 상태:** model I/O contract 검증과 forward-key 선택만 구현되어 있습니다. 외부 model
> import, checkpoint load, adapter forward와 output 표준화는 아직 구현되지 않았습니다.

`backend`는 향후 checkpoint와 tensor adapter의 기본 실행환경입니다. 각 branch가 최종적으로
제공해야 할 항목은 다음 네 가지입니다.

1. `source_dir`: repository 외부 모델 코드가 있는 선택 경로. package로 설치되어 있으면 null이어도 됩니다.
2. `entrypoint`: 예: `my_model.factory:create_model`처럼 import 가능한 factory 주소입니다.
3. `init_args`: factory에 넘길 모델별 config입니다. 공통 pipeline schema 밖의 임의 인자는 여기만 허용합니다.
4. `adapter_entrypoint`: pipeline batch를 모델 인자로 바꾸고 model-specific 출력을 표준 key로 변환합니다.

`pretrained.path`, `sha256`, `strict`는 초기 weight의 위치, 무결성, key 일치 정책입니다. 외부 코드/weight는 출처와 라이선스를 확인해야 합니다.

### 입력 계약

기본 branch 입력은 다음과 같습니다.

```text
front_image 또는 side_image
shape: [B, 3, 224, 224]
dtype: float32
color: RGB
range: [0, 1]
```

모델이 eye patch, head pose, landmarks 등 추가 입력을 요구하면 adapter contract에 key/shape/dtype/unit을 추가하고 전처리 stage의 output과 연결합니다.

### 출력 계약

최소 출력은 `gaze_xy: [B,2]`입니다. late fusion을 사용하려면 각 branch가 `front_embedding` 또는 `side_embedding`도 반환합니다. embedding 차원은 branch마다 달라도 되지만 fusion module의 init args가 이를 알아야 합니다. 불확실성은 별도 output key로 정의합니다.

모델 주소만 바꿔서 학습하려면 runner가 import 전에 다음을 검증해야 합니다.

- source/entrypoint 존재 및 허용된 코드인지
- model factory가 `init_args`를 받는지
- synthetic batch의 input/output contract
- task 좌표계와 model output 좌표계가 일치하거나 adapter가 명시적으로 변환하는지

## 7. `fusion`: late fusion

> **구현 상태:** `fusion.enabled=true`일 때 pairing과 두 branch 의존성을 검사하는 config
> validation만 구현되어 있습니다. fusion module과 학습 코드는 아직 없습니다.

향후 late fusion은 front/side encoder가 각각 예측 또는 embedding을 만든 **뒤**에 결합합니다.
early fusion처럼 두 이미지를 channel 방향으로 바로 붙이지 않습니다.

`enabled: true`로 바꾸기 위한 전제는 다음과 같습니다.

```text
data.pairing.enabled = true
model.front.enabled = true
model.side.enabled = true
명시적 pair_id가 있는 dual-view manifest 존재
```

- `stage: late`: branch별 표현 생성 후 결합함을 뜻합니다.
- `method: axis_aware_gated`: x/y축마다 front와 side의 신뢰도를 다르게 학습합니다.
- `entrypoint`: custom fusion module factory입니다.
- `input_keys`: branch prediction/embedding 중 fusion이 읽는 표준 key입니다.
- `output_key`, `output_shape`: 최종 `(x,y)` 계약입니다.
- `axis_priors`: 학습 시작 시의 축별 가중치 prior입니다. 고정된 진실이 아니며 validation으로 검증합니다.
- `learnable_gate`: sample별 gate를 학습할지 정합니다.
- `use_quality`, `use_uncertainty`: detector quality나 predicted uncertainty를 gate 입력으로 쓸지 정합니다.
- `missing_branch_policy`: 한 view가 없을 때 available branch로 fallback할지, sample을 버릴지 정합니다.

예를 들어 정면 모델이 수평 방향에 강하고 측면 모델이 수직 residual에 도움을 준다는 가설이라면 x gate는 front 비중을 높게 시작할 수 있습니다. 이 수치는 논문에서 자동으로 보장되는 값이 아니므로 branch-only와 fusion metric을 같이 보고 판단해야 합니다.

## 8. 학습, optimizer, scheduler, loss

> **구현 상태:** 이 section은 향후 trainer 계약입니다. 현재 CLI에는 `train`이 없으며 아래
> 값을 소비하는 optimizer/scheduler/loss executor도 없습니다.

### `training`

- `max_epochs`: 전체 train dataset 반복 횟수입니다.
- `accelerator`, `devices`: CPU/CUDA/MPS 자동 선택과 device 수입니다.
- `precision`: 기본 32-bit. mixed precision을 사용할 때 `16-mixed` 또는 구현이 허용한 enum으로 변경합니다.
- `gradient_accumulation_steps`: 여러 mini-batch gradient를 모아 effective batch를 키웁니다.
- `gradient_clip_norm`: exploding gradient 방지를 위한 전체 norm 상한입니다.
- `log_every_n_steps`, `validate_every_n_epochs`: logging/validation 주기입니다.
- `sanity_validation_batches`: 본 학습 전에 validation pipeline을 몇 batch 확인할지 정합니다.
- `early_stopping`: monitor metric 개선이 `patience_epochs` 동안 없으면 중지합니다. cm calibration이 없는 dataset profile에서는 `metrics.selection_metric`을 fallback normalized metric으로 override합니다.

### `optimizer`, `scheduler`

AdamW의 `learning_rate`, `weight_decay`, `betas`를 config에서 바꿉니다. ReduceLROnPlateau는 validation metric이 개선되지 않을 때 LR에 `factor`를 곱하고 `min_learning_rate` 아래로 내리지 않습니다. scheduler의 monitor와 checkpoint/early stopping monitor는 같은 단위인지 확인해야 합니다.

### `loss`

기본 `huber_xy`는 작은 오차에 L2처럼, 큰 outlier에는 L1처럼 동작합니다. `delta`는 centered-normalized 좌표 단위이고 `axis_weights`로 x/y 중요도를 조절합니다. `uncertainty_nll`은 모델이 분산을 함께 예측할 때만 켭니다. `branch_auxiliary`는 fusion을 학습하면서 각 encoder의 독립 예측도 유지할 때 사용합니다.

## 9. 결과 metric

> **구현 상태:** 아래 값은 향후 평가 계약입니다. 현재 pipeline은 prediction metric을
> 계산하거나 best checkpoint를 선택하지 않습니다.

2D 화면 응시점 모델의 sample 오차는 다음과 같이 계산합니다.

```text
e_norm = sqrt((pred_x_norm - gt_x_norm)^2 +
              (pred_y_norm - gt_y_norm)^2)

dx_cm = (pred_x_norm - gt_x_norm) * screen_width_cm
dy_cm = (pred_y_norm - gt_y_norm) * screen_height_cm
e_cm  = sqrt(dx_cm^2 + dy_cm^2)
```

pixel metric은 각각 screen width/height pixel을 곱해 같은 방식으로 계산합니다.

### 기본 선택 metric

`subject_macro_euclidean_cm`를 권장합니다. 각 subject 안에서 평균 cm 오차를 구한 뒤 subject별 값을 동일 가중치로 평균하므로 사진 수가 많은 한 사람이 전체 결과를 지배하지 않습니다. screenSize calibration이 없을 때만 normalized metric을 fallback으로 씁니다.

### 함께 보고할 값

- normalized Euclidean mean/median
- pixel 및 cm Euclidean mean
- x/y 방향 MAE
- normalized RMSE
- cm error p90/p95
- out-of-bounds prediction 비율
- sample-micro와 subject-macro 집계
- `front/*`, `side/*`, `fusion/*` namespace별 같은 metric

평균만 있으면 일부 큰 오류를 보기 어려워 percentile을 함께 남깁니다. fusion 성능은 반드시 각 branch 단독 결과와 비교합니다.

`angular_error_deg`는 새 DB에 `gaze_target_3d`와 `face_center_3d`가 있고 모델이 3D gaze direction을 예측할 때만 켭니다. screen `(x,y)` 회귀의 기본 metric으로 angular error를 섞지 않습니다.

## 10. Checkpoint 저장 위치와 `.pt` 대 `.pkl`

> **구현 상태:** checkpoint/export runner가 아직 없으므로 아래 파일은 현재 생성되지
> 않습니다. 형식과 경로는 향후 구현 시 지켜야 할 계약입니다.

### 저장 디렉터리

`checkpoint.dir`는 `${paths.run_dir}/checkpoints`, 최종 추론 weight는 `${paths.run_dir}/models`입니다. 둘을 나누는 이유는 다음과 같습니다.

- `best_weights.pt`: validation 선택 metric이 가장 좋은 model `state_dict`만 저장, 추론용
- `last_checkpoint.pt`: model뿐 아니라 optimizer/scheduler/scaler, epoch/step, best metric, RNG, config/dataset hash를 저장, 학습 재개용
- `final_weights.pt`: 종료 시점에 export한 weight와 resolved config/model contract 묶음

### 형식 선택

이 PyTorch 파이프라인에서는 `.pt`가 낫습니다. `.pt`와 `.pth`는 확장자 convention이고, 실제 핵심은 **전체 모델 객체가 아니라 `state_dict`를 저장하는 것**입니다.

`.pkl`은 Python class/module 경로와 실행 코드에 강하게 묶이고 신뢰하지 않은 파일을 load할 때 보안 위험이 있으므로 model 배포 형식으로 사용하지 않습니다. `.pt`도 내부적으로 pickle을 사용할 수 있으므로 외부 파일은 무조건 안전하다고 간주하면 안 됩니다. 가능한 경우 weight-only load, checksum 검증, `map_location="cpu"`, strict state-dict load를 사용합니다.

단순히 확장자를 `.pt`로 바꾸는 것만으로 안전해지는 것은 아닙니다. 저장 payload와 신뢰 경계를 문서화해야 합니다.

## 11. MLflow

현재 구현은 `prepare`가 만드는 **data-preparation run**만 기록합니다.

- `tracking_uri`: 기본은 MLflow 3.14 호환 local SQLite `sqlite:///./mlflow.db`입니다. 팀 server에서는 환경 변수로 바꿉니다. deprecated filesystem tracking URI인 `file:./mlruns`는 사용하지 않습니다.
- `experiment_name`, `run_name`: config와 output 폴더 naming을 맞춥니다.
- `log_system_metrics`: CPU/GPU/memory 사용량을 기록합니다. 기본값 `true`에 필요한 `psutil`은 runtime requirements에 포함됩니다.
- `log_resolved_config`: 실제 사용된 override 중 credential과 로컬 경로를 제거한 JSON 사본 및 local YAML의 hash를 저장합니다.
- `log_dataset_manifest`, `log_split_manifest`: 어떤 image가 어느 split에 들어갔는지 고정합니다.
- `log_environment`: Python/platform과 설치된 주요 package version을 기록합니다.
- `tags`: schema, data mode, pairing strategy처럼 run 검색에 필요한 작은 문자열입니다.

현재는 model weight와 checkpoint를 log하지 않습니다. 향후 training tracker는 Git
commit/dirty 여부, resolved config SHA-256, dataset/split hash, model weight SHA-256,
loss/metric과 best/last checkpoint를 별도 training run에 기록해야 합니다.

API token/password/secret/credential과 tracking URI의 userinfo·민감 query/fragment는 local
snapshot 전에 `<redacted>`로 치환합니다. `dataset_manifest_hash`는 path를 포함한
`dataset_manifest.csv` 바이트의 SHA-256이며 원본 이미지 내용 전체의 hash라는 뜻이 아닙니다.
이전 `dataset_hash` result property는 호환 alias일 뿐 사용자 출력과 MLflow parameter에는 새
이름을 사용합니다. 원본 얼굴 이미지는 기본 artifact로 업로드하지 않습니다.

manifest CSV 자체에는 로컬 image path와 `subject_id`가 있으므로 신뢰하지 않는 원격 server에서는 `mlflow.log_dataset_manifest=false`와 `mlflow.log_split_manifest=false`로 끕니다. 집계 summary, hash, 비식별 config와 환경 정보는 그대로 기록됩니다.
