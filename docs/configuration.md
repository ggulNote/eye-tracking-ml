# Configuration reference

이 문서는 현재 학습에 사용하는 Config만 설명합니다. 전체 기본값은
[`config.yaml`](../configs/config.yaml), 실제 실행 예시는 [README](../README.md)에 있습니다.

## 1. 병합 순서

뒤의 파일과 CLI override가 앞의 값을 덮어씁니다.

```text
configs/config.yaml
  → configs/profiles/blazegaze.yaml
  → configs/profiles/side_profile_90.yaml
  → configs/profiles/side_precomputed_roi.yaml
  → configs/profiles/measured_head_down_neutral.yaml
  → configs/models/front_webeyetrack.yaml
  → configs/models/<selected-side-model>.yaml
  → CLI override
```

Production 순서를 검증하려면 다음 명령을 사용합니다.

```bash
make validate-config \
  PROFILES="configs/profiles/blazegaze.yaml configs/profiles/side_profile_90.yaml configs/profiles/side_precomputed_roi.yaml configs/profiles/measured_head_down_neutral.yaml configs/models/front_webeyetrack.yaml configs/models/side_mobilenet_v4.yaml"
```

## 2. 주요 그룹

| 그룹 | 역할 |
|---|---|
| `experiment`, `paths` | run 이름, seed, data/output 경로 |
| `task` | 시선 좌표와 target 계약 |
| `data` | reader, pairing, session 선택, split, DataLoader |
| `preprocessing` | Front/Side 전처리, 선택 feature, cache |
| `model` | factory, adapter, pretrained weight, I/O contract |
| `fusion` | Side의 Y축 residual 결합 |
| `training`, `optimizer`, `scheduler` | epoch와 최적화 설정 |
| `loss`, `metrics` | 학습 objective와 평가·checkpoint 기준 |
| `checkpoint`, `model_export` | `.pt` 저장·resume |
| `mlflow` | Config, metric, checkpoint lineage 기록 |

개인 절대경로는 tracked YAML에 쓰지 않고 환경 변수나 Make 변수로 전달합니다.

## 3. 화면 좌표

Target은 화면 중심 기준 좌표입니다.

```text
x_norm = target_x_px / screen_width_px  - 0.5
y_norm = target_y_px / screen_height_px - 0.5
```

분모는 카메라 image 크기가 아니라 화면 크기입니다. Nominal 범위 `[-0.5,0.5]`는 label 좌표계이며
activation 범위가 아닙니다. 음수 좌표가 필요하므로 출력에 ReLU를 사용하면 안 됩니다. 기본
회귀 head는 linear이며, bound가 꼭 필요한 외부 모델은 adapter에서 `0.5*tanh(raw)`처럼 직접
구현해야 합니다.

## 4. Data

### Reader와 session 선택

```yaml
data:
  reader:
    type: generic_csv
    manifest_path: "${oc.env:MEASURED_DATA_MANIFEST,./data/measured_manifest.csv}"
  selection:
    session_ids: [head_down, neutral]
```

`image_path`, `sample_id`, `subject_id`, `pair_id`, 화면 target은 manifest가 제공합니다. 측정 DB
builder는 `inputs.csv.valid=0`인 pair를 manifest 생성 전에 제외합니다. Pipeline 내부에서 눈 상태
지표를 다시 계산하지 않습니다.

### Pairing과 split

```yaml
data:
  pairing:
    enabled: true
    require_same_subject: true
    require_same_target: true
    unpaired_policy: error
  split:
    strategy: grouped_ratio
    group_key: subject_id
    ratios: {train: 0.70, validation: 0.15, test: 0.15}
    seed: 42
    shuffle_groups: true
```

분할 단위는 sample이 아니라 사람입니다. 같은 피험자는 하나의 split에만 들어갑니다. 피험자가
5명이면 `70/15/15` 목표가 실제로는 `3/1/1`, 즉 `60/20/20`이 될 수 있습니다.

### DataLoader

- `batch_size`: optimization step당 pair 수
- `num_workers`: image loader process 수
- `pin_memory`: CUDA 전송 최적화
- `persistent_workers`, `prefetch_factor`: worker 재사용과 선읽기
- `train_shuffle`: train 순서만 섞을지 여부

현재 공식 Front Keras wrapper를 CPU에서 시작할 때는 `batch_size: 8`, `num_workers: 0`이 안전한
기본값입니다.

## 5. 전처리

현재 stage 순서는 다음과 같습니다.

```text
decode → exif_orientation → validate → face_landmarks → eye_selection
→ metric_head_pose → eye_region_warp → face_roi → background_mask
→ resize → normalize → augment
```

로컬 눈 상태 계산 stage는 없습니다. `inputs.csv.valid`는 upstream 승인 결과로만 사용합니다.

### Front WebEyeTrack 입력

[`blazegaze.yaml`](../configs/profiles/blazegaze.yaml)은 corrected web frame에서 다음 값을 만듭니다.

| 입력 | Shape | 의미 |
|---|---|---|
| `front_image` | `[B,3,128,512]` | homography로 정렬한 양쪽 눈 strip |
| `front_head_vector` | `[B,3]` | Front camera 좌표계 머리 방향 |
| `front_face_origin_3d` | `[B,3]` | metric 얼굴 중심, cm |

측정 profile의 `metric_head_pose.source: precomputed_only`는 `inputs.csv` pose만 사용합니다. 기존
저장 ROI는 읽지 않고 corrected frame에서 Front image를 다시 만듭니다.

### Side ROI

Production에서는 [`side_precomputed_roi.yaml`](../configs/profiles/side_precomputed_roi.yaml)이
외부 `side_eye_roi` 이미지를 직접 사용합니다.

```yaml
preprocessing:
  branch_overrides:
    side:
      eye_region_warp:
        method: precomputed_side_roi
        content_size_hw: [128, 128]
        size_hw: [128, 256]
        pad_rgb: [0, 0, 0]
```

입력이 `128×128`보다 크면 중앙 crop하고 작으면 검정 padding합니다. Resize나 interpolation은
하지 않습니다. 결과 ROI를 `128×256` canvas 중앙에 배치하므로 좌우에는 각각 64픽셀 검정 영역이
생깁니다.

### Side 선택 feature

`side_image [B,3,128,256]`은 항상 모델 입력입니다. 추가 feature는 각각 선택합니다.

```yaml
preprocessing:
  branch_overrides:
    side:
      eye_region_warp:
        feature_extraction:
          side_headpose:
            enabled: false
            source: side_2d       # 또는 front_3d
          side_eyeangle:
            enabled: false        # a0→a1, a0→a2 방향각 [B,2]
          side_eyelidangle:
            enabled: false        # iris 상대 위치 [B,2]
            vertical_only: true
```

| Toggle | 필요한 값 | 전달되는 key |
|---|---|---|
| `side_headpose`, `source: side_2d` | `profile_head_origin_xy`, `profile_head_forward_xy` | `side_head_pose_2d` |
| `side_headpose`, `source: front_3d` | paired Front pose | `front_head_vector` |
| `side_eyeangle` | `visible_eye_keypoints_xy` 6점 | `side_eye_angles` |
| `side_eyelidangle` | 눈꺼풀 6점, `iris_center_xy` | `side_iris_pose_2d` |

`model.side.input_contract.forward_keys: auto`가 활성 feature만 선택합니다. 필요한 annotation 없이
toggle을 켜면 sample은 무효 처리됩니다.

### Deterministic `.pkl` cache

측정 profile은 동일한 전처리를 매 epoch 반복하지 않도록 cache를 켭니다.

```yaml
preprocessing:
  cache:
    enabled: true
    dir: "${paths.output_root}/preprocessed_cache"
    mode: read_write       # read_write | read_only | refresh
    format: pickle
    schema_version: 1
    implementation_id: ordered-gaze-preprocessor-v2
    trusted_local: true
```

Cache key는 preprocessing/task Config, manifest row, view, dataset split, source path와 image bytes
SHA-256을 포함합니다. Config, bbox 또는 source image가 바뀌면 자동으로 새 shard를 만듭니다.
Augmentation은 cache에 넣지 않고 cache hit 뒤 매 epoch 다시 적용합니다.

`refresh`는 Dataset 실행 중 각 sample의 최초 접근에서 다시 계산·저장한 뒤 새 shard를
재사용하며, `read_only`는 miss가 나도 쓰지 않습니다. Pickle은
신뢰하지 않은 파일을 열 때 위험하므로 pipeline이 만든 비공유 로컬 cache에서만
`trusted_local: true`를 사용합니다. 같은 cache 폴더에서 여러 process가 동시에 `refresh`를
실행하지 않습니다.

## 6. 모델 교체

```yaml
model:
  side:
    enabled: true
    source_dir: /absolute/path/to/model-repository
    entrypoint: my_models.side:create_model
    adapter_entrypoint: my_models.side:create_adapter
    init_args:
      embedding_dim: 256
    pretrained:
      path: /absolute/path/to/weights.pt
      sha256: <64-hex>
      strict: true
```

- `source_dir`: 외부 코드 root. 설치된 package면 `null` 가능
- `entrypoint`: `create_model(**init_args)` factory
- `adapter_entrypoint`: canonical batch와 모델 고유 signature/output 사이 변환
- `pretrained`: PyTorch state dict와 checksum 정책

Front 최소 출력은 `gaze_xy [B,2]`, Side 최소 출력은 `delta_y_side [B,1]`입니다. Generic
`pretrained.path`는 `.pt`/`.pth` 전용입니다. 공식 Front `.keras`는
[`front_webeyetrack.yaml`](../configs/models/front_webeyetrack.yaml)의 검증된 전용 factory가
로드합니다.

## 7. Fusion

```yaml
fusion:
  enabled: true
  method: y_axis_residual
  residual_weight: 1.0
  learnable_weight: true
  missing_branch_policy: use_available_branch
```

```text
x_final = x_front
y_final = y_front + residual_weight * delta_y_side
```

Side는 x축을 변경하지 않습니다. `learnable_weight=true`이면 residual weight도 optimizer가
갱신합니다. Side가 무효이면 residual을 0으로 두고 Front 결과를 사용합니다.

## 8. 학습 설정과 Loss

```yaml
training:
  max_epochs: 50
  gradient_accumulation_steps: 1
  gradient_clip_norm: 1.0
optimizer:
  name: AdamW          # Adam | AdamW
  learning_rate: 0.0001
  weight_decay: 0.0001
scheduler:
  enabled: true
  name: ReduceLROnPlateau  # 또는 ExponentialLR
```

지원 primary loss는 다음과 같습니다.

| `loss.primary.name` | 의미 | 주요 옵션 |
|---|---|---|
| `huber_xy` | 작은 오차 L2, 큰 오차 L1 | `delta`, `axis_weights` |
| `mse_xy` | x/y squared error | `axis_weights` |
| `weighted_l2_xy` | 화면 cell 빈도의 역수로 가중한 L2 | `frequency_grid_size`, `axis_weights` |

CLI에서 바꾸는 예:

```bash
make measured-train OVERRIDES="loss.primary.name=huber_xy loss.primary.delta=0.05"
make measured-train OVERRIDES="loss.primary.name=mse_xy"
make measured-train \
  OVERRIDES="loss.primary.name=weighted_l2_xy loss.primary.frequency_grid_size=[30,30]"
```

`loss.branch_auxiliary.side_weight`는 Side residual target에 추가할 보조 loss 가중치입니다.

## 9. Metric과 checkpoint 기준

시선 좌표는 회귀이므로 분류 accuracy 대신 거리 오차를 사용합니다.

```yaml
metrics:
  selection_metric: subject_macro_euclidean_normalized
  fallback_selection_metric: euclidean_normalized_mean
  report:
    - euclidean_normalized_mean
    - mae_x_normalized
    - mae_y_normalized
    - rmse_normalized
    - within_0_05_normalized_rate
  threshold_rates:
    - name: within_0_05_normalized_rate
      unit: normalized
      threshold: 0.05
      enabled: true
```

지원 metric:

- normalized: Euclidean mean/median, x/y MAE, RMSE, OOB, subject macro
- pixel: 화면 해상도가 있을 때 Euclidean mean과 subject macro
- cm: 실제 화면 크기(mm)가 있을 때 Euclidean mean, p90/p95, subject macro
- threshold rate: `normalized`, `pixel`, `cm` 거리 이내 비율

`metrics.report`는 기록할 값을, `selection_metric`은 best checkpoint 기준을 정합니다. 요청한 물리
metric을 계산할 metadata가 없으면 `fallback_selection_metric`을 사용합니다. 현재 측정 profile은
실제 화면 크기가 없어 normalized subject-macro metric을 선택합니다.

## 10. Checkpoint와 MLflow

```text
checkpoints/best_weights.pt       # best component state
checkpoints/last_checkpoint.pt    # resume용 optimizer/scheduler/RNG 포함 가능
models/final_weights.pt           # 학습 종료 weight
preprocessed_cache/**/*.pkl       # 로컬 전처리 cache
```

모델 저장은 Python 객체 전체가 아니라 state dict 기반 `.pt`를 사용합니다. `.pkl`은 전처리 cache
전용이며 모델 배포 형식이 아닙니다.

MLflow는 resolved Config, dataset/split hash, epoch loss/metric, best/last/final checkpoint와 평가
lineage를 기록합니다. 측정 profile은 피험자 이름과 로컬 경로를 보호하기 위해 manifest,
prediction과 raw image 업로드를 기본으로 끕니다.
