# Local Data Pipeline

## 목적

webcam과 phone 영상을 한 컴퓨터에서 재현 가능하게 전처리하고, 모델과 독립적인 versioned processed dataset을 생성합니다. S3/MinIO나 중앙 데이터 서버는 사용하지 않습니다.

## 전체 흐름

```text
GGULNOTE_DATA_ROOT/raw
  + GGULNOTE_DATA_ROOT/manifests/manifest.csv
  -> manifest validation
  -> synchronized frame/window decoding
  -> participant-level split
  -> canonical full-frame preprocessing
  -> GGULNOTE_DATA_ROOT/processed/<dataset_version>
```

## 저장 구조

```text
GGULNOTE_DATA_ROOT/
├── raw/
│   ├── webcam/<participant>/<session>.mp4
│   └── phone/<participant>/<session>.mp4
├── manifests/manifest.csv
└── processed/dataset-v001/
    ├── train.npz
    ├── train.metadata.json
    ├── validation.npz
    ├── validation.metadata.json
    ├── test.npz
    ├── test.metadata.json
    └── dataset.json
```

## Raw manifest 계약

필수 열은 `video_path`, `source`, `participant_id`, `session_id`, `target_x`, `target_y`입니다. 실제 학습 데이터에는 다음 항목도 기록하는 것을 권장합니다.

- `sample_id`: 전체 dataset에서 유일한 샘플 ID
- `label_timestamp_ms` 또는 `label_frame_index`: 정답과 영상 시점 동기화
- `target_coordinate_space`: 좌표 형식
- `screen_width_px`, `screen_height_px`: 화면 pixel 크기
- `screen_width_cm`, `screen_height_cm`: cm 오차 계산용 실제 크기
- `device_id`: 장치별 성능 분석
- `rotation_degrees`: phone 방향 보정
- `camera_intrinsics_path`: 선택적 카메라 calibration JSON
- `calibration_point_id`: 9-point 보정 위치
- `sample_weight`, `valid`: sampling/loss와 제외 여부

동일 영상에 정답 시점이 여러 개면 같은 `video_path`를 여러 행에 기록합니다. loader는 한 행마다 지정된 시점의 frame/window만 디코딩합니다.

## Canonical output

```text
frames: float32[B,T,C,H,W]
targets: float32[B,2], top-left normalized [0,1]
sample_weights: float32[B]
```

`H`, `W`, RGB/gray, mean/std는 config로 관리합니다. WebEyeTrack에서는 RGB full frame을 유지하고, `128×512` eye patch 변환은 다음 model-specific processor에서 수행합니다.

## 분할과 재현성

split은 frame이 아닌 `participant_id` 단위입니다. 동일 사용자가 train과 validation/test에 동시에 들어가지 않습니다. split seed와 비율은 config에 기록됩니다.

processed dataset은 다음 식별 정보를 갖습니다.

- `dataset_version`: 사람이 관리하는 불변 version
- `manifest_hash`: raw manifest SHA-256
- `content_hashes`: split별 tensor/metadata 내용 hash
- `dataset_hash`: manifest, contract, split content를 합친 SHA-256
- `preprocessor_version`: 전처리 코드 version

기존 version을 기본적으로 덮어쓸 수 없습니다. 데이터나 전처리가 달라졌다면 `dataset-v002`처럼 version을 올립니다.

## 실행

```bash
cp .env.example .env
make setup-video
make preprocess CONFIG=configs/manifest-local.example.yaml
```

전처리 결과의 `dataset.json`에서 sample 수, participant 수, shape, dtype, hash를 확인합니다.

## 현재 한계

현재 baseline은 split별 NPZ를 메모리에 적재합니다. 실제 데이터 규모가 메모리 용량을 넘는 시점에는 processed format을 sharded NPZ/HDF5/Zarr로 확장하고 model loader를 streaming 방식으로 바꿔야 합니다. manifest와 model I/O 계약은 유지합니다.
