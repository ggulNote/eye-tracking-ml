# Model Pipeline

## 목적

현재 local data pipeline과 향후 WebEyeTrack/BlazeGaze 구현 사이의 경계를 고정합니다. MediaPipe processor와 BlazeGaze 학습 코드는 아직 구현하지 않습니다.

## 단계

```text
canonical RGB full frame
  -> MediaPipe Face Landmarker
  -> landmark/eye-state validation
  -> metric face reconstruction and head pose
  -> homography-normalized bilateral eye patch
  -> BlazeGaze encoder
  -> gaze MLP
  -> screen-centered normalized point of gaze
```

## Processor input

원본 frame은 RGB여야 하며 얼굴 landmark를 검출할 수 있는 해상도를 유지해야 합니다. `128×512`는 full frame 크기가 아니라 homography 적용 후 생성되는 양쪽 눈 patch 크기입니다.

권장 capture 기준:

- 기본: `1280×720 RGB`
- 정밀도 우선: `1920×1080 RGB`
- 저사양 장치: `640×480 RGB`

학습과 배포에서 색상 순서, landmark version, crop 좌표, homography 방법을 동일하게 유지해야 합니다.

## BlazeGaze model input

```python
model_inputs = {
    "image": eye_patch,              # float32[B,128,512,3], [0,1]
    "head_vector": head_vector,      # float32[B,3]
    "face_origin_3d": face_origin,   # float32[B,3], camera coordinates in cm
}
```

공개 구현과 사전학습 가중치를 사용할 때 image shape는 변경하지 않습니다. `WebEyeTrackInputBatch.validate()`가 shape, dtype, finite value, image range를 검사합니다.

## Training target

```text
norm_pog: float32[B,2]
coordinate system: screen-centered normalized
range: [-0.5,0.5]
```

raw manifest의 좌상단 `[0,1]` 좌표는 학습 경계에서 `to_screen_centered_normalized()`로 변환합니다.

`WebEyeTrackTrainingBatch`는 model inputs 외에 다음 정보를 보존합니다.

- `valid_mask`: 얼굴 미검출·눈 감김 샘플 제외
- `participant_ids`: split 및 MAML task 단위
- `sample_ids`: 중복/추적 방지
- `calibration_point_ids`: 보정 지점 확인
- `sample_weights`: 불균형 loss 보정

## 학습 단계

### Stage 1: representation learning

- Encoder + Decoder + gaze estimator 학습
- eye patch reconstruction loss
- weighted 2D PoG loss
- embedding consistency loss
- 완료 후 Decoder는 배포 대상에서 제거

### Stage 2: personalization meta-learning

- Encoder 고정
- 사용자 한 명을 하나의 task로 구성
- support set: 최대 9개 calibration sample
- query set: 같은 사용자의 겹치지 않는 평가 sample
- gaze MLP를 MAML 초기값으로 학습

### Inference

- eye patch, head vector, face origin 입력
- linear head의 raw prediction 계산
- 선택적 personalization/affine correction
- Kalman smoothing
- 최종 좌표를 `[-0.5,0.5]`로 clip

## MLflow 기록

- dataset/manifest hash
- preprocessor, landmark, eye-patch contract version
- encoder/gaze model version
- input/output schema
- hyperparameters와 support/query 크기
- train/validation/test metric
- model artifact와 synthetic zero input signature

원본 영상, eye patch tensor, processed dataset, 실제 image-derived input example은 MLflow에 업로드하지 않습니다.

## 구현 체크리스트

1. MediaPipe 실패와 blink가 `valid_mask`에 반영되는가
2. eye patch가 RGB `[128,512,3]`인가
3. image가 `float32 [0,1]`인가
4. head와 face origin이 eye patch와 동일 timestamp인가
5. target이 화면 중심 `[-0.5,0.5]`인가
6. participant split과 support/query가 누수 없이 분리되는가
7. 학습/배포 전처리 version이 일치하는가
8. 계약과 model signature가 MLflow에 기록되는가
