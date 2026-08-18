# WebEyeTrack 데이터 계약

이 문서는 모델 구현보다 먼저 고정해야 하는 데이터 경계를 정의합니다. 기준은 WebEyeTrack 논문과 공개 Python/JavaScript 구현의 `webeyetrack_v1` 프로필입니다.

## 책임 경계

```text
webcam / phone video + synchronized gaze labels
  -> video loader (구현 완료)
  -> canonical full frames (구현 완료)
  -> MediaPipe + eye/head processor (구현 완료)
  -> BlazeGaze encoder + gaze MLP (추후 구현)
  -> centered normalized (x, y)
```

원본 영상을 바로 `128 x 512`로 바꾸면 안 됩니다. 이 크기는 얼굴 랜드마크를 검출하고 얼굴을 정규화한 뒤 얻는 **양쪽 눈 영역 패치**의 크기입니다. 원본 full frame은 얼굴 검출에 필요한 정보를 유지한 채 설정된 공통 크기로만 변환합니다.

## 1. 원본 영상 입력

웹캠과 폰 모두 다음 계약으로 통일합니다.

| 필드 | 계약 |
|---|---|
| frames | RGB `uint8[T,H,W,3]` |
| target | 화면 좌상단 기준 정규화 `float32[2]`, 범위 `[0,1]` |
| frame_indices | 정답 시점 주변에서 선택한 프레임 번호 `T`개 |
| timestamps_ms | 프레임별 영상 시각 `T`개 |
| source | `webcam` 또는 `phone` |
| participant/session/device | 사용자 누수 방지와 장치별 분석용 ID |
| screen size | pixel 크기 필수 권장, cm 크기 강력 권장 |
| camera intrinsics | 알고 있으면 JSON 경로 기록, 없으면 추후 추정 가능 |

영상 해상도, 코덱, FPS, 회전 방향이 달라도 loader가 BGR을 RGB로 바꾸고 `rotation_degrees`를 적용합니다. 같은 영상의 여러 시점에 정답이 있다면 manifest에 같은 `video_path`를 여러 행으로 기록합니다.

카메라 보정값이 있으면 `examples/camera-intrinsics.example.json` 형식으로 `camera_matrix`, `distortion_coefficients`, 보정 당시 영상 크기를 기록합니다. 예제 숫자를 실제 값처럼 재사용하지 말고 각 장치의 calibration 결과를 넣어야 합니다. 회전 또는 resize 시 intrinsics도 같은 좌표 변환에 맞게 조정하는 책임은 추후 head-pose processor에 있습니다.

## 2. 정답 시점 동기화

manifest 한 행은 “영상 하나”가 아니라 **정답이 붙은 한 시점 또는 한 window**입니다.

- `label_timestamp_ms`: 영상 시작 기준 millisecond
- `label_frame_index`: 0부터 시작하는 frame 번호
- 둘 중 하나만 사용
- `sequence_length=1`이면 해당 시점의 한 프레임
- `sequence_length>1`이면 해당 시점을 중심으로 `frame_stride` 간격의 window

둘 다 생략하면 이전 clip-level 형식과의 호환을 위해 영상 전체에서 균등 추출하지만, 실제 WebEyeTrack 데이터에는 권장하지 않습니다.

## 3. 좌표계

manifest 입력은 아래 세 형식을 허용하고 loader가 `[0,1]` 좌상단 좌표로 통일합니다.

| `target_coordinate_space` | 입력 예 | 변환 |
|---|---|---|
| `top_left_normalized` | `(0.25, 0.40)` | 그대로 사용 |
| `screen_centered_normalized` | `(-0.25, -0.10)` | 각 축에 `0.5` 추가 |
| `screen_pixels` | `(960, 540)` | `(screen_width_px, screen_height_px)`로 나눔 |

WebEyeTrack 모델 경계에서는 `to_screen_centered_normalized()`로 변환하여 화면 중심 `(0,0)`, 좌상단 `(-0.5,-0.5)`, 우하단 `(0.5,0.5)`를 사용합니다.

## 4. 공통 full-frame 출력

`VideoPreprocessor`의 출력은 다음과 같습니다.

```text
float32[B,T,C,H,W]
```

- `B`: labeled sample 수
- `T`: `data.sequence_length`
- `C`: `rgb=3`, `gray=1`
- `H,W`: config의 `preprocessing.output_height/output_width`

WebEyeTrack을 연결할 때는 `color_mode: rgb`를 사용해야 합니다. `gray`는 다른 모델 실험을 위한 일반 파이프라인 옵션입니다.

## 5. 구현된 WebEyeTrack 모델 입력

`make collect`의 마지막 단계 또는 `make webeyetrack-inputs`가 정면 web 이미지에
MediaPipe와 카메라 내부 보정을 적용해 아래 입력을 생성합니다.

| 이름 | shape / dtype | 의미 |
|---|---|---|
| `image` | `float32[B,128,512,3]`, BHWC, `[0,1]` | 정규화된 양쪽 눈 패치 |
| `head_vector` | `float32[B,3]` | 카메라 좌표계 머리 방향 벡터 |
| `face_origin_3d` | `float32[B,3]` | 카메라 좌표계 얼굴 원점, cm |

참가자·자세별 출력은 `feature_maps/webeyetrack/`에 저장됩니다.

```text
feature_maps/webeyetrack/
├── eye_roi/*.png       # 512x128 양쪽 눈 패치
├── inputs.csv          # 유효/무효 행 전체와 실패 사유
├── training.csv        # 학습용 유효 행만
├── evaluation.csv      # 최종 평가용 유효 행만
└── summary.json        # 유효율·거리·재투영 오차 요약
```

`head_vector`와 `face_origin_3d`는 정면 카메라의 `Camera.mat`, MediaPipe canonical
face 좌표(cm), OpenCV `solvePnP`로 같은 프레임에서 계산합니다. training과 evaluation
각각의 유효율이 config 기준보다 낮으면 부분 결과를 삭제하고 실패 처리하며 원본 영상과
기존 `feature_maps`는 보존합니다.

`WebEyeTrackInputBatch.validate()`가 이 경계를 검사합니다. 논문의 수식은 head pose를 `[R|t]`로 표현하지만 현재 공개 학습/배포 코드는 실제 MLP 입력을 `head_vector[3] + face_origin_3d[3]`로 구성하므로 이 저장소도 공개 코드 계약을 따릅니다.

학습 데이터 경계인 `WebEyeTrackTrainingBatch`는 여기에 정답, `valid_mask`, 참가자 ID, calibration point ID, sample weight를 함께 묶습니다. `valid_mask=False`는 얼굴 검출 실패나 눈 감김처럼 모델 loss와 평가에서 제외할 샘플을 의미합니다.

## 6. 모델 출력과 개인화 단위

```text
gaze: float32[B,2], screen-centered normalized, range [-0.5,0.5]
```

공개 모델의 마지막 layer는 linear이므로 raw prediction이 범위를 잠시 벗어날 수 있습니다. 외부에 노출하는 최종 출력은 smoothing/후처리 뒤 각 축을 `[-0.5,0.5]`로 clip한 다음 `WebEyeTrackOutputBatch`로 검증합니다.

`participant_id`는 train/validation/test 분할 단위이자 MAML task 단위입니다. 한 참가자의 calibration support set은 config의 `model_io.support_size`로 관리하며 `webeyetrack_v1`에서는 최대 9개입니다. `calibration_point_id`에는 `grid-01`부터 `grid-09`처럼 보정 위치를 기록하면 중복점과 누락점을 확인하기 쉽습니다.

## 7. 새 모델을 연결할 때의 체크리스트

1. full-frame 입력이 RGB인지 확인
2. MediaPipe 실패·눈 감김 샘플을 valid mask로 제외
3. eye patch가 정확히 `[128,512,3]`인지 확인
4. image 값이 `[0,1]`인지 확인
5. `head_vector`와 `face_origin_3d`가 같은 프레임에서 계산됐는지 확인
6. target을 `[-0.5,0.5]`로 변환했는지 확인
7. participant 단위 split을 유지했는지 확인
8. support/query가 같은 참가자이되 겹치지 않는지 확인
9. config와 실제 입출력 schema를 MLflow artifact로 기록
