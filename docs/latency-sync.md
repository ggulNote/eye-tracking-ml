# 레이턴시·프레임 동기화 계약

이 문서는 `feat/latency-sync` 브랜치에서 구현하는 A 담당 영역의 입출력 경계를 정의합니다. MediaPipe 특징 추출과 얼굴·눈 전처리는 이 모듈의 범위가 아닙니다.

## 원본 수집 계약

`labels/labels.csv`는 보정 전 원본 기록입니다. 수집 중 화면에 실제로 표시한 좌표를 저장하며, 측정되거나 추정된 레이턴시로 좌표를 이동하지 않습니다.

필수 시간 열은 모두 Unix 나노초 정수입니다.

```text
display_timestamp
webcam_timestamp
phonecam_timestamp
```

- `display_timestamp`: `imshow` 이후 GUI 이벤트를 처리한 직후 기록한 소프트웨어 표시 시각
- `webcam_timestamp`: 웹캠 프레임을 OpenCV에서 받은 시각
- `phonecam_timestamp`: phonecam 프레임을 OpenCV에서 받은 시각

`display_timestamp`는 하드웨어 VSync나 광자 방출 시각이 아니라 소프트웨어 기준점입니다. 검정·흰색 전환을 카메라 영상에서 검출해 이 기준점과의 차이를 반복 측정하고 중앙값으로 보정합니다.

## 원본 보존 원칙

- `labels.csv`를 후속 처리에서 덮어쓰지 않습니다.
- 레이턴시 측정 결과는 `Calibration/latency.json`에 저장합니다.
- 보정 프레임과 좌표는 `synchronized/synchronized_frames.csv`에 새로 저장합니다.
- 모든 계산은 정수 나노초로 수행하고, 표시용 통계만 밀리초로 변환합니다.

## 검정·흰색 전환 측정

기본 프로토콜은 검은 화면에서 시작해 800ms마다 검정과 흰색을 15회 전환합니다. 처음 5초는 착석·카메라 노출 안정화를 위한 검정 기준 구간이며 마지막 전환 후 800ms를 추가 기록합니다. 참가자는 화면 중앙의 작은 반대색 점을 계속 응시합니다.

```text
초기 검정: 5.0초
전환: 15회 × 0.8초 간격
마지막 기록: 0.8초
전체: 약 17초
```

카메라는 참가자 얼굴을 계속 촬영합니다. 검정·흰색 화면의 빛이 얼굴에 반사되면서 생기는 중앙 ROI 평균 밝기 변화를 사용하므로, 측정 중 카메라를 화면 쪽으로 돌리지 않습니다.

실제 측정:

```bash
python -m ggulnote_ml.synchronization \
  --participant p00 \
  --capture-config configs/capture.yaml \
  --latency-config configs/latency.yaml
```

카메라 없는 simulation:

```bash
python -m ggulnote_ml.synchronization \
  --simulate \
  --participant p00 \
  --dataset-root /private/tmp/gaze-latency-simulation
```

## 밝기 변화 검출

정면 웹캠과 측면 phonecam은 얼굴 위치가 다르므로 설정에 카메라별 얼굴 ROI와 최소 밝기 변화량을 따로 둡니다. 각 후보 프레임에서 설정된 개수의 이전·이후 프레임 평균을 비교합니다. 흰색 전환은 양의 밝기 변화, 검정 전환은 음의 밝기 변화를 찾습니다.

- `min_brightness_change`보다 작은 변화는 무효
- `max_latency_ms` 이후의 프레임은 검색하지 않음
- 같은 프레임을 여러 전환에 사용하지 않음
- 유효 latency의 중앙값에서 MAD 기반으로 먼 값은 outlier 처리
- `min_valid_events`보다 적으면 최종 `latency.json`을 만들지 않음

최종 통계는 카메라마다 별도로 계산합니다.

```text
median_ms
mad_ms
p95_ms
valid_events
total_events
```

## 출력 구조

```text
p00/Calibration/
├── latency.json                         # 유효한 최종 결과만 생성
└── latency_runs/latency_<UTC>/
    ├── capture_config.yaml
    ├── latency_config.yaml
    ├── display_events.csv
    ├── webcam_brightness.csv
    ├── phonecam_brightness.csv
    ├── detections.csv
    ├── webcam.mp4                       # 실제 측정만 생성
    ├── phonecam.mp4                     # 실제 측정만 생성
    ├── webcam_timestamps.csv            # 실제 측정만 생성
    ├── phonecam_timestamps.csv          # 실제 측정만 생성
    └── result.json
```

유효한 `Calibration/latency.json`은 덮어쓰지 않습니다. 검출에 실패하거나 중단한 실행은 run 디렉터리에 진단 자료만 남기며, 설정을 조정한 뒤 새 run으로 다시 측정할 수 있습니다.

## Simulation

simulation의 `display_timestamp`는 다음 식으로 생성합니다.

```text
base_unix_timestamp_ns + frame * frame_interval_ns
```

같은 설정으로 실행하면 같은 timestamp가 생성되어야 합니다. simulation에서는 웹캠 timestamp를 표시 시각과 같게 두고 phonecam timestamp에 설정의 `phone_delay_ms`를 더합니다.

## 다음 구현 단계

유효한 `Calibration/latency.json`과 새 형식의 `labels/labels.csv`가 있으면 다음 명령으로 보정·매칭합니다.

```bash
python -m ggulnote_ml.synchronization \
  --synchronize \
  --participant p00
```

장비, 해상도, FPS, 카메라 연결 방식과 위치가 모두 동일한 상태에서 바로 이어서
새 참가자를 촬영했다면 기존 측정값을 명시적으로 재사용할 수 있습니다. 경로를 직접
지정한 경우에만 참가자 ID가 다른 레이턴시 파일을 허용하며, 출력 요약에는 사용한
원본 경로가 남습니다.

```bash
python -m ggulnote_ml.synchronization \
  --synchronize \
  --participant p02 \
  --latency-json data/raw/participants/p00/Calibration/latency.json
```

처리 순서:

1. 웹캠 timestamp에서 웹캠 median latency를 뺍니다.
2. phonecam timestamp에서 phonecam median latency를 뺍니다.
3. 보정 timestamp가 가장 가까운 프레임을 시간순·일대일로 선택합니다.
4. `max_pair_diff_ms`를 넘는 쌍은 `valid_sync=0`으로 표시합니다.
5. 보정 시각에 해당하는 화면 target을 찾습니다.
6. 과거 `dynamic_*` 데이터가 입력될 때만 같은 segment의 target을 선형 보간합니다. 현재 `vertical_click_3col_6row`는 정적 클릭 좌표이므로 보간하지 않습니다.
7. 원본을 건드리지 않고 새 CSV와 요약 JSON을 저장합니다.

```text
p00/synchronized/
├── synchronized_frames.csv
└── synchronization.json
```

주요 출력 열:

```text
webcam_corrected_timestamp
phonecam_corrected_timestamp
corrected_time_diff_ms
reference_timestamp
target_timestamp
x_norm, y_norm
target_interpolated
valid_sync
valid_target
valid
invalid_reason
```

완료 메시지 구간처럼 target이 없거나 가장 가까운 target과 설정 시간 이상 떨어진 프레임은 원본 행을 보존하되 `valid_target=0`으로 표시합니다.

## 남은 구현 단계

1. 실제 dot test 데이터로 보정 프레임 쌍 검증
2. 동기화 CSV와 MediaPipe 특징 CSV 결합
3. 무작위 정적 학습점과 세로 왕복 클릭점의 균형 샘플링
