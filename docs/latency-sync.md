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

## Simulation

simulation의 `display_timestamp`는 다음 식으로 생성합니다.

```text
base_unix_timestamp_ns + frame * frame_interval_ns
```

같은 설정으로 실행하면 같은 timestamp가 생성되어야 합니다. simulation에서는 웹캠 timestamp를 표시 시각과 같게 두고 phonecam timestamp에 설정의 `phone_delay_ms`를 더합니다.

## 다음 구현 단계

1. 검정·흰색 전환 이벤트와 영상 밝기 변화 검출
2. 카메라별 median, MAD, p95 레이턴시 계산
3. 보정 timestamp 계산
4. 단조·일대일 최근접 프레임 매칭
5. 동적 target 좌표 보간
6. `synchronized_frames.csv` 생성
