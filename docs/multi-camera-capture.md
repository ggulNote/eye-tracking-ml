# WebEyeTrack 참고 2카메라 데이터 수집

이 모듈은 [WebEyeTrack 논문](https://arxiv.org/abs/2508.19544)을 참고하되, 정면 웹캠과 참가자 왼쪽 30–45°의 iPhone을 사용하는 자체 데이터 수집 확장입니다. 논문의 단안 장비와 수집 조건을 그대로 재현하지는 않습니다.

## 저장 원칙

- 익명 참가자 ID는 `p00`부터 `p99`까지 사용합니다.
- 참가자 한 명은 한 번만 촬영합니다.
- 이미 수집 결과가 있는 참가자 폴더는 덮어쓰지 않습니다.
- 선택한 프로토콜이 여러 개여도 카메라마다 하나의 연속 MP4만 생성합니다.
- 학습 좌표와 동기화 정보는 하나의 `labels/labels.csv`에 기록합니다.
- 프로토콜 구간은 CSV의 `protocol`과 `split`로 구분합니다.

```text
p00/
├── Calibration/
│   ├── screenSize.mat
│   ├── stereoCalibration.mat          # 선택 사항
│   ├── webcam/
│   │   ├── Camera.mat
│   │   └── monitorPose.mat
│   └── phonecam/
│       ├── Camera.mat
│       └── monitorPose.mat
├── webcam/
│   ├── capture.mp4
│   └── timestamps.csv
├── phonecam/
│   ├── capture.mp4
│   └── timestamps.csv
├── labels/
│   └── labels.csv
├── events/
│   ├── capture_config.yaml
│   └── <protocol>.csv
└── participant.json
```

## 설치와 실행

```bash
make setup-capture
make list-cameras
make suggest-participant
```

카메라 없이 기본 학습 수집 구조를 검증합니다.

```bash
make simulate PARTICIPANT=p00 DATASET_ROOT=/private/tmp/gaze-simulation
```

기본 실행은 전체 105초 dot test를 한 번에 실행합니다.

```bash
make collect PARTICIPANT=p00
```

실제 점 프로토콜 전에 같은 카메라 연결을 유지한 채 5초간 `webcam`과 `phonecam`을 나란히 표시합니다. 두 영상의 역할·구도·초점과 실시간 갱신 여부를 확인하며, 사전 검사가 끝나면 카메라를 다시 열지 않고 바로 점 테스트로 넘어갑니다.

일부 구간만 개발 테스트할 때는 프로토콜을 명시합니다. 실제 참가자 수집에는 기본 `all`을 사용합니다.

```bash
make collect PARTICIPANT=p00 PROTOCOL=evaluation_static_3x6
make collect PARTICIPANT=p00 PROTOCOL=dynamic_vertical_3col
make collect PARTICIPANT=p00 PROTOCOL=all
```

아직 보정값 없이 카메라 배치와 저장만 시험할 때는 다음 override를 사용합니다. 이 결과는 학습 데이터로 사용하지 않습니다.

```bash
make collect PARTICIPANT=p00 ALLOW_MISSING_CALIBRATION=1
```

실행 중 `Q` 또는 `Esc`를 누르면 파일을 닫고 `participant.json` 상태를 `aborted`로 기록합니다.
카메라가 `max_identical_frames`보다 오래 동일 프레임을 반환하면 연결 정지로 판단해 즉시 실패 처리합니다. 해당 촬영본을 사용하지 말고 카메라를 다시 연결한 뒤 레이턴시부터 재측정합니다.

## 카메라 역할

- `webcam_front`: 모니터 정면 웹캠 → `webcam/`
- `iphone_left`: 참가자 왼쪽 30–45° iPhone → `phonecam/`

노트북마다 장치 번호가 다를 수 있습니다. `make list-cameras`로 확인하고 `configs/capture.yaml`의 `device_index`만 조정합니다. 역할명은 저장 폴더와 연결되므로 교환하지 않습니다.

## 프로토콜

### 전체 순서와 시간

1. `intro_center`: 중앙 응시 5초
2. `train_static_3x9`: 정적 학습 27점, 32.4초
3. `protocol_transition`: 중앙 전환 3초
4. `dynamic_vertical_3col`: 동적 세로 이동과 전환, 36초
5. `center_refix`: 중앙 재응시 4초
6. `evaluation_static_3x6`: 정적 평가 18점 21.6초와 종료 3초

전체 시간은 105초입니다.

### 학습용 정적 점

- 3열 × 9행, 총 27개 위치
- x는 20%, 50%, 80%, y는 8%부터 92%까지 9단계
- 무작위 순서로 1회
- 점당 1.2초
- 처음 0.4초는 `settling=1`이며 학습에서 제외
- 0.4~1.05초는 `usable=1`, `training=1`

### 정적 평가

- 학습 y 위치 사이의 3열 × 6행, 총 18개 위치
- `split=evaluation`, `training=0`
- 모델 학습과 레이턴시 추정에 섞지 않고 최종 MAE 계산에만 사용

### dynamic 평가

- 왼쪽·중앙·오른쪽 열에서 위→아래, 아래→위
- 각 방향의 순수 이동은 5초이며 방향·열 전환은 각각 1초
- 이동 시작과 끝 0.4초는 학습 후보에서 제외
- 전체 동적 구간은 36초
- 매 프레임 움직이는 점 좌표 저장
- 세로 안내선은 dynamic에서만 선택적으로 표시

## CSV 규격

`labels/labels.csv`의 열은 다음과 같습니다.

```text
participant,protocol,split,frame,
display_timestamp,
webcam_frame,webcam_timestamp,
phonecam_frame,phonecam_timestamp,time_diff_ms,
x_px,y_px,x_norm,y_norm,x_centered,y_centered,
segment,repeat,target,direction,settling,usable,training
```

- `display_timestamp`: 화면 상태를 갱신한 직후 기록한 Unix 나노초 정수
- `webcam_timestamp`, `phonecam_timestamp`: 프레임을 받은 Unix 나노초 정수
- `time_diff_ms`: `phonecam_timestamp - webcam_timestamp`
- `x_norm`, `y_norm`: 좌상단 `(0,0)`, 우하단 `(1,1)`
- `x_centered`, `y_centered`: 화면 중심 `(0,0)`, 범위 `[-0.5,0.5]`
- `repeat`: 현재 프로토콜은 1회이지만 규격 확장성을 위해 유지

각 카메라의 `timestamps.csv`는 `frame,elapsed_ms,timestamp` 형식입니다. 후속 동기화와 프레임 선택은 MP4 재생 시간이 아니라 이 Unix timestamp를 사용합니다.

## MAT 보정 자산

수집기는 실제 loader가 소비하는 다음 변수를 검사하고 결과를 `participant.json`에 기록합니다.

- `Camera.mat`: `cameraMatrix`, `distCoeffs`, `retval`
- `monitorPose.mat`: `rvects`, `tvecs`
- `screenSize.mat`: `width_pixel`, `height_pixel`, `width_mm`, `height_mm`
- `stereoCalibration.mat`(선택): 두 카메라 내부 파라미터, `R_iphone_to_webcam`, `T_iphone_to_webcam`, stereo reprojection error

웹캠과 iPhone은 렌즈와 위치가 다르므로 각자의 `Camera.mat`과 `monitorPose.mat`을 별도로 생성해야 합니다.

고정 장비 보정 도구의 인쇄·측정·실행 순서는 [카메라 기하 보정 문서](camera-geometry-calibration.md)를 참고합니다. 장비 공통 보정값은 `data/calibration/setups/macbook_air_m5_13_iphone16/`에 한 번 생성하고, 실제 참가자 수집 시 최종 MAT 파일만 참가자 `Calibration/` 폴더로 복사합니다.

## 개인정보와 후속 파이프라인

얼굴 영상은 생체·개인정보에 해당할 수 있으므로 연구 동의, 접근 권한, 암호화, 보관 기간 및 폐기 정책을 먼저 확정해야 합니다. 안경·렌즈, 시력 조건, 주사용 손, 조명, 눈–화면 거리 같은 익명 메타데이터는 `--participant-metadata` JSON으로 전달할 수 있습니다.

현재 모듈은 지속 학습 파이프라인의 **원본 데이터 수집 단계**입니다. `labels.csv`에는 실제 표시 좌표와 `display_timestamp`를 보정 없이 기록합니다. 실제 lag는 별도 레이턴시 측정 단계에서 계산하고, 보정 결과는 원본을 덮어쓰지 않고 별도의 synchronized CSV로 생성합니다. 세부 계약은 [레이턴시·프레임 동기화 문서](latency-sync.md)를 참고합니다.

이후 MediaPipe 특징 추출, 눈 감김·얼굴 미검출 제거, y축 15~20구간 균등 샘플링, 정적/동적 50:50 배치 구성, 학습 dataset 변환, MLflow 기반 학습·평가·모델 버전 연결을 구현해야 새 참가자 폴더를 추가하는 것만으로 재학습할 수 있습니다. 평가점은 각 점의 안정 구간 예측 중앙값으로 MAE_x, MAE_y, 상·중·하 MAE_y와 predicted-target y 기울기를 계산합니다.
