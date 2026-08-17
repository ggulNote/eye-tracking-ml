# WebEyeTrack 참고 2카메라 데이터 수집

이 모듈은 [WebEyeTrack 논문](https://arxiv.org/abs/2508.19544)을 참고하되, 정면 웹캠과 참가자 왼쪽 30–45°의 iPhone을 사용하는 자체 데이터 수집 확장입니다. 논문의 단안 장비와 수집 조건을 그대로 재현하지는 않습니다.

## 저장 원칙

- 실행 시작 시 입력한 참가자 이름을 최상위 폴더명과 CSV 참가자 값으로 사용합니다.
- 한 참가자는 `neutral`, `head_up`, `head_down` 자세를 각각 한 번 촬영합니다.
- 이미 수집 결과가 있는 참가자 폴더는 덮어쓰지 않습니다.
- 선택한 프로토콜이 여러 개여도 카메라마다 하나의 연속 MP4만 생성합니다.
- 점당 최종 학습 좌표와 이미지 쌍은 하나의 `labels/labels.csv`에 기록합니다.
- 동기화용 전체 프레임 기록은 `metadata/frame_log.csv`로 구분합니다.
- 각 클릭 지점의 안정 구간에서 가장 좋은 프레임 한 쌍만 같은 sample ID로 저장합니다.
- 프로토콜 구간은 CSV의 `protocol`과 `split`로 구분합니다.

```text
안은제/
├── neutral/<전체 촬영 데이터>/
├── head_up/<전체 촬영 데이터>/
└── head_down/<전체 촬영 데이터>/
```

상세 구조와 모든 CSV 열은
[참가자 데이터 폴더와 CSV 설명](participant-data-layout.md)을 기준으로 합니다.

## 설치와 실행

```bash
make setup-capture
make list-cameras
```

카메라 없이 기본 학습 수집 구조를 검증합니다.

```bash
make simulate PARTICIPANT=테스트참가자 HEAD_POSE=neutral DATASET_ROOT=/private/tmp/gaze-simulation
```

기본 실행은 전체 dot test를 한 번에 실행합니다. 모든 학습·세로 왕복·평가 점은 참가자 클릭 속도에 따라 시간이 달라지며, 각 점을 허용되는 즉시 확정하면 최소 약 112.2초입니다.

Dot Test 전체 화면은 현재 macOS가 OpenCV에 제공하는 실제 활성
크기인 `1470x956` 캔버스를 사용합니다. OpenCV의 원본 비율 유지로 여백이
생기지 않도록 캔버스를 전체 화면에 맞춰 표시합니다. macOS의 화면 해상도
조절을 바꾸면 캔버스 크기도 달라질 수 있습니다. 실제 수집 전에는 카메라와
참가자 데이터를 만들지 않는 화면 검사부터 실행합니다.

```bash
make check-display
```

초록 테두리가 화면 네 면에 모두 닿아야 합니다. 회색 또는 검은 여백이 한쪽에
보이면 수집하지 말고 `Q` 또는 `Esc`로 닫은 뒤 화면 설정과 캔버스 크기를 다시
확인합니다.

```bash
make collect
# 이름 적어주세요: 안은제
# headpose를 입력해주세요 [neutral/head_up/head_down]: neutral
```

입력은 한 번만 받으며, 실행 순서는
`레이턴시 측정 → Dot Test → 프레임 동기화 → 정면 MediaPipe 특징 추출`입니다.
레이턴시가 실패하거나 사용자가 중단하면 다음 단계로 넘어가지 않습니다. 결과는
각 자세 폴더의 `calibration/latency.json`과 `feature_maps/`에 따로 저장됩니다.

## iCloud 공유 폴더 자동 백업

공유 폴더 소유자가 사용자를 `편집 가능` 참여자로 추가하고, Finder의 iCloud Drive에
해당 폴더가 나타난 뒤 절대 경로를 전달합니다.

```bash
make collect ICLOUD_BACKUP_ROOT="/Users/ahn-eunje/Library/Mobile Documents/com~apple~CloudDocs/공유폴더이름"
```

후처리까지 정상 완료된 참가자·head pose만 복사합니다. 임시 폴더로 복사한 뒤 원본과
복사본의 파일 크기와 SHA-256을 모두 비교하고, 검증이 통과하면 최종 폴더명으로
변경합니다. 그 뒤 다음 질문에서 `네`를 입력한 경우에만 로컬 head pose 폴더를
삭제합니다.

```text
iCloud 복사·검증이 완료되었습니다. 로컬 데이터를 삭제하시겠습니까? [네/아니요]:
```

`아니요` 또는 Enter를 입력하면 로컬 데이터가 유지됩니다. 기존 iCloud 백업은
덮어쓰지 않으며, iCloud 서버에 실제 업로드됐는지는 Finder 상태 아이콘으로도
확인해야 합니다.

실제 점 프로토콜 전에 같은 카메라 연결을 유지한 채 `webcam`과 `phonecam`을 나란히 표시합니다. 정면 webcam은 MediaPipe face+iris가 설정된 프레임 수만큼 연속 검출되어야 통과합니다. phonecam은 MediaPipe를 적용하지 않는 측면 이미지 모델 입력이므로, 한쪽 눈이 선명하고 노출이 적절한지 육안으로 확인합니다. `q` 또는 `Esc`로 안전하게 중단할 수 있습니다.

레이턴시 측정 전에 데이터 저장 없이 같은 검사를 먼저 실행할 수 있습니다.

```bash
make check-cameras
```

검사를 통과한 뒤에는 phonecam 위치·줌·해상도·연결 방식을 바꾸지 않고
`make collect`를 실행합니다. 이 명령이 레이턴시 측정부터 촬영 후 전처리까지
이어 실행합니다.

일부 구간만 개발 테스트할 때는 프로토콜을 명시합니다. 실제 참가자 수집에는 기본 `all`을 사용합니다.

```bash
make collect PARTICIPANT=안은제 HEAD_POSE=neutral PROTOCOL=evaluation_static_3x6
make collect PARTICIPANT=안은제 HEAD_POSE=head_up PROTOCOL=vertical_click_3col_6row
make collect PARTICIPANT=안은제 HEAD_POSE=head_down PROTOCOL=all
```

기본 `intrinsics_2d` 모드는 장비 공통 폴더의 웹캠·폰캠 `Camera.mat`을 검사하고
참가자 `calibration/` 폴더에 복사합니다. 두 파일의 보정 영상 크기가 촬영 설정과
다르면 수집 전에 실패합니다.

```bash
make collect PARTICIPANT=안은제 HEAD_POSE=neutral
```

`fixed_rig_2d`는 명시적으로 렌즈 보정을 끄는 개발용 선택이고,
`calibrated_3d`는 실제 `Camera.mat`과 `monitorPose.mat`이 모두 필요합니다. 측정하지
않은 값을 가짜 MAT 파일로 채우지 않습니다.

정적 학습·세로 왕복·평가 점에서는 회색 원이 줄어든 뒤 점을 계속 보면서 마우스 왼쪽 버튼을 한 번 클릭합니다. Space 또는 Enter도 같은 확정 입력으로 사용할 수 있습니다. 너무 이른 입력은 무시되며, 확정 뒤 초록 원이 보이는 0.65초 동안에도 점을 계속 봐야 합니다.

실행 중 `Q` 또는 `Esc`를 누르면 파일을 닫고 `participant.json` 상태를 `aborted`로 기록합니다.
카메라가 순간적으로 프레임을 읽지 못하면 `read_retry_count`와
`read_retry_delay_ms`에 따라 최대 1초 동안 재시도합니다. 잠깐의 iPhone 연속성
카메라 끊김은 같은 촬영에서 자동 복구됩니다. 그래도 읽지 못하거나
`max_identical_frames`보다 오래 동일 프레임을 반환하면 실제 연결 정지로 판단해 실패
처리합니다. 이 경우에만 카메라를 다시 연결한 뒤 레이턴시부터 재측정합니다.

이름은 한글을 그대로 지원하지만 `/`, `\\`, `:`, 제어문자와 숨김 폴더 형태는
거부합니다. 같은 이름의 기존 수집 데이터는 덮어쓰지 않습니다. 얼굴 영상과 실명을
함께 저장하는 경우 직접 식별 정보가 되므로 연구용 별칭 사용을 권장합니다.

head pose는 `neutral`, `head_up`, `head_down`만 허용합니다. 세 자세의 CSV
`participant` 값은 모두 같은 이름이고 `head_pose` 열만 달라집니다. 같은 사람을
세 명처럼 나누지 않으므로 학습·평가 분할은 항상 참가자 이름 단위로 수행합니다.

## 카메라 역할

- `webcam_front`: 모니터 정면 웹캠 → `video/web/`, `images/web/`
- `iphone_left`: 참가자 왼쪽 30–45° iPhone → `video/phone/`, `images/phone/`

노트북마다 장치 번호가 다를 수 있습니다. `make list-cameras`로 확인하고 `configs/capture.yaml`의 `device_index`만 조정합니다. 역할명은 저장 폴더와 연결되므로 교환하지 않습니다.

## 프로토콜

### Head pose 조건

- `neutral`: 화면 중앙을 향하고 턱을 자연스럽게 유지
- `head_up`: 정면 방향은 유지하면서 턱을 약 10도 올림
- `head_down`: 정면 방향은 유지하면서 턱을 약 10도 내림

각 촬영에서는 입력한 자세를 유지하고 눈만 점을 따라갑니다. 너무 큰 각도로 한쪽
눈이 가려지거나 phonecam의 한쪽 눈 영상이 사라지지 않게 합니다. 촬영 준비 화면과
프로토콜 시작 화면에 현재 head pose가 표시됩니다.

### 전체 순서와 시간

1. `intro_center`: 중앙 응시 5초
2. `train_static_3x9`: 정적 학습 27점, 참가자 클릭 방식(최소 32.4초)
3. `protocol_transition`: 중앙 전환 3초
4. `vertical_click_3col_6row`: 3개 열에서 6점 상하 왕복 클릭(총 36회, 최소 43.2초)
5. `center_refix`: 중앙 재응시 4초
6. `evaluation_static_3x6`: 정적 평가 18점 클릭 방식(최소 21.6초)과 종료 3초

전체 시간은 최소 112.2초이며, 참가자가 점을 확인하고 클릭하는 시간만큼 늘어납니다.

### 학습용 정적 점

- 3열 × 9행, 총 27개 위치
- x는 20%, 50%, 80%, y는 8%부터 92%까지 9단계
- 무작위 순서로 1회
- 점이 나온 뒤 최소 0.4초 동안은 확정할 수 없으며 학습 후보가 아닙니다.
- 참가자가 마우스 왼쪽 버튼 또는 Space/Enter로 점을 확정합니다.
- 확정 뒤 0.65초는 `usable=1`인 안정 구간입니다.
- 이후 0.15초 전환을 거쳐 다음 무작위 점으로 이동합니다.
- 확정 전 프레임은 원본 MP4와 `metadata/frame_log.csv`에는 남지만 학습 이미지로 저장하지 않습니다.

### 정적 평가

- 학습 y 위치 사이의 3열 × 6행, 총 18개 위치
- `split=evaluation`이며 학습에는 넣지 않습니다.
- 학습점과 동일하게 클릭 확정 뒤 0.65초의 이미지만 저장합니다.
- 모델 학습과 레이턴시 추정에 섞지 않고 최종 MAE 계산에만 사용

### 세로 왕복 클릭 학습

- x는 왼쪽 20%, 중앙 50%, 오른쪽 80% 순서입니다.
- 각 열에는 화면 높이 8~92% 범위의 점 6개가 표시됩니다.
- 각 열에서 위→아래 6점을 클릭한 뒤 아래→위 6점을 다시 클릭합니다.
- 열마다 12회, 전체 36회이며 `split=train`입니다.
- 움직이는 점과 세로 안내선은 표시하지 않습니다.
- 각 위치는 정적인 클릭 점이며 다른 클릭 프로토콜과 같은 안정 구간을 사용합니다.

## CSV 규격

`labels/labels.csv`는 점당 품질 최적 이미지 한 쌍과 정답을 합친 81행의 단일
학습 라벨입니다. `metadata/frame_log.csv`는 촬영 중 모든 프레임과 화면 상태를
기록하며 후속 동기화에 사용합니다. 각 카메라의 `timestamps.csv`는
`frame,elapsed_ms,timestamp` 형식입니다. 후속 동기화와 프레임 선택은 MP4 재생
시간이 아니라 Unix timestamp를 사용합니다.

### 논문 참고 이미지 샘플

[MPIIGaze 공식 수집 설명](https://collaborative-ai.org/research/datasets/MPIIGaze/)의 축소되는 회색 원과 참가자 확정 입력 방식을 참고했습니다. 원 연구는 Space 입력을 사용하지만 이 프로젝트는 요청에 맞춰 마우스 왼쪽 버튼도 지원합니다.

`labels/labels.csv`는 클릭 한 번당 **가장 좋은 카메라 프레임 쌍 한 개**를 기록합니다. 두 이미지 파일명은 같은 sample ID이며, 원본 MP4의 카메라별 프레임 번호와 timestamp, 표적 좌표, train/evaluation 구분을 함께 저장합니다. 기본 전체 실행은 학습 27개 + 세로 왕복 36개 + 평가 18개이므로 정상 완료 시 카메라별 이미지가 81장입니다.

후보 선택 우선순위는 OpenCV Haar 기반 눈 열림 증거, 얼굴 검출, Laplacian 선명도, 적정 밝기입니다. 웹캠과 폰캠을 따로 고르면 fusion 시점이 어긋나므로 두 카메라의 합산 품질 점수가 가장 높은 **동일 pair**를 선택합니다. `candidate_count`, 합산 품질 점수와 카메라별 얼굴·눈 검출 수, 선명도, 밝기를 manifest에 남깁니다.

Haar 눈 검출은 측면 폰캠, 안경, 강한 반사에서 실패할 수 있으므로 선명도·노출과 함께 보조 점수로만 사용합니다. 클릭 안정 구간의 대표 프레임을 고를 때 MediaPipe FaceMesh+iris 성공 점수는 정면 webcam에만 적용합니다. phonecam은 동일 시각의 측면 이미지를 그대로 저장하고 OpenCV 품질 점수만 반영합니다.

모든 CSV 열과 의미는 [CSV 스키마 전체 목록](csv-schemas.md)을 기준으로 합니다.

## MAT 보정 자산

기본 `intrinsics_2d`에서는 카메라별 `Camera.mat`이 필수입니다. B 전처리가 원본
프레임에 `cv2.undistort`를 적용한 후 PNG와 MediaPipe 8차원 특징을 생성합니다.
`monitorPose.mat`은 3D 시선 벡터나 카메라–화면 좌표 변환을 구현할 때만 필요합니다.

수집기는 실제 loader가 소비하는 다음 변수를 검사하고 결과를 `participant.json`에 기록합니다.

- `Camera.mat`: `cameraMatrix`, `distCoeffs`, `retval`, `image_width`, `image_height`
- `monitorPose.mat`: `rvects`, `tvecs`
- `screenSize.mat`: `width_pixel`, `height_pixel`, `width_mm`, `height_mm`
- `stereoCalibration.mat`(선택): 두 카메라 내부 파라미터, `R_iphone_to_webcam`, `T_iphone_to_webcam`, stereo reprojection error

웹캠과 iPhone은 렌즈와 위치가 다르므로 각자의 `Camera.mat`과 `monitorPose.mat`을 별도로 생성해야 합니다.

## 개인정보와 후속 파이프라인

얼굴 영상은 생체·개인정보에 해당할 수 있으므로 연구 동의, 접근 권한, 암호화, 보관 기간 및 폐기 정책을 먼저 확정해야 합니다. 안경·렌즈, 시력 조건, 주사용 손, 조명, 눈–화면 거리 같은 익명 메타데이터는 `--participant-metadata` JSON으로 전달할 수 있습니다.

현재 모듈은 지속 학습 파이프라인의 **원본 데이터 수집 단계**입니다. 실제 표시
좌표와 모든 프레임 시각은 `metadata/frame_log.csv`에 보정 없이 기록합니다. 실제
lag는 별도 레이턴시 측정 단계에서 계산하고, 보정 결과는 원본을 덮어쓰지 않고
`feature_maps/synchronized.csv`로 생성합니다. 세부 계약은
[레이턴시·프레임 동기화 문서](latency-sync.md)를 참고합니다.

후속 `ggulnote-video-preprocess`는 점당 하나의 레이턴시 보정 pair에서 MediaPipe 8차원 특징을 추출하고 눈 감김·얼굴 미검출을 표시합니다. 자세한 연결 계약은 [영상 전처리 문서](video-preprocessing-handoff.md)를 참고합니다. 이후 학습 dataset 변환, MLflow 기반 학습·평가·모델 버전 연결이 필요합니다. 평가점은 각 점의 안정 구간 예측값으로 MAE_x, MAE_y, 상·중·하 MAE_y와 predicted-target y 기울기를 계산합니다.
