# 고정 장비 카메라 기하 보정

이 도구는 현재 고정 장비인 **MacBook Air 13-inch (M5) + iPhone 16**의 보정값을 한 번 만들고, 이후 참가자 데이터에 동일한 최종 MAT 파일만 복사하기 위한 도구입니다.

장비를 옮기거나 카메라 렌즈·줌·해상도를 변경하면 해당 보정을 다시 해야 합니다. 참가자가 바뀌는 것만으로는 다시 하지 않습니다.

## 설정과 출력 위치

- 설정: `configs/geometry_calibration.yaml`
- 장비 공통 보정값: `data/calibration/setups/macbook_air_m5_13_iphone16/`
- 출력용 체커보드: `outputs/calibration/`

장비 공통 보정 폴더와 촬영 이미지는 로컬 데이터이므로 Git에서 제외합니다. 코드와 설정 형식만 Git으로 공유합니다.

## 1. 체커보드 생성과 인쇄

```bash
make geometry-board
open outputs/calibration/checkerboard_9x6_25.0mm.svg
```

인쇄 조건:

- A4 가로
- 배율 `100%` 또는 `실제 크기`
- `페이지에 맞춤` 해제
- 흑백 출력 가능
- 출력 후 평평하고 단단한 판에 부착
- 반사되는 코팅은 사용하지 않기

기본 패턴은 내부 코너 `9x6`, 실제 사각형 `10x7`입니다. 웹캠 보정에서는 iPhone
화면의 한 칸 `8mm`를 사용했고, phonecam 보정에서는 MacBook 화면의 한 칸
`16mm`를 사용했습니다. 두 측정값은 `checkerboard.camera_measurements`에 카메라별로
분리되어 있고 각 `Camera.mat`에도 촬영 당시 크기가 기록됩니다.

출력 후 연속된 8칸의 길이를 자로 측정하고 8로 나눕니다. 예를 들어 8칸이 `199.2mm`이면 실제 한 칸은 `24.9mm`입니다. 이 값을 `checkerboard.square_size_mm`에 입력하고 다음 값을 바꿉니다.

```yaml
checkerboard:
  square_size_mm: 24.9
  verification_source: manual_measurement
```

## 2. 화면 크기 확인

현재 MacBook Air M5 13-inch는 Apple 공식 `2560x1664`, `224ppi` 사양으로 계산한 `290.286 x 188.686mm`를 사용합니다. 실제 측정값을 사용하고 싶을 때만 발광 영역을 베젤을 제외하고 다시 측정합니다.

```yaml
screen:
  width_pixel: 1470
  height_pixel: 956
  width_mm: 290.286
  height_mm: 188.686
  verification_source: apple_official_224ppi
```

화면 픽셀 좌표는 dot test의 캔버스와 동일한 `1470x956`입니다. 물리 화면과 픽셀 좌표의 종횡비가 2%보다 크게 다르면 실행 초기에 오류가 발생합니다.

설정 수정 후 다음 파일을 생성합니다.

```bash
make geometry-screen
```

출력:

```text
data/calibration/setups/macbook_air_m5_13_iphone16/screenSize.mat
```

## 3. 웹캠 내부 보정

웹캠과 phonecam 역할이 `configs/capture.yaml`에서 올바른지 먼저 확인합니다. 체커보드 전체가 영상에 들어오게 한 뒤 실행합니다.

```bash
make geometry-intrinsics CAMERA=webcam
```

화면 조작:

- 초록색 `DETECTED`: 코너 검출 성공
- 미리보기 창 왼쪽 클릭, `Space` 또는 `Enter`: 현재 프레임 저장
- `Q` 또는 `Esc`: 촬영 종료

키보드 포커스 문제로 입력이 되지 않으면 자동 촬영을 사용합니다. 첫 검출을 저장한
뒤 체커보드 위치가 충분히 달라지고 새 위치에서 안정됐을 때만 다음 사진을
저장하므로, 매 촬영 후 위치·거리·기울기를 바꾸고 잠시 멈춥니다. 계산 시 흔들린
단일 이상치는 최소 촬영 수를 유지하는 범위에서 제외하며 `result.json`에 남깁니다.

```bash
make geometry-intrinsics CAMERA=phonecam AUTO_CAPTURE=1
```

20장을 권장하고 최소 12장이 필요합니다. 같은 위치에서 20장을 찍지 말고 다음 구도를 골고루 포함합니다.

- 화면 중앙, 왼쪽, 오른쪽, 위, 아래
- 가까운 거리와 먼 거리
- 상하·좌우로 기울인 각도
- 체커보드 전체 코너가 선명하게 보이는 장면

RMS 재투영 오차가 설정의 `max_rms_error_px`보다 크면 `Camera.mat`을 만들지 않습니다.

완성된 두 `Camera.mat`은 실제 수집 시작 시 각 참가자 `calibration/` 폴더에 자동
복사됩니다. B 전처리는 보정 당시 해상도와 원본 MP4 해상도가 정확히 같을 때만
OpenCV 왜곡 보정을 적용합니다.

## 4. iPhone 내부 보정

iPhone을 실제 수집에 사용할 렌즈·방향·해상도·연결 방식으로 놓고 실행합니다.

```bash
make geometry-intrinsics CAMERA=phonecam
```

웹캠 값을 phonecam에 복사하면 안 됩니다. 두 카메라는 각각 자신의 `Camera.mat`을 생성해야 합니다.

## 5. 결과 확인

```bash
make geometry-inspect
```

필수 최종 파일:

```text
screenSize.mat
webcam/Camera.mat
webcam/monitorPose.mat
phonecam/Camera.mat
phonecam/monitorPose.mat
```

`stereoCalibration.mat`은 실제 3D 삼각측량을 사용할 때 필요한 선택 파일입니다.

## 6. 아직 남은 monitor pose

`monitorPose.mat`은 체커보드를 손에 들고 찍는 내부 보정과 다른 작업입니다. 화면 평면이 카메라 좌표계에서 어디에 있는지를 계산해야 합니다.

MPIIGaze 방식처럼 카메라가 자기 뒤쪽 화면을 직접 볼 수 없는 배치에서는 화면에 체커보드를 표시하고 **평면 거울에 비친 화면을 촬영하는 외부 보정**이 필요합니다. 웹캠과 참가자 쪽을 바라보는 iPhone 모두 동일한 문제가 있습니다. 거울 없이 임의의 회전·이동값을 생성하면 3D 정답 좌표가 틀어지므로 이 도구는 가짜 `monitorPose.mat`을 만들지 않습니다.

현재 단계에서는 체커보드 출력, `screenSize.mat`, 두 카메라의 `Camera.mat`까지 진행합니다. monitor pose 도구는 평면 거울 크기와 설치 방법을 정한 뒤 별도 단계로 추가합니다.
