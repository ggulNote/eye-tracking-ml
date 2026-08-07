# 다중 카메라 캡처

`ggulnote_ml.capture`는 노트북 웹캠과 운영체제에 등록된 휴대폰 카메라를 동시에 읽고 카메라별 원본 영상과 프레임 시각을 저장합니다.

## 설치

```bash
make setup-capture
```

`opencv-python-headless`는 학습 서버의 영상 디코딩용이고 `opencv-python`은 로컬 GUI 미리보기용입니다. 캡처 환경에서는 `requirements-capture.txt`를 사용합니다.

## 실행

다른 카메라 앱을 종료한 뒤 장치 번호를 검색합니다.

```bash
make list-cameras
```

`configs/capture.yaml`을 환경에 맞게 수정하고 실행합니다.

```bash
make capture
```

macOS에서 VS Code 또는 Terminal의 카메라 접근 권한을 허용해야 합니다. iPhone은 동일한 Apple 계정, Wi-Fi/Bluetooth 활성화, Mac 신뢰 및 연속성 카메라 활성화가 필요합니다.

## 설정 계약

각 `cameras` 항목은 다음 값을 모두 가져야 합니다.

| 필드 | 형식 | 설명 |
|---|---|---|
| `name` | 문자열 | 창과 파일명에 사용하는 고유 이름 |
| `device_index` | 0 이상 정수 | 운영체제 카메라 번호 |
| `backend` | 문자열 | `auto`, `avfoundation`, `v4l2`, `dshow` |
| `width`, `height` | 양의 정수 | 후속 모듈에 전달할 고정 프레임 크기 |
| `fps` | 양수 | 요청 및 저장 FPS |
| `warmup_frames` | 0 이상 정수 | 세션 시작 전에 버릴 프레임 수 |
| `mirror` | boolean | 미리보기만 좌우 반전 |

## 프레임 출력 계약

```text
frame_index:     int, 카메라별 0부터 증가
captured_at_ms:  float, 공통 세션 시작 기준 단조 증가 시간(ms)
frame:           numpy.ndarray, uint8, (height, width, 3), BGR
```

카메라는 같은 세션 시계를 공유합니다. 각 프레임은 실제 읽기가 끝난 직후 시각을 기록하므로 CSV 사이의 차이를 이후 레이턴시 분석에 사용할 수 있습니다.

## 저장 결과

```text
data/raw/captures/session_YYYYMMDD_HHMMSS_macbook.mp4
data/raw/captures/session_YYYYMMDD_HHMMSS_macbook_timestamps.csv
data/raw/captures/session_YYYYMMDD_HHMMSS_iphone.mp4
data/raw/captures/session_YYYYMMDD_HHMMSS_iphone_timestamps.csv
```

영상은 저장소에 커밋하지 않습니다. 캡처 설정과 세션 메타데이터를 MLflow에 연결하는 작업은 캘리브레이션·전처리 단계에서 추가합니다.
