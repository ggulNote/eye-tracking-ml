# CSV 스키마 전체 목록

현재 참가자 데이터 스키마는 `csv_schema_version=4`입니다. 참가자 이름 아래
`neutral`, `head_up`, `head_down` 촬영을 분리하고 주요 CSV에 `head_pose`를
기록합니다. 학습용 정답표는 `labels/labels.csv` 하나입니다.

모든 CSV의 정확한 위치, 모든 열의 의미, Calibration과 feature_maps 설명은
[참가자 데이터 폴더와 CSV 설명](participant-data-layout.md)에 정리되어 있습니다.

핵심 구분은 다음과 같습니다.

| 파일 | 역할 | 모델 학습에 직접 사용 |
|---|---|---|
| `labels/labels.csv` | 점당 최적 web/phone 이미지와 정답 좌표를 합친 단일 라벨 | 예 |
| `metadata/frame_log.csv` | 동기화를 위한 촬영 중 전체 프레임 기록 | 아니요 |
| `metadata/protocol.csv` | 계획된 점 순서와 좌표 | 아니요 |
| `video/web/timestamps.csv` | 정면 MP4 프레임과 Unix ns 시각 매핑 | 동기화에 사용 |
| `video/phone/timestamps.csv` | 측면 MP4 프레임과 Unix ns 시각 매핑 | 동기화에 사용 |
| `feature_maps/synchronized.csv` | 레이턴시 보정 후 같은 순간의 프레임과 정답 | 전처리에 사용 |
| `feature_maps/web/features.csv` | 정면 MediaPipe 8차원 특징과 유효성 | 예 |
| `feature_maps/phone/features.csv` | 측면 특징 계약; 현재 MediaPipe 미사용으로 헤더만 생성 | 후속 측면 모델용 |
| `feature_maps/training.csv` | 학습 이미지 manifest | 예 |
| `feature_maps/evaluation.csv` | 최종 평가 이미지 manifest | 평가 전용 |

모든 일반 timestamp는 별도 표기가 없으면 Unix nanosecond 정수입니다.
`elapsed_ms`, `*_latency_ms`, `corrected_time_diff_ms`만 millisecond입니다.
