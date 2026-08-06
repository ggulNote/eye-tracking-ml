# Branch & Collaboration Guide

## 브랜치 역할

- `main`: 검증이 끝난 배포·공유 기준 브랜치
  - 기능 개발 커밋을 직접 push하지 않습니다.
  - Pull Request와 필수 검증을 통과한 변경만 병합합니다.
- `develop`: 여러 기능을 통합하는 개발 브랜치
  - 팀에서 `develop`을 운영하기 시작하면 일반 기능 PR의 기본 대상입니다.
  - 아직 `develop`이 없는 초기 저장소에서는 첫 기능 PR을 `main`으로 보낼 수 있습니다.
- `feat/*`, `fix/*`, `docs/*`, `chore/*`: 작업 브랜치
  - `feat/local-data-pipeline`
  - `fix/manifest-validation`
  - `docs/model-contract`

## 개발 환경

```bash
make setup
make setup-video
cp .env.example .env
```

실제 영상과 processed 데이터는 저장소 밖의 `GGULNOTE_DATA_ROOT`에 둡니다. `.env`, 얼굴 영상, MLflow DB, model output은 commit하지 않습니다.

## 작업 순서

```bash
git switch main
git pull --ff-only origin main
git switch -c feat/<description>
```

코드 변경 후 다음 검증을 실행합니다.

```bash
make validate
make test
make smoke
```

실제 manifest 전처리 계약을 변경했다면 로컬 데이터로 다음 명령도 확인합니다.

```bash
make preprocess CONFIG=configs/manifest-local.example.yaml
```

## 커밋 메시지

`type(scope): description` 형식을 사용합니다.

- `feat(data): add versioned local processed dataset`
- `fix(manifest): validate synchronized frame labels`
- `docs(pipeline): document BlazeGaze input contract`
- `test(data): cover processed dataset hash stability`
- `chore(deps): update MLflow runtime pin`

하나의 commit에는 하나의 논리적 변경을 담습니다. 실제 데이터, 실험 output 또는 unrelated 파일을 함께 stage하지 않습니다.

## Python 의존성

- `pyproject.toml`: 패키지 metadata와 지원 의존성 범위
- `requirements.txt`: 검증된 runtime 고정 버전
- `requirements-dev.txt`: runtime + test 도구
- `requirements-video.txt`: runtime + OpenCV 영상 adapter

의존성을 추가하면 사용 목적에 맞는 requirements 파일과 `pyproject.toml`을 함께 갱신하고 `make setup` 또는 `make setup-video`로 재검증합니다.

## Pull Request 기준

PR 본문에는 다음 내용을 포함합니다.

- 변경 내용과 변경 이유
- 사용자·개발자에게 미치는 영향
- 입출력 shape 또는 config 변경 여부
- dataset/preprocessor/model version 변경 여부
- 실행한 검증 명령과 결과

최소 체크리스트:

- [ ] `make validate`
- [ ] `make test`
- [ ] `make smoke`
- [ ] 실제 영상과 `.env`가 포함되지 않음
- [ ] manifest/config/schema 변경이 문서에 반영됨
- [ ] MLflow에 원본 영상이나 processed tensor를 업로드하지 않음

## 데이터·개인정보 규칙

- participant ID는 익명화합니다.
- `*.mp4`, `*.mov`, `*.avi`, `*.mkv`를 commit하지 않습니다.
- 원본·processed 데이터는 `GGULNOTE_DATA_ROOT`에만 저장합니다.
- MLflow에는 config, hash, 통계, metric, schema, model만 기록합니다.
- 데이터 계약을 바꾸면 기존 dataset version을 덮어쓰지 말고 새 version을 생성합니다.
