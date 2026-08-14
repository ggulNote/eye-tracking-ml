# Contributing

이 문서는 **파이프라인 코드를 수정하거나 기능을 추가하는 개발자**를 위한 안내입니다. 현재 구현된 설치·data preparation·전처리·학습 사용법은 [README.md](README.md)를 참고하세요.

현재 config/data package와 generic PyTorch trainer/evaluator가 구현되어 있습니다. 기능을
추가할 때 config와 문서에 정의된 데이터·모델·artifact 계약을 유지해야 합니다.

## 개발 환경

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

의존성을 변경하면 `requirements.txt` 또는 `requirements-dev.txt`를 함께 갱신하고 Python 3.12 clean environment에서 resolver와 최소 smoke test를 확인합니다.

## 브랜치와 커밋

브랜치 이름은 변경 목적이 보이도록 작성합니다.

```text
feature/<name>     새 기능
fix/<name>         버그 수정
docs/<name>        문서 변경
experiment/<name> 실험용 config/profile
```

커밋은 한 가지 논리 변경만 포함하고 명령형 제목을 사용합니다.

```text
feat: add dual-view CSV manifest field
fix: prevent subject leakage in grouped split
docs: explain BlazeGaze input contract
test: add checkpoint round-trip test
```

얼굴 이미지, dataset, credential, checkpoint 또는 대형 binary를 commit에 포함하지 않습니다.

## 코드 스타일과 기본 검사

제출 전 다음 명령을 실행합니다.

```bash
ruff check .
ruff format --check .
pytest --cov
```

자동 formatting이 필요하면 다음을 사용합니다.

```bash
ruff format .
```

공개 함수와 adapter에는 type hint를 사용하고, shape·dtype·단위가 중요한 tensor는 docstring 또는 contract에 명시합니다. validation/test 코드에는 비결정적 augmentation을 넣지 않습니다.

데이터 contract, config, split, parser, 전처리, Dataset, model runtime, 학습·평가와 tracking
테스트를 모두 통과해야 합니다.

## Config 변경 방법

`configs/config.yaml`은 공통 실행 계약입니다. 필드를 추가하거나 의미를 변경할 때 다음을 함께 수정합니다.

1. config의 기본값과 주석
2. `docs/configuration.md`의 의미·단위·허용값 설명
3. config validation 또는 schema
4. 정상/실패 case 테스트
5. 필요한 model/dataset profile

기존 run을 해석할 수 없게 만드는 변경은 `schema_version`을 올립니다. 모델별 임의 constructor 인자는 `model.<branch>.init_args` 안에만 둡니다.

실행 시에는 base YAML이 아니라 모든 override가 반영된 `resolved_config.yaml`을 저장해야 합니다.

## 새로운 전처리 모듈 추가

1. stage의 이름과 역할을 정합니다.
2. 필요한 input key와 생성할 output key를 정의합니다.
3. `preprocessing.stage_order`에서 실행 위치를 지정합니다.
4. enable flag와 model/dataset별 parameter를 config에 추가합니다.
5. 실패 정책을 `drop`, `fallback`, `mark_invalid` 중에서 명시합니다.
6. train-only augmentation과 deterministic eval transform을 분리합니다.
7. shape/landmark/target 변환 테스트를 작성합니다.

crop, resize, flip처럼 공간 좌표를 바꾸는 전처리는 image뿐 아니라 landmark와 gaze 관련 좌표도 같은 convention에 맞춰 처리해야 합니다.

새 source image의 배경 형태를 가정하지 않습니다. 모델이 요구하는 ROI, mask와 normalization은 branch profile에 명시하고 테스트합니다.

## 새로운 모델과 adapter 추가

새 모델은 가능한 한 pipeline dataset이나 trainer를 수정하지 않고 factory contract로
연결합니다. 기본 호출 signature와 표준 출력이 맞으면 adapter는 생략할 수 있습니다.

1. import 가능한 model factory를 구현합니다.
2. 호출 signature가 다를 때 canonical batch를 model 입력으로 바꾸는 adapter를 구현합니다.
3. 출력 형식이 다를 때 raw model 출력을 표준 output으로 변환합니다.
4. model profile config를 추가합니다.
5. one-batch contract 및 checkpoint round-trip 테스트를 작성합니다.
6. 모델 코드·weight의 출처, commit, 라이선스, SHA-256을 기록합니다.

필수 config 정보는 다음과 같습니다.

```yaml
model:
  front:
    source_dir: /path/to/model
    entrypoint: package.factory:create_model
    adapter_entrypoint: package.adapter:FrontAdapter
    init_args: {}
    input_contract: {}
    output_contract: {}
```

Front의 표준 출력은 다음과 같습니다.

```text
gaze_xy
shape: [B, 2]
dtype: float32
order: (x, y)
coordinate: task.coordinate_system
```

Y축 residual Side 모델의 필수 출력은 `delta_y_side [B,1]`입니다. embedding은 선택이며
불확실성은 좌표나 residual에 섞지 않고 별도 key로 반환합니다.

최소 model 통합 test는 다음을 포함합니다. runtime은 primary/선택 auxiliary tensor의 선언된
key·shape·dtype·finite/value range와 표준 출력 tensor/shape를 검사하지만, 좌표계 의미와
state-dict round trip은 자동으로 증명하지 않습니다.

- synthetic 또는 실제 한 batch forward
- input/output key, shape, dtype 검사
- 좌표계 변환 수치 검사
- `state_dict` 저장 후 strict reload
- CPU `map_location` reload
- 고정 seed eval 결과 확인

## 새로운 loss 또는 metric 추가

- loss는 gradient를 생성하는 학습 목적이고 metric은 평가용이라는 경계를 유지합니다.
- 단위가 있는 metric은 normalized/pixel/cm/degree를 이름에 포함합니다.
- metric 계산 전에 prediction을 clamp하지 않습니다.
- 평균뿐 아니라 subject-macro 집계를 지원합니다.
- 새 selection metric을 추가하면 checkpoint, scheduler, early stopping monitor도 함께 검증합니다.
- 3D angular error는 3D gaze-vector task에서만 사용합니다.

구현마다 수치가 손으로 계산 가능한 작은 unit test를 추가합니다.

## 새로운 fusion 추가

1. fusion이 요구하는 branch output key를 정의합니다.
2. paired sample이 없을 때의 동작을 정합니다.
3. `pair_id`, subject, target 일치 validation을 추가합니다.
4. 필요한 branch/fusion metric namespace와 선택 metric을 정의하고 generic evaluator를
   함께 확장합니다. 현재 evaluator는 final `gaze_xy` metric만 계산합니다.
5. missing branch와 unpaired sample 테스트를 작성합니다.

파일명, 정렬 순서 또는 같은 subject라는 이유만으로 두 image를 pair로 추정하지 않습니다. fusion에는 수집 단계의 명시적 `pair_id`가 필요합니다.

## Split과 재현성

- 기본 split group은 `subject_id`입니다.
- 같은 subject와 같은 `pair_id`는 여러 split에 걸칠 수 없습니다.
- 한 row에서 파생된 face/eye crop도 같은 split을 유지합니다.
- seed와 실제 split manifest를 모두 저장합니다.
- Python/PyTorch/platform 환경과 path-free resolved-config/dataset/split hash를 MLflow와
  checkpoint lineage에 기록합니다.
- raw 얼굴 image를 MLflow artifact로 자동 업로드하지 않습니다.

## Checkpoint 규칙

- 추론용 `best_weights.pt`에는 format version, component state, epoch, metric, 안전한 model
  contract와 path-free lineage hash를 저장합니다.
- 재개용 `last_checkpoint.pt`에는 같은 공통 payload와 config가 허용한
  optimizer/scheduler/RNG state를 저장합니다.
- `final_weights.pt`에는 component state와 선택적인 안전한 model contract/lineage hash를
  저장하며 raw resolved config를 넣지 않습니다.
- `torch.save(model)`이나 `.pkl` 전체 model 저장은 사용하지 않습니다.
- 외부 checkpoint는 출처와 checksum을 확인합니다.
- 가능한 경우 `weights_only=True`, `map_location="cpu"`, strict state-dict load를 사용합니다.

## Pull Request 규칙

PR 설명에는 다음을 포함합니다.

- 변경 목적과 해결하는 문제
- 영향을 받는 config 및 contract
- 실행한 테스트와 결과
- metric 변화가 있다면 비교 기준과 결과
- 호환성 또는 migration 주의사항

제출 전 확인합니다.

- [ ] 변경 범위가 하나의 명확한 목적을 가지는가?
- [ ] config, 문서, 테스트가 함께 갱신되었는가?
- [ ] 입력·출력 shape, dtype, 좌표계, 단위가 명시되었는가?
- [ ] subject/pair leakage가 없는가?
- [ ] validation/test가 deterministic한가?
- [ ] resolved config와 lineage가 MLflow에 남는가?
- [ ] 얼굴 이미지, 실제 식별자, credential, checkpoint가 포함되지 않았는가?
- [ ] 외부 model/dataset의 출처·라이선스·checksum이 기록되었는가?
