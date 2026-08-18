SHELL := /bin/bash
.DEFAULT_GOAL := help

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
export PYTHONPATH := $(PROJECT_ROOT)/src$(if $(PYTHONPATH),:$(PYTHONPATH))
PYTHON ?= python3.12
VENV ?= $(PROJECT_ROOT)/.venv
VENV_PYTHON := $(VENV)/bin/python
VENV_MLFLOW := $(VENV)/bin/mlflow
CONFIG ?= $(PROJECT_ROOT)/configs/config.yaml
PROFILE ?=
PROFILES ?=
DUAL_VIEW_MANIFEST ?= $(PROJECT_ROOT)/data/dual_view_manifest.csv
GAZE_OUTPUT_ROOT ?= $(PROJECT_ROOT)/outputs
MLFLOW_DB ?= $(PROJECT_ROOT)/mlflow.db
MLFLOW_TRACKING_URI ?= sqlite:///$(MLFLOW_DB)
MLFLOW_PORT ?= 5000
DUAL_DEMO_ROOT ?= $(PROJECT_ROOT)/.demo/dual_view_training
DUAL_DEMO_DB ?= $(DUAL_DEMO_ROOT)/mlflow-training-demo.db
DUAL_DEMO_PROFILE := $(PROJECT_ROOT)/configs/profiles/demo_dual_view_training.yaml
MEASURED_DATA_ROOT ?= $(PROJECT_ROOT)/../project_data/data
MEASURED_RUN_ROOT ?= $(PROJECT_ROOT)/.demo/measured_head_down_neutral
MEASURED_DATA_MANIFEST ?= $(MEASURED_RUN_ROOT)/manifest.csv
MEASURED_OUTPUT ?= $(MEASURED_RUN_ROOT)/outputs
MEASURED_DB ?= $(MEASURED_RUN_ROOT)/mlflow.db
MEASURED_PREVIEW_OUTPUT ?= $(PROJECT_ROOT)/outputs/measured_preprocessing_preview
MEASURED_DATA_PROFILE := $(PROJECT_ROOT)/configs/profiles/measured_head_down_neutral.yaml
SIDE_ANNOTATIONS ?= $(MEASURED_DATA_ROOT)/side_annotations.csv
FRONT_PREPROCESS_PROFILE := $(PROJECT_ROOT)/configs/profiles/blazegaze.yaml
FRONT_MODEL_PROFILE := $(PROJECT_ROOT)/configs/models/front_webeyetrack.yaml
SIDE_PREPROCESS_PROFILE := $(PROJECT_ROOT)/configs/profiles/side_profile_90.yaml
SIDE_ROI_PROFILE := $(PROJECT_ROOT)/configs/profiles/side_roi_only.yaml
SIDE_MODEL_PROFILE ?= $(PROJECT_ROOT)/configs/models/side_mobilenet_v4.yaml
WEBEYETRACK_ASSET_DIR ?= $(PROJECT_ROOT)/models
MEDIAPIPE_FACE_MODEL ?= $(WEBEYETRACK_ASSET_DIR)/face_landmarker_v2_with_blendshapes.task
WEBEYETRACK_WEIGHTS ?= $(WEBEYETRACK_ASSET_DIR)/blazegaze_mpiifacegaze.keras
CHECKPOINT ?=
EVAL_SPLIT ?= test
OVERRIDES ?=
PROFILE_ARG = $(if $(strip $(PROFILE)),--profile "$(PROFILE)",) $(foreach item,$(PROFILES),--profile "$(item)")
CHECKPOINT_ARG = $(if $(strip $(CHECKPOINT)),--checkpoint "$(CHECKPOINT)",)

.PHONY: help paths show-paths setup setup-dev require-venv require-data \
	check-setup validate-config validate prepare train evaluate mlflow-check check-mlflow \
	mlflow-ui demo-dual-train \
	measured-manifest measured-prepare measured-preview measured-train measured-evaluate \
	measured-mlflow-ui \
	webeyetrack-assets check-webeyetrack-assets \
	test lint format-check check clean-cache

help:
	@echo "주요 명령"
	@echo "  make paths          절대경로와 현재 설정 확인"
	@echo "  make setup          Python 3.12 가상환경과 runtime 의존성 설치"
	@echo "  make setup-dev      runtime + 개발/테스트 의존성 설치"
	@echo "  make check-setup    Python 3.12 및 전체 runtime import 확인"
	@echo "  make validate       실제 데이터 경로까지 config 검증"
	@echo "  make prepare        manifest/split 생성 및 MLflow 기록"
	@echo "  make train          모델 학습, validation, checkpoint와 MLflow 기록"
	@echo "  make evaluate       CHECKPOINT 또는 config 기본 checkpoint의 validation/test 평가"
	@echo "  make mlflow-check   SQLite DB와 FINISHED run 확인"
	@echo "  make mlflow-ui      http://127.0.0.1:5000 UI 실행"
	@echo "  make demo-dual-train 합성 Front/Side DB와 내장 fallback 모델의 Y축 fusion smoke test"
	@echo "  make measured-manifest 이름별 DB의 head_down·neutral canonical manifest 생성"
	@echo "  make measured-prepare 피험자 단위 train/validation/test split과 pairing 검증"
	@echo "  make measured-preview inputs.csv 기반 실제 Front 전처리와 Side 상태 plot"
	@echo "  make measured-train Side annotation 준비 후 선택한 Side Encoder로 전체 학습"
	@echo "  make measured-evaluate CHECKPOINT=... 같은 Config로 validation/test 평가"
	@echo "  make measured-mlflow-ui 측정 DB 전용 MLflow UI 실행"
	@echo "  make webeyetrack-assets 공식 MediaPipe/BlazeGaze asset 다운로드+SHA 검증"
	@echo "  make clean-cache    Python/test/lint cache와 egg-info 제거"
	@echo "  make check          config/unit/lint/format 전체 검사"
	@echo
	@echo "예: DUAL_VIEW_MANIFEST='/absolute/manifest.csv' make prepare GAZE_DATA_ROOT='/absolute/dual_view'"

paths:
	@echo "PROJECT_ROOT=$(PROJECT_ROOT)"
	@echo "VENV=$(VENV)"
	@echo "BOOTSTRAP_PYTHON=$(PYTHON)"
	@echo "VENV_PYTHON=$(VENV_PYTHON)"
	@echo "CONFIG=$(CONFIG)"
	@echo "PROFILE=$(if $(strip $(PROFILE)),$(PROFILE),<none>)"
	@echo "PROFILES=$(if $(strip $(PROFILES)),$(PROFILES),<none>)"
	@echo "GAZE_DATA_ROOT=$(if $(strip $(GAZE_DATA_ROOT)),$(GAZE_DATA_ROOT),<not-set>)"
	@echo "DUAL_VIEW_MANIFEST=$(DUAL_VIEW_MANIFEST)"
	@echo "GAZE_OUTPUT_ROOT=$(GAZE_OUTPUT_ROOT)"
	@echo "MLFLOW_DB=$(MLFLOW_DB)"
	@echo "MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI)"
	@echo "DUAL_DEMO_ROOT=$(DUAL_DEMO_ROOT)"
	@echo "DUAL_DEMO_DB=$(DUAL_DEMO_DB)"
	@echo "MEASURED_DATA_ROOT=$(MEASURED_DATA_ROOT)"
	@echo "MEASURED_DATA_MANIFEST=$(MEASURED_DATA_MANIFEST)"
	@echo "MEASURED_OUTPUT=$(MEASURED_OUTPUT)"
	@echo "MEASURED_DB=$(MEASURED_DB)"
	@echo "MEASURED_PREVIEW_OUTPUT=$(MEASURED_PREVIEW_OUTPUT)"
	@echo "SIDE_ANNOTATIONS=$(SIDE_ANNOTATIONS)"
	@echo "SIDE_MODEL_PROFILE=$(SIDE_MODEL_PROFILE)"
	@echo "MEDIAPIPE_FACE_MODEL=$(MEDIAPIPE_FACE_MODEL)"
	@echo "WEBEYETRACK_WEIGHTS=$(WEBEYETRACK_WEIGHTS)"
	@echo "CHECKPOINT=$(if $(strip $(CHECKPOINT)),$(CHECKPOINT),<config-default>)"
	@echo "EVAL_SPLIT=$(EVAL_SPLIT)"

show-paths: paths

setup:
	@if ! command -v "$(PYTHON)" >/dev/null 2>&1; then echo "Python 실행 파일이 없습니다: $(PYTHON)"; echo "Python 3.12 절대경로를 지정하세요: make setup PYTHON='/absolute/path/to/python3.12'"; exit 2; fi
	@if ! "$(PYTHON)" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then echo "Python 3.12가 필요합니다. 현재 버전: $$("$(PYTHON)" --version 2>&1)"; exit 2; fi
	@if [[ ! -x "$(VENV_PYTHON)" ]]; then "$(PYTHON)" -m venv "$(VENV)"; fi
	@"$(VENV_PYTHON)" -m pip install --upgrade pip
	@"$(VENV_PYTHON)" -m pip install -r "$(PROJECT_ROOT)/requirements.txt"
	@"$(VENV_PYTHON)" -m pip install -e "$(PROJECT_ROOT)"
	@$(MAKE) --no-print-directory check-setup VENV="$(VENV)"

setup-dev: setup
	@"$(VENV_PYTHON)" -m pip install -r "$(PROJECT_ROOT)/requirements-dev.txt"

require-venv:
	@if [[ ! -x "$(VENV_PYTHON)" ]]; then echo "가상환경이 없습니다: $(VENV)"; echo "먼저 make setup을 실행하세요."; exit 2; fi

require-data:
	@if [[ -z "$(strip $(GAZE_DATA_ROOT))" ]]; then echo "GAZE_DATA_ROOT가 필요합니다."; echo "예: DUAL_VIEW_MANIFEST='/absolute/manifest.csv' make prepare GAZE_DATA_ROOT='/absolute/dual_view'"; exit 2; fi
	@if [[ ! -d "$(GAZE_DATA_ROOT)" ]]; then echo "데이터 폴더가 없습니다: $(GAZE_DATA_ROOT)"; exit 2; fi
	@if [[ ! -f "$(DUAL_VIEW_MANIFEST)" ]]; then echo "DB manifest가 없습니다: $(DUAL_VIEW_MANIFEST)"; exit 2; fi

check-setup: require-venv
	@"$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/check_environment.py"
	@MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" "$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/check_mlflow.py" --imports-only

validate-config: require-venv
	@"$(VENV_PYTHON)" -m gaze_pipeline validate-config --config "$(CONFIG)" $(PROFILE_ARG) --skip-path-checks $(OVERRIDES)

validate: require-venv require-data
	@GAZE_DATA_ROOT="$(GAZE_DATA_ROOT)" DUAL_VIEW_MANIFEST="$(DUAL_VIEW_MANIFEST)" GAZE_OUTPUT_ROOT="$(GAZE_OUTPUT_ROOT)" MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" "$(VENV_PYTHON)" -m gaze_pipeline validate-config --config "$(CONFIG)" $(PROFILE_ARG) $(OVERRIDES)

prepare: require-venv require-data
	@GAZE_DATA_ROOT="$(GAZE_DATA_ROOT)" DUAL_VIEW_MANIFEST="$(DUAL_VIEW_MANIFEST)" GAZE_OUTPUT_ROOT="$(GAZE_OUTPUT_ROOT)" MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" "$(VENV_PYTHON)" -m gaze_pipeline prepare --config "$(CONFIG)" $(PROFILE_ARG) $(OVERRIDES)

train: require-venv require-data
	@GAZE_DATA_ROOT="$(GAZE_DATA_ROOT)" DUAL_VIEW_MANIFEST="$(DUAL_VIEW_MANIFEST)" GAZE_OUTPUT_ROOT="$(GAZE_OUTPUT_ROOT)" MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" "$(VENV_PYTHON)" -m gaze_pipeline train --config "$(CONFIG)" $(PROFILE_ARG) $(OVERRIDES)

evaluate: require-venv require-data
	@GAZE_DATA_ROOT="$(GAZE_DATA_ROOT)" DUAL_VIEW_MANIFEST="$(DUAL_VIEW_MANIFEST)" GAZE_OUTPUT_ROOT="$(GAZE_OUTPUT_ROOT)" MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" "$(VENV_PYTHON)" -m gaze_pipeline evaluate --config "$(CONFIG)" $(PROFILE_ARG) $(CHECKPOINT_ARG) --split "$(EVAL_SPLIT)" $(OVERRIDES)

mlflow-check: require-venv
	@MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" "$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/check_mlflow.py" --require-db

check-mlflow: mlflow-check

mlflow-ui: require-venv
	@echo "MLflow UI: http://127.0.0.1:$(MLFLOW_PORT)"
	@"$(VENV_MLFLOW)" ui --backend-store-uri "$(MLFLOW_TRACKING_URI)" --port "$(MLFLOW_PORT)"

demo-dual-train: require-venv
	@"$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/create_dual_view_training_demo.py" --output-root "$(DUAL_DEMO_ROOT)"
	@DUAL_DEMO_MANIFEST="$(DUAL_DEMO_ROOT)/manifest.csv" GAZE_DATA_ROOT="$(DUAL_DEMO_ROOT)" GAZE_OUTPUT_ROOT="$(DUAL_DEMO_ROOT)/outputs" MLFLOW_TRACKING_URI="sqlite:///$(DUAL_DEMO_DB)" "$(VENV_PYTHON)" -m gaze_pipeline train --config "$(CONFIG)" --profile "$(DUAL_DEMO_PROFILE)" $(OVERRIDES)
	@MLFLOW_TRACKING_URI="sqlite:///$(DUAL_DEMO_DB)" "$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/check_mlflow.py" --require-db

measured-manifest: require-venv
	@if [[ ! -d "$(MEASURED_DATA_ROOT)" ]]; then echo "측정 DB 폴더가 없습니다: $(MEASURED_DATA_ROOT)"; exit 2; fi
	@if [[ ! -f "$(SIDE_ANNOTATIONS)" ]]; then echo "Side bbox annotation CSV가 없습니다: $(SIDE_ANNOTATIONS)"; echo "필수 열: sample_id, visible_eye, visible_eye_bbox_xyxy, eye_annotation_valid"; exit 2; fi
	@mkdir -p "$(MEASURED_RUN_ROOT)"
	@"$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/create_measured_data_manifest.py" \
		--source-root "$(MEASURED_DATA_ROOT)" \
		--output-manifest "$(MEASURED_DATA_MANIFEST)" \
		--sessions head_down neutral --side-annotations "$(SIDE_ANNOTATIONS)" \
		--require-front-pose --require-side-annotations --force

measured-prepare: measured-manifest
	@GAZE_DATA_ROOT="$(MEASURED_DATA_ROOT)" MEASURED_DATA_MANIFEST="$(MEASURED_DATA_MANIFEST)" \
		GAZE_OUTPUT_ROOT="$(MEASURED_OUTPUT)" MLFLOW_TRACKING_URI="sqlite:///$(MEASURED_DB)" \
		"$(VENV_PYTHON)" -m gaze_pipeline prepare --config "$(CONFIG)" \
		--profile "$(FRONT_PREPROCESS_PROFILE)" \
		--profile "$(SIDE_PREPROCESS_PROFILE)" \
		--profile "$(SIDE_ROI_PROFILE)" \
		--profile "$(MEASURED_DATA_PROFILE)" \
		--profile "$(FRONT_MODEL_PROFILE)" \
		--profile "$(SIDE_MODEL_PROFILE)"

measured-preview: require-venv check-webeyetrack-assets
	@mkdir -p "$(MEASURED_RUN_ROOT)/matplotlib"
	@KERAS_BACKEND=torch MPLCONFIGDIR="$(MEASURED_RUN_ROOT)/matplotlib" \
		"$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/preview_measured_inputs_preprocessing.py" \
		--data-root "$(MEASURED_DATA_ROOT)" \
		--output-dir "$(MEASURED_PREVIEW_OUTPUT)" \
		--config "$(CONFIG)" \
		--front-profile "$(FRONT_PREPROCESS_PROFILE)" \
		--model-asset "$(MEDIAPIPE_FACE_MODEL)" \
		--side-annotations "$(SIDE_ANNOTATIONS)" \
		--samples 2 --sessions head_down neutral --overwrite

measured-train: require-venv check-webeyetrack-assets
	@if [[ ! -d "$(MEASURED_DATA_ROOT)" ]]; then echo "측정 DB 폴더가 없습니다: $(MEASURED_DATA_ROOT)"; exit 2; fi
	@mkdir -p "$(MEASURED_RUN_ROOT)"
	@"$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/create_measured_data_manifest.py" \
		--source-root "$(MEASURED_DATA_ROOT)" \
		--output-manifest "$(MEASURED_DATA_MANIFEST)" \
		--sessions head_down neutral --side-annotations "$(SIDE_ANNOTATIONS)" \
		--require-front-pose --require-side-annotations --force
	@GAZE_DATA_ROOT="$(MEASURED_DATA_ROOT)" MEASURED_DATA_MANIFEST="$(MEASURED_DATA_MANIFEST)" \
		GAZE_OUTPUT_ROOT="$(MEASURED_OUTPUT)" MLFLOW_TRACKING_URI="sqlite:///$(MEASURED_DB)" \
		MEDIAPIPE_FACE_MODEL="$(MEDIAPIPE_FACE_MODEL)" \
		WEBEYETRACK_WEIGHTS="$(WEBEYETRACK_WEIGHTS)" KERAS_BACKEND=torch \
		KERAS_TORCH_DEVICE=cpu "$(VENV_PYTHON)" -m gaze_pipeline train --config "$(CONFIG)" \
		--profile "$(FRONT_PREPROCESS_PROFILE)" \
		--profile "$(SIDE_PREPROCESS_PROFILE)" \
		--profile "$(SIDE_ROI_PROFILE)" \
		--profile "$(MEASURED_DATA_PROFILE)" \
		--profile "$(FRONT_MODEL_PROFILE)" \
		--profile "$(SIDE_MODEL_PROFILE)" $(OVERRIDES)

measured-evaluate: require-venv check-webeyetrack-assets
	@if [[ ! -f "$(MEASURED_DATA_MANIFEST)" ]]; then echo "먼저 make measured-manifest를 실행하세요: $(MEASURED_DATA_MANIFEST)"; exit 2; fi
	@if [[ -z "$(strip $(CHECKPOINT))" ]]; then echo "CHECKPOINT 절대경로가 필요합니다."; exit 2; fi
	@if [[ ! -f "$(CHECKPOINT)" ]]; then echo "checkpoint가 없습니다: $(CHECKPOINT)"; exit 2; fi
	@GAZE_DATA_ROOT="$(MEASURED_DATA_ROOT)" MEASURED_DATA_MANIFEST="$(MEASURED_DATA_MANIFEST)" \
		GAZE_OUTPUT_ROOT="$(MEASURED_OUTPUT)" MLFLOW_TRACKING_URI="sqlite:///$(MEASURED_DB)" \
		MEDIAPIPE_FACE_MODEL="$(MEDIAPIPE_FACE_MODEL)" \
		WEBEYETRACK_WEIGHTS="$(WEBEYETRACK_WEIGHTS)" KERAS_BACKEND=torch \
		KERAS_TORCH_DEVICE=cpu "$(VENV_PYTHON)" -m gaze_pipeline evaluate --config "$(CONFIG)" \
		--profile "$(FRONT_PREPROCESS_PROFILE)" \
		--profile "$(SIDE_PREPROCESS_PROFILE)" \
		--profile "$(SIDE_ROI_PROFILE)" \
		--profile "$(MEASURED_DATA_PROFILE)" \
		--profile "$(FRONT_MODEL_PROFILE)" \
		--profile "$(SIDE_MODEL_PROFILE)" \
		--checkpoint "$(CHECKPOINT)" --split "$(EVAL_SPLIT)" $(OVERRIDES)

measured-mlflow-ui: require-venv
	@echo "Measured-data MLflow UI: http://127.0.0.1:$(MLFLOW_PORT)"
	@"$(VENV_MLFLOW)" ui --backend-store-uri "sqlite:///$(MEASURED_DB)" --port "$(MLFLOW_PORT)"

webeyetrack-assets: require-venv
	@"$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/fetch_webeyetrack_assets.py" --output-dir "$(WEBEYETRACK_ASSET_DIR)"

check-webeyetrack-assets: require-venv
	@"$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/fetch_webeyetrack_assets.py" --output-dir "$(WEBEYETRACK_ASSET_DIR)" --check-only

test: require-venv
	@"$(VENV_PYTHON)" -m pytest -q

lint: require-venv
	@"$(VENV)/bin/ruff" check --no-cache .

format-check: require-venv
	@"$(VENV)/bin/ruff" format --check .

check: check-setup validate-config test lint format-check

clean-cache:
	@rm -rf "$(PROJECT_ROOT)/.pytest_cache" "$(PROJECT_ROOT)/.ruff_cache" "$(PROJECT_ROOT)/src/gaze_pipeline.egg-info"
	@find "$(PROJECT_ROOT)/src" "$(PROJECT_ROOT)/scripts" "$(PROJECT_ROOT)/tests" -type d -name __pycache__ -prune -exec rm -rf {} +
	@find "$(PROJECT_ROOT)/src" "$(PROJECT_ROOT)/scripts" "$(PROJECT_ROOT)/tests" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	@find "$(PROJECT_ROOT)" -maxdepth 3 -type f -name .DS_Store -delete
	@echo "Reproducible caches removed. .venv/outputs와 MLflow SQLite DB는 별도 관리합니다."
