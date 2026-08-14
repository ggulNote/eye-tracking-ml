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
DEMO_ROOT ?= $(PROJECT_ROOT)/.demo/two_images
DEMO_DB ?= $(DEMO_ROOT)/mlflow-demo.db
DEMO_PROFILE := $(PROJECT_ROOT)/configs/profiles/demo_two_images.yaml
DUAL_DEMO_ROOT ?= $(PROJECT_ROOT)/.demo/dual_view_training
DUAL_DEMO_DB ?= $(DUAL_DEMO_ROOT)/mlflow-training-demo.db
DUAL_DEMO_PROFILE := $(PROJECT_ROOT)/configs/profiles/demo_dual_view_training.yaml
DATA_VER1_ROOT ?= $(PROJECT_ROOT)/../project_data/data(ver1)
DATA_VER1_SMOKE_ROOT ?= $(PROJECT_ROOT)/.demo/data_ver1_smoke
DATA_VER1_MANIFEST ?= $(DATA_VER1_SMOKE_ROOT)/manifest.csv
DATA_VER1_OUTPUT ?= $(DATA_VER1_SMOKE_ROOT)/outputs
DATA_VER1_DB ?= $(DATA_VER1_SMOKE_ROOT)/mlflow.db
DATA_VER1_PROFILE := $(PROJECT_ROOT)/configs/profiles/data_ver1_smoke.yaml
DATA_VER1_PREVIEW_OUTPUT ?= $(PROJECT_ROOT)/outputs/preprocessing_ver1_preview
EXAMPLE_ROOT ?= $(PROJECT_ROOT)/../project_data/example
PREVIEW_OUTPUT ?= $(PROJECT_ROOT)/outputs/preprocessing_preview
PREVIEW_ONLY ?=
PROFILE90_PREVIEW_OUTPUT ?= $(PROJECT_ROOT)/outputs/profile90_preprocessing
PROFILE90_ANNOTATIONS ?= $(PROJECT_ROOT)/configs/examples/profile90_annotations.yaml
PROFILE90_FRONT_PROFILE ?= $(PROJECT_ROOT)/configs/profiles/blazegaze.yaml
PROFILE90_POSE_GRID_ANNOTATIONS ?= $(PROJECT_ROOT)/configs/examples/profile90_pose_grid_annotations.yaml
PROFILE90_POSE_GRID_OUTPUT ?= $(PROJECT_ROOT)/outputs/profile90_pose_grid
WEBEYETRACK_ASSET_DIR ?= $(PROJECT_ROOT)/models
MEDIAPIPE_FACE_MODEL ?= $(WEBEYETRACK_ASSET_DIR)/face_landmarker_v2_with_blendshapes.task
WEBEYETRACK_WEIGHTS ?= $(WEBEYETRACK_ASSET_DIR)/blazegaze_mpiifacegaze.keras
CHECKPOINT ?=
EVAL_SPLIT ?= test
OVERRIDES ?=
PROFILE_ARG = $(if $(strip $(PROFILE)),--profile "$(PROFILE)",) $(foreach item,$(PROFILES),--profile "$(item)")
PREVIEW_ONLY_ARG = $(if $(strip $(PREVIEW_ONLY)),--only-profile "$(PREVIEW_ONLY)",)
CHECKPOINT_ARG = $(if $(strip $(CHECKPOINT)),--checkpoint "$(CHECKPOINT)",)

.PHONY: help paths show-paths setup setup-dev require-venv require-data \
	check-setup validate-config validate prepare train evaluate mlflow-check check-mlflow \
	mlflow-ui demo-two-images demo-dual-train demo-mlflow-check demo-mlflow-ui \
	data-ver1-manifest data-ver1-prepare data-ver1-preview data-ver1-smoke \
	webeyetrack-assets check-webeyetrack-assets preprocess-preview \
	preprocess-profile90-preview preprocess-profile90-pose-grid \
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
	@echo "  make demo-two-images 합성 이미지 2장으로 별도 MLflow DB smoke test"
	@echo "  make demo-dual-train 합성 Front/Side DB와 내장 fallback 모델의 Y축 fusion smoke test"
	@echo "  make data-ver1-preview p00/p03 실제 전처리 단계별 이미지 생성"
	@echo "  make data-ver1-smoke 실제 WebEyeTrack Front + 간단한 Side의 학습·평가·MLflow 검사"
	@echo "  make webeyetrack-assets 공식 MediaPipe/BlazeGaze asset 다운로드+SHA 검증"
	@echo "  make preprocess-preview 실제 BlazeGaze/front·side 단계별 전처리 plot"
	@echo "  make preprocess-profile90-preview front 2장 + 등록된 strict 90° side 전체 plot"
	@echo "  make preprocess-profile90-pose-grid strict side head×eye 3×3 vector plot"
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
	@echo "DEMO_ROOT=$(DEMO_ROOT)"
	@echo "DEMO_DB=$(DEMO_DB)"
	@echo "DUAL_DEMO_ROOT=$(DUAL_DEMO_ROOT)"
	@echo "DUAL_DEMO_DB=$(DUAL_DEMO_DB)"
	@echo "DATA_VER1_ROOT=$(DATA_VER1_ROOT)"
	@echo "DATA_VER1_MANIFEST=$(DATA_VER1_MANIFEST)"
	@echo "DATA_VER1_OUTPUT=$(DATA_VER1_OUTPUT)"
	@echo "DATA_VER1_DB=$(DATA_VER1_DB)"
	@echo "DATA_VER1_PREVIEW_OUTPUT=$(DATA_VER1_PREVIEW_OUTPUT)"
	@echo "EXAMPLE_ROOT=$(EXAMPLE_ROOT)"
	@echo "PREVIEW_OUTPUT=$(PREVIEW_OUTPUT)"
	@echo "PROFILE90_PREVIEW_OUTPUT=$(PROFILE90_PREVIEW_OUTPUT)"
	@echo "PROFILE90_ANNOTATIONS=$(PROFILE90_ANNOTATIONS)"
	@echo "PROFILE90_FRONT_PROFILE=$(PROFILE90_FRONT_PROFILE)"
	@echo "PROFILE90_POSE_GRID_ANNOTATIONS=$(PROFILE90_POSE_GRID_ANNOTATIONS)"
	@echo "PROFILE90_POSE_GRID_OUTPUT=$(PROFILE90_POSE_GRID_OUTPUT)"
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

demo-two-images: require-venv
	@"$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/create_two_image_demo.py" --output-root "$(DEMO_ROOT)"
	@cd "$(DEMO_ROOT)" && GAZE_DATA_ROOT="$(DEMO_ROOT)" GAZE_OUTPUT_ROOT="$(DEMO_ROOT)/outputs" DEMO_MANIFEST="$(DEMO_ROOT)/manifest.csv" MLFLOW_TRACKING_URI="sqlite:///$(DEMO_DB)" "$(VENV_PYTHON)" -m gaze_pipeline prepare --config "$(CONFIG)" --profile "$(DEMO_PROFILE)"
	@$(MAKE) --no-print-directory demo-mlflow-check VENV="$(VENV)" DEMO_ROOT="$(DEMO_ROOT)" DEMO_DB="$(DEMO_DB)"
	@echo "Demo UI: make demo-mlflow-ui"

demo-dual-train: require-venv
	@"$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/create_dual_view_training_demo.py" --output-root "$(DUAL_DEMO_ROOT)"
	@DUAL_DEMO_MANIFEST="$(DUAL_DEMO_ROOT)/manifest.csv" GAZE_DATA_ROOT="$(DUAL_DEMO_ROOT)" GAZE_OUTPUT_ROOT="$(DUAL_DEMO_ROOT)/outputs" MLFLOW_TRACKING_URI="sqlite:///$(DUAL_DEMO_DB)" "$(VENV_PYTHON)" -m gaze_pipeline train --config "$(CONFIG)" --profile "$(DUAL_DEMO_PROFILE)" $(OVERRIDES)
	@MLFLOW_TRACKING_URI="sqlite:///$(DUAL_DEMO_DB)" "$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/check_mlflow.py" --require-db

data-ver1-manifest: require-venv
	@if [[ ! -d "$(DATA_VER1_ROOT)" ]]; then echo "data(ver1) 폴더가 없습니다: $(DATA_VER1_ROOT)"; exit 2; fi
	@"$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/create_data_ver1_manifest.py" \
		--source-root "$(DATA_VER1_ROOT)" \
		--output-manifest "$(DATA_VER1_MANIFEST)" \
		--dummy-side-annotations --force

data-ver1-preview: require-venv check-webeyetrack-assets
	@MPLCONFIGDIR="$(DATA_VER1_SMOKE_ROOT)/matplotlib" "$(VENV_PYTHON)" \
		"$(PROJECT_ROOT)/scripts/preview_data_ver1_preprocessing.py" \
		--data-root "$(DATA_VER1_ROOT)" \
		--output-dir "$(DATA_VER1_PREVIEW_OUTPUT)" \
		--samples 2 --overwrite-generated

data-ver1-prepare: data-ver1-manifest check-webeyetrack-assets
	@GAZE_DATA_ROOT="$(DATA_VER1_ROOT)" DATA_VER1_MANIFEST="$(DATA_VER1_MANIFEST)" \
		GAZE_OUTPUT_ROOT="$(DATA_VER1_OUTPUT)" MLFLOW_TRACKING_URI="sqlite:///$(DATA_VER1_DB)" \
		MEDIAPIPE_FACE_MODEL="$(MEDIAPIPE_FACE_MODEL)" WEBEYETRACK_WEIGHTS="$(WEBEYETRACK_WEIGHTS)" \
		KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cpu \
		"$(VENV_PYTHON)" -m gaze_pipeline prepare --config "$(CONFIG)" \
		--profile "$(PROJECT_ROOT)/configs/profiles/blazegaze.yaml" \
		--profile "$(PROJECT_ROOT)/configs/profiles/side_profile_90.yaml" \
		--profile "$(DATA_VER1_PROFILE)"

data-ver1-smoke: data-ver1-prepare
	@GAZE_DATA_ROOT="$(DATA_VER1_ROOT)" DATA_VER1_MANIFEST="$(DATA_VER1_MANIFEST)" \
		GAZE_OUTPUT_ROOT="$(DATA_VER1_OUTPUT)" MLFLOW_TRACKING_URI="sqlite:///$(DATA_VER1_DB)" \
		MEDIAPIPE_FACE_MODEL="$(MEDIAPIPE_FACE_MODEL)" WEBEYETRACK_WEIGHTS="$(WEBEYETRACK_WEIGHTS)" \
		KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cpu MPLCONFIGDIR="$(DATA_VER1_SMOKE_ROOT)/matplotlib" \
		"$(VENV_PYTHON)" -m gaze_pipeline train --config "$(CONFIG)" \
		--profile "$(PROJECT_ROOT)/configs/profiles/blazegaze.yaml" \
		--profile "$(PROJECT_ROOT)/configs/profiles/side_profile_90.yaml" \
		--profile "$(DATA_VER1_PROFILE)"
	@GAZE_DATA_ROOT="$(DATA_VER1_ROOT)" DATA_VER1_MANIFEST="$(DATA_VER1_MANIFEST)" \
		GAZE_OUTPUT_ROOT="$(DATA_VER1_OUTPUT)" MLFLOW_TRACKING_URI="sqlite:///$(DATA_VER1_DB)" \
		MEDIAPIPE_FACE_MODEL="$(MEDIAPIPE_FACE_MODEL)" WEBEYETRACK_WEIGHTS="$(WEBEYETRACK_WEIGHTS)" \
		KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cpu MPLCONFIGDIR="$(DATA_VER1_SMOKE_ROOT)/matplotlib" \
		"$(VENV_PYTHON)" -m gaze_pipeline evaluate --config "$(CONFIG)" \
		--profile "$(PROJECT_ROOT)/configs/profiles/blazegaze.yaml" \
		--profile "$(PROJECT_ROOT)/configs/profiles/side_profile_90.yaml" \
		--profile "$(DATA_VER1_PROFILE)" \
		--checkpoint "$(DATA_VER1_OUTPUT)/data_ver1_smoke/data_ver1_smoke_run/checkpoints/best_weights.pt" \
		--split validation
	@MLFLOW_TRACKING_URI="sqlite:///$(DATA_VER1_DB)" "$(VENV_PYTHON)" \
		"$(PROJECT_ROOT)/scripts/check_mlflow.py" --require-db

demo-mlflow-check: require-venv
	@MLFLOW_TRACKING_URI="sqlite:///$(DEMO_DB)" "$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/check_mlflow.py" --require-db

demo-mlflow-ui: require-venv
	@echo "Demo MLflow UI: http://127.0.0.1:$(MLFLOW_PORT)"
	@"$(VENV_MLFLOW)" ui --backend-store-uri "sqlite:///$(DEMO_DB)" --port "$(MLFLOW_PORT)"

webeyetrack-assets: require-venv
	@"$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/fetch_webeyetrack_assets.py" --output-dir "$(WEBEYETRACK_ASSET_DIR)"

check-webeyetrack-assets: require-venv
	@"$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/fetch_webeyetrack_assets.py" --output-dir "$(WEBEYETRACK_ASSET_DIR)" --check-only

preprocess-preview: require-venv
	@if [[ ! -f "$(MEDIAPIPE_FACE_MODEL)" ]]; then echo "MediaPipe model이 없습니다: $(MEDIAPIPE_FACE_MODEL)"; echo "먼저 make webeyetrack-assets를 실행하세요."; exit 2; fi
	@MEDIAPIPE_FACE_MODEL="$(MEDIAPIPE_FACE_MODEL)" WEBEYETRACK_WEIGHTS="$(WEBEYETRACK_WEIGHTS)" "$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/plot_webeyetrack_preprocessing.py" --input-root "$(EXAMPLE_ROOT)" --output-dir "$(PREVIEW_OUTPUT)" --config "$(CONFIG)" --model-asset "$(MEDIAPIPE_FACE_MODEL)" $(PREVIEW_ONLY_ARG)

preprocess-profile90-preview: require-venv
	@if [[ ! -f "$(MEDIAPIPE_FACE_MODEL)" ]]; then echo "MediaPipe model이 없습니다: $(MEDIAPIPE_FACE_MODEL)"; echo "front 전처리를 위해 먼저 make webeyetrack-assets를 실행하세요."; exit 2; fi
	@if [[ ! -f "$(PROFILE90_ANNOTATIONS)" ]]; then echo "strict-profile annotation이 없습니다: $(PROFILE90_ANNOTATIONS)"; exit 2; fi
	@MEDIAPIPE_FACE_MODEL="$(MEDIAPIPE_FACE_MODEL)" WEBEYETRACK_WEIGHTS="$(WEBEYETRACK_WEIGHTS)" "$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/plot_front_and_profile90_preprocessing.py" --input-root "$(EXAMPLE_ROOT)" --output-dir "$(PROFILE90_PREVIEW_OUTPUT)" --config "$(CONFIG)" --front-profile "$(PROFILE90_FRONT_PROFILE)" --annotations "$(PROFILE90_ANNOTATIONS)" --model-asset "$(MEDIAPIPE_FACE_MODEL)"

preprocess-profile90-pose-grid: require-venv
	@if [[ ! -f "$(PROFILE90_POSE_GRID_ANNOTATIONS)" ]]; then echo "3x3 pose annotation이 없습니다: $(PROFILE90_POSE_GRID_ANNOTATIONS)"; exit 2; fi
	@"$(VENV_PYTHON)" "$(PROJECT_ROOT)/scripts/plot_profile90_pose_grid.py" --input-root "$(EXAMPLE_ROOT)" --annotations "$(PROFILE90_POSE_GRID_ANNOTATIONS)" --output-dir "$(PROFILE90_POSE_GRID_OUTPUT)"

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
