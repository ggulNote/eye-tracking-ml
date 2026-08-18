-include Makefile.local

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
CONFIG ?= configs/base.yaml
MODEL ?=
PREPROCESS_FLAGS ?=
PARTICIPANT ?=
HEAD_POSE ?=
PROTOCOL ?=all
ALLOW_MISSING_CALIBRATION ?=
DATASET_ROOT ?=
DATASET_ROOT_EFFECTIVE = $(if $(strip $(DATASET_ROOT)),$(DATASET_ROOT),data/raw/participants)
LATENCY_JSON ?=
OUTPUT_ROOT ?=
EAR_THRESHOLD ?=0.20
INTRINSICS_MODE ?=required
FEATURE_CAMERAS ?=webcam
CAPTURE_ONLY ?=
ICLOUD_BACKUP_ROOT ?=
GEOMETRY_CONFIG ?=configs/geometry_calibration.yaml
WEBEYETRACK_CONFIG ?=configs/webeyetrack_preprocessing.yaml
CAMERA ?=
AUTO_CAPTURE ?=
FORCE ?=

.PHONY: setup setup-video setup-capture collect simulate suggest-participant list-cameras check-cameras check-display measure-latency sync-participant video-features webeyetrack-inputs webeyetrack-batch prepare-participant migrate-layout geometry-board geometry-screen geometry-intrinsics geometry-inspect validate preprocess smoke train evaluate predict test check mlflow

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e . --no-deps

setup-video:
	$(PIP) install -r requirements-video.txt
	$(PIP) install -e . --no-deps

setup-capture:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements-capture.txt
	$(PIP) install -e . --no-deps

collect:
	$(PYTHON) -m ggulnote_ml.capture --config configs/capture.yaml $(if $(PARTICIPANT),--participant "$(PARTICIPANT)",) $(if $(HEAD_POSE),--head-pose "$(HEAD_POSE)",) --protocol $(PROTOCOL) $(if $(ALLOW_MISSING_CALIBRATION),--allow-missing-calibration,) $(if $(CAPTURE_ONLY),--capture-only,) --dataset-root "$(DATASET_ROOT_EFFECTIVE)" $(if $(ICLOUD_BACKUP_ROOT),--backup-root "$(ICLOUD_BACKUP_ROOT)",)

simulate:
	$(PYTHON) -m ggulnote_ml.capture --config configs/capture.yaml --simulate $(if $(PARTICIPANT),--participant "$(PARTICIPANT)",) $(if $(HEAD_POSE),--head-pose "$(HEAD_POSE)",) --protocol $(PROTOCOL) --dataset-root "$(DATASET_ROOT_EFFECTIVE)"

suggest-participant:
	$(PYTHON) -m ggulnote_ml.capture --config configs/capture.yaml --suggest-participant --dataset-root "$(DATASET_ROOT_EFFECTIVE)"

list-cameras:
	$(PYTHON) -m ggulnote_ml.capture --config configs/capture.yaml --list-cameras

check-cameras:
	$(PYTHON) -m ggulnote_ml.capture --config configs/capture.yaml --check-cameras

check-display:
	$(PYTHON) -m ggulnote_ml.capture --config configs/capture.yaml --check-display

geometry-board:
	$(PYTHON) -m ggulnote_ml.capture.geometry_cli --config $(GEOMETRY_CONFIG) generate-board

geometry-screen:
	$(PYTHON) -m ggulnote_ml.capture.geometry_cli --config $(GEOMETRY_CONFIG) write-screen

geometry-intrinsics:
	@test -n "$(CAMERA)" || (echo "CAMERA=webcam or CAMERA=phonecam is required" && exit 2)
	$(PYTHON) -m ggulnote_ml.capture.geometry_cli --config $(GEOMETRY_CONFIG) capture-intrinsics --camera $(CAMERA) $(if $(AUTO_CAPTURE),--auto-capture,)

geometry-inspect:
	$(PYTHON) -m ggulnote_ml.capture.geometry_cli --config $(GEOMETRY_CONFIG) inspect

measure-latency:
	@test -n "$(PARTICIPANT)" || (echo "PARTICIPANT=참가자이름 is required" && exit 2)
	@test -n "$(HEAD_POSE)" || (echo "HEAD_POSE=neutral, head_up, or head_down is required" && exit 2)
	$(PYTHON) -m ggulnote_ml.synchronization --participant "$(PARTICIPANT)" --head-pose "$(HEAD_POSE)" --dataset-root "$(DATASET_ROOT_EFFECTIVE)"

sync-participant:
	@test -n "$(PARTICIPANT)" || (echo "PARTICIPANT=참가자이름 is required" && exit 2)
	@test -n "$(HEAD_POSE)" || (echo "HEAD_POSE=neutral, head_up, or head_down is required" && exit 2)
	$(PYTHON) -m ggulnote_ml.synchronization --synchronize --participant "$(PARTICIPANT)" --head-pose "$(HEAD_POSE)" --dataset-root "$(DATASET_ROOT_EFFECTIVE)" $(if $(LATENCY_JSON),--latency-json "$(LATENCY_JSON)",)

video-features:
	@test -n "$(PARTICIPANT)" || (echo "PARTICIPANT=참가자이름 is required" && exit 2)
	@test -n "$(HEAD_POSE)" || (echo "HEAD_POSE=neutral, head_up, or head_down is required" && exit 2)
	$(PYTHON) -m ggulnote_ml.video_preprocessing --participant "$(PARTICIPANT)" --head-pose "$(HEAD_POSE)" --dataset-root "$(DATASET_ROOT_EFFECTIVE)" $(if $(OUTPUT_ROOT),--output-root "$(OUTPUT_ROOT)",) --ear-threshold $(EAR_THRESHOLD) --intrinsics-mode $(INTRINSICS_MODE) --feature-cameras $(FEATURE_CAMERAS)

webeyetrack-inputs:
	@test -n "$(PARTICIPANT)" || (echo "PARTICIPANT=참가자이름 is required" && exit 2)
	@test -n "$(HEAD_POSE)" || (echo "HEAD_POSE=neutral, head_up, or head_down is required" && exit 2)
	$(PYTHON) -m ggulnote_ml.video_preprocessing.webeyetrack_cli --participant "$(PARTICIPANT)" --head-pose "$(HEAD_POSE)" --dataset-root "$(DATASET_ROOT_EFFECTIVE)" --config $(WEBEYETRACK_CONFIG) $(if $(OUTPUT_ROOT),--output-root "$(OUTPUT_ROOT)",) $(if $(FORCE),--force,)

webeyetrack-batch:
	$(PYTHON) -m ggulnote_ml.video_preprocessing.webeyetrack_cli --all --dataset-root "$(DATASET_ROOT_EFFECTIVE)" --config $(WEBEYETRACK_CONFIG) $(if $(OUTPUT_ROOT),--output-root "$(OUTPUT_ROOT)",) $(if $(FORCE),--force,)

prepare-participant: sync-participant video-features

migrate-layout:
	$(PYTHON) -m ggulnote_ml.capture.migrate_layout $(if $(PARTICIPANT),--participant $(PARTICIPANT),--all) $(if $(APPLY),--apply,)

validate:
	$(PYTHON) -m ggulnote_ml validate-config --config $(CONFIG)

preprocess:
	$(PYTHON) -m ggulnote_ml preprocess --config $(CONFIG) $(PREPROCESS_FLAGS)

smoke:
	$(PYTHON) -m ggulnote_ml train --config configs/local-no-mlflow.yaml

train:
	$(PYTHON) -m ggulnote_ml train --config $(CONFIG)

evaluate:
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.npz is required" && exit 2)
	$(PYTHON) -m ggulnote_ml evaluate --config $(CONFIG) --model-path $(MODEL)

predict:
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.npz is required" && exit 2)
	$(PYTHON) -m ggulnote_ml predict --config $(CONFIG) --model-path $(MODEL)

test:
	$(PYTHON) -m pytest

check: validate test smoke

mlflow:
	.venv/bin/mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 127.0.0.1 \
		--port 5000
