PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
CONFIG ?= configs/base.yaml
MODEL ?=
PREPROCESS_FLAGS ?=
PARTICIPANT ?=
PROTOCOL ?=all
ALLOW_MISSING_CALIBRATION ?=
DATASET_ROOT ?=
LATENCY_JSON ?=
OUTPUT_ROOT ?=data/interim/dual_view
EAR_THRESHOLD ?=0.20

.PHONY: setup setup-video setup-capture collect simulate suggest-participant list-cameras check-cameras measure-latency sync-participant video-features validate preprocess smoke train evaluate predict test check mlflow

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
	$(PYTHON) -m ggulnote_ml.capture --config configs/capture.yaml $(if $(PARTICIPANT),--participant $(PARTICIPANT),) --protocol $(PROTOCOL) $(if $(ALLOW_MISSING_CALIBRATION),--allow-missing-calibration,) $(if $(DATASET_ROOT),--dataset-root $(DATASET_ROOT),)

simulate:
	$(PYTHON) -m ggulnote_ml.capture --config configs/capture.yaml --simulate $(if $(PARTICIPANT),--participant $(PARTICIPANT),) --protocol $(PROTOCOL) $(if $(DATASET_ROOT),--dataset-root $(DATASET_ROOT),)

suggest-participant:
	$(PYTHON) -m ggulnote_ml.capture --config configs/capture.yaml --suggest-participant

list-cameras:
	$(PYTHON) -m ggulnote_ml.capture --config configs/capture.yaml --list-cameras

check-cameras:
	$(PYTHON) -m ggulnote_ml.capture --config configs/capture.yaml --check-cameras

measure-latency:
	@test -n "$(PARTICIPANT)" || (echo "PARTICIPANT=p00 is required" && exit 2)
	$(PYTHON) -m ggulnote_ml.synchronization --participant $(PARTICIPANT)

sync-participant:
	@test -n "$(PARTICIPANT)" || (echo "PARTICIPANT=p00 is required" && exit 2)
	$(PYTHON) -m ggulnote_ml.synchronization --synchronize --participant $(PARTICIPANT) $(if $(LATENCY_JSON),--latency-json $(LATENCY_JSON),)

video-features:
	@test -n "$(PARTICIPANT)" || (echo "PARTICIPANT=p00 is required" && exit 2)
	$(PYTHON) -m ggulnote_ml.video_preprocessing --participant $(PARTICIPANT) --dataset-root data/raw/participants --output-root $(OUTPUT_ROOT) --ear-threshold $(EAR_THRESHOLD)

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
