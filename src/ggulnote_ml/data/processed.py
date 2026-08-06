from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

from ggulnote_ml.config import DataConfig, PreprocessingConfig
from ggulnote_ml.contracts import CanonicalBatch
from ggulnote_ml.exceptions import ContractError
from ggulnote_ml.utils import sha256_file, sha256_json, write_json


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class StoredCanonicalSplits:
    train: CanonicalBatch
    validation: CanonicalBatch
    test: CanonicalBatch
    metadata: Dict[str, object]


class ProcessedDatasetStore:
    """Versioned local processed dataset; it never uploads image data to MLflow."""

    format_version = "canonical-npz-v1"

    def __init__(
        self,
        data_root: Path,
        data_config: DataConfig,
        preprocessing_config: PreprocessingConfig,
    ) -> None:
        self.data_root = data_root.resolve()
        self.data_config = data_config
        self.preprocessing_config = preprocessing_config
        processed_dir = Path(data_config.processed_dir)
        if processed_dir.is_absolute():
            raise ContractError("data.processed_dir must be relative to GGULNOTE_DATA_ROOT.")
        self.version_dir = (
            self.data_root / processed_dir / data_config.dataset_version
        ).resolve()
        if self.data_root not in self.version_dir.parents:
            raise ContractError("Processed dataset path escapes GGULNOTE_DATA_ROOT.")

    def save(
        self,
        splits: Mapping[str, CanonicalBatch],
        manifest_path: Optional[Path],
        preprocessor_version: str,
        force: bool = False,
    ) -> Dict[str, object]:
        if self.version_dir.exists() and any(self.version_dir.iterdir()) and not force:
            raise ContractError(
                "Processed dataset version already exists: %s. Use --force or increment dataset_version."
                % self.version_dir
            )
        self.version_dir.mkdir(parents=True, exist_ok=True)
        split_metadata: Dict[str, object] = {}
        file_hashes: Dict[str, str] = {}
        content_hashes: Dict[str, str] = {}
        all_participants = set()
        sample_count = 0

        for split_name in SPLIT_NAMES:
            try:
                batch = splits[split_name]
            except KeyError as exc:
                raise ContractError("Missing processed split: %s" % split_name) from exc
            npz_path = self.version_dir / (split_name + ".npz")
            metadata_path = self.version_dir / (split_name + ".metadata.json")
            np.savez_compressed(
                npz_path,
                frames=batch.frames,
                targets=batch.targets,
                sample_weights=batch.sample_weights,
            )
            write_json(metadata_path, _batch_metadata(batch))
            file_hashes[npz_path.name] = sha256_file(npz_path)
            file_hashes[metadata_path.name] = sha256_file(metadata_path)
            content_hashes[split_name] = _batch_content_hash(batch)
            participants = sorted(set(batch.participant_ids))
            all_participants.update(participants)
            sample_count += batch.frames.shape[0]
            split_metadata[split_name] = {
                "sample_count": batch.frames.shape[0],
                "participant_ids": participants,
                "sources": sorted(set(batch.sources)),
                "frames_shape": list(batch.frames.shape),
                "frames_dtype": str(batch.frames.dtype),
                "targets_shape": list(batch.targets.shape),
            }

        manifest_hash = sha256_file(manifest_path) if manifest_path is not None else None
        contract = {
            "sequence_length": self.data_config.sequence_length,
            "preprocessing": asdict(self.preprocessing_config),
        }
        dataset_hash = sha256_json(
            {
                "manifest_hash": manifest_hash,
                "content": content_hashes,
                "contract": contract,
            }
        )
        metadata: Dict[str, object] = {
            "format_version": self.format_version,
            "dataset_version": self.data_config.dataset_version,
            "dataset_hash": dataset_hash,
            "manifest_hash": manifest_hash,
            "preprocessor_version": preprocessor_version,
            "participant_count": len(all_participants),
            "sample_count": sample_count,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract": contract,
            "splits": split_metadata,
            "file_hashes": file_hashes,
            "content_hashes": content_hashes,
        }
        write_json(self.version_dir / "dataset.json", metadata)
        return metadata

    def load(self) -> StoredCanonicalSplits:
        metadata_path = self.version_dir / "dataset.json"
        if not metadata_path.is_file():
            raise ContractError(
                "Processed dataset is missing: %s. Run make preprocess first."
                % self.version_dir
            )
        metadata = _read_json(metadata_path)
        self._validate_metadata(metadata)
        loaded = {
            split_name: self._load_split(split_name) for split_name in SPLIT_NAMES
        }
        return StoredCanonicalSplits(
            train=loaded["train"],
            validation=loaded["validation"],
            test=loaded["test"],
            metadata=metadata,
        )

    def _load_split(self, split_name: str) -> CanonicalBatch:
        npz_path = self.version_dir / (split_name + ".npz")
        metadata_path = self.version_dir / (split_name + ".metadata.json")
        if not npz_path.is_file() or not metadata_path.is_file():
            raise ContractError("Processed split files are missing for %s." % split_name)
        expected_hashes = _read_json(self.version_dir / "dataset.json")["file_hashes"]
        for path in (npz_path, metadata_path):
            if sha256_file(path) != expected_hashes[path.name]:
                raise ContractError("Processed dataset hash mismatch: %s" % path)
        values = _read_json(metadata_path)
        with np.load(npz_path, allow_pickle=False) as arrays:
            batch = CanonicalBatch(
                frames=arrays["frames"].astype(np.float32, copy=False),
                targets=arrays["targets"].astype(np.float32, copy=False),
                participant_ids=tuple(values["participant_ids"]),
                session_ids=tuple(values["session_ids"]),
                sources=tuple(values["sources"]),
                sample_ids=tuple(values["sample_ids"]),
                frame_indices=tuple(tuple(item) for item in values["frame_indices"]),
                timestamps_ms=tuple(tuple(item) for item in values["timestamps_ms"]),
                screen_sizes_px=tuple(
                    tuple(item) if item is not None else None
                    for item in values["screen_sizes_px"]
                ),
                screen_sizes_cm=tuple(
                    tuple(item) if item is not None else None
                    for item in values["screen_sizes_cm"]
                ),
                device_ids=tuple(values["device_ids"]),
                camera_intrinsics_paths=tuple(values["camera_intrinsics_paths"]),
                calibration_point_ids=tuple(values["calibration_point_ids"]),
                sample_weights=arrays["sample_weights"].astype(np.float32, copy=False),
            )
        channels = 3 if self.preprocessing_config.color_mode == "rgb" else 1
        batch.validate(
            sequence_length=self.data_config.sequence_length,
            channels=channels,
            height=self.preprocessing_config.output_height,
            width=self.preprocessing_config.output_width,
        )
        return batch

    def _validate_metadata(self, metadata: Dict[str, object]) -> None:
        if metadata.get("format_version") != self.format_version:
            raise ContractError("Unsupported processed dataset format.")
        if metadata.get("dataset_version") != self.data_config.dataset_version:
            raise ContractError("Processed dataset version does not match config.")
        expected_contract = {
            "sequence_length": self.data_config.sequence_length,
            "preprocessing": asdict(self.preprocessing_config),
        }
        if metadata.get("contract") != expected_contract:
            raise ContractError(
                "Processed dataset contract does not match current config; preprocess a new version."
            )


def _batch_metadata(batch: CanonicalBatch) -> Dict[str, object]:
    return {
        "participant_ids": list(batch.participant_ids),
        "session_ids": list(batch.session_ids),
        "sources": list(batch.sources),
        "sample_ids": list(batch.sample_ids),
        "frame_indices": [list(item) for item in batch.frame_indices],
        "timestamps_ms": [list(item) for item in batch.timestamps_ms],
        "screen_sizes_px": [list(item) if item is not None else None for item in batch.screen_sizes_px],
        "screen_sizes_cm": [list(item) if item is not None else None for item in batch.screen_sizes_cm],
        "device_ids": list(batch.device_ids),
        "camera_intrinsics_paths": list(batch.camera_intrinsics_paths),
        "calibration_point_ids": list(batch.calibration_point_ids),
    }


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError("Expected a JSON object: %s" % path)
    return value


def _batch_content_hash(batch: CanonicalBatch) -> str:
    digest = hashlib.sha256()
    for array in (batch.frames, batch.targets, batch.sample_weights):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    digest.update(sha256_json(_batch_metadata(batch)).encode("ascii"))
    return digest.hexdigest()
