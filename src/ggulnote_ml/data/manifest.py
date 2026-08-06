from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Mapping

import numpy as np

from ggulnote_ml.config import DataConfig
from ggulnote_ml.contracts import RawDataset, RawVideoSample
from ggulnote_ml.data.base import DataSource
from ggulnote_ml.exceptions import ContractError, OptionalDependencyError


REQUIRED_COLUMNS = {
    "video_path",
    "source",
    "participant_id",
    "session_id",
    "target_x",
    "target_y",
}


class ManifestVideoDataSource(DataSource):
    """Load one labeled gaze window per video listed in a CSV manifest."""

    def __init__(self, config: DataConfig, manifest_path: Path) -> None:
        self.config = config
        self.manifest_path = manifest_path

    def load(self) -> RawDataset:
        try:
            import cv2
        except ImportError as exc:
            raise OptionalDependencyError(
                "Manifest video loading requires the 'video' extra: pip install -e '.[video]'."
            ) from exc

        if not self.manifest_path.exists():
            raise ContractError("Manifest does not exist: %s" % self.manifest_path)

        samples: List[RawVideoSample] = []
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
            if missing:
                raise ContractError(
                    "Manifest is missing required columns: %s" % ", ".join(sorted(missing))
                )
            for row_number, row in enumerate(reader, start=2):
                samples.append(self._load_row(cv2, row, row_number))

        dataset = RawDataset(samples=tuple(samples))
        dataset.validate(self.config.sequence_length)
        return dataset

    def _load_row(self, cv2: object, row: Mapping[str, str], row_number: int) -> RawVideoSample:
        video_path = Path(row["video_path"])
        if not video_path.is_absolute():
            video_path = (self.manifest_path.parent / video_path).resolve()
        if not video_path.exists():
            raise ContractError("Manifest row %d video does not exist: %s" % (row_number, video_path))

        source = row["source"].strip().lower()
        if source not in {"webcam", "phone"}:
            raise ContractError("Manifest row %d source must be webcam or phone." % row_number)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ContractError("Could not open video at manifest row %d: %s" % (row_number, video_path))
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise ContractError("Video has no readable frames: %s" % video_path)
            indices = np.linspace(
                0,
                max(0, frame_count - 1),
                self.config.sequence_length,
                dtype=int,
            )
            frames = []
            rotation = int(row.get("rotation_degrees") or 0)
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, bgr = capture.read()
                if not ok:
                    raise ContractError(
                        "Could not decode frame %d from %s" % (frame_index, video_path)
                    )
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = _rotate(rgb, rotation)
                frames.append(rgb.astype(np.uint8, copy=False))
        finally:
            capture.release()

        target = np.asarray(
            [float(row["target_x"]), float(row["target_y"])], dtype=np.float32
        )
        sample = RawVideoSample(
            frames=np.stack(frames, axis=0),
            target=target,
            participant_id=row["participant_id"].strip(),
            session_id=row["session_id"].strip(),
            source=source,
        )
        sample.validate(self.config.sequence_length)
        return sample


def _rotate(frame: np.ndarray, rotation_degrees: int) -> np.ndarray:
    rotations = {0: 0, 90: 3, 180: 2, 270: 1}
    if rotation_degrees not in rotations:
        raise ContractError("rotation_degrees must be one of 0, 90, 180, 270.")
    turns = rotations[rotation_degrees]
    return np.rot90(frame, k=turns).copy() if turns else frame

