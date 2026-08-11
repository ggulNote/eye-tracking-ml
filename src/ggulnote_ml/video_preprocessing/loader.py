from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ggulnote_ml.exceptions import ContractError
from ggulnote_ml.synchronization.matching import SYNC_COLUMNS
from ggulnote_ml.video_preprocessing.contracts import (
    SelectedSample,
    SelectedSynchronizedPair,
    SynchronizedPair,
)


SELECTED_SAMPLE_REQUIRED_COLUMNS = frozenset(
    {
        "sample",
        "participant",
        "protocol",
        "split",
        "webcam_frame",
        "webcam_timestamp",
        "phonecam_frame",
        "phonecam_timestamp",
        "segment",
        "target",
        "direction",
    }
)


def load_synchronized_pairs(path: Path) -> Tuple[SynchronizedPair, ...]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Synchronized CSV does not exist: %s" % resolved)
    pairs: List[SynchronizedPair] = []
    participant: Optional[str] = None
    seen_pairs = set()
    with resolved.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        _validate_header(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            row_participant = (row.get("participant") or "").strip()
            if not row_participant:
                raise ContractError("Row %d has an empty participant." % row_number)
            if participant is None:
                participant = row_participant
            elif participant != row_participant:
                raise ContractError("Synchronized CSV must contain one participant.")
            pair = _required_int(row, "pair", row_number, minimum=0)
            if pair in seen_pairs:
                raise ContractError("Synchronized CSV contains duplicate pair %d." % pair)
            seen_pairs.add(pair)
            pairs.append(
                SynchronizedPair(
                    participant=row_participant,
                    pair=pair,
                    webcam_frame=_optional_int(row, "webcam_frame", row_number, 0),
                    phonecam_frame=_optional_int(row, "phonecam_frame", row_number, 0),
                    webcam_timestamp=_optional_int(row, "webcam_timestamp", row_number, 1),
                    phonecam_timestamp=_optional_int(row, "phonecam_timestamp", row_number, 1),
                    webcam_corrected_timestamp=_optional_int(
                        row, "webcam_corrected_timestamp", row_number, 1
                    ),
                    phonecam_corrected_timestamp=_optional_int(
                        row, "phonecam_corrected_timestamp", row_number, 1
                    ),
                    reference_timestamp=_required_int(
                        row, "reference_timestamp", row_number, minimum=1
                    ),
                    target_timestamp=_optional_int(
                        row, "target_timestamp", row_number, 1
                    ),
                    x_norm=_optional_float(row, "x_norm", row_number),
                    y_norm=_optional_float(row, "y_norm", row_number),
                    protocol=row["protocol"],
                    split=row["split"],
                    segment=row["segment"],
                    target=row["target"],
                    direction=row["direction"],
                    usable=_required_flag(row, "usable", row_number),
                    valid=_required_flag(row, "valid", row_number),
                    invalid_reason=row["invalid_reason"],
                )
            )
    if not pairs:
        raise ContractError("Synchronized CSV contains no data rows.")
    return tuple(pairs)


def load_selected_samples(path: Path) -> Tuple[SelectedSample, ...]:
    """Load the one-best-pair manifest created by the click capture stage."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Selected sample CSV does not exist: %s" % resolved)
    samples: List[SelectedSample] = []
    seen_ids = set()
    with resolved.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        _validate_selected_sample_header(reader.fieldnames)
        fieldnames = set(reader.fieldnames or ())
        pair_column = "pair" if "pair" in fieldnames else "pair_frame"
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample") or "").strip()
            participant = (row.get("participant") or "").strip()
            if not sample_id:
                raise ContractError("Row %d has an empty sample." % row_number)
            if sample_id in seen_ids:
                raise ContractError("Selected sample CSV contains duplicate sample %s." % sample_id)
            if not participant:
                raise ContractError("Row %d has an empty participant." % row_number)
            seen_ids.add(sample_id)
            sample = SelectedSample(
                sample=sample_id,
                participant=participant,
                protocol=row["protocol"],
                split=row["split"],
                source_pair=_required_int(row, pair_column, row_number, minimum=0),
                webcam_frame=_required_int(row, "webcam_frame", row_number, minimum=0),
                phonecam_frame=_required_int(row, "phonecam_frame", row_number, minimum=0),
                webcam_timestamp=_required_int(
                    row, "webcam_timestamp", row_number, minimum=1
                ),
                phonecam_timestamp=_required_int(
                    row, "phonecam_timestamp", row_number, minimum=1
                ),
                segment=(row.get("segment") or "").strip(),
                target=(row.get("target") or "").strip(),
                direction=(row.get("direction") or "").strip(),
            )
            if sample.export_partition is None:
                raise ContractError(
                    "Row %d split must be train/training or evaluation." % row_number
                )
            samples.append(sample)
    if not samples:
        raise ContractError("Selected sample CSV contains no data rows.")
    return tuple(samples)


def select_synchronized_samples(
    pairs: Sequence[SynchronizedPair], samples: Sequence[SelectedSample]
) -> Tuple[SelectedSynchronizedPair, ...]:
    """Map each best-frame sample to the nearest valid synchronized frame pair.

    The selected images were scored before camera-latency correction.  Matching
    within the same protocol/segment keeps the click label fixed, while choosing
    the closest camera frame numbers preserves the quality-selected instant.
    The returned frames always come from ``synchronized_frames.csv``.
    """

    selected: List[SelectedSynchronizedPair] = []
    used_pairs = set()
    for sample in samples:
        candidates = [
            pair
            for pair in pairs
            if pair.pair not in used_pairs
            and pair.participant == sample.participant
            and pair.export_partition == sample.export_partition
            and pair.protocol == sample.protocol
            and pair.segment == sample.segment
            and pair.target == sample.target
            and pair.direction == sample.direction
        ]
        if not candidates:
            raise ContractError(
                "No valid synchronized pair matches selected sample %s." % sample.sample
            )
        chosen = min(candidates, key=lambda pair: _sample_distance(pair, sample))
        used_pairs.add(chosen.pair)
        selected.append(SelectedSynchronizedPair(sample=sample, synchronized=chosen))
    return tuple(selected)


def load_screen_size(participant_json: Path) -> Tuple[int, int]:
    resolved = participant_json.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Participant metadata does not exist: %s" % resolved)
    with resolved.open(encoding="utf-8") as file:
        data = json.load(file)
    try:
        width = int(data["screen"]["canvas_width_pixel"])
        height = int(data["screen"]["canvas_height_pixel"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(
            "participant.json is missing screen canvas pixel dimensions."
        ) from exc
    if width <= 0 or height <= 0:
        raise ContractError("Screen pixel dimensions must be positive.")
    return width, height


def _validate_header(fieldnames: Optional[Sequence[str]]) -> None:
    if not fieldnames:
        raise ContractError("Synchronized CSV must contain a header.")
    duplicates = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
    if duplicates:
        raise ContractError("Synchronized CSV contains duplicate columns.")
    missing = sorted(set(SYNC_COLUMNS) - set(fieldnames))
    if missing:
        raise ContractError(
            "Synchronized CSV is missing columns: %s." % ", ".join(missing)
        )


def _validate_selected_sample_header(fieldnames: Optional[Sequence[str]]) -> None:
    if not fieldnames:
        raise ContractError("Selected sample CSV must contain a header.")
    duplicates = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
    if duplicates:
        raise ContractError("Selected sample CSV contains duplicate columns.")
    missing = sorted(SELECTED_SAMPLE_REQUIRED_COLUMNS - set(fieldnames))
    if missing:
        raise ContractError(
            "Selected sample CSV is missing columns: %s." % ", ".join(missing)
        )
    if "pair" not in fieldnames and "pair_frame" not in fieldnames:
        raise ContractError(
            "Selected sample CSV is missing columns: pair (or legacy pair_frame)."
        )


def _sample_distance(pair: SynchronizedPair, sample: SelectedSample) -> Tuple[int, int, int]:
    assert pair.webcam_frame is not None and pair.phonecam_frame is not None
    assert pair.webcam_timestamp is not None and pair.phonecam_timestamp is not None
    frame_distance = abs(pair.webcam_frame - sample.webcam_frame) + abs(
        pair.phonecam_frame - sample.phonecam_frame
    )
    timestamp_distance = abs(pair.webcam_timestamp - sample.webcam_timestamp) + abs(
        pair.phonecam_timestamp - sample.phonecam_timestamp
    )
    return frame_distance, timestamp_distance, pair.pair


def _required_int(
    row: Mapping[str, str], name: str, row_number: int, minimum: int
) -> int:
    value = _optional_int(row, name, row_number, minimum)
    if value is None:
        raise ContractError("Row %d has an empty %s." % (row_number, name))
    return value


def _optional_int(
    row: Mapping[str, str], name: str, row_number: int, minimum: int
) -> Optional[int]:
    text = (row.get(name) or "").strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise ContractError("Row %d has invalid integer %s." % (row_number, name)) from exc
    if value < minimum:
        raise ContractError("Row %d has %s below %d." % (row_number, name, minimum))
    return value


def _optional_float(
    row: Mapping[str, str], name: str, row_number: int
) -> Optional[float]:
    text = (row.get(name) or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ContractError("Row %d has invalid float %s." % (row_number, name)) from exc
    if not np.isfinite(value):
        raise ContractError("Row %d has non-finite %s." % (row_number, name))
    if name in {"x_norm", "y_norm"} and not 0.0 <= value <= 1.0:
        raise ContractError("Row %d has %s outside [0,1]." % (row_number, name))
    return value


def _required_flag(row: Mapping[str, str], name: str, row_number: int) -> bool:
    text = (row.get(name) or "").strip()
    if text not in {"0", "1"}:
        raise ContractError("Row %d has invalid 0/1 flag %s." % (row_number, name))
    return text == "1"
