from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ggulnote_ml.exceptions import ContractError
from ggulnote_ml.synchronization.matching import SYNC_COLUMNS
from ggulnote_ml.video_preprocessing.contracts import SynchronizedPair


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
                    usable=_required_flag(row, "usable", row_number),
                    training=_required_flag(row, "training", row_number),
                    valid=_required_flag(row, "valid", row_number),
                    invalid_reason=row["invalid_reason"],
                )
            )
    if not pairs:
        raise ContractError("Synchronized CSV contains no data rows.")
    return tuple(pairs)


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
