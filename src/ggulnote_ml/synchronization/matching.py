from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import MatchingConfig


REQUIRED_LABEL_COLUMNS = {
    "participant",
    "protocol",
    "split",
    "display_timestamp",
    "webcam_frame",
    "webcam_timestamp",
    "phonecam_frame",
    "phonecam_timestamp",
    "x_norm",
    "y_norm",
    "x_centered",
    "y_centered",
    "segment",
    "target",
    "direction",
    "usable",
}

SYNC_COLUMNS = (
    "participant",
    "pair",
    "webcam_frame",
    "phonecam_frame",
    "webcam_timestamp",
    "phonecam_timestamp",
    "webcam_latency_ms",
    "phonecam_latency_ms",
    "webcam_corrected_timestamp",
    "phonecam_corrected_timestamp",
    "corrected_time_diff_ms",
    "reference_timestamp",
    "target_timestamp",
    "x_norm",
    "y_norm",
    "x_centered",
    "y_centered",
    "protocol",
    "split",
    "segment",
    "target",
    "direction",
    "usable",
    "target_interpolated",
    "valid_sync",
    "valid_target",
    "valid",
    "invalid_reason",
)


@dataclass(frozen=True)
class CameraFrameTime:
    frame: int
    timestamp_ns: int
    corrected_timestamp_ns: int


@dataclass(frozen=True)
class TargetSample:
    source_pair: int
    display_timestamp_ns: int
    x_norm: Optional[float]
    y_norm: Optional[float]
    x_centered: Optional[float]
    y_centered: Optional[float]
    protocol: str
    split: str
    segment: str
    target: str
    direction: str
    usable: int


@dataclass(frozen=True)
class TargetLookup:
    sample: Optional[TargetSample]
    target_timestamp_ns: Optional[int]
    x_norm: Optional[float]
    y_norm: Optional[float]
    x_centered: Optional[float]
    y_centered: Optional[float]
    interpolated: bool
    valid: bool
    invalid_reason: str


def _required_int(row: Dict[str, str], name: str, row_number: int) -> int:
    try:
        return int(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("labels row %d has invalid integer %s." % (row_number, name)) from error


def _optional_float(value: str, name: str, row_number: int) -> Optional[float]:
    if value == "":
        return None
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError("labels row %d has invalid float %s." % (row_number, name)) from error
    if not np.isfinite(result):
        raise ValueError("labels row %d has non-finite %s." % (row_number, name))
    return result


def load_raw_labels(
    path: Path, webcam_latency_ms: float, phonecam_latency_ms: float
) -> Tuple[str, Tuple[CameraFrameTime, ...], Tuple[CameraFrameTime, ...], Tuple[TargetSample, ...]]:
    if not path.is_file():
        raise FileNotFoundError("Raw labels CSV does not exist: %s" % path)
    webcam_latency_ns = round(webcam_latency_ms * 1_000_000)
    phonecam_latency_ns = round(phonecam_latency_ms * 1_000_000)
    webcam_frames: List[CameraFrameTime] = []
    phonecam_frames: List[CameraFrameTime] = []
    targets: List[TargetSample] = []
    participant = None
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or ())
        missing = REQUIRED_LABEL_COLUMNS - fieldnames
        if missing:
            raise ValueError("Raw labels CSV is missing columns: %s" % ", ".join(sorted(missing)))
        pair_column = "pair" if "pair" in fieldnames else "frame" if "frame" in fieldnames else None
        if pair_column is None:
            raise ValueError("Raw labels CSV must contain pair (or legacy frame).")
        for row_number, row in enumerate(reader, start=2):
            row_participant = row["participant"].strip()
            if not row_participant:
                raise ValueError("labels row %d has an empty participant." % row_number)
            if participant is None:
                participant = row_participant
            elif row_participant != participant:
                raise ValueError("Raw labels CSV must contain exactly one participant.")
            webcam_timestamp = _required_int(row, "webcam_timestamp", row_number)
            phonecam_timestamp = _required_int(row, "phonecam_timestamp", row_number)
            display_timestamp = _required_int(row, "display_timestamp", row_number)
            webcam_frames.append(
                CameraFrameTime(
                    _required_int(row, "webcam_frame", row_number),
                    webcam_timestamp,
                    webcam_timestamp - webcam_latency_ns,
                )
            )
            phonecam_frames.append(
                CameraFrameTime(
                    _required_int(row, "phonecam_frame", row_number),
                    phonecam_timestamp,
                    phonecam_timestamp - phonecam_latency_ns,
                )
            )
            targets.append(
                TargetSample(
                    source_pair=_required_int(row, pair_column, row_number),
                    display_timestamp_ns=display_timestamp,
                    x_norm=_optional_float(row["x_norm"], "x_norm", row_number),
                    y_norm=_optional_float(row["y_norm"], "y_norm", row_number),
                    x_centered=_optional_float(row["x_centered"], "x_centered", row_number),
                    y_centered=_optional_float(row["y_centered"], "y_centered", row_number),
                    protocol=row["protocol"],
                    split=row["split"],
                    segment=row["segment"],
                    target=row["target"],
                    direction=row["direction"],
                    usable=_required_int(row, "usable", row_number),
                )
            )
    if participant is None:
        raise ValueError("Raw labels CSV contains no data rows.")
    _validate_strictly_increasing(webcam_frames, "webcam")
    _validate_strictly_increasing(phonecam_frames, "phonecam")
    if any(
        right.display_timestamp_ns < left.display_timestamp_ns
        for left, right in zip(targets, targets[1:])
    ):
        raise ValueError("display_timestamp values must be non-decreasing.")
    return participant, tuple(webcam_frames), tuple(phonecam_frames), tuple(targets)


def _validate_strictly_increasing(frames: Sequence[CameraFrameTime], camera: str) -> None:
    if any(
        right.corrected_timestamp_ns <= left.corrected_timestamp_ns
        for left, right in zip(frames, frames[1:])
    ):
        raise ValueError("%s corrected timestamps must be strictly increasing." % camera)


def load_latency_medians(path: Path, participant: Optional[str] = None) -> Tuple[float, float]:
    if not path.is_file():
        raise FileNotFoundError("Latency calibration JSON does not exist: %s" % path)
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if data.get("status") != "valid":
        raise ValueError("Latency calibration status must be valid.")
    if participant is not None and data.get("participant") != participant:
        raise ValueError("Latency calibration participant does not match labels participant.")
    try:
        webcam = float(data["cameras"]["webcam"]["median_ms"])
        phonecam = float(data["cameras"]["phonecam"]["median_ms"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Latency calibration is missing camera median_ms values.") from error
    if not np.isfinite(webcam) or not np.isfinite(phonecam) or webcam < 0 or phonecam < 0:
        raise ValueError("Latency median values must be finite and non-negative.")
    return webcam, phonecam


def _nearest_target(
    targets: Sequence[TargetSample], reference_ns: int, config: MatchingConfig
) -> TargetLookup:
    valid_targets = [
        sample
        for sample in targets
        if sample.x_norm is not None
        and sample.y_norm is not None
        and sample.x_centered is not None
        and sample.y_centered is not None
    ]
    if not valid_targets:
        return TargetLookup(None, None, None, None, None, None, False, False, "no_target_samples")
    timestamps = [sample.display_timestamp_ns for sample in valid_targets]
    position = int(np.searchsorted(timestamps, reference_ns, side="left"))
    left = valid_targets[position - 1] if position > 0 else None
    right = valid_targets[position] if position < len(valid_targets) else None
    nearest = min(
        (sample for sample in (left, right) if sample is not None),
        key=lambda sample: abs(sample.display_timestamp_ns - reference_ns),
    )
    max_gap_ns = round(config.max_target_gap_ms * 1_000_000)
    if abs(nearest.display_timestamp_ns - reference_ns) > max_gap_ns:
        return TargetLookup(
            nearest, None, None, None, None, None, False, False, "target_time_gap_exceeded"
        )

    can_interpolate = (
        config.interpolate_dynamic_targets
        and left is not None
        and right is not None
        and left.display_timestamp_ns < reference_ns < right.display_timestamp_ns
        and left.protocol == right.protocol
        and left.segment == right.segment
        and "dynamic" in left.protocol
        and left.direction != "transition"
        and right.direction != "transition"
    )
    if can_interpolate:
        width = right.display_timestamp_ns - left.display_timestamp_ns
        ratio = (reference_ns - left.display_timestamp_ns) / width
        return TargetLookup(
            sample=nearest,
            target_timestamp_ns=reference_ns,
            x_norm=float(left.x_norm + (right.x_norm - left.x_norm) * ratio),
            y_norm=float(left.y_norm + (right.y_norm - left.y_norm) * ratio),
            x_centered=float(
                left.x_centered + (right.x_centered - left.x_centered) * ratio
            ),
            y_centered=float(
                left.y_centered + (right.y_centered - left.y_centered) * ratio
            ),
            interpolated=True,
            valid=True,
            invalid_reason="",
        )
    return TargetLookup(
        sample=nearest,
        target_timestamp_ns=nearest.display_timestamp_ns,
        x_norm=nearest.x_norm,
        y_norm=nearest.y_norm,
        x_centered=nearest.x_centered,
        y_centered=nearest.y_centered,
        interpolated=False,
        valid=True,
        invalid_reason="",
    )


def _match_frame_indices(
    webcam: Sequence[CameraFrameTime],
    phonecam: Sequence[CameraFrameTime],
    config: MatchingConfig,
) -> Tuple[Tuple[int, Optional[int], float, bool, str], ...]:
    maximum_ns = round(config.max_pair_diff_ms * 1_000_000)
    matches = []
    phone_cursor = 0
    for webcam_index, webcam_frame in enumerate(webcam):
        if phone_cursor >= len(phonecam):
            matches.append((webcam_index, None, float("nan"), False, "no_phonecam_frame"))
            continue
        candidate = phone_cursor
        while candidate + 1 < len(phonecam) and abs(
            phonecam[candidate + 1].corrected_timestamp_ns
            - webcam_frame.corrected_timestamp_ns
        ) <= abs(
            phonecam[candidate].corrected_timestamp_ns
            - webcam_frame.corrected_timestamp_ns
        ):
            candidate += 1
        difference_ns = (
            phonecam[candidate].corrected_timestamp_ns - webcam_frame.corrected_timestamp_ns
        )
        valid = abs(difference_ns) <= maximum_ns
        reason = "" if valid else "corrected_time_diff_exceeded"
        matches.append((webcam_index, candidate, difference_ns / 1_000_000.0, valid, reason))
        if config.enforce_one_to_one:
            if valid or phonecam[candidate].corrected_timestamp_ns <= webcam_frame.corrected_timestamp_ns:
                phone_cursor = candidate + 1
        else:
            phone_cursor = candidate
    return tuple(matches)


def synchronize_labels(
    labels_path: Path,
    latency_path: Path,
    output_path: Path,
    config: MatchingConfig,
) -> Dict[str, object]:
    webcam_latency_ms, phonecam_latency_ms = load_latency_medians(latency_path)
    participant, webcam, phonecam, targets = load_raw_labels(
        labels_path, webcam_latency_ms, phonecam_latency_ms
    )
    load_latency_medians(latency_path, participant=participant)
    if output_path.exists():
        raise FileExistsError("Synchronized CSV already exists and will not be overwritten: %s" % output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches = _match_frame_indices(webcam, phonecam, config)
    valid_sync_count = 0
    valid_target_count = 0
    valid_count = 0
    with output_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SYNC_COLUMNS)
        writer.writeheader()
        for pair_index, (webcam_index, phonecam_index, difference_ms, valid_sync, sync_reason) in enumerate(matches):
            webcam_frame = webcam[webcam_index]
            phonecam_frame = phonecam[phonecam_index] if phonecam_index is not None else None
            reference_ns = (
                (webcam_frame.corrected_timestamp_ns + phonecam_frame.corrected_timestamp_ns) // 2
                if phonecam_frame is not None
                else webcam_frame.corrected_timestamp_ns
            )
            target = _nearest_target(targets, reference_ns, config)
            valid = bool(valid_sync and target.valid)
            reasons = [reason for reason in (sync_reason, target.invalid_reason) if reason]
            sample = target.sample
            writer.writerow(
                {
                    "participant": participant,
                    "pair": pair_index,
                    "webcam_frame": webcam_frame.frame,
                    "phonecam_frame": phonecam_frame.frame if phonecam_frame else "",
                    "webcam_timestamp": webcam_frame.timestamp_ns,
                    "phonecam_timestamp": phonecam_frame.timestamp_ns if phonecam_frame else "",
                    "webcam_latency_ms": "%.6f" % webcam_latency_ms,
                    "phonecam_latency_ms": "%.6f" % phonecam_latency_ms,
                    "webcam_corrected_timestamp": webcam_frame.corrected_timestamp_ns,
                    "phonecam_corrected_timestamp": (
                        phonecam_frame.corrected_timestamp_ns if phonecam_frame else ""
                    ),
                    "corrected_time_diff_ms": (
                        "%.6f" % difference_ms if phonecam_frame else ""
                    ),
                    "reference_timestamp": reference_ns,
                    "target_timestamp": target.target_timestamp_ns or "",
                    "x_norm": "%.6f" % target.x_norm if target.x_norm is not None else "",
                    "y_norm": "%.6f" % target.y_norm if target.y_norm is not None else "",
                    "x_centered": (
                        "%.6f" % target.x_centered if target.x_centered is not None else ""
                    ),
                    "y_centered": (
                        "%.6f" % target.y_centered if target.y_centered is not None else ""
                    ),
                    "protocol": sample.protocol if sample else "",
                    "split": sample.split if sample else "",
                    "segment": sample.segment if sample else "",
                    "target": sample.target if sample else "",
                    "direction": sample.direction if sample else "",
                    "usable": sample.usable if sample else 0,
                    "target_interpolated": int(target.interpolated),
                    "valid_sync": int(valid_sync),
                    "valid_target": int(target.valid),
                    "valid": int(valid),
                    "invalid_reason": ";".join(reasons),
                }
            )
            valid_sync_count += int(valid_sync)
            valid_target_count += int(target.valid)
            valid_count += int(valid)
    return {
        "schema_version": 1,
        "participant": participant,
        "labels": str(labels_path),
        "latency": str(latency_path),
        "output": str(output_path),
        "webcam_latency_ms": webcam_latency_ms,
        "phonecam_latency_ms": phonecam_latency_ms,
        "total_pairs": len(matches),
        "valid_sync_pairs": valid_sync_count,
        "valid_target_pairs": valid_target_count,
        "valid_pairs": valid_count,
    }


def write_synchronization_summary(path: Path, summary: Dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(
            "Synchronization summary already exists and will not be overwritten: %s" % path
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    temporary.replace(path)
