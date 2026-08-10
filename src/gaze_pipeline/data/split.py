"""Deterministic grouped splitting and explicit dual-view pair validation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .records import CanonicalRecord, DataContractError

SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")


class SplitError(DataContractError):
    """Raised when a split would violate ratios or leakage constraints."""


class PairValidationError(DataContractError):
    """Raised when explicit dual-view pairing is missing or ambiguous."""


@dataclass(frozen=True, slots=True)
class PairValidationResult:
    """Validated records and pair completeness metadata."""

    records: tuple[CanonicalRecord, ...]
    complete_pair_ids: frozenset[str]
    incomplete_pair_ids: frozenset[str]
    unpaired_sample_ids: frozenset[str]
    dropped_sample_ids: frozenset[str]

    def is_complete(self, record: CanonicalRecord) -> bool:
        return bool(record.pair_id and record.pair_id in self.complete_pair_ids)


def deterministic_group_split(
    records: Sequence[CanonicalRecord],
    *,
    ratios: Mapping[str, float] | None = None,
    seed: int = 42,
    group_key: str = "subject_id",
    shuffle_groups: bool = True,
) -> dict[str, list[CanonicalRecord]]:
    """Split records without allowing a group into more than one partition.

    For the default 15 MPIIFaceGaze subjects, largest-remainder allocation for
    70/15/15 produces 11/2/2 subject groups.  The realized sample ratios may
    differ because group sizes are intentionally not fragmented.
    """

    if not records:
        raise SplitError("cannot split an empty record sequence")
    normalized_ratios = _validate_ratios(
        ratios or {"train": 0.70, "validation": 0.15, "test": 0.15}
    )

    groups: dict[str, list[CanonicalRecord]] = {}
    for record in records:
        try:
            raw_group = getattr(record, group_key)
        except AttributeError as exc:
            raise SplitError(f"canonical record has no split group key {group_key!r}") from exc
        if raw_group is None or not str(raw_group).strip():
            raise SplitError(f"record {record.sample_id!r} has an empty {group_key}")
        groups.setdefault(str(raw_group), []).append(record)

    group_names = sorted(groups)
    if shuffle_groups:
        group_names.sort(key=lambda group: _stable_group_key(seed, group))
    group_counts = _allocate_group_counts(len(group_names), normalized_ratios)

    group_to_split: dict[str, str] = {}
    cursor = 0
    for split_name in SPLIT_NAMES:
        next_cursor = cursor + group_counts[split_name]
        for group in group_names[cursor:next_cursor]:
            group_to_split[group] = split_name
        cursor = next_cursor
    if cursor != len(group_names):
        raise AssertionError("internal group allocation did not consume all groups")

    splits: dict[str, list[CanonicalRecord]] = {split_name: [] for split_name in SPLIT_NAMES}
    for group, group_records in groups.items():
        split_name = group_to_split[group]
        splits[split_name].extend(group_records)
    for split_name in SPLIT_NAMES:
        splits[split_name].sort(key=lambda record: record.sample_id)

    validate_no_group_leakage(splits, group_key=group_key)
    validate_no_pair_leakage(splits)
    return splits


def _validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    aliases = dict(ratios)
    if "val" in aliases and "validation" not in aliases:
        aliases["validation"] = aliases.pop("val")
    unknown = set(aliases) - set(SPLIT_NAMES)
    missing = set(SPLIT_NAMES) - set(aliases)
    if unknown or missing:
        raise SplitError(
            f"split ratios require exactly {SPLIT_NAMES}; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    converted = {name: float(aliases[name]) for name in SPLIT_NAMES}
    if any(not math.isfinite(value) or value < 0 for value in converted.values()):
        raise SplitError("split ratios must be finite and non-negative")
    total = sum(converted.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise SplitError(f"split ratios must sum to 1.0, got {total}")
    if converted["train"] <= 0:
        raise SplitError("train split ratio must be positive")
    return converted


def _allocate_group_counts(number_of_groups: int, ratios: Mapping[str, float]) -> dict[str, int]:
    positive_splits = [name for name in SPLIT_NAMES if ratios[name] > 0]
    if number_of_groups < len(positive_splits):
        raise SplitError(
            f"{number_of_groups} groups cannot populate all {len(positive_splits)} "
            f"positive-ratio splits {positive_splits}; collect more groups or set "
            "an unused split ratio to 0"
        )

    ideals = {name: number_of_groups * ratios[name] for name in SPLIT_NAMES}
    counts = {name: math.floor(ideals[name]) for name in SPLIT_NAMES}
    remaining = number_of_groups - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(ideals[name] - counts[name]), SPLIT_NAMES.index(name)),
    )
    for name in order[:remaining]:
        counts[name] += 1

    for empty_name in [name for name in positive_splits if counts[name] == 0]:
        donors = [name for name in positive_splits if counts[name] > 1]
        if not donors:
            raise SplitError("could not allocate at least one group to every positive-ratio split")
        donor = max(
            donors,
            key=lambda name: (
                counts[name] - ideals[name],
                counts[name],
                -SPLIT_NAMES.index(name),
            ),
        )
        counts[donor] -= 1
        counts[empty_name] += 1
    if sum(counts.values()) != number_of_groups:
        raise AssertionError("internal split count allocation error")
    return counts


def _stable_group_key(seed: int, group: str) -> tuple[bytes, str]:
    digest = hashlib.sha256(f"{int(seed)}\0{group}".encode()).digest()
    return digest, group


def validate_no_group_leakage(
    splits: Mapping[str, Sequence[CanonicalRecord]], *, group_key: str
) -> None:
    memberships: dict[str, str] = {}
    for split_name, records in splits.items():
        for record in records:
            value = str(getattr(record, group_key))
            previous = memberships.setdefault(value, split_name)
            if previous != split_name:
                raise SplitError(
                    f"{group_key}={value!r} occurs in both {previous} and {split_name}"
                )


def validate_no_pair_leakage(
    splits: Mapping[str, Sequence[CanonicalRecord]],
) -> None:
    memberships: dict[str, str] = {}
    for split_name, records in splits.items():
        for record in records:
            if not record.pair_id:
                continue
            previous = memberships.setdefault(record.pair_id, split_name)
            if previous != split_name:
                raise SplitError(
                    f"pair_id={record.pair_id!r} occurs in both {previous} and {split_name}"
                )


def validate_explicit_pairs(
    records: Iterable[CanonicalRecord],
    *,
    enabled: bool,
    require_same_subject: bool = True,
    require_same_target: bool = True,
    max_target_distance_normalized: float = 0.0,
    unpaired_policy: str = "branch_only",
) -> PairValidationResult:
    """Validate explicit pair IDs without ever inferring pairs from filenames.

    A complete pair contains exactly one ``front`` and one ``side`` record.
    Empty or singleton pairs follow ``branch_only``, ``drop``, or ``error``.
    Duplicate records for a view are always ambiguous and therefore errors.
    """

    records_tuple = tuple(records)
    if not enabled:
        return PairValidationResult(
            records=records_tuple,
            complete_pair_ids=frozenset(),
            incomplete_pair_ids=frozenset(),
            unpaired_sample_ids=frozenset(
                record.sample_id for record in records_tuple if not record.pair_id
            ),
            dropped_sample_ids=frozenset(),
        )

    if unpaired_policy not in {"branch_only", "drop", "error"}:
        raise PairValidationError("unpaired_policy must be branch_only, drop, or error")
    tolerance = float(max_target_distance_normalized)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise PairValidationError("max_target_distance_normalized must be finite and non-negative")

    groups: dict[str, list[CanonicalRecord]] = {}
    no_pair_id: list[CanonicalRecord] = []
    for record in records_tuple:
        if record.pair_id:
            groups.setdefault(record.pair_id, []).append(record)
        else:
            no_pair_id.append(record)

    complete: set[str] = set()
    incomplete: set[str] = set()
    invalid_unpaired_ids = {record.sample_id for record in no_pair_id}
    for pair_id, pair_records in sorted(groups.items()):
        by_view: dict[str, list[CanonicalRecord]] = {}
        for record in pair_records:
            by_view.setdefault(record.view, []).append(record)
        duplicate_views = {view: items for view, items in by_view.items() if len(items) > 1}
        if duplicate_views:
            details = ", ".join(
                f"{view}={len(items)}" for view, items in sorted(duplicate_views.items())
            )
            raise PairValidationError(
                f"pair_id {pair_id!r} is ambiguous ({details}); expected at most one per view"
            )
        if set(by_view) != {"front", "side"}:
            incomplete.add(pair_id)
            invalid_unpaired_ids.update(record.sample_id for record in pair_records)
            continue

        front = by_view["front"][0]
        side = by_view["side"][0]
        if require_same_subject and front.subject_id != side.subject_id:
            raise PairValidationError(
                f"pair_id {pair_id!r} crosses subjects {front.subject_id!r} and {side.subject_id!r}"
            )
        if require_same_target:
            distance = _normalized_target_distance(front, side)
            if distance > tolerance + 1e-12:
                raise PairValidationError(
                    f"pair_id {pair_id!r} target mismatch: normalized distance "
                    f"{distance:.12g} exceeds {tolerance:.12g}"
                )
        complete.add(pair_id)

    if unpaired_policy == "error" and invalid_unpaired_ids:
        preview = sorted(invalid_unpaired_ids)[:10]
        suffix = "..." if len(invalid_unpaired_ids) > len(preview) else ""
        raise PairValidationError(
            f"pairing requires complete front/side pairs; unpaired samples: {preview}{suffix}"
        )

    dropped: set[str] = set()
    if unpaired_policy == "drop":
        dropped = invalid_unpaired_ids
        kept = tuple(record for record in records_tuple if record.sample_id not in dropped)
    else:
        kept = records_tuple

    return PairValidationResult(
        records=kept,
        complete_pair_ids=frozenset(complete),
        incomplete_pair_ids=frozenset(incomplete),
        unpaired_sample_ids=frozenset(invalid_unpaired_ids),
        dropped_sample_ids=frozenset(dropped),
    )


def _normalized_target_distance(first: CanonicalRecord, second: CanonicalRecord) -> float:
    first_xy = first.gaze_screen_xy_normalized
    second_xy = second.gaze_screen_xy_normalized
    return math.hypot(first_xy[0] - second_xy[0], first_xy[1] - second_xy[1])


# Concise alias for callers that do not need to know the implementation name.
split_records = deterministic_group_split
