from pathlib import Path

import pytest

from gaze_pipeline.data.records import CanonicalRecord, ScreenCalibration
from gaze_pipeline.data.split import (
    PairValidationError,
    SplitError,
    deterministic_group_split,
    validate_explicit_pairs,
)


def record(
    subject: str,
    *,
    view: str = "front",
    pair_id: str | None = None,
    target: tuple[float, float] = (50.0, 50.0),
) -> CanonicalRecord:
    suffix = f"{view}-{pair_id or 'single'}"
    return CanonicalRecord(
        sample_id=f"{subject}-{suffix}",
        subject_id=subject,
        session_id="day01",
        view=view,
        pair_id=pair_id,
        image_path=Path(f"/data/{subject}/{suffix}.jpg"),
        image_relative_path=f"{subject}/{suffix}.jpg",
        image_width_px=100,
        image_height_px=100,
        image_channels=3,
        gaze_screen_xy_px=target,
        screen=ScreenCalibration(width_px=100, height_px=100),
    )


def test_grouped_ratio_split_is_deterministic_and_subject_disjoint() -> None:
    records = [record(f"p{index:02d}") for index in range(6)]

    first = deterministic_group_split(records, seed=42)
    second = deterministic_group_split(list(reversed(records)), seed=42)

    assert {name: [item.sample_id for item in items] for name, items in first.items()} == {
        name: [item.sample_id for item in items] for name, items in second.items()
    }
    assert {name: len(items) for name, items in first.items()} == {
        "train": 4,
        "validation": 1,
        "test": 1,
    }
    subjects = {name: {item.subject_id for item in items} for name, items in first.items()}
    assert subjects["train"].isdisjoint(subjects["validation"])
    assert subjects["train"].isdisjoint(subjects["test"])
    assert subjects["validation"].isdisjoint(subjects["test"])


def test_five_subject_ratio_is_three_one_one_and_seed_controls_assignment() -> None:
    records = [record(f"p{index:02d}") for index in range(5)]

    seed_42 = deterministic_group_split(records, seed=42)
    seed_42_again = deterministic_group_split(list(reversed(records)), seed=42)
    seed_1 = deterministic_group_split(records, seed=1)

    subjects_42 = {name: {item.subject_id for item in items} for name, items in seed_42.items()}
    subjects_42_again = {
        name: {item.subject_id for item in items} for name, items in seed_42_again.items()
    }
    subjects_1 = {name: {item.subject_id for item in items} for name, items in seed_1.items()}
    assert {name: len(items) for name, items in subjects_42.items()} == {
        "train": 3,
        "validation": 1,
        "test": 1,
    }
    assert subjects_42_again == subjects_42
    assert subjects_1 != subjects_42
    assert subjects_42["train"].isdisjoint(subjects_42["validation"] | subjects_42["test"])
    assert subjects_42["validation"].isdisjoint(subjects_42["test"])


def test_explicit_front_side_pair_is_accepted() -> None:
    records = [
        record("p01", view="front", pair_id="g1"),
        record("p01", view="side", pair_id="g1"),
    ]

    result = validate_explicit_pairs(records, enabled=True)

    assert result.complete_pair_ids == {"g1"}
    assert not result.incomplete_pair_ids


def test_pair_target_mismatch_is_rejected() -> None:
    records = [
        record("p01", view="front", pair_id="g1", target=(50.0, 50.0)),
        record("p01", view="side", pair_id="g1", target=(60.0, 50.0)),
    ]

    with pytest.raises(PairValidationError, match="target mismatch"):
        validate_explicit_pairs(records, enabled=True)


def test_positive_ratio_splits_require_at_least_one_group_each() -> None:
    records = [record("p00"), record("p01")]

    with pytest.raises(SplitError, match="2 groups cannot populate all 3 positive-ratio splits"):
        deterministic_group_split(records)
