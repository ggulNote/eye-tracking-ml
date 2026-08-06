from __future__ import annotations

from typing import Dict, List

import numpy as np

from ggulnote_ml.config import DataConfig
from ggulnote_ml.contracts import DatasetSplits, RawDataset
from ggulnote_ml.exceptions import ContractError


def split_by_participant(dataset: RawDataset, config: DataConfig, seed: int) -> DatasetSplits:
    participant_ids = sorted({sample.participant_id for sample in dataset.samples})
    if len(participant_ids) < 3:
        raise ContractError(
            "Participant-level train/validation/test split requires at least three participants."
        )

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(participant_ids))
    train_count = max(1, int(round(len(shuffled) * config.train_fraction)))
    validation_count = max(1, int(round(len(shuffled) * config.validation_fraction)))
    if train_count + validation_count >= len(shuffled):
        train_count = max(1, len(shuffled) - 2)
        validation_count = 1

    groups: Dict[str, set] = {
        "train": set(shuffled[:train_count]),
        "validation": set(shuffled[train_count : train_count + validation_count]),
        "test": set(shuffled[train_count + validation_count :]),
    }
    indices: Dict[str, List[int]] = {name: [] for name in groups}
    for index, sample in enumerate(dataset.samples):
        for split_name, members in groups.items():
            if sample.participant_id in members:
                indices[split_name].append(index)
                break

    if any(not split_indices for split_indices in indices.values()):
        raise ContractError("Participant-level split produced an empty dataset split.")

    return DatasetSplits(
        train=dataset.subset(indices["train"]),
        validation=dataset.subset(indices["validation"]),
        test=dataset.subset(indices["test"]),
    )

