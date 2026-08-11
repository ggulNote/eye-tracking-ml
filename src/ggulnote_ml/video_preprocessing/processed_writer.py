from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

from ggulnote_ml.video_preprocessing.contracts import PROCESSED_FEATURE_COLUMNS


def write_processed_feature_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PROCESSED_FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
