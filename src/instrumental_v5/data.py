from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.instrumental_v5.representation import (
    CADENCE_TYPE_TO_ID,
    CONTOUR_BUCKET_TO_ID,
    HARMONIC_FUNCTION_TO_ID,
    PHRASE_ROLE_TO_ID,
    RHYTHM_BUCKET_TO_ID,
    SPEAC_LABEL_TO_ID,
    V5_FEATURE_SPECS,
    V5_FIELD_NAMES,
)


class V5SliceDataset(Dataset[torch.Tensor]):
    def __init__(self, events: pd.DataFrame, *, seq_len: int, split: str = "train") -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        missing = [name for name in V5_FIELD_NAMES if name not in events.columns]
        if missing:
            raise ValueError(f"events dataframe missing v5 fields: {missing}")

        self.seq_len = seq_len
        self.events = events[events["split"] == split].copy() if "split" in events.columns else events.copy()
        self.events.sort_values(["piece_id", "row_index"], inplace=True)
        self.groups: list[pd.DataFrame] = [group for _, group in self.events.groupby("piece_id", sort=False)]
        self.windows: list[tuple[int, int]] = []
        stride = max(1, seq_len // 2)
        for group_idx, group in enumerate(self.groups):
            n = len(group)
            if n < seq_len:
                continue
            for start in range(0, n - seq_len + 1, stride):
                self.windows.append((group_idx, start))
            if self.windows and self.windows[-1] != (group_idx, n - seq_len):
                self.windows.append((group_idx, n - seq_len))
        if not self.windows:
            raise ValueError("no v5 training windows; reduce seq_len or add longer pieces")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        group_idx, start = self.windows[index]
        rows = self.groups[group_idx].iloc[start : start + self.seq_len][V5_FIELD_NAMES]
        return torch.tensor(rows.to_numpy(dtype="int64"), dtype=torch.long)


def load_events(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_events(path: str | Path, events: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(path, index=False)


def build_v5_vocab() -> dict[str, Any]:
    return {
        "field_names": V5_FIELD_NAMES,
        "feature_specs": V5_FEATURE_SPECS,
        "phrase_role": PHRASE_ROLE_TO_ID,
        "speac_label": SPEAC_LABEL_TO_ID,
        "cadence_target": CADENCE_TYPE_TO_ID,
        "harmonic_function": HARMONIC_FUNCTION_TO_ID,
        "local_key_pc": {str(value): value for value in range(13)},
        "retrieved_contour_bucket": CONTOUR_BUCKET_TO_ID,
        "retrieved_rhythm_bucket": RHYTHM_BUCKET_TO_ID,
    }


def save_vocab(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(build_v5_vocab(), f, indent=2, sort_keys=True)
