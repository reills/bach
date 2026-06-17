from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.instrumental_v5.representation import V5_FEATURE_SPECS, V5_FIELD_NAMES


@dataclass(frozen=True)
class TokenizedSplit:
    windows: torch.Tensor
    mask: torch.Tensor
    piece_ids: list[str]
    starts: torch.Tensor
    lengths: torch.Tensor


def load_v5_vocab(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    validate_v5_vocab(vocab)
    return vocab


def validate_v5_vocab(vocab: dict[str, Any]) -> None:
    if list(vocab.get("field_names", [])) != V5_FIELD_NAMES:
        raise ValueError("v5 vocab field_names do not match current representation")
    if dict(vocab.get("feature_specs", {})) != V5_FEATURE_SPECS:
        raise ValueError("v5 vocab feature_specs do not match current representation")
    raw = json.dumps(vocab)
    if "fragment_id" in raw or "FRAGMENT_ID" in raw:
        raise ValueError("raw fragment IDs must not enter the v5 LM vocab")


def validate_events(events: pd.DataFrame) -> None:
    missing = [field for field in [*V5_FIELD_NAMES, "piece_id", "split", "row_index"] if field not in events.columns]
    if missing:
        raise ValueError(f"events table missing required columns: {missing}")
    for field in V5_FIELD_NAMES:
        values = events[field]
        if not pd.api.types.is_integer_dtype(values):
            raise ValueError(f"events field {field!r} must have integer dtype")
        if len(values) == 0:
            continue
        low = int(values.min())
        high = int(values.max())
        if low < 0 or high >= V5_FEATURE_SPECS[field]:
            raise ValueError(
                f"events field {field!r} has value range [{low}, {high}], "
                f"expected [0, {V5_FEATURE_SPECS[field] - 1}]"
            )


def build_tokenized_split(
    events: pd.DataFrame,
    *,
    split: str,
    seq_len: int,
    stride: int | None = None,
    drop_short: bool = False,
) -> TokenizedSplit:
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    validate_events(events)
    stride = stride or max(1, seq_len // 2)
    if stride <= 0:
        raise ValueError("stride must be positive")

    split_events = events[events["split"] == split].copy()
    split_events.sort_values(["piece_id", "row_index"], inplace=True)

    windows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    piece_ids: list[str] = []
    starts: list[int] = []
    lengths: list[int] = []

    for piece_id, group in split_events.groupby("piece_id", sort=False):
        rows = torch.tensor(group[V5_FIELD_NAMES].to_numpy(dtype="int64"), dtype=torch.long)
        n_rows = int(rows.shape[0])
        if n_rows == 0:
            continue
        for start in _window_starts(n_rows, seq_len=seq_len, stride=stride, drop_short=drop_short):
            length = min(seq_len, n_rows - start)
            if length < 2:
                continue
            window = torch.zeros(seq_len, len(V5_FIELD_NAMES), dtype=torch.long)
            mask = torch.zeros(seq_len, dtype=torch.bool)
            window[:length] = rows[start : start + length]
            mask[:length] = True
            windows.append(window)
            masks.append(mask)
            piece_ids.append(str(piece_id))
            starts.append(start)
            lengths.append(length)

    if not windows:
        raise ValueError(f"no tokenized {split!r} windows; reduce seq_len or check split")

    return TokenizedSplit(
        windows=torch.stack(windows),
        mask=torch.stack(masks),
        piece_ids=piece_ids,
        starts=torch.tensor(starts, dtype=torch.long),
        lengths=torch.tensor(lengths, dtype=torch.long),
    )


def save_tokenized_split(path: str | Path, split: TokenizedSplit) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "windows": split.windows,
            "mask": split.mask,
            "piece_ids": split.piece_ids,
            "starts": split.starts,
            "lengths": split.lengths,
            "field_names": V5_FIELD_NAMES,
            "feature_specs": V5_FEATURE_SPECS,
        },
        path,
    )


def load_tokenized_split(path: str | Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    if list(data.get("field_names", [])) != V5_FIELD_NAMES:
        raise ValueError("tokenized split field_names do not match current representation")
    if dict(data.get("feature_specs", {})) != V5_FEATURE_SPECS:
        raise ValueError("tokenized split feature_specs do not match current representation")
    return data


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _window_starts(n_rows: int, *, seq_len: int, stride: int, drop_short: bool) -> list[int]:
    if n_rows < seq_len:
        return [] if drop_short else [0]
    starts = list(range(0, n_rows - seq_len + 1, stride))
    final = n_rows - seq_len
    if starts[-1] != final:
        starts.append(final)
    return starts
