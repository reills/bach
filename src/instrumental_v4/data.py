from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.instrumental_v4.representation import V4Piece, V4_FIELD_NAMES, PLAN_FIELD_NAMES


class V4SliceDataset(Dataset[torch.Tensor]):
    def __init__(self, pieces: list[V4Piece], *, seq_len: int) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.pieces = pieces
        self.seq_len = seq_len
        self.windows: list[tuple[int, int]] = []
        stride = max(1, seq_len // 2)
        for piece_idx, piece in enumerate(pieces):
            n = len(piece.rows)
            if n < seq_len:
                continue
            for start in range(0, n - seq_len + 1, stride):
                self.windows.append((piece_idx, start))
            if self.windows and self.windows[-1] != (piece_idx, n - seq_len):
                self.windows.append((piece_idx, n - seq_len))
        if not self.windows:
            raise ValueError("no slice windows; reduce seq_len or add longer pieces")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        piece_idx, start = self.windows[index]
        rows = self.pieces[piece_idx].rows[start : start + self.seq_len]
        return torch.tensor(rows, dtype=torch.long)


class V4PlanDataset(Dataset[torch.Tensor]):
    def __init__(self, pieces: list[V4Piece], *, seq_len: int) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.pieces = pieces
        self.seq_len = seq_len
        self.windows: list[tuple[int, int]] = []
        stride = max(1, seq_len // 2)
        for piece_idx, piece in enumerate(pieces):
            n = len(piece.plans)
            if n < seq_len:
                continue
            for start in range(0, n - seq_len + 1, stride):
                self.windows.append((piece_idx, start))
            if self.windows and self.windows[-1] != (piece_idx, n - seq_len):
                self.windows.append((piece_idx, n - seq_len))
        if not self.windows:
            raise ValueError("no plan windows; reduce seq_len or add longer pieces")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        piece_idx, start = self.windows[index]
        rows = [plan.values for plan in self.pieces[piece_idx].plans[start : start + self.seq_len]]
        return torch.tensor(rows, dtype=torch.long)


def save_v4_dataset(path: str | Path, pieces: list[V4Piece], *, meta: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "v4_field_names": V4_FIELD_NAMES,
            "plan_field_names": PLAN_FIELD_NAMES,
            **(meta or {}),
        },
        "pieces": [piece.to_dict() for piece in pieces],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_v4_dataset(path: str | Path) -> tuple[list[V4Piece], dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    meta = dict(data.get("meta", {}))
    if meta.get("v4_field_names") and list(meta["v4_field_names"]) != V4_FIELD_NAMES:
        raise ValueError("v4 field names do not match current representation")
    if meta.get("plan_field_names") and list(meta["plan_field_names"]) != PLAN_FIELD_NAMES:
        raise ValueError("plan field names do not match current representation")
    return [V4Piece.from_dict(item) for item in data["pieces"]], meta
