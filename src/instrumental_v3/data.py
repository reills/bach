from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.instrumental_v3.representation import FIELD_NAMES, InstrumentalV3Piece


class InstrumentalV3Dataset(Dataset[torch.Tensor]):
    def __init__(self, pieces: list[InstrumentalV3Piece], *, seq_len: int) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.pieces = pieces
        self.seq_len = seq_len
        self.windows: list[tuple[int, int]] = []
        stride = max(1, seq_len // 2)
        for piece_idx, piece in enumerate(pieces):
            n = len(piece.slices)
            if n < seq_len:
                continue
            for start in range(0, n - seq_len + 1, stride):
                self.windows.append((piece_idx, start))
            if self.windows and self.windows[-1] != (piece_idx, n - seq_len):
                self.windows.append((piece_idx, n - seq_len))
        if not self.windows:
            raise ValueError("no training windows; reduce seq_len or add longer pieces")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        piece_idx, start = self.windows[index]
        rows = [slice_.values for slice_ in self.pieces[piece_idx].slices[start : start + self.seq_len]]
        return torch.tensor(rows, dtype=torch.long)


def load_dataset(path: str | Path) -> tuple[list[InstrumentalV3Piece], dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    pieces = [InstrumentalV3Piece.from_dict(item) for item in data["pieces"]]
    meta = dict(data.get("meta", {}))
    if meta.get("field_names") and list(meta["field_names"]) != FIELD_NAMES:
        raise ValueError("dataset field_names do not match current representation")
    return pieces, meta


def save_dataset(path: str | Path, pieces: list[InstrumentalV3Piece], *, meta: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"field_names": FIELD_NAMES, **(meta or {})},
        "pieces": [piece.to_dict() for piece in pieces],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
