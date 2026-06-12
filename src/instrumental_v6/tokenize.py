from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.instrumental_v6.representation import (
    GLOBAL_FEATURE_SPECS,
    GLOBAL_FIELD_NAMES,
    PAIR_FEATURE_SPECS,
    PAIR_FIELD_NAMES,
    VOICE_FEATURE_SPECS,
    VOICE_FIELD_NAMES,
    InstrumentalV6Piece,
)


@dataclass(frozen=True)
class TokenizedSplit:
    global_values: torch.Tensor
    voice_values: torch.Tensor
    pair_values: torch.Tensor
    mask: torch.Tensor
    piece_ids: list[str]


def build_tokenized_split(
    pieces: list[InstrumentalV6Piece],
    *,
    piece_ids: set[str],
    seq_len: int,
    stride: int | None = None,
) -> TokenizedSplit:
    stride = stride or max(1, seq_len // 2)
    global_windows: list[torch.Tensor] = []
    voice_windows: list[torch.Tensor] = []
    pair_windows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    window_piece_ids: list[str] = []
    selected = [piece for piece in pieces if piece.piece_id in piece_ids]
    if not selected:
        raise ValueError("no pieces selected for tokenization")
    max_voices = selected[0].max_voices
    if any(piece.max_voices != max_voices for piece in selected):
        raise ValueError("all v6 pieces must share max_voices")
    for piece in selected:
        length = len(piece.global_rows)
        for start in _window_starts(length, seq_len=seq_len, stride=stride):
            window_length = min(seq_len, length - start)
            if window_length < 2:
                continue
            global_window = torch.zeros((seq_len, len(GLOBAL_FIELD_NAMES)), dtype=torch.long)
            voice_window = torch.zeros(
                (seq_len, max_voices, len(VOICE_FIELD_NAMES)),
                dtype=torch.long,
            )
            pair_window = torch.zeros(
                (seq_len, max_voices, max_voices, len(PAIR_FIELD_NAMES)),
                dtype=torch.long,
            )
            mask = torch.zeros(seq_len, dtype=torch.bool)
            global_window[:window_length] = torch.tensor(
                piece.global_rows[start : start + window_length],
                dtype=torch.long,
            )
            voice_window[:window_length] = torch.tensor(
                piece.voice_rows[start : start + window_length],
                dtype=torch.long,
            )
            pair_window[:window_length] = torch.tensor(
                piece.pair_rows[start : start + window_length],
                dtype=torch.long,
            )
            mask[:window_length] = True
            global_windows.append(global_window)
            voice_windows.append(voice_window)
            pair_windows.append(pair_window)
            masks.append(mask)
            window_piece_ids.append(piece.piece_id)
    if not global_windows:
        raise ValueError("no v6 tokenized windows")
    return TokenizedSplit(
        global_values=torch.stack(global_windows),
        voice_values=torch.stack(voice_windows),
        pair_values=torch.stack(pair_windows),
        mask=torch.stack(masks),
        piece_ids=window_piece_ids,
    )


def save_tokenized_split(path: str | Path, split: TokenizedSplit, *, max_voices: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_values": split.global_values,
            "voice_values": split.voice_values,
            "pair_values": split.pair_values,
            "mask": split.mask,
            "piece_ids": split.piece_ids,
            "max_voices": max_voices,
            "global_field_names": GLOBAL_FIELD_NAMES,
            "voice_field_names": VOICE_FIELD_NAMES,
            "pair_field_names": PAIR_FIELD_NAMES,
            "global_feature_specs": GLOBAL_FEATURE_SPECS,
            "voice_feature_specs": VOICE_FEATURE_SPECS,
            "pair_feature_specs": PAIR_FEATURE_SPECS,
        },
        path,
    )


def load_tokenized_split(path: str | Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("global_field_names") != GLOBAL_FIELD_NAMES:
        raise ValueError("v6 global fields do not match")
    if data.get("voice_field_names") != VOICE_FIELD_NAMES:
        raise ValueError("v6 voice fields do not match")
    if data.get("pair_field_names") != PAIR_FIELD_NAMES:
        raise ValueError("v6 pair fields do not match")
    return data


def _window_starts(length: int, *, seq_len: int, stride: int) -> list[int]:
    if length <= seq_len:
        return [0]
    starts = list(range(0, length - seq_len + 1, stride))
    final = length - seq_len
    if starts[-1] != final:
        starts.append(final)
    return starts
