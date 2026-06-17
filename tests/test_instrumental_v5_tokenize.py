from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from scripts.make_instrumental_v5_dataset import build_v5_outputs
from src.instrumental_v3.representation import FIELD_NAMES as V3_FIELD_NAMES, InstrumentalV3Piece, SliceEvent
from src.instrumental_v4.representation import build_v4_piece
from src.instrumental_v5.model import (
    build_generator,
    generation_field_weights,
    generation_target_masks,
    masked_multihead_loss,
)
from src.instrumental_v5.representation import V5_FEATURE_SPECS, V5_FIELD_NAMES
from src.instrumental_v5.tokenize import (
    build_tokenized_split,
    load_tokenized_split,
    load_v5_vocab,
    save_tokenized_split,
    validate_v5_vocab,
)
from src.instrumental_v4.model import CompoundConfig


def _toy_v3_piece(piece_id: str) -> InstrumentalV3Piece:
    rows = []
    for idx in range(20):
        bar = idx // 5
        pos = idx % 5
        note = pos in {0, 2, 4}
        state0 = 2 if note else 1
        state1 = 2 if note else 1
        p0 = 48 + (idx % 7)
        p1 = 60 + (idx % 5)
        row = [
            bar,
            pos,
            bar % 8,
            1 if bar >= 2 else 0,
            0,
            0,
            2,
            state0,
            p0,
            25,
            2,
            0 if note else 1,
            1,
            state1,
            p1,
            25,
            2,
            0 if note else 1,
            1,
            abs(p1 - p0) + 1,
            2,
            abs(p1 - p0) + 1,
        ]
        assert len(row) == len(V3_FIELD_NAMES)
        rows.append(row)
    return InstrumentalV3Piece(
        piece_id=piece_id,
        source_path=f"{piece_id}.musicxml",
        tpq=24,
        grid_ticks=6,
        time_signature="5/4",
        key="C",
        key_pc=0,
        mode=0,
        bar_len_ticks=30,
        steps_per_bar=5,
        slices=[SliceEvent(row) for row in rows],
    )


def _write_v5_dataset(tmp_path: Path) -> Path:
    pieces = [build_v4_piece(_toy_v3_piece("tok_a")), build_v4_piece(_toy_v3_piece("tok_b"))]
    build_v5_outputs(
        pieces,
        output_dir=tmp_path,
        source_dataset="toy_v4.json",
        length_slices=4,
        hop_slices=2,
        val_split=0.5,
        seed=11,
    )
    return tmp_path


def test_tokenized_split_builds_padded_fixed_length_windows(tmp_path: Path) -> None:
    data_dir = _write_v5_dataset(tmp_path)
    events = pd.read_parquet(data_dir / "events.parquet")

    train = build_tokenized_split(events, split="train", seq_len=32, stride=16)

    assert train.windows.shape[1:] == (32, len(V5_FIELD_NAMES))
    assert train.mask.dtype == torch.bool
    assert train.mask.shape == train.windows.shape[:2]
    assert train.lengths.max().item() <= 32
    assert train.lengths.min().item() >= 2
    for field_idx, field in enumerate(V5_FIELD_NAMES):
        assert int(train.windows[..., field_idx].max()) < V5_FEATURE_SPECS[field]


def test_tokenized_split_round_trip_metadata_and_no_fragment_ids(tmp_path: Path) -> None:
    data_dir = _write_v5_dataset(tmp_path)
    events = pd.read_parquet(data_dir / "events.parquet")
    train = build_tokenized_split(events, split="train", seq_len=16, stride=8)
    path = data_dir / "tokenized" / "train.pt"

    save_tokenized_split(path, train)
    restored = load_tokenized_split(path)

    assert restored["windows"].shape == train.windows.shape
    assert restored["field_names"] == V5_FIELD_NAMES
    raw_metadata = json.dumps(
        {
            "piece_ids": restored["piece_ids"],
            "field_names": restored["field_names"],
            "feature_specs": restored["feature_specs"],
        }
    )
    assert "fragment_id" not in raw_metadata
    assert "tok_a_v0" not in raw_metadata


def test_v5_vocab_rejects_raw_fragment_id_entries(tmp_path: Path) -> None:
    data_dir = _write_v5_dataset(tmp_path)
    vocab = load_v5_vocab(data_dir / "vocab.json")
    vocab["fragment_id"] = {"tok_a_v0_s0_l4": 0}

    with pytest.raises(ValueError, match="raw fragment IDs"):
        validate_v5_vocab(vocab)


def test_masked_v5_loss_ignores_padded_tail(tmp_path: Path) -> None:
    data_dir = _write_v5_dataset(tmp_path)
    events = pd.read_parquet(data_dir / "events.parquet")
    train = build_tokenized_split(events, split="train", seq_len=16, stride=16)
    config = CompoundConfig(d_model=32, n_heads=4, n_layers=1, dropout=0.0, max_seq_len=16)
    model = build_generator(config)

    logits = model(train.windows[:1, :-1, :])
    loss, metrics = masked_multihead_loss(logits, train.windows[:1, 1:, :], train.mask[:1, 1:])

    assert loss.item() > 0
    assert "phrase_role_acc" in metrics
    assert "speac_label_acc" in metrics
    assert "retrieved_contour_bucket_acc" in metrics


def test_v5_generation_target_masks_focus_note_fields_on_onsets() -> None:
    targets = torch.zeros((1, 4, len(V5_FIELD_NAMES)), dtype=torch.long)
    mask = torch.ones((1, 4), dtype=torch.bool)
    state_i = V5_FIELD_NAMES.index("v0_state")
    mel_i = V5_FIELD_NAMES.index("v0_mel")
    targets[0, :, state_i] = torch.tensor([0, 1, 2, 2])
    targets[0, :, mel_i] = torch.tensor([0, 0, 0, 27])

    field_masks = generation_target_masks(targets, mask)

    assert field_masks["v0_state"].sum().item() == 4
    assert field_masks["v0_pitch"].sum().item() == 2
    assert field_masks["v0_dur"].sum().item() == 2
    assert field_masks["v0_mel"].sum().item() == 1


def test_v5_loss_reports_onset_conditioned_target_counts() -> None:
    targets = torch.zeros((1, 3, len(V5_FIELD_NAMES)), dtype=torch.long)
    mask = torch.ones((1, 3), dtype=torch.bool)
    targets[0, :, V5_FIELD_NAMES.index("v0_state")] = torch.tensor([1, 2, 2])
    targets[0, :, V5_FIELD_NAMES.index("v0_mel")] = torch.tensor([0, 0, 27])
    logits = {
        name: torch.zeros((1, 3, V5_FEATURE_SPECS[name]), dtype=torch.float32)
        for name in V5_FIELD_NAMES
    }

    loss, metrics = masked_multihead_loss(
        logits,
        targets,
        mask,
        field_weights=generation_field_weights(),
        field_masks=generation_target_masks(targets, mask),
    )

    assert loss.item() > 0
    assert metrics["v0_state_count"] == 3
    assert metrics["v0_pitch_count"] == 2
    assert metrics["v0_mel_count"] == 1
