from __future__ import annotations

import torch

from src.instrumental_v3.representation import FIELD_NAMES as V3_FIELD_NAMES, InstrumentalV3Piece, SliceEvent
from src.instrumental_v4.data import V4PlanDataset, V4SliceDataset
from src.instrumental_v4.model import CompoundConfig, build_generator, build_planner
from src.instrumental_v4.representation import PLAN_FIELD_NAMES, V4_FIELD_NAMES, build_v4_piece


def _toy_v3_piece() -> InstrumentalV3Piece:
    rows = []
    for idx in range(16):
        bar = idx // 4
        pos = idx % 4
        state0 = 2 if pos in {0, 2} else 1
        state1 = 2 if pos in {1, 3} else 1
        p0 = 48 + bar + pos
        p1 = 60 + bar - pos
        row = [
            bar,
            pos,
            bar % 8,
            0,
            0,
            0,
            2,
            state0,
            p0,
            25,
            1,
            0 if state0 == 2 else 1,
            1,
            state1,
            p1,
            25,
            1,
            0 if state1 == 2 else 1,
            1,
            abs(p1 - p0) + 1,
            2,
            abs(p1 - p0) + 1,
        ]
        assert len(row) == len(V3_FIELD_NAMES)
        rows.append(row)
    return InstrumentalV3Piece(
        piece_id="toy",
        source_path="toy.musicxml",
        tpq=24,
        grid_ticks=6,
        time_signature="1/1",
        key="C",
        key_pc=0,
        mode=0,
        bar_len_ticks=24,
        steps_per_bar=4,
        slices=[SliceEvent(row) for row in rows],
    )


def test_build_v4_piece_repeats_measure_plan_on_rows() -> None:
    piece = build_v4_piece(_toy_v3_piece())
    assert len(piece.plans) == 4
    assert len(piece.rows) == 16
    assert len(piece.rows[0]) == len(V4_FIELD_NAMES)
    for bar in range(4):
        expected = piece.plans[bar].values
        for row in piece.rows[bar * 4 : (bar + 1) * 4]:
            assert row[-len(PLAN_FIELD_NAMES) :] == expected


def test_v4_datasets_and_models_forward() -> None:
    piece = build_v4_piece(_toy_v3_piece())
    plan_batch = V4PlanDataset([piece], seq_len=3)[0].unsqueeze(0)
    slice_batch = V4SliceDataset([piece], seq_len=8)[0].unsqueeze(0)
    config = CompoundConfig(d_model=32, n_heads=4, n_layers=1, dropout=0.0, max_seq_len=8)
    planner = build_planner(config)
    generator = build_generator(config)
    plan_logits = planner(plan_batch)
    slice_logits = generator(slice_batch)
    assert set(plan_logits) == set(PLAN_FIELD_NAMES)
    assert set(slice_logits) == set(V4_FIELD_NAMES)
    assert plan_logits[PLAN_FIELD_NAMES[0]].shape[:2] == torch.Size([1, 3])
    assert slice_logits[V4_FIELD_NAMES[0]].shape[:2] == torch.Size([1, 8])
