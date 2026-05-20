from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.make_v5_listening_batch import (
    RATING_COLUMNS,
    _reset_positions,
    _sanity_metrics,
    _write_notes,
    _write_ratings_template,
)
from scripts.generate_instrumental_v5 import _repair_generated_row
from scripts.generate_instrumental_v5 import _score_candidate_rows
from src.instrumental_v3.representation import InstrumentalV3Piece
from src.instrumental_v3.representation import STATE_NOTE
from src.instrumental_v5.representation import V5_FIELD_NAMES


class _Args:
    out_dir = "out/listening_batches/test_batch"
    checkpoint = "out/instrumental_v5_overfit_sample/checkpoint_step1000.pt"
    temperature = 0.8
    top_p = 0.95
    seed = 42
    samples = 2
    max_new_tokens = 512


class _Template:
    steps_per_bar = 16
    key_pc = 0
    mode = 0


def test_listening_batch_writes_rating_template_and_notes(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings_template.csv"
    notes_path = tmp_path / "notes.md"

    _write_ratings_template(ratings_path, ["sample_001", "sample_002"])
    _write_notes(notes_path, args=_Args(), sample_ids=["sample_001", "sample_002"])

    with ratings_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == RATING_COLUMNS
    assert [row["sample_id"] for row in rows] == ["sample_001", "sample_002"]
    assert "playable_on_guitar_1_5" not in rows[0]

    notes = notes_path.read_text(encoding="utf-8")
    assert "# Listening batch: test_batch" in notes
    assert "Does it sound like intentional Baroque counterpoint?" in notes
    assert "continuation-only" in notes
    assert "### sample_001" in notes
    assert "Main failure:" in notes
    assert "Guitar playable:" not in notes


def test_listening_batch_sanity_metrics_are_slim() -> None:
    report = {
        "invalid_pitch_state_rate": 0.0,
        "voice_crossing_rate": 0.125,
        "v0_stuck_rate": 0.05,
        "v1_stuck_rate": 0.12,
        "repeated_sonority_rate": 0.08,
        "slice_count": 128,
        "v0_note_rate": 0.7,
        "v1_note_rate": 0.6,
    }

    metrics = _sanity_metrics(report, template=_Template())

    assert metrics == {
        "invalid_pitch_state_rate": 0.0,
        "voice_crossing_rate": 0.125,
        "stuck_voice_rate": 0.12,
        "repeated_sonority_rate": 0.08,
        "num_bars_estimate": 8,
        "num_voices_active": 2,
    }
    assert "melodic_interval_distribution" not in json.dumps(metrics)


def test_reset_positions_removes_prompt_offset() -> None:
    rows = []
    for idx in range(3):
        row = [0] * len(V5_FIELD_NAMES)
        row[0] = 20
        row[1] = 10 + idx
        rows.append(row)

    reset = _reset_positions(rows, template=_Template())

    assert [row[0] for row in reset] == [0, 0, 0]
    assert [row[1] for row in reset] == [0, 1, 2]


def test_repair_generated_v5_row_clamps_sampled_pitch_to_valid_midi() -> None:
    previous = [0] * len(V5_FIELD_NAMES)
    row = [0] * len(V5_FIELD_NAMES)
    for voice in (0, 1):
        previous[V5_FIELD_NAMES.index(f"v{voice}_state")] = STATE_NOTE
        previous[V5_FIELD_NAMES.index(f"v{voice}_pitch")] = 60 + voice * 12
        row[V5_FIELD_NAMES.index(f"v{voice}_state")] = STATE_NOTE
        row[V5_FIELD_NAMES.index(f"v{voice}_pitch")] = 128

    _repair_generated_row(row, [previous], _Template())

    assert row[V5_FIELD_NAMES.index("v0_pitch")] == 127
    assert row[V5_FIELD_NAMES.index("v1_pitch")] == 127


def test_candidate_score_prefers_valid_non_crossed_two_voice_motion() -> None:
    template = InstrumentalV3Piece(
        piece_id="toy",
        source_path="toy.musicxml",
        tpq=24,
        grid_ticks=6,
        time_signature="4/4",
        key="C",
        key_pc=0,
        mode=0,
        bar_len_ticks=96,
        steps_per_bar=16,
        slices=[],
    )
    good_rows = [_candidate_row(60 + idx, 84 - idx) for idx in range(8)]
    crossed_rows = [_candidate_row(72 + idx, 60 - idx) for idx in range(8)]

    good_score, good_diagnostics = _score_candidate_rows(
        good_rows,
        template=template,
        prompt_row_count=0,
        source_pieces=[],
    )
    crossed_score, crossed_diagnostics = _score_candidate_rows(
        crossed_rows,
        template=template,
        prompt_row_count=0,
        source_pieces=[],
    )

    assert good_score > crossed_score
    assert good_diagnostics["voice_crossing_rate"] == 0.0
    assert crossed_diagnostics["voice_crossing_rate"] == 1.0


def _candidate_row(v0_pitch: int, v1_pitch: int) -> list[int]:
    row = [0] * len(V5_FIELD_NAMES)
    row[V5_FIELD_NAMES.index("key_pc")] = 0
    row[V5_FIELD_NAMES.index("mode")] = 0
    row[V5_FIELD_NAMES.index("voice_count")] = 2
    for voice, pitch in [(0, v0_pitch), (1, v1_pitch)]:
        row[V5_FIELD_NAMES.index(f"v{voice}_state")] = STATE_NOTE
        row[V5_FIELD_NAMES.index(f"v{voice}_pitch")] = pitch
        row[V5_FIELD_NAMES.index(f"v{voice}_mel")] = 25
        row[V5_FIELD_NAMES.index(f"v{voice}_dur")] = 1
    interval = abs(v1_pitch - v0_pitch) + 1
    row[V5_FIELD_NAMES.index("vertical_interval")] = interval
    row[V5_FIELD_NAMES.index("spacing")] = interval
    return row
