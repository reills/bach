from __future__ import annotations

from src.emi.fragments import FragmentQuery, extract_fragments, fragment_from_jsonl, fragment_to_jsonl, rank_fragments
from src.instrumental_v3.representation import FIELD_NAMES, InstrumentalV3Piece, SliceEvent


def _toy_piece() -> InstrumentalV3Piece:
    rows = []
    p0 = [48, 50, 52, 53, 55, 57, 59, 60]
    p1 = [60, 62, 64, 65, 67, 69, 71, 72]
    for idx in range(16):
        bar = idx // 4
        pos = idx % 4
        note = idx % 2 == 0
        low = p0[min(idx // 2, len(p0) - 1)]
        high = p1[min(idx // 2, len(p1) - 1)]
        state0 = 2 if note else 1
        state1 = 2 if note else 1
        row = [
            bar,
            pos,
            bar % 8,
            1 if bar >= 2 else 0,
            0,
            0,
            2,
            state0,
            low,
            25,
            2,
            0 if note else 1,
            1,
            state1,
            high,
            25,
            2,
            0 if note else 1,
            1,
            abs(high - low) + 1,
            2,
            abs(high - low) + 1,
        ]
        assert len(row) == len(FIELD_NAMES)
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


def test_extract_fragments_records_interval_rhythm_signature() -> None:
    fragments = extract_fragments(_toy_piece(), length_slices=4, hop_slices=4, min_notes=2)

    assert fragments
    first = fragments[0]
    assert first.id == "toy_v0_s0_l4"
    assert first.phrase_role == "ANSWER_ENTRY"
    assert first.melodic_intervals == [2]
    assert first.rhythm_steps == [2, 2]
    assert first.vertical_intervals == [12, 12, 12, 12]
    assert first.fingerprint


def test_fragment_jsonl_round_trip() -> None:
    fragment = extract_fragments(_toy_piece(), length_slices=4, hop_slices=4, min_notes=2)[0]

    restored = fragment_from_jsonl(fragment_to_jsonl(fragment))

    assert restored == fragment


def test_rank_fragments_prefers_compatible_query() -> None:
    fragments = extract_fragments(_toy_piece(), length_slices=4, hop_slices=4, min_notes=2)
    query = FragmentQuery(
        voice=1,
        phrase_role="CADENCE",
        key_pc=0,
        mode=0,
        previous_end_pitch=69,
        previous_end_degree=5,
        avoid_piece_id="other",
    )

    matches = rank_fragments(query, fragments, limit=3)

    assert matches[0].fragment.voice == 1
    assert matches[0].fragment.phrase_role == "CADENCE"
    assert matches[0].score > matches[-1].score
