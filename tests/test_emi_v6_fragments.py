from __future__ import annotations

from src.emi.fragments import FragmentQuery, fragment_from_jsonl, fragment_to_jsonl, rank_fragments
from src.emi.v6_fragments import extract_v6_fragments
from src.instrumental_v6.representation import (
    FORM_TO_ID,
    GLOBAL_FIELD_NAMES,
    ROLE_TO_ID,
    STATE_HOLD,
    STATE_NOTE,
    InstrumentalV6Piece,
)


def _toy_v6_piece() -> InstrumentalV6Piece:
    global_rows = []
    voice_rows = []
    for index in range(8):
        role = "SUBJECT_ENTRY" if index < 4 else "SEQUENCE"
        global_row = [0] * len(GLOBAL_FIELD_NAMES)
        global_row[GLOBAL_FIELD_NAMES.index("bar")] = index // 4
        global_row[GLOBAL_FIELD_NAMES.index("pos")] = index % 4
        global_row[GLOBAL_FIELD_NAMES.index("key_pc")] = 0
        global_row[GLOBAL_FIELD_NAMES.index("mode")] = 0
        global_row[GLOBAL_FIELD_NAMES.index("voice_count")] = 2
        global_row[GLOBAL_FIELD_NAMES.index("form")] = FORM_TO_ID["INVENTION"]
        global_row[GLOBAL_FIELD_NAMES.index("section_role")] = ROLE_TO_ID[role]
        global_row[GLOBAL_FIELD_NAMES.index("local_key_pc")] = 0
        global_rows.append(global_row)
        note = index % 2 == 0
        low_pitch = 60 + index
        high_pitch = 72 + index
        voice_rows.append(
            [
                [STATE_NOTE if note else STATE_HOLD, low_pitch, 25 if note else 0, 2, 0 if note else 1, 1],
                [STATE_NOTE if note else STATE_HOLD, high_pitch, 25 if note else 0, 2, 0 if note else 1, 5],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
    pair_rows = [
        [[[0] * 8 for _ in range(6)] for _ in range(6)]
        for _ in global_rows
    ]
    return InstrumentalV6Piece(
        piece_id="toy_v6",
        source_path="toy.musicxml",
        form="INVENTION",
        movement_index=0,
        tpq=24,
        grid_ticks=6,
        time_signature="1/1",
        key="C",
        key_pc=0,
        mode=0,
        voice_count=2,
        max_voices=6,
        bar_len_ticks=24,
        steps_per_bar=4,
        global_rows=global_rows,
        voice_rows=voice_rows,
        pair_rows=pair_rows,
    )


def test_extract_v6_fragments_supports_variable_voice_piece() -> None:
    fragments = extract_v6_fragments(_toy_v6_piece(), length_slices=4, hop_slices=4)

    assert fragments
    first = fragments[0]
    assert first.id == "toy_v6_v0_s0_l4"
    assert first.phrase_role == "SUBJECT_ENTRY"
    assert first.melodic_intervals == [2]
    assert first.rhythm_steps == [2, 2]
    assert first.vertical_intervals == [12, 12, 12, 12]
    assert first.contour_bucket == "ASCENDING_STEPWISE"
    assert first.rhythm_bucket == "EVEN_8THS"
    assert first.speac_label in {"S", "P", "E", "A", "C"}
    assert first.copy_hash
    assert first.transposition_hash
    assert fragment_from_jsonl(fragment_to_jsonl(first)) == first


def test_rank_v6_fragments_prefers_matching_voice_and_role() -> None:
    fragments = extract_v6_fragments(_toy_v6_piece(), length_slices=4, hop_slices=4)

    matches = rank_fragments(
        FragmentQuery(
            voice=1,
            phrase_role="SEQUENCE",
            key_pc=0,
            mode=0,
            local_key_pc=0,
        ),
        fragments,
        limit=2,
    )

    assert matches[0].fragment.voice == 1
    assert matches[0].fragment.phrase_role == "SEQUENCE"
