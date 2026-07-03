from __future__ import annotations

from src.instrumental_v6.global_coherence import evaluate_global_coherence
from src.instrumental_v6.representation import (
    DEVELOPMENT_TO_ID,
    GLOBAL_FIELD_NAMES,
    ROLE_TO_ID,
    STATE_NOTE,
)


def _global_row(bar: int, pos: int, role: str, development: str, local_key: int = 0) -> list[int]:
    row = [0] * len(GLOBAL_FIELD_NAMES)
    row[GLOBAL_FIELD_NAMES.index("bar")] = bar
    row[GLOBAL_FIELD_NAMES.index("pos")] = pos
    row[GLOBAL_FIELD_NAMES.index("cadence_zone")] = int(role in {"CADENTIAL_PREP", "CADENCE"})
    row[GLOBAL_FIELD_NAMES.index("key_pc")] = 0
    row[GLOBAL_FIELD_NAMES.index("mode")] = 0
    row[GLOBAL_FIELD_NAMES.index("voice_count")] = 2
    row[GLOBAL_FIELD_NAMES.index("section_role")] = ROLE_TO_ID[role]
    row[GLOBAL_FIELD_NAMES.index("development")] = DEVELOPMENT_TO_ID[development]
    row[GLOBAL_FIELD_NAMES.index("local_key_pc")] = local_key
    return row


def _rows(*, strong: bool) -> tuple[list[list[int]], list[list[list[int]]]]:
    steps_per_bar = 4
    bars = 24
    roles = []
    for bar in range(bars):
        if bar == 0:
            roles.append(("SUBJECT_ENTRY", "SUBJECT"))
        elif bar == 1:
            roles.append(("ANSWER_ENTRY", "ANSWER"))
        elif bar in {7, 15, 23}:
            roles.append(("CADENCE", "CADENCE"))
        elif bar in {6, 14, 22}:
            roles.append(("CADENTIAL_PREP", "EPISODE"))
        elif bar % 2 == 0:
            roles.append(("SEQUENCE", "SEQUENCE_DOWN"))
        else:
            roles.append(("EPISODE", "EPISODE"))
    global_rows = [
        _global_row(
            bar,
            pos,
            roles[bar][0],
            roles[bar][1],
            local_key=7 if 8 <= bar < 16 else 0,
        )
        for bar in range(bars)
        for pos in range(steps_per_bar)
    ]
    voice_rows = [
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        for _ in global_rows
    ]
    subject_pitches = [60, 62, 63, 62, 64]
    starts = [0, 32, 80] if strong else [0]
    for start in starts:
        for offset, pitch in enumerate(subject_pitches):
            row_index = start + offset
            voice_rows[row_index][1] = [STATE_NOTE, pitch, 0, 1, 0, 1]
    return global_rows, voice_rows


def test_global_coherence_rewards_subject_recurrence_cadences_and_key_arc() -> None:
    global_rows, voice_rows = _rows(strong=True)

    report = evaluate_global_coherence(
        global_rows,
        voice_rows,
        voice_count=2,
        steps_per_bar=4,
        subject=[2, 1, -1, 2],
    )

    assert report["subject_coverage_score"] == 1.0
    assert report["cadence_count_score"] == 1.0
    assert report["local_key_change_count"] == 2
    assert report["development_score"] == 1.0
    assert report["coherence_score"] > 80.0


def test_global_coherence_penalizes_missing_middle_and_closing_subjects() -> None:
    strong_global, strong_voice = _rows(strong=True)
    weak_global, weak_voice = _rows(strong=False)

    strong = evaluate_global_coherence(
        strong_global,
        strong_voice,
        voice_count=2,
        steps_per_bar=4,
        subject=[2, 1, -1, 2],
    )
    weak = evaluate_global_coherence(
        weak_global,
        weak_voice,
        voice_count=2,
        steps_per_bar=4,
        subject=[2, 1, -1, 2],
    )

    assert weak["subject_coverage_score"] < strong["subject_coverage_score"]
    assert weak["coherence_score"] < strong["coherence_score"]
