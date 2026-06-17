import pytest

from src.api.canonical import (
    CanonicalScore,
    Event,
    FingeringSelection,
    GuitarFingering,
    Measure,
    Part,
    PartInfo,
    ScoreHeader,
    apply_fingering_selections,
)


def _build_score() -> CanonicalScore:
    return CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=[Measure(id="m0", index=0, start_tick=0, length_ticks=24)],
        parts=[
            Part(
                info=PartInfo(
                    id="guitar",
                    instrument="classical_guitar",
                    tuning=[40, 45, 50, 55, 59, 64],
                    midi_program=24,
                ),
                events=[
                    Event(
                        id="ev-1",
                        start_tick=0,
                        dur_tick=12,
                        pitch_midi=64,
                        voice_id=0,
                        fingering=GuitarFingering(string_index=5, fret=0),
                    ),
                    Event(
                        id="ev-2",
                        start_tick=12,
                        dur_tick=12,
                        pitch_midi=67,
                        voice_id=0,
                        fingering=GuitarFingering(string_index=3, fret=0),
                    ),
                ],
            )
        ],
    )


def test_apply_fingering_selections_updates_only_selected_event_fingering():
    score = _build_score()

    updated_score = apply_fingering_selections(
        score,
        [
            FingeringSelection(
                event_id="ev-1",
                pitch_midi=64,
                start_tick=0,
                dur_tick=12,
                fingering=GuitarFingering(string_index=2, fret=5),
            )
        ],
    )

    updated_event = updated_score.parts[0].events[0]

    assert updated_event.fingering == GuitarFingering(string_index=2, fret=5)
    assert updated_event.pitch_midi == score.parts[0].events[0].pitch_midi
    assert updated_event.start_tick == score.parts[0].events[0].start_tick
    assert updated_event.dur_tick == score.parts[0].events[0].dur_tick
    assert updated_score.parts[0].events[1] == score.parts[0].events[1]
    assert score.parts[0].events[0].fingering == GuitarFingering(string_index=5, fret=0)


def test_apply_fingering_selections_rejects_unknown_event_id():
    score = _build_score()

    with pytest.raises(ValueError, match="unknown event id: missing-event"):
        apply_fingering_selections(
            score,
            [
                FingeringSelection(
                    event_id="missing-event",
                    pitch_midi=64,
                    start_tick=0,
                    dur_tick=12,
                    fingering=GuitarFingering(string_index=2, fret=5),
                )
            ],
        )
