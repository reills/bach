from src.api.canonical import (
    CanonicalScore,
    Event,
    Measure,
    Part,
    PartInfo,
    ScoreHeader,
    carry_in_events_for_measure,
    event_by_id,
    events_starting_in_measure,
    measure_by_id,
    replace_events_in_measure,
)


def _build_score() -> CanonicalScore:
    measures = [
        Measure(id="m0", index=0, start_tick=0, length_ticks=24),
        Measure(id="m1", index=1, start_tick=24, length_ticks=24),
        Measure(id="m2", index=2, start_tick=48, length_ticks=24),
    ]
    part = Part(
        info=PartInfo(id="part-0", instrument="classical_guitar", tuning=[40, 45, 50, 55, 59, 64], midi_program=24),
        events=[
            Event(id="carry", start_tick=0, dur_tick=30, voice_id=0, pitch_midi=60),
            Event(id="m0-note", start_tick=12, dur_tick=6, voice_id=1, pitch_midi=64),
            Event(id="m1-note-a", start_tick=24, dur_tick=12, voice_id=0, pitch_midi=62),
            Event(id="m1-note-b", start_tick=36, dur_tick=18, voice_id=1, pitch_midi=65),
            Event(id="m2-note", start_tick=48, dur_tick=12, voice_id=0, pitch_midi=67),
        ],
    )
    return CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=measures,
        parts=[part],
    )


def test_measure_and_event_lookup_return_canonical_objects():
    score = _build_score()

    measure = measure_by_id(score, "m1")
    event = event_by_id(score, "m1-note-b")

    assert measure == score.measures[1]
    assert event == score.parts[0].events[3]


def test_carry_in_and_measure_event_queries_split_measure_boundary_correctly():
    score = _build_score()
    part = score.parts[0]
    measure = score.measures[1]

    assert [event.id for event in carry_in_events_for_measure(part, measure)] == ["carry"]
    assert [event.id for event in events_starting_in_measure(part, measure)] == [
        "m1-note-a",
        "m1-note-b",
    ]


def test_replace_events_in_measure_only_replaces_events_that_start_in_target_measure():
    score = _build_score()
    part = score.parts[0]
    measure = score.measures[1]

    replacement = [
        Event(id="replacement-a", start_tick=24, dur_tick=18, voice_id=0, pitch_midi=69),
        Event(id="replacement-b", start_tick=42, dur_tick=6, voice_id=1, pitch_midi=71),
    ]

    updated_part = replace_events_in_measure(part, measure, replacement)

    assert [event.id for event in updated_part.events] == [
        "carry",
        "m0-note",
        "replacement-a",
        "replacement-b",
        "m2-note",
    ]
    assert updated_part.events[0].end_tick == 30
    assert updated_part.events[-1].start_tick == 48
