from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.api.services import preview_window_inpaint
from src.api.store import InMemoryScoreRepository


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
            Event(id="m1-note-b", start_tick=36, dur_tick=12, voice_id=1, pitch_midi=65),
            Event(id="m2-note", start_tick=48, dur_tick=12, voice_id=0, pitch_midi=67),
        ],
    )
    return CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=measures,
        parts=[part],
    )


def test_preview_window_inpaint_preserves_carry_in_events_and_locks_them():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored_score = repository.create_score(_build_score())

    result = preview_window_inpaint(
        repository,
        stored_score.score_id,
        revision=stored_score.revision,
        measure_id="m1",
    )

    assert result.draft_id == "draft-1"
    assert result.base_revision == stored_score.revision
    assert result.highlight_measure_id == "m1"
    assert result.changed_measure_ids == ["m1"]
    assert result.locked_event_ids == ["carry"]

    updated_events = result.score.parts[0].events
    assert [event.id for event in updated_events] == [
        "carry",
        "m0-note",
        "part-0-m1-regen-0",
        "part-0-m1-regen-1",
        "m2-note",
    ]
    assert updated_events[0] == Event(id="carry", start_tick=0, dur_tick=30, voice_id=0, pitch_midi=60)
    assert updated_events[2].start_tick == 24
    assert updated_events[2].pitch_midi == 63
    assert updated_events[3].start_tick == 36
    assert updated_events[3].pitch_midi == 66

    saved_draft = repository.get_draft(result.draft_id)
    assert saved_draft.score == result.score
    assert 'id="m1"' in result.score_xml


def test_preview_window_inpaint_replaces_only_events_starting_in_selected_measure():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored_score = repository.create_score(_build_score())
    captured: dict[str, list[str]] = {}

    def replacement_planner(part, measure, measure_events, carry_in_events):
        del part
        captured["measure_ids"] = [event.id for event in measure_events]
        captured["carry_ids"] = [event.id for event in carry_in_events]
        return [
            Event(id="replacement-window", start_tick=measure.start_tick, dur_tick=measure.length_ticks, voice_id=0, pitch_midi=70)
        ]

    result = preview_window_inpaint(
        repository,
        stored_score.score_id,
        revision=stored_score.revision,
        measure_id="m1",
        locked_event_ids=["user-lock"],
        replacement_planner=replacement_planner,
    )

    assert captured == {
        "measure_ids": ["m1-note-a", "m1-note-b"],
        "carry_ids": ["carry"],
    }
    assert result.locked_event_ids == ["user-lock", "carry"]
    assert [event.id for event in result.score.parts[0].events] == [
        "carry",
        "m0-note",
        "replacement-window",
        "m2-note",
    ]
    assert result.changed_measure_ids == ["m1"]
    assert result.score.parts[0].events[0].end_tick == 30
    assert result.score.parts[0].events[-1].start_tick == 48


def test_preview_window_inpaint_reports_downstream_measures_touched_by_replacement_span():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored_score = repository.create_score(_build_score())

    def replacement_planner(part, measure, measure_events, carry_in_events):
        del part, measure_events, carry_in_events
        return [
            Event(
                id="cross-bar-window",
                start_tick=measure.start_tick,
                dur_tick=measure.length_ticks + 6,
                voice_id=0,
                pitch_midi=70,
            )
        ]

    result = preview_window_inpaint(
        repository,
        stored_score.score_id,
        revision=stored_score.revision,
        measure_id="m1",
        replacement_planner=replacement_planner,
    )

    assert result.changed_measure_ids == ["m1", "m2"]
    assert [event.id for event in result.score.parts[0].events] == [
        "carry",
        "m0-note",
        "cross-bar-window",
        "m2-note",
    ]
    assert result.score.parts[0].events[2].end_tick == 54
