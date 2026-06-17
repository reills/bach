from dataclasses import replace
import xml.etree.ElementTree as ET

import pytest

from src.api.canonical import CanonicalScore, Event, GuitarFingering, Measure, Part, PartInfo, ScoreHeader
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
    carry = updated_events[0]
    assert carry.id == "carry"
    assert carry.start_tick == 0
    assert carry.dur_tick == 30
    assert carry.voice_id == 0
    assert carry.pitch_midi == 60
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


def _build_piano_score() -> CanonicalScore:
    measures = [
        Measure(id="m0", index=0, start_tick=0, length_ticks=24),
        Measure(id="m1", index=1, start_tick=24, length_ticks=24),
    ]
    part = Part(
        info=PartInfo(id="piano-0", instrument="piano", tuning=[]),
        events=[
            Event(id="treble", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=64),
            Event(id="bass", start_tick=24, dur_tick=24, voice_id=0, pitch_midi=48),
        ],
    )
    return CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=measures,
        parts=[part],
    )


def test_guitar_inpaint_preview_assigns_fingering_to_regenerated_events():
    """After inpaint, regenerated guitar events should have fingering (tab-capable)."""
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored = repository.create_score(_build_score())

    result = preview_window_inpaint(
        repository,
        stored.score_id,
        revision=stored.revision,
        measure_id="m1",
    )

    # Regenerated events in m1 should have fingering assigned
    regen_events = [
        e for e in result.score.parts[0].events
        if e.id.startswith("part-0-m1-regen") and e.pitch_midi is not None
    ]
    assert regen_events, "expected regenerated events in m1"
    for event in regen_events:
        assert event.fingering is not None, f"event {event.id!r} missing fingering after guitar inpaint"

    # The exported XML should have <technical><string/><fret/></technical> for those events
    root = ET.fromstring(result.score_xml)
    for event in regen_events:
        note_el = root.find(f".//*[@{{http://www.w3.org/XML/1998/namespace}}id='{event.id}']")
        assert note_el is not None
        assert note_el.findtext("./notations/technical/string") is not None


def test_guitar_inpaint_preview_preserves_unchanged_explicit_fingerings():
    repository = InMemoryScoreRepository[CanonicalScore]()
    original_score = _build_score()
    selected_fingering = GuitarFingering(string_index=0, fret=20)
    original_part = original_score.parts[0]
    original_score = replace(
        original_score,
        parts=[
            replace(
                original_part,
                events=[
                    replace(
                        original_part.events[0],
                        fingering=selected_fingering,
                    ),
                    *original_part.events[1:],
                ],
            )
        ],
    )
    stored = repository.create_score(original_score)

    result = preview_window_inpaint(
        repository,
        stored.score_id,
        revision=stored.revision,
        measure_id="m1",
    )

    carry_event = result.score.parts[0].events[0]
    assert carry_event.id == "carry"
    assert carry_event.fingering == selected_fingering


def test_guitar_inpaint_preview_normalizes_low_regenerated_pitches_before_tabbing():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored = repository.create_score(_build_score())

    def replacement_planner(part, measure, measure_events, carry_in_events):
        del part, measure_events, carry_in_events
        return [
            Event(id="low-note", start_tick=measure.start_tick, dur_tick=measure.length_ticks, voice_id=0, pitch_midi=28)
        ]

    result = preview_window_inpaint(
        repository,
        stored.score_id,
        revision=stored.revision,
        measure_id="m1",
        replacement_planner=replacement_planner,
    )

    low_note = next(event for event in result.score.parts[0].events if event.id == "low-note")
    assert low_note.pitch_midi == 40
    assert low_note.fingering is not None


def test_guitar_inpaint_preview_raises_when_regenerated_notes_cannot_be_tabbed():
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored = repository.create_score(_build_score())

    def replacement_planner(part, measure, measure_events, carry_in_events):
        del part, measure_events, carry_in_events
        return [
            Event(id="too-high-note", start_tick=measure.start_tick, dur_tick=measure.length_ticks, voice_id=0, pitch_midi=127)
        ]

    with pytest.raises(ValueError, match="guitar inpaint retab failed"):
        preview_window_inpaint(
            repository,
            stored.score_id,
            revision=stored.revision,
            measure_id="m1",
            replacement_planner=replacement_planner,
        )


def test_piano_inpaint_preview_preserves_grand_staff_export_shape():
    """Piano inpaint keeps the two-staff export — <staff> tags and staves=2 in measure 1."""
    repository = InMemoryScoreRepository[CanonicalScore]()
    stored = repository.create_score(_build_piano_score())

    result = preview_window_inpaint(
        repository,
        stored.score_id,
        revision=stored.revision,
        measure_id="m1",
    )

    root = ET.fromstring(result.score_xml)
    # Measure 1 attributes: staves=2
    attrs = root.find("./part/measure/attributes")
    assert attrs is not None
    assert attrs.findtext("staves") == "2"
    # All pitched notes have a <staff> element
    notes = root.findall(".//note")
    pitched = [n for n in notes if n.find("./pitch") is not None]
    assert pitched, "expected pitched notes in piano export"
    for n in pitched:
        assert n.findtext("staff") is not None, "piano note missing <staff> tag"


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
