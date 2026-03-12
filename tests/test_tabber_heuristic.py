import pytest

from src.api.canonical import Event, GuitarFingering
from src.tabber.heuristic import tab_events


def test_tab_events_prefers_open_string_when_available():
    events = [
        Event(id="note-0", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=64),
    ]

    tabbed = tab_events(events)

    assert tabbed[0].fingering == GuitarFingering(string_index=5, fret=0)


def test_tab_events_assigns_unique_strings_for_basic_chord():
    events = [
        Event(id="note-0", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=52),
        Event(id="note-1", start_tick=0, dur_tick=24, voice_id=1, pitch_midi=59),
        Event(id="note-2", start_tick=0, dur_tick=24, voice_id=2, pitch_midi=64),
    ]

    tabbed = tab_events(events)

    assert [event.fingering for event in tabbed] == [
        GuitarFingering(string_index=2, fret=2),
        GuitarFingering(string_index=4, fret=0),
        GuitarFingering(string_index=5, fret=0),
    ]


def test_tab_events_rejects_same_onset_voicing_that_needs_one_string_twice():
    events = [
        Event(id="note-0", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=40),
        Event(id="note-1", start_tick=0, dur_tick=24, voice_id=1, pitch_midi=41),
    ]

    with pytest.raises(ValueError, match="onset 0"):
        tab_events(events)
