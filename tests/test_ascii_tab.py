import pytest

from src.api.canonical import Event, GuitarFingering
from src.tabber import render_ascii_tab


def test_render_ascii_tab_renders_short_phrase_across_multiple_strings():
    events = [
        Event(
            id="note-0",
            start_tick=0,
            dur_tick=24,
            voice_id=0,
            pitch_midi=64,
            fingering=GuitarFingering(string_index=5, fret=0),
        ),
        Event(
            id="note-1",
            start_tick=24,
            dur_tick=24,
            voice_id=0,
            pitch_midi=60,
            fingering=GuitarFingering(string_index=4, fret=1),
        ),
        Event(
            id="note-2",
            start_tick=48,
            dur_tick=24,
            voice_id=0,
            pitch_midi=57,
            fingering=GuitarFingering(string_index=3, fret=2),
        ),
        Event(
            id="note-3",
            start_tick=48,
            dur_tick=24,
            voice_id=1,
            pitch_midi=52,
            fingering=GuitarFingering(string_index=2, fret=2),
        ),
        Event(
            id="note-4",
            start_tick=72,
            dur_tick=24,
            voice_id=0,
            pitch_midi=55,
            fingering=GuitarFingering(string_index=1, fret=10),
        ),
    ]

    rendered = render_ascii_tab(events)

    assert rendered == "\n".join(
        [
            "e|0--------|",
            "B|--1------|",
            "G|----2----|",
            "D|----2----|",
            "A|------10-|",
            "E|---------|",
        ]
    )


def test_render_ascii_tab_rejects_pitched_event_without_fingering():
    events = [Event(id="note-0", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=64)]

    with pytest.raises(ValueError, match="fingering"):
        render_ascii_tab(events)
