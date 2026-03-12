from src.api.canonical import Event, GuitarFingering
from src.tabber import alternate_fingerings_for_event


def test_alternate_fingerings_for_event_returns_all_valid_positions_in_stable_order():
    event = Event(id="note-0", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=64)

    alternates = alternate_fingerings_for_event(event)

    assert alternates == [
        GuitarFingering(string_index=5, fret=0),
        GuitarFingering(string_index=4, fret=5),
        GuitarFingering(string_index=3, fret=9),
        GuitarFingering(string_index=2, fret=14),
        GuitarFingering(string_index=1, fret=19),
    ]


def test_alternate_fingerings_for_event_respects_fret_limit_for_high_pitch():
    event = Event(id="note-0", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=84)

    alternates = alternate_fingerings_for_event(event)

    assert alternates == [GuitarFingering(string_index=5, fret=20)]
