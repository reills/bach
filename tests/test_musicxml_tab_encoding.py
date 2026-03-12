import xml.etree.ElementTree as ET

from src.api.canonical import (
    CanonicalScore,
    Event,
    GuitarFingering,
    Measure,
    Part,
    PartInfo,
    ScoreHeader,
)
from src.api.render.musicxml import canonical_score_to_musicxml


def _render_single_note(*, event_id: str, pitch_midi: int, string_index: int, fret: int) -> ET.Element:
    score = CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "1/4"}),
        measures=[Measure(id="measure-0", index=0, start_tick=0, length_ticks=24)],
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
                        id=event_id,
                        start_tick=0,
                        dur_tick=24,
                        voice_id=0,
                        pitch_midi=pitch_midi,
                        fingering=GuitarFingering(string_index=string_index, fret=fret),
                    )
                ],
            )
        ],
    )

    xml_text = canonical_score_to_musicxml(score)
    root = ET.fromstring(xml_text)
    note_el = root.find("./part/measure/note")
    assert note_el is not None
    return note_el


def test_musicxml_export_encodes_high_e_string_as_1():
    note_el = _render_single_note(event_id="note-high-e", pitch_midi=64, string_index=5, fret=0)

    assert note_el.findtext("./notations/technical/string") == "1"
    assert note_el.findtext("./notations/technical/fret") == "0"


def test_musicxml_export_encodes_low_e_string_as_6():
    note_el = _render_single_note(event_id="note-low-e", pitch_midi=40, string_index=0, fret=0)

    assert note_el.findtext("./notations/technical/string") == "6"
    assert note_el.findtext("./notations/technical/fret") == "0"
