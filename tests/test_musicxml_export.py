import xml.etree.ElementTree as ET

from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.api.render.musicxml import canonical_score_to_musicxml

XML_NS = "http://www.w3.org/XML/1998/namespace"


def test_musicxml_export_includes_divisions_and_xml_ids():
    score = CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=[Measure(id="measure-0", index=0, start_tick=0, length_ticks=96)],
        parts=[
            Part(
                info=PartInfo(
                    id="guitar",
                    instrument="classical_guitar",
                    tuning=[40, 45, 50, 55, 59, 64],
                    midi_program=24,
                ),
                events=[Event(id="note-0", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=60)],
            )
        ],
    )

    xml_text = canonical_score_to_musicxml(score)
    root = ET.fromstring(xml_text)

    measure_el = root.find("./part/measure")
    assert measure_el is not None
    assert measure_el.attrib[f"{{{XML_NS}}}id"] == "measure-0"
    assert measure_el.findtext("./attributes/divisions") == "24"

    note_el = measure_el.find("./note")
    assert note_el is not None
    assert note_el.attrib[f"{{{XML_NS}}}id"] == "note-0"


def test_musicxml_export_splits_cross_bar_note_with_ties_and_shared_event_id():
    score = CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "1/4", 24: "1/4"}),
        measures=[
            Measure(id="measure-0", index=0, start_tick=0, length_ticks=24),
            Measure(id="measure-1", index=1, start_tick=24, length_ticks=24),
        ],
        parts=[
            Part(
                info=PartInfo(
                    id="guitar",
                    instrument="classical_guitar",
                    tuning=[40, 45, 50, 55, 59, 64],
                    midi_program=24,
                ),
                events=[Event(id="note-cross", start_tick=0, dur_tick=48, voice_id=0, pitch_midi=60)],
            )
        ],
    )

    xml_text = canonical_score_to_musicxml(score)
    root = ET.fromstring(xml_text)

    measures = root.findall("./part/measure")
    notes = [measure.find("./note") for measure in measures]

    assert len(measures) == 2
    assert [note.attrib[f"{{{XML_NS}}}id"] for note in notes if note is not None] == [
        "note-cross",
        "note-cross",
    ]
    assert [note.findtext("./duration") for note in notes if note is not None] == ["24", "24"]
    assert [note.find("./tie").attrib["type"] for note in notes if note is not None] == ["start", "stop"]
