import xml.etree.ElementTree as ET

from src.api.canonical import CanonicalScore, Event, GuitarFingering, Measure, Part, PartInfo, ScoreHeader
from src.api.render.musicxml import canonical_score_to_musicxml

XML_NS = "http://www.w3.org/XML/1998/namespace"

_STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4


def _guitar_score(events: list[Event] | None = None) -> CanonicalScore:
    return CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=[Measure(id="m0", index=0, start_tick=0, length_ticks=96)],
        parts=[
            Part(
                info=PartInfo(
                    id="guitar",
                    instrument="classical_guitar",
                    tuning=_STANDARD_TUNING,
                    midi_program=24,
                ),
                events=events or [
                    Event(id="n0", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=60)
                ],
            )
        ],
    )


def _piano_score(events: list[Event] | None = None) -> CanonicalScore:
    return CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=[Measure(id="m0", index=0, start_tick=0, length_ticks=96)],
        parts=[
            Part(
                info=PartInfo(id="piano", instrument="piano", tuning=[]),
                events=events or [
                    Event(id="n0", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=60)
                ],
            )
        ],
    )


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


def test_guitar_measure1_contains_staff_details_and_tuning():
    root = ET.fromstring(canonical_score_to_musicxml(_guitar_score()))
    attrs = root.find("./part/measure/attributes")
    assert attrs is not None

    # Guitar treble clef with octave-down
    clef = attrs.find("./clef")
    assert clef is not None
    assert clef.findtext("sign") == "G"
    assert clef.findtext("line") == "2"
    assert clef.findtext("clef-octave-change") == "-1"

    # staff-details
    staff_details = attrs.find("./staff-details")
    assert staff_details is not None
    assert staff_details.findtext("staff-lines") == "6"

    tuning_els = staff_details.findall("./staff-tuning")
    assert len(tuning_els) == 6
    # line="1" = highest string = E4
    line1 = next(el for el in tuning_els if el.attrib["line"] == "1")
    assert line1.findtext("tuning-step") == "E"
    assert line1.findtext("tuning-octave") == "4"
    # line="6" = lowest string = E2
    line6 = next(el for el in tuning_els if el.attrib["line"] == "6")
    assert line6.findtext("tuning-step") == "E"
    assert line6.findtext("tuning-octave") == "2"

    # No capo element when capo==0
    assert staff_details.find("./capo") is None


def test_guitar_measure1_emits_capo_when_nonzero():
    score = CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=[Measure(id="m0", index=0, start_tick=0, length_ticks=96)],
        parts=[
            Part(
                info=PartInfo(
                    id="guitar",
                    instrument="classical_guitar",
                    tuning=_STANDARD_TUNING,
                    capo=2,
                ),
                events=[Event(id="n0", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=62)],
            )
        ],
    )
    root = ET.fromstring(canonical_score_to_musicxml(score))
    capo = root.find("./part/measure/attributes/staff-details/capo")
    assert capo is not None
    assert capo.text == "2"


def test_piano_measure1_contains_staves2_and_dual_clefs():
    root = ET.fromstring(canonical_score_to_musicxml(_piano_score()))
    attrs = root.find("./part/measure/attributes")
    assert attrs is not None
    assert attrs.findtext("staves") == "2"

    clef1 = attrs.find("./clef[@number='1']")
    assert clef1 is not None
    assert clef1.findtext("sign") == "G"
    assert clef1.findtext("line") == "2"

    clef2 = attrs.find("./clef[@number='2']")
    assert clef2 is not None
    assert clef2.findtext("sign") == "F"
    assert clef2.findtext("line") == "4"


def test_piano_notes_emit_staff_tag():
    events = [
        Event(id="treble", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=64),   # above split → staff 1
        Event(id="bass", start_tick=24, dur_tick=24, voice_id=0, pitch_midi=48),    # below split → staff 2
    ]
    root = ET.fromstring(canonical_score_to_musicxml(_piano_score(events)))
    notes = root.findall("./part/measure/note")
    pitched = [n for n in notes if n.find("./pitch") is not None]
    assert len(pitched) == 2
    # First note E4 (64 >= 60) → staff 1
    assert pitched[0].attrib[f"{{{XML_NS}}}id"] == "treble"
    assert pitched[0].findtext("staff") == "1"
    # Second note C3 (48 < 60) → staff 2
    assert pitched[1].attrib[f"{{{XML_NS}}}id"] == "bass"
    assert pitched[1].findtext("staff") == "2"


def test_piano_leading_rest_uses_staff_of_first_pitched_note():
    events = [
        Event(id="bass-entry", start_tick=24, dur_tick=24, voice_id=0, pitch_midi=48),
    ]
    root = ET.fromstring(canonical_score_to_musicxml(_piano_score(events)))
    notes = root.findall("./part/measure/note")

    assert len(notes) == 3
    assert notes[0].find("./rest") is not None
    assert notes[0].findtext("staff") == "2"
    assert notes[1].attrib[f"{{{XML_NS}}}id"] == "bass-entry"
    assert notes[1].findtext("staff") == "2"
    assert notes[2].find("./rest") is not None
    assert notes[2].findtext("staff") == "2"


def test_piano_chord_staff_determined_by_lowest_pitch():
    """A chord with lowest pitch < 60 must be entirely on staff 2."""
    events = [
        Event(id="lo", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=55),  # G3
        Event(id="hi", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=67),  # G4
    ]
    root = ET.fromstring(canonical_score_to_musicxml(_piano_score(events)))
    notes = root.findall("./part/measure/note")
    pitched = [n for n in notes if n.find("./pitch") is not None]
    assert len(pitched) == 2
    # Lowest pitch is 55 < 60, so whole chord goes to staff 2
    for n in pitched:
        assert n.findtext("staff") == "2"


def test_piano_simultaneous_voices_use_backup_and_both_staves():
    """Two voices at the same onset end up on different staves with a backup."""
    events = [
        Event(id="rh", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=64),  # staff 1
        Event(id="lh", start_tick=0, dur_tick=24, voice_id=1, pitch_midi=48),  # staff 2
    ]
    root = ET.fromstring(canonical_score_to_musicxml(_piano_score(events)))
    children = list(root.find("./part/measure"))
    # Skip the attributes element
    content = [el for el in children if el.tag != "attributes"]
    # Should have: note(rh), [rest], backup, note(lh), [rest]
    backup_els = [el for el in content if el.tag == "backup"]
    assert len(backup_els) >= 1
    note_els = [el for el in content if el.tag == "note"]
    pitched = [n for n in note_els if n.find("./pitch") is not None]
    staffs = {n.attrib.get(f"{{{XML_NS}}}id"): n.findtext("staff") for n in pitched}
    assert staffs["rh"] == "1"
    assert staffs["lh"] == "2"


def test_piano_same_onset_chord_emits_musicxml_chord_tags_without_breaking_backup():
    events = [
        Event(id="rh-c", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=60),
        Event(id="rh-e", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=64),
        Event(id="rh-g", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=67),
        Event(id="lh-c", start_tick=0, dur_tick=24, voice_id=1, pitch_midi=48),
    ]
    root = ET.fromstring(canonical_score_to_musicxml(_piano_score(events)))

    measure = root.find("./part/measure")
    assert measure is not None
    assert measure.findtext("./attributes/staves") == "2"

    content = [el for el in list(measure) if el.tag != "attributes"]
    backup_index = next(i for i, el in enumerate(content) if el.tag == "backup")
    assert content[backup_index].findtext("duration") == "96"

    before_backup = content[:backup_index]
    rh_pitched = [el for el in before_backup if el.tag == "note" and el.find("./pitch") is not None]
    assert [note.attrib[f"{{{XML_NS}}}id"] for note in rh_pitched] == ["rh-c", "rh-e", "rh-g"]
    assert [note.findtext("staff") for note in rh_pitched] == ["1", "1", "1"]
    assert rh_pitched[0].find("./chord") is None
    assert rh_pitched[1].find("./chord") is not None
    assert rh_pitched[2].find("./chord") is not None
    assert [note.findtext("duration") for note in rh_pitched] == ["24", "24", "24"]

    after_backup = content[backup_index + 1 :]
    lh_pitched = [el for el in after_backup if el.tag == "note" and el.find("./pitch") is not None]
    assert len(lh_pitched) == 1
    assert lh_pitched[0].attrib[f"{{{XML_NS}}}id"] == "lh-c"
    assert lh_pitched[0].findtext("staff") == "2"
    assert lh_pitched[0].find("./chord") is None


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
