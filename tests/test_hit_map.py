from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.api.compose_service import build_event_hit_map, build_measure_map, export_score


def _build_polyphonic_score() -> CanonicalScore:
    measures = [
        Measure(id="m0", index=0, start_tick=0, length_ticks=24),
        Measure(id="m1", index=1, start_tick=24, length_ticks=24),
    ]
    part = Part(
        info=PartInfo(
            id="part-0",
            instrument="classical_guitar",
            tuning=[40, 45, 50, 55, 59, 64],
            midi_program=24,
        ),
        events=[
            Event(id="v0-open", start_tick=0, dur_tick=6, voice_id=0, pitch_midi=60),
            Event(id="v1-carry", start_tick=0, dur_tick=30, voice_id=1, pitch_midi=67),
            Event(id="v0-answer", start_tick=12, dur_tick=6, voice_id=0, pitch_midi=62),
            Event(id="v0-next", start_tick=24, dur_tick=12, voice_id=0, pitch_midi=64),
            Event(id="v1-late", start_tick=36, dur_tick=6, voice_id=1, pitch_midi=69),
        ],
    )
    return CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}),
        measures=measures,
        parts=[part],
    )


def test_hit_maps_follow_exported_polyphonic_musicxml_structure():
    score = _build_polyphonic_score()
    exported = export_score(score)
    score_xml = exported.score_xml

    expected_measure_map = {
        "0": "m0",
        "1": "m1",
    }
    expected_event_hit_map = {
        "0|0|0|0": "v0-open",
        "0|0|2|0": "v0-answer",
        "0|1|0|0": "v1-carry",
        "1|0|0|0": "v0-next",
        "1|1|0|0": "v1-carry",
        "1|1|2|0": "v1-late",
    }

    assert build_measure_map(score) == expected_measure_map
    assert build_measure_map(score_xml) == expected_measure_map
    assert build_event_hit_map(score) == expected_event_hit_map
    assert build_event_hit_map(score_xml) == expected_event_hit_map
    assert exported.measure_map == expected_measure_map
    assert exported.event_hit_map == expected_event_hit_map


def test_event_hit_map_tracks_note_index_for_chord_notes_in_musicxml():
    score_xml = """
    <score-partwise version="4.0">
      <part id="part-0">
        <measure number="1" xml:id="m0">
          <note xml:id="lead-a">
            <pitch><step>C</step><octave>4</octave></pitch>
            <duration>12</duration>
            <voice>1</voice>
          </note>
          <note xml:id="lead-b">
            <chord />
            <pitch><step>E</step><octave>4</octave></pitch>
            <duration>12</duration>
            <voice>1</voice>
          </note>
          <note>
            <rest />
            <duration>12</duration>
            <voice>1</voice>
          </note>
          <note xml:id="lead-c">
            <pitch><step>G</step><octave>4</octave></pitch>
            <duration>12</duration>
            <voice>1</voice>
          </note>
          <backup>
            <duration>24</duration>
          </backup>
          <note xml:id="bass-a">
            <pitch><step>C</step><octave>3</octave></pitch>
            <duration>24</duration>
            <voice>2</voice>
          </note>
        </measure>
      </part>
    </score-partwise>
    """.strip()

    assert build_measure_map(score_xml) == {"0": "m0"}
    assert build_event_hit_map(score_xml) == {
        "0|0|0|0": "lead-a",
        "0|0|0|1": "lead-b",
        "0|0|2|0": "lead-c",
        "0|1|0|0": "bass-a",
    }
