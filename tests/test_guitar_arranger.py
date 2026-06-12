from __future__ import annotations

import pytest

from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.arrangers.guitar import GuitarArrangementSettings, convert_piano_score_to_guitar


def _score(events: list[Event]) -> CanonicalScore:
    return CanonicalScore(
        header=ScoreHeader(tpq=24, key_sig_map={0: "C"}, time_sig_map={0: "4/4"}, tempo_map={0: 96}),
        measures=[Measure(id="m0", index=0, start_tick=0, length_ticks=96)],
        parts=[
            Part(
                info=PartInfo(id="P1", instrument="piano", midi_program=0),
                events=events,
            )
        ],
    )


def test_convert_piano_score_to_guitar_returns_independent_arrangement() -> None:
    score = _score(
        [
            Event(id="n0", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=52),
            Event(id="n1", start_tick=24, dur_tick=24, voice_id=0, pitch_midi=55),
        ]
    )

    arrangement = convert_piano_score_to_guitar(score)

    guitar_part = arrangement.score.parts[0]
    assert guitar_part.info.instrument == "classical_guitar"
    assert guitar_part.info.tuning == [40, 45, 50, 55, 59, 64]
    assert [event.id for event in guitar_part.events] == ["gtr-n0", "gtr-n1"]
    assert all(event.fingering is not None for event in guitar_part.events)
    assert all(event.fingering is None for event in score.parts[0].events)
    assert [note_map.source_event_id for note_map in arrangement.source_map.notes] == ["n0", "n1"]
    assert [note_map.target_event_id for note_map in arrangement.source_map.notes] == ["gtr-n0", "gtr-n1"]
    assert arrangement.diagnostics.to_dict()["droppedNotes"] == []


def test_convert_piano_score_to_guitar_tracks_octave_range_changes() -> None:
    score = _score(
        [
            Event(id="low", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=35),
        ]
    )

    arrangement = convert_piano_score_to_guitar(score)

    event = arrangement.score.parts[0].events[0]
    assert event.pitch_midi == 47
    assert arrangement.source_map.notes[0].semitone_shift == 12
    assert arrangement.diagnostics.octave_shifted_notes[0].source_event_id == "low"
    assert arrangement.diagnostics.range_changes[0].arranged_pitch_midi == 47


def test_convert_piano_score_to_guitar_drops_notes_for_oversized_chord() -> None:
    score = _score(
        [
            Event(id=f"n{index}", start_tick=0, dur_tick=24, voice_id=index, pitch_midi=pitch)
            for index, pitch in enumerate([40, 45, 50, 55, 59, 64, 67])
        ]
    )

    arrangement = convert_piano_score_to_guitar(score)

    assert len(arrangement.score.parts[0].events) == 4
    assert len(arrangement.diagnostics.dropped_notes) == 3
    assert arrangement.diagnostics.impossible_chords[0].onset_tick == 0
    assert arrangement.diagnostics.warnings
    dropped_maps = [note_map for note_map in arrangement.source_map.notes if note_map.dropped]
    assert len(dropped_maps) == 3
    assert all(note_map.target_event_id is None for note_map in dropped_maps)


def test_convert_piano_score_to_guitar_preserves_melody_bass_and_harmony_notes() -> None:
    score = _score(
        [
            Event(id=f"n{index}", start_tick=0, dur_tick=24, voice_id=index, pitch_midi=pitch)
            for index, pitch in enumerate([48, 52, 55, 58, 60, 64, 67])
        ]
    )

    arrangement = convert_piano_score_to_guitar(score)

    output_pitches = [event.pitch_midi for event in arrangement.score.parts[0].events]
    assert len(output_pitches) == 4
    assert 48 in output_pitches
    assert 67 in output_pitches
    assert 58 in output_pitches
    assert any(pitch % 12 == 4 for pitch in output_pitches)
    assert all(event.fingering is not None for event in arrangement.score.parts[0].events)


def test_convert_piano_score_to_guitar_uses_difficulty_density() -> None:
    score = _score(
        [
            Event(id=f"n{index}", start_tick=0, dur_tick=24, voice_id=index, pitch_midi=pitch)
            for index, pitch in enumerate([48, 52, 55, 58, 60, 64])
        ]
    )

    easy = convert_piano_score_to_guitar(score, settings=GuitarArrangementSettings(difficulty="easy"))
    hard = convert_piano_score_to_guitar(score, settings=GuitarArrangementSettings(difficulty="hard"))

    assert len(easy.score.parts[0].events) == 3
    assert len(hard.score.parts[0].events) > len(easy.score.parts[0].events)


def test_convert_piano_score_to_guitar_drops_low_note_when_octave_shift_would_invert_bass() -> None:
    score = _score(
        [
            Event(id="low", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=39),
            Event(id="bass", start_tick=0, dur_tick=24, voice_id=1, pitch_midi=40),
            Event(id="third", start_tick=0, dur_tick=24, voice_id=2, pitch_midi=52),
            Event(id="fifth", start_tick=0, dur_tick=24, voice_id=3, pitch_midi=55),
        ]
    )

    arrangement = convert_piano_score_to_guitar(score)

    output_pitches = [event.pitch_midi for event in arrangement.score.parts[0].events]
    assert 51 not in output_pitches
    assert 40 in output_pitches
    assert arrangement.diagnostics.dropped_notes[0].source_event_id == "low"
    assert "invert the bass line" in arrangement.diagnostics.dropped_notes[0].reason


def test_convert_piano_score_to_guitar_strict_mode_raises_without_composing() -> None:
    score = _score(
        [
            Event(id="too-high", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=90),
        ]
    )

    with pytest.raises(ValueError, match="note is outside the fret range"):
        convert_piano_score_to_guitar(
            score,
            settings=GuitarArrangementSettings.for_legacy_compose(),
        )
