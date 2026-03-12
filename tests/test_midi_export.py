from music21.midi import MidiFile

from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.api.render import canonical_score_to_midi


def test_midi_export_returns_parseable_midi_bytes():
    score = CanonicalScore(
        header=ScoreHeader(tpq=24, time_sig_map={0: "4/4"}, tempo_map={0: 96}),
        measures=[Measure(id="measure-0", index=0, start_tick=0, length_ticks=96)],
        parts=[
            Part(
                info=PartInfo(
                    id="guitar",
                    instrument="classical_guitar",
                    tuning=[40, 45, 50, 55, 59, 64],
                    midi_program=24,
                ),
                events=[
                    Event(id="note-0", start_tick=0, dur_tick=24, voice_id=0, pitch_midi=60, velocity=90),
                    Event(id="rest-0", start_tick=24, dur_tick=24, voice_id=0, pitch_midi=None),
                    Event(id="note-1", start_tick=48, dur_tick=24, voice_id=0, pitch_midi=64, velocity=96),
                ],
            )
        ],
    )

    midi_bytes = canonical_score_to_midi(score)

    assert midi_bytes
    assert midi_bytes.startswith(b"MThd")
    assert b"MTrk" in midi_bytes

    midi_file = MidiFile()
    midi_file.readstr(midi_bytes)

    assert midi_file.ticksPerQuarterNote == score.header.tpq
    assert len(midi_file.tracks) >= 1
