from __future__ import annotations

import music21
from music21.midi import translate

from src.api.canonical.types import CanonicalScore, Part, PartInfo


def canonical_score_to_midi(score: CanonicalScore) -> bytes:
    midi_file = translate.streamToMidiFile(_to_music21_score(score))
    midi_file.ticksPerQuarterNote = score.header.tpq
    return midi_file.writestr()


def score_to_midi(score: CanonicalScore) -> bytes:
    return canonical_score_to_midi(score)


def _to_music21_score(score: CanonicalScore) -> music21.stream.Score:
    rendered = music21.stream.Score(id="canonical-score")
    _append_global_metadata(rendered, score)

    for part in score.parts:
        rendered.insert(0, _to_music21_part(score, part))

    return rendered


def _append_global_metadata(rendered: music21.stream.Score, score: CanonicalScore) -> None:
    for tick, time_signature in score.header.time_sig_map.items():
        rendered.insert(_offset_quarters(score, tick), music21.meter.TimeSignature(time_signature))

    for tick, bpm in score.header.tempo_map.items():
        rendered.insert(_offset_quarters(score, tick), music21.tempo.MetronomeMark(number=bpm))

    for tick, key_signature in score.header.key_sig_map.items():
        rendered.insert(_offset_quarters(score, tick), _parse_key_signature(key_signature))


def _to_music21_part(score: CanonicalScore, part: Part) -> music21.stream.Part:
    rendered_part = music21.stream.Part(id=part.info.id)
    rendered_part.partName = part.info.instrument
    rendered_part.insert(0, _instrument_for_part(part.info))

    for event in part.events:
        if event.pitch_midi is None:
            continue

        note = music21.note.Note(event.pitch_midi)
        note.duration = music21.duration.Duration(_offset_quarters(score, event.dur_tick))
        if event.velocity is not None:
            note.volume.velocity = event.velocity
        rendered_part.insert(_offset_quarters(score, event.start_tick), note)

    return rendered_part


def _instrument_for_part(part_info: PartInfo) -> music21.instrument.Instrument:
    instrument = music21.instrument.Instrument()
    instrument.partName = part_info.instrument
    instrument.instrumentName = part_info.instrument
    if part_info.midi_program is not None:
        instrument.midiProgram = part_info.midi_program
    return instrument


def _parse_key_signature(key_signature: str) -> music21.key.Key:
    mode = "major"
    tonic = key_signature
    if key_signature.endswith("m"):
        tonic = key_signature[:-1]
        mode = "minor"
    return music21.key.Key(tonic.replace("b", "-"), mode)


def _offset_quarters(score: CanonicalScore, ticks: int) -> float:
    return ticks / score.header.tpq
