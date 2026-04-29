from __future__ import annotations

import io

import mido
import music21

from src.api.canonical.types import CanonicalScore, Part, PartInfo


def canonical_score_to_midi(score: CanonicalScore) -> bytes:
    midi_file = mido.MidiFile(type=1, ticks_per_beat=score.header.tpq)

    meta_track = mido.MidiTrack()
    midi_file.tracks.append(meta_track)
    _append_meta_events(meta_track, score)

    for part in score.parts:
        midi_file.tracks.append(_part_to_midi_track(score, part))

    output = io.BytesIO()
    midi_file.save(file=output)
    return output.getvalue()


def score_to_midi(score: CanonicalScore) -> bytes:
    return canonical_score_to_midi(score)


def _append_meta_events(track: mido.MidiTrack, score: CanonicalScore) -> None:
    events: list[tuple[int, mido.MetaMessage]] = []
    for tick, time_signature in score.header.time_sig_map.items():
        numerator, denominator = _parse_time_signature(time_signature)
        events.append(
            (
                tick,
                mido.MetaMessage(
                    "time_signature",
                    numerator=numerator,
                    denominator=denominator,
                    time=0,
                ),
            )
        )
    for tick, bpm in score.header.tempo_map.items():
        events.append((tick, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0)))
    for tick, key_signature in score.header.key_sig_map.items():
        try:
            events.append((tick, mido.MetaMessage("key_signature", key=key_signature, time=0)))
        except ValueError:
            continue

    _append_absolute_events(track, events)
    score_end = score.measures[-1].end_tick if score.measures else 0
    track.append(mido.MetaMessage("end_of_track", time=max(0, score_end - _track_absolute_time(track))))


def _part_to_midi_track(score: CanonicalScore, part: Part) -> mido.MidiTrack:
    track = mido.MidiTrack()
    program = part.info.midi_program if part.info.midi_program is not None else 0
    channels = sorted({event.voice_id for event in part.events if event.pitch_midi is not None})
    events: list[tuple[int, mido.Message]] = []
    for voice_id in channels:
        events.append(
            (
                0,
                mido.Message(
                    "program_change",
                    channel=_voice_channel(voice_id),
                    program=program,
                    time=0,
                ),
            )
        )

    for event in part.events:
        if event.pitch_midi is None:
            continue
        channel = _voice_channel(event.voice_id)
        velocity = event.velocity if event.velocity is not None else 80
        events.append(
            (
                event.start_tick,
                mido.Message(
                    "note_on",
                    channel=channel,
                    note=event.pitch_midi,
                    velocity=velocity,
                    time=0,
                ),
            )
        )
        events.append(
            (
                event.start_tick + event.dur_tick,
                mido.Message(
                    "note_off",
                    channel=channel,
                    note=event.pitch_midi,
                    velocity=0,
                    time=0,
                ),
            )
        )

    _append_absolute_events(track, events)
    score_end = score.measures[-1].end_tick if score.measures else 0
    track.append(mido.MetaMessage("end_of_track", time=max(0, score_end - _track_absolute_time(track))))
    return track


def _append_absolute_events(
    track: mido.MidiTrack,
    events: list[tuple[int, mido.Message | mido.MetaMessage]],
) -> None:
    absolute_time = 0
    for tick, message in sorted(events, key=lambda item: (item[0], _event_order(item[1]))):
        delta = max(0, tick - absolute_time)
        absolute_time = max(absolute_time, tick)
        track.append(message.copy(time=delta))


def _event_order(message: mido.Message | mido.MetaMessage) -> int:
    if getattr(message, "type", "") == "note_off":
        return 0
    if getattr(message, "type", "") == "program_change":
        return 1
    if getattr(message, "type", "") == "note_on":
        return 2
    return 1


def _track_absolute_time(track: mido.MidiTrack) -> int:
    return sum(int(message.time) for message in track)


def _voice_channel(voice_id: int) -> int:
    channel = voice_id % 15
    if channel >= 9:
        channel += 1
    return min(channel, 15)


def _parse_time_signature(value: str) -> tuple[int, int]:
    numerator, denominator = value.split("/", 1)
    return int(numerator), int(denominator)


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
