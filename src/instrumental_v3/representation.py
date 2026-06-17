from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import music21

from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.tokens import eventizer as legacy_eventizer

TPQ = 24
GRID_TICKS = 6
MAX_VOICES = 2
MAX_INTERVAL = 24
MAX_DURATION_STEPS = 32
MAX_VERTICAL_INTERVAL = 48
MAX_STEPS_PER_BAR = 32
MAX_BARS = 128
PHRASE_BARS = 8

STATE_REST = 0
STATE_HOLD = 1
STATE_NOTE = 2

FIELD_NAMES = [
    "bar",
    "pos",
    "phrase_pos",
    "cadence_zone",
    "key_pc",
    "mode",
    "voice_count",
    "v0_state",
    "v0_pitch",
    "v0_mel",
    "v0_dur",
    "v0_tie",
    "v0_degree",
    "v1_state",
    "v1_pitch",
    "v1_mel",
    "v1_dur",
    "v1_tie",
    "v1_degree",
    "vertical_interval",
    "consonance",
    "spacing",
]

FEATURE_SPECS: dict[str, int] = {
    "bar": MAX_BARS,
    "pos": MAX_STEPS_PER_BAR,
    "phrase_pos": PHRASE_BARS,
    "cadence_zone": 2,
    "key_pc": 13,
    "mode": 3,
    "voice_count": 5,
    "v0_state": 3,
    "v0_pitch": 129,
    "v0_mel": (MAX_INTERVAL * 2) + 2,
    "v0_dur": MAX_DURATION_STEPS + 1,
    "v0_tie": 2,
    "v0_degree": 13,
    "v1_state": 3,
    "v1_pitch": 129,
    "v1_mel": (MAX_INTERVAL * 2) + 2,
    "v1_dur": MAX_DURATION_STEPS + 1,
    "v1_tie": 2,
    "v1_degree": 13,
    "vertical_interval": MAX_VERTICAL_INTERVAL + 2,
    "consonance": 4,
    "spacing": MAX_VERTICAL_INTERVAL + 2,
}

_MAJOR_SCALE = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
_MINOR_SCALE = {0: 1, 2: 2, 3: 3, 5: 4, 7: 5, 8: 6, 10: 7, 11: 7}
_PERFECT_CONSONANCES = {0, 7}
_IMPERFECT_CONSONANCES = {3, 4, 8, 9}


@dataclass(frozen=True)
class SliceEvent:
    values: list[int]

    def field(self, name: str) -> int:
        return self.values[FIELD_NAMES.index(name)]


@dataclass(frozen=True)
class InstrumentalV3Piece:
    piece_id: str
    source_path: str
    tpq: int
    grid_ticks: int
    time_signature: str
    key: str | None
    key_pc: int
    mode: int
    bar_len_ticks: int
    steps_per_bar: int
    slices: list[SliceEvent]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["slices"] = [slice_.values for slice_ in self.slices]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstrumentalV3Piece":
        return cls(
            piece_id=str(data["piece_id"]),
            source_path=str(data["source_path"]),
            tpq=int(data["tpq"]),
            grid_ticks=int(data["grid_ticks"]),
            time_signature=str(data["time_signature"]),
            key=data.get("key"),
            key_pc=int(data["key_pc"]),
            mode=int(data["mode"]),
            bar_len_ticks=int(data["bar_len_ticks"]),
            steps_per_bar=int(data["steps_per_bar"]),
            slices=[SliceEvent([int(v) for v in row]) for row in data["slices"]],
        )


def parse_musicxml_to_piece(
    path: str | Path,
    *,
    max_bars: int | None = None,
    normalize_key: bool = False,
) -> InstrumentalV3Piece:
    path = Path(path)
    score = music21.converter.parse(str(path))
    key_token = legacy_eventizer.detect_key_token(score, None)
    original_key_pc, original_mode = _key_token_to_context(key_token)
    if normalize_key and original_key_pc < 12:
        score, key_token = _transpose_score_to_c_tonic(score, original_key_pc, original_mode)

    time_sig = legacy_eventizer.detect_time_signature(score)
    time_signature = f"{time_sig[0]}/{time_sig[1]}"
    bar_len_ticks = int(round(time_sig[0] * (4.0 / time_sig[1]) * TPQ))
    if bar_len_ticks <= 0:
        raise ValueError(f"invalid bar length for {path}")
    steps_per_bar = bar_len_ticks // GRID_TICKS
    if steps_per_bar > MAX_STEPS_PER_BAR:
        raise ValueError(f"time signature {time_signature} exceeds max grid steps")

    key_pc, mode = _key_token_to_context(key_token)

    if len(score.parts) == MAX_VOICES:
        scored_parts = []
        for part in score.parts:
            events = [_snap_event(ev) for ev in legacy_eventizer._extract_events(part, TPQ, shift_ticks=0)]
            events = [ev for ev in events if ev.duration_tick > 0 and ev.onset_tick >= 0]
            if not events:
                continue
            median_pitch = sorted(ev.pitch for ev in events)[len(events) // 2]
            scored_parts.append((median_pitch, events))
        scored_parts.sort(key=lambda item: item[0])
        events_by_voice = {idx: events for idx, (_, events) in enumerate(scored_parts)}
    else:
        raw_events = []
        for part in score.parts:
            raw_events.extend(legacy_eventizer._extract_events(part, TPQ, shift_ticks=0))
        snapped = [_snap_event(ev) for ev in raw_events]
        snapped = [ev for ev in snapped if ev.duration_tick > 0 and ev.onset_tick >= 0]
        if not snapped:
            raise ValueError(f"no notes found in {path}")

        by_tick = legacy_eventizer._collapse_unison_octaves(
            snapped,
            max_octaves_per_pitch_class=1,
        )
        events_by_voice = legacy_eventizer._assign_events_by_continuity(by_tick, max_voices=MAX_VOICES)
    if len(events_by_voice) != MAX_VOICES:
        raise ValueError(f"expected exactly 2 active voices in {path}, got {len(events_by_voice)}")

    max_end = max(ev.onset_tick + ev.duration_tick for events in events_by_voice.values() for ev in events)
    num_bars = max(1, (max_end + bar_len_ticks - 1) // bar_len_ticks)
    if max_bars is not None:
        num_bars = min(num_bars, max_bars)
    num_bars = min(num_bars, MAX_BARS)
    total_ticks = num_bars * bar_len_ticks

    onset_maps: list[dict[int, legacy_eventizer.NoteEvent]] = []
    for voice in range(MAX_VOICES):
        voice_events = [ev for ev in events_by_voice.get(voice, []) if ev.onset_tick < total_ticks]
        onset_maps.append({ev.onset_tick: ev for ev in voice_events})

    prev_note_pitch: list[int | None] = [None, None]
    active_pitch: list[int | None] = [None, None]
    active_until = [0, 0]
    slices: list[SliceEvent] = []

    for tick in range(0, total_ticks, GRID_TICKS):
        bar = min(tick // bar_len_ticks, MAX_BARS - 1)
        pos = min((tick % bar_len_ticks) // GRID_TICKS, MAX_STEPS_PER_BAR - 1)
        phrase_pos = bar % PHRASE_BARS
        cadence_zone = 1 if phrase_pos in {PHRASE_BARS - 2, PHRASE_BARS - 1} else 0
        row = [bar, pos, phrase_pos, cadence_zone, key_pc, mode, MAX_VOICES]
        current_pitches: list[int | None] = []

        for voice in range(MAX_VOICES):
            ev = onset_maps[voice].get(tick)
            if ev is not None:
                state = STATE_NOTE
                pitch = _clip_pitch(ev.pitch)
                mel = _encode_melody(None if prev_note_pitch[voice] is None else pitch - prev_note_pitch[voice])
                dur = min(MAX_DURATION_STEPS, max(1, ev.duration_tick // GRID_TICKS))
                tie = 0
                prev_note_pitch[voice] = pitch
                active_pitch[voice] = pitch
                active_until[voice] = tick + ev.duration_tick
            elif active_until[voice] > tick and active_pitch[voice] is not None:
                state = STATE_HOLD
                pitch = active_pitch[voice]
                mel = 0
                dur = min(MAX_DURATION_STEPS, max(1, (active_until[voice] - tick) // GRID_TICKS))
                tie = 1
            else:
                state = STATE_REST
                pitch = 0
                mel = 0
                dur = 0
                tie = 0
                active_pitch[voice] = None

            degree = _scale_degree(pitch, key_pc, mode) if pitch else 0
            row.extend([state, pitch, mel, dur, tie, degree])
            current_pitches.append(pitch if pitch else None)

        row.extend(_vertical_features(current_pitches[0], current_pitches[1]))
        slices.append(SliceEvent(row))

    return InstrumentalV3Piece(
        piece_id=path.stem,
        source_path=str(path),
        tpq=TPQ,
        grid_ticks=GRID_TICKS,
        time_signature=time_signature,
        key=key_token[4:] if key_token else None,
        key_pc=key_pc,
        mode=mode,
        bar_len_ticks=bar_len_ticks,
        steps_per_bar=steps_per_bar,
        slices=slices,
    )


def _transpose_score_to_c_tonic(
    score: music21.stream.Score,
    original_key_pc: int,
    original_mode: int,
) -> tuple[music21.stream.Score, str]:
    target_pc = 0
    semitones = ((target_pc - original_key_pc + 6) % 12) - 6
    transposed = score.transpose(semitones, inPlace=False)
    key_token = "KEY_Cm" if original_mode == 1 else "KEY_C"
    return transposed, key_token


def piece_to_canonical_score(piece: InstrumentalV3Piece, *, title: str | None = None) -> CanonicalScore:
    events: list[Event] = []
    active: list[tuple[int, int] | None] = [None, None]
    total_ticks = len(piece.slices) * piece.grid_ticks
    measure_count = max(1, (total_ticks + piece.bar_len_ticks - 1) // piece.bar_len_ticks)

    for idx, slice_ in enumerate(piece.slices):
        tick = idx * piece.grid_ticks
        for voice in range(MAX_VOICES):
            prefix = f"v{voice}_"
            state = slice_.field(prefix + "state")
            pitch = slice_.field(prefix + "pitch")
            if state == STATE_NOTE and pitch > 0:
                if active[voice] is not None:
                    start, old_pitch = active[voice]
                    if tick > start:
                        events.append(_event(start, tick - start, voice, old_pitch, len(events)))
                active[voice] = (tick, pitch)
            elif state != STATE_HOLD:
                if active[voice] is not None:
                    start, old_pitch = active[voice]
                    if tick > start:
                        events.append(_event(start, tick - start, voice, old_pitch, len(events)))
                    active[voice] = None

    for voice, current in enumerate(active):
        if current is None:
            continue
        start, pitch = current
        if total_ticks > start:
            events.append(_event(start, total_ticks - start, voice, pitch, len(events)))

    present_voices = {event.voice_id for event in events}
    for voice in range(MAX_VOICES):
        if voice not in present_voices:
            events.append(
                Event(
                    id=f"rest-v{voice}",
                    start_tick=0,
                    dur_tick=total_ticks,
                    voice_id=voice,
                    pitch_midi=None,
                )
            )

    events.sort(key=lambda ev: (ev.start_tick, ev.voice_id, ev.pitch_midi or 0, ev.id))
    measures = [
        Measure(
            id=f"m{i}",
            index=i,
            start_tick=i * piece.bar_len_ticks,
            length_ticks=piece.bar_len_ticks,
        )
        for i in range(measure_count)
    ]
    header = ScoreHeader(
        tpq=piece.tpq,
        key_sig_map={0: piece.key} if piece.key else {},
        time_sig_map={0: piece.time_signature},
        tempo_map={0: 92},
    )
    return CanonicalScore(
        header=header,
        measures=measures,
        parts=[Part(PartInfo(id="P1", instrument="piano", midi_program=0), events=events)],
    )


def slice_rows_to_piece(
    rows: list[list[int]],
    *,
    template: InstrumentalV3Piece,
    piece_id: str,
    source_path: str = "generated",
) -> InstrumentalV3Piece:
    return InstrumentalV3Piece(
        piece_id=piece_id,
        source_path=source_path,
        tpq=template.tpq,
        grid_ticks=template.grid_ticks,
        time_signature=template.time_signature,
        key=template.key,
        key_pc=template.key_pc,
        mode=template.mode,
        bar_len_ticks=template.bar_len_ticks,
        steps_per_bar=template.steps_per_bar,
        slices=[SliceEvent([_clip_field(name, int(value)) for name, value in zip(FIELD_NAMES, row)]) for row in rows],
    )


def _snap_event(ev: legacy_eventizer.NoteEvent) -> legacy_eventizer.NoteEvent:
    onset = int(round(ev.onset_tick / GRID_TICKS) * GRID_TICKS)
    dur = max(GRID_TICKS, int(round(ev.duration_tick / GRID_TICKS) * GRID_TICKS))
    return legacy_eventizer.NoteEvent(onset_tick=onset, duration_tick=dur, pitch=ev.pitch)


def _event(start: int, dur: int, voice: int, pitch: int, idx: int) -> Event:
    return Event(
        id=f"n{idx}",
        start_tick=start,
        dur_tick=max(1, dur),
        voice_id=voice,
        pitch_midi=pitch,
        velocity=82 if voice == 1 else 74,
    )


def _key_token_to_context(key_token: str | None) -> tuple[int, int]:
    if not key_token:
        return 12, 2
    key = key_token[4:]
    mode = 1 if key.endswith("m") else 0
    tonic = key[:-1] if mode == 1 else key
    try:
        pc = music21.pitch.Pitch(tonic.replace("b", "-")).pitchClass
    except Exception:
        return 12, 2
    return int(pc), mode


def _scale_degree(pitch: int, key_pc: int, mode: int) -> int:
    if key_pc >= 12:
        return 0
    rel = (pitch - key_pc) % 12
    scale = _MINOR_SCALE if mode == 1 else _MAJOR_SCALE
    return scale.get(rel, 8 + rel % 5)


def _vertical_features(low_pitch: int | None, high_pitch: int | None) -> list[int]:
    if low_pitch is None or high_pitch is None:
        return [0, 0, 0]
    spacing = abs(high_pitch - low_pitch)
    interval = spacing % 12
    if interval in _PERFECT_CONSONANCES:
        consonance = 1
    elif interval in _IMPERFECT_CONSONANCES:
        consonance = 2
    else:
        consonance = 3
    encoded = min(MAX_VERTICAL_INTERVAL, spacing) + 1
    return [encoded, consonance, encoded]


def _encode_melody(delta: int | None) -> int:
    if delta is None:
        return 0
    clipped = max(-MAX_INTERVAL, min(MAX_INTERVAL, int(delta)))
    return clipped + MAX_INTERVAL + 1


def _clip_pitch(pitch: int) -> int:
    return max(1, min(128, int(pitch)))


def _clip_field(name: str, value: int) -> int:
    return max(0, min(FEATURE_SPECS[name] - 1, value))
