from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from src.tokens.roundtrip import (
    parse_abs_voice_token,
    parse_last_token_int,
    parse_time_sig_token,
    parse_token_int,
)
from src.tokens.tokenizer import parse_voice_event
from src.tokens.validator import validate_harm_tokens


CONSONANT_INTERVAL_CLASSES = {0, 3, 4, 7, 8, 9}


@dataclass(frozen=True)
class PitchedEvent:
    voice: int
    start_tick: int
    dur_tick: int
    pitch: int
    bar_index: int
    pos_tick: int

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.dur_tick


@dataclass(frozen=True)
class _Sonority:
    tick: int
    bar_index: int
    pos_tick: int
    pitches: dict[int, int]


@dataclass(frozen=True)
class CounterpointMetrics:
    parallel_fifths: int
    parallel_octaves: int
    parallel_unisons: int
    direct_fifths: int
    direct_octaves: int
    voice_crossings: int
    spacing_violations: int
    dissonance_on_strong_beat: int
    unresolved_dissonances: int
    avg_active_voices: float | None
    monophonic_position_rate: float | None
    static_voice_rate: float | None
    harmonic_metadata_mismatches: int | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def evaluate_counterpoint_tokens(tokens: Sequence[str], *, tpq: int = 24) -> CounterpointMetrics:
    """Reconstruct token pitches and score basic contrapuntal failure modes."""
    events = pitched_events_from_tokens(tokens, tpq=tpq)
    try:
        mismatch_count: int | None = len(validate_harm_tokens(list(tokens), tpq=tpq))
    except Exception:
        mismatch_count = None
    return evaluate_counterpoint_events(
        events,
        tpq=tpq,
        harmonic_metadata_mismatches=mismatch_count,
    )


def evaluate_counterpoint_events(
    events: Sequence[PitchedEvent],
    *,
    tpq: int = 24,
    harmonic_metadata_mismatches: int | None = None,
    max_adjacent_spacing: int = 19,
) -> CounterpointMetrics:
    sonorities = _build_sonorities(events)
    position_count = len(sonorities)

    active_counts = [len(sonority.pitches) for sonority in sonorities]
    avg_active_voices = round(sum(active_counts) / position_count, 3) if position_count else None
    monophonic_position_rate = (
        round(sum(1 for count in active_counts if count <= 1) / position_count, 4)
        if position_count
        else None
    )

    voice_crossings = 0
    spacing_violations = 0
    dissonance_on_strong_beat = 0
    for sonority in sonorities:
        voices = sorted(sonority.pitches)
        for lower, upper in zip(voices, voices[1:]):
            lower_pitch = sonority.pitches[lower]
            upper_pitch = sonority.pitches[upper]
            if lower_pitch > upper_pitch:
                voice_crossings += 1
            if upper_pitch - lower_pitch > max_adjacent_spacing:
                spacing_violations += 1

        if sonority.pos_tick == 0:
            for left_idx, left in enumerate(voices):
                for right in voices[left_idx + 1 :]:
                    if _interval_class(sonority.pitches[left], sonority.pitches[right]) not in CONSONANT_INTERVAL_CLASSES:
                        dissonance_on_strong_beat += 1

    parallel_fifths = 0
    parallel_octaves = 0
    parallel_unisons = 0
    direct_fifths = 0
    direct_octaves = 0
    unresolved_dissonances = 0

    for previous, current in zip(sonorities, sonorities[1:]):
        common_voices = sorted(set(previous.pitches) & set(current.pitches))
        for left_idx, left in enumerate(common_voices):
            for right in common_voices[left_idx + 1 :]:
                prev_interval = _interval_kind(previous.pitches[left], previous.pitches[right])
                curr_interval = _interval_kind(current.pitches[left], current.pitches[right])
                left_motion = current.pitches[left] - previous.pitches[left]
                right_motion = current.pitches[right] - previous.pitches[right]
                same_direction = _same_nonzero_direction(left_motion, right_motion)

                if same_direction and curr_interval == "fifth":
                    if prev_interval == "fifth":
                        parallel_fifths += 1
                    elif abs(left_motion) > 2 or abs(right_motion) > 2:
                        direct_fifths += 1
                elif same_direction and curr_interval == "octave":
                    if prev_interval == "octave":
                        parallel_octaves += 1
                    elif abs(left_motion) > 2 or abs(right_motion) > 2:
                        direct_octaves += 1
                elif same_direction and curr_interval == "unison" and prev_interval == "unison":
                    parallel_unisons += 1

                if prev_interval == "dissonance" and curr_interval == "dissonance":
                    unresolved_dissonances += 1

    return CounterpointMetrics(
        parallel_fifths=parallel_fifths,
        parallel_octaves=parallel_octaves,
        parallel_unisons=parallel_unisons,
        direct_fifths=direct_fifths,
        direct_octaves=direct_octaves,
        voice_crossings=voice_crossings,
        spacing_violations=spacing_violations,
        dissonance_on_strong_beat=dissonance_on_strong_beat,
        unresolved_dissonances=unresolved_dissonances,
        avg_active_voices=avg_active_voices,
        monophonic_position_rate=monophonic_position_rate,
        static_voice_rate=_static_voice_rate(events),
        harmonic_metadata_mismatches=harmonic_metadata_mismatches,
    )


def pitched_events_from_tokens(tokens: Sequence[str], *, tpq: int = 24) -> list[PitchedEvent]:
    events: list[PitchedEvent] = []
    prev_pitch: dict[int, int] = {}
    bar_len_ticks = tpq * 4
    bar_start_tick = 0
    bar_index = -1
    saw_bar = False
    current_pos_tick: int | None = None
    idx = 0

    while idx < len(tokens):
        token = tokens[idx]

        if token == "BAR":
            if saw_bar:
                bar_start_tick += bar_len_ticks
            else:
                saw_bar = True
            bar_index += 1
            current_pos_tick = None
            idx += 1
            continue

        if token.startswith("TIME_SIG_"):
            try:
                numerator, denominator = parse_time_sig_token(token)
                bar_len_ticks = int(round(numerator * (4.0 / denominator) * tpq))
            except ValueError:
                pass
            idx += 1
            continue

        if token.startswith("POS_"):
            try:
                current_pos_tick = parse_token_int(token)
            except ValueError:
                current_pos_tick = None
            idx += 1
            continue

        if token.startswith("ABS_VOICE_"):
            try:
                voice, pitch = parse_abs_voice_token(token)
            except ValueError:
                idx += 1
                continue
            prev_pitch[voice] = pitch
            idx += 1
            continue

        if token.startswith("ABS_BASS_"):
            prev_pitch[0] = parse_last_token_int(token)
            idx += 1
            continue

        if token.startswith("ABS_SOP_"):
            prev_pitch[3] = parse_last_token_int(token)
            idx += 1
            continue

        if token.startswith("VOICE_"):
            try:
                event, next_idx = parse_voice_event(tokens, idx)
            except ValueError:
                idx += 1
                continue

            if current_pos_tick is None or event.is_rest:
                idx = next_idx
                continue

            previous_pitch = prev_pitch.get(event.voice)
            if previous_pitch is None:
                idx = next_idx
                continue

            pitch = previous_pitch + event.mel_int
            prev_pitch[event.voice] = pitch
            events.append(
                PitchedEvent(
                    voice=event.voice,
                    start_tick=bar_start_tick + current_pos_tick,
                    dur_tick=event.duration_ticks,
                    pitch=pitch,
                    bar_index=max(bar_index, 0),
                    pos_tick=current_pos_tick,
                )
            )
            idx = next_idx
            continue

        idx += 1

    return events


def _build_sonorities(events: Sequence[PitchedEvent]) -> list[_Sonority]:
    if not events:
        return []
    sorted_events = sorted(events, key=lambda event: (event.start_tick, event.voice, event.pitch))
    events_by_tick = {event.start_tick: event for event in sorted_events}
    sonorities: list[_Sonority] = []

    for tick in sorted({event.start_tick for event in sorted_events}):
        active: dict[int, PitchedEvent] = {}
        for event in sorted_events:
            if event.start_tick <= tick < event.end_tick:
                current = active.get(event.voice)
                if current is None or event.start_tick >= current.start_tick:
                    active[event.voice] = event
        anchor = events_by_tick[tick]
        sonorities.append(
            _Sonority(
                tick=tick,
                bar_index=anchor.bar_index,
                pos_tick=anchor.pos_tick,
                pitches={voice: event.pitch for voice, event in active.items()},
            )
        )
    return sonorities


def _interval_class(left_pitch: int, right_pitch: int) -> int:
    return abs(right_pitch - left_pitch) % 12


def _interval_kind(left_pitch: int, right_pitch: int) -> str:
    semitones = abs(right_pitch - left_pitch)
    interval_class = semitones % 12
    if semitones == 0:
        return "unison"
    if interval_class == 0:
        return "octave"
    if interval_class == 7:
        return "fifth"
    if interval_class in CONSONANT_INTERVAL_CLASSES:
        return "consonance"
    return "dissonance"


def _same_nonzero_direction(left_motion: int, right_motion: int) -> bool:
    return (left_motion > 0 and right_motion > 0) or (left_motion < 0 and right_motion < 0)


def _static_voice_rate(events: Sequence[PitchedEvent]) -> float | None:
    transitions = 0
    static = 0
    by_voice: dict[int, list[PitchedEvent]] = {}
    for event in events:
        by_voice.setdefault(event.voice, []).append(event)

    for voice_events in by_voice.values():
        ordered = sorted(voice_events, key=lambda event: (event.start_tick, event.pitch))
        for previous, current in zip(ordered, ordered[1:]):
            transitions += 1
            if previous.pitch == current.pitch:
                static += 1

    if transitions == 0:
        return None
    return round(static / transitions, 4)
