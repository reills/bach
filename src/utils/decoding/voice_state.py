from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch

from src.tokens.tokenizer import parse_voice_event


VOICE_RANGES: dict[int, tuple[int, int]] = {
    0: (36, 60),
    1: (48, 67),
    2: (55, 74),
    3: (60, 84),
}

DEFAULT_VOICE_RANGE = (36, 84)
DEFAULT_ADJACENT_SPACING = 19
BASS_TENOR_SPACING = 24
VoiceLeadingMode = Literal["off", "balanced", "best"]
VOICE_LEADING_DEFAULT: VoiceLeadingMode = "balanced"


@dataclass(frozen=True)
class _PitchedEvent:
    voice: int
    start_tick: int
    end_tick: int
    pitch: int


@dataclass(frozen=True)
class VoiceLeadingState:
    pending_voice: int | None
    previous_pitch: Mapping[int, int]
    current_tick: int | None
    current_sonority: Mapping[int, int]
    previous_sonority: Mapping[int, int]

    @property
    def is_complete_for_melody(self) -> bool:
        return (
            self.pending_voice is not None
            and self.current_tick is not None
            and self.pending_voice in self.previous_pitch
        )


def build_voice_leading_mask(
    prefix_tokens: Sequence[str],
    vocab: Mapping[str, int],
    *,
    allowed_categories: set[str] | None = None,
    tpq: int = 24,
) -> torch.Tensor:
    """Return a vocab mask that hard-blocks obvious bad MEL_INT12 choices only."""
    vocab_size = max(vocab.values()) + 1 if vocab else 0
    mask = torch.ones(vocab_size, dtype=torch.bool)
    if allowed_categories is not None and "MEL_INT12" not in allowed_categories:
        return mask

    mel_candidates = _mel_candidates(vocab)
    if not mel_candidates:
        return mask

    state = voice_leading_state_from_tokens(prefix_tokens, tpq=tpq)
    if not state.is_complete_for_melody:
        return mask

    blocked_ids: list[int] = []
    allowed_mel_count = 0
    voice = state.pending_voice
    assert voice is not None
    base_pitch = state.previous_pitch[voice]
    for token, token_id, interval in mel_candidates:
        candidate_pitch = base_pitch + interval
        if _candidate_is_blocked(state, voice, candidate_pitch):
            blocked_ids.append(token_id)
        else:
            allowed_mel_count += 1

    if allowed_mel_count == 0:
        return mask

    for token_id in blocked_ids:
        mask[token_id] = False
    return mask


def normalize_voice_leading(value: str | None, *, default: VoiceLeadingMode = VOICE_LEADING_DEFAULT) -> VoiceLeadingMode:
    if value is None:
        return default
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"off", "none", "disabled", "disable", "false", "0", "fast"}:
        return "off"
    if normalized in {"balanced", "on", "enabled", "enable", "true", "1"}:
        return "balanced"
    if normalized in {"best", "strict"}:
        return "best"
    raise ValueError("voiceLeading must be one of: fast, off, balanced, best")


def voice_leading_enabled(mode: VoiceLeadingMode) -> bool:
    return mode != "off"


def voice_leading_state_from_tokens(
    tokens: Sequence[str],
    *,
    tpq: int = 24,
) -> VoiceLeadingState:
    previous_pitch: dict[int, int] = {}
    events: list[_PitchedEvent] = []
    bar_len_ticks = tpq * 4
    bar_start_tick = 0
    saw_bar = False
    current_pos_tick: int | None = None
    current_tick: int | None = None
    idx = 0

    while idx < len(tokens):
        token = tokens[idx]
        if token == "BAR":
            if saw_bar:
                bar_start_tick += bar_len_ticks
            else:
                saw_bar = True
            current_pos_tick = None
            current_tick = None
            idx += 1
            continue

        if token.startswith("TIME_SIG_"):
            try:
                numerator, denominator = _parse_time_sig_token(token)
                bar_len_ticks = int(round(numerator * (4.0 / denominator) * tpq))
            except ValueError:
                pass
            idx += 1
            continue

        if token.startswith("POS_"):
            try:
                current_pos_tick = int(token.split("_", 1)[1])
                current_tick = bar_start_tick + current_pos_tick
            except ValueError:
                current_pos_tick = None
                current_tick = None
            idx += 1
            continue

        if token.startswith("ABS_VOICE_"):
            try:
                voice, pitch = _parse_abs_voice_token(token)
            except ValueError:
                idx += 1
                continue
            previous_pitch[voice] = pitch
            idx += 1
            continue

        if token.startswith("ABS_BASS_"):
            try:
                previous_pitch[0] = int(token.split("_")[-1])
            except ValueError:
                pass
            idx += 1
            continue

        if token.startswith("ABS_SOP_"):
            try:
                previous_pitch[3] = int(token.split("_")[-1])
            except ValueError:
                pass
            idx += 1
            continue

        if token.startswith("VOICE_"):
            try:
                event, next_idx = parse_voice_event(tokens, idx)
            except ValueError:
                idx += 1
                continue

            if not event.is_rest and current_tick is not None:
                base_pitch = previous_pitch.get(event.voice)
                if base_pitch is not None:
                    pitch = base_pitch + event.mel_int
                    previous_pitch[event.voice] = pitch
                    events.append(
                        _PitchedEvent(
                            voice=event.voice,
                            start_tick=current_tick,
                            end_tick=current_tick + event.duration_ticks,
                            pitch=pitch,
                        )
                    )
            idx = next_idx
            continue

        idx += 1

    pending_voice = _pending_melody_voice(tokens)
    current_sonority = _sonority_at_tick(events, current_tick) if current_tick is not None else {}
    previous_tick = _previous_event_tick(events, current_tick)
    previous_sonority = _sonority_at_tick(events, previous_tick) if previous_tick is not None else {}
    return VoiceLeadingState(
        pending_voice=pending_voice,
        previous_pitch=previous_pitch,
        current_tick=current_tick,
        current_sonority=current_sonority,
        previous_sonority=previous_sonority,
    )


def _candidate_is_blocked(state: VoiceLeadingState, voice: int, pitch: int) -> bool:
    low, high = VOICE_RANGES.get(voice, DEFAULT_VOICE_RANGE)
    if pitch < low or pitch > high:
        return True

    current = dict(state.current_sonority)
    current.pop(voice, None)
    lower = current.get(voice - 1)
    upper = current.get(voice + 1)

    if lower is not None:
        if pitch < lower:
            return True
        if pitch - lower > _max_spacing(voice - 1, voice):
            return True

    if upper is not None:
        if pitch > upper:
            return True
        if upper - pitch > _max_spacing(voice, voice + 1):
            return True

    return _creates_parallel_perfect_interval(state, voice, pitch, current)


def _creates_parallel_perfect_interval(
    state: VoiceLeadingState,
    voice: int,
    pitch: int,
    current_without_voice: Mapping[int, int],
) -> bool:
    previous_voice_pitch = state.previous_sonority.get(voice)
    if previous_voice_pitch is None:
        return False

    voice_motion = pitch - previous_voice_pitch
    if voice_motion == 0:
        return False

    for other_voice, other_pitch in current_without_voice.items():
        previous_other_pitch = state.previous_sonority.get(other_voice)
        if previous_other_pitch is None:
            continue
        other_motion = other_pitch - previous_other_pitch
        if not _same_nonzero_direction(voice_motion, other_motion):
            continue
        previous_kind = _perfect_interval_kind(previous_voice_pitch, previous_other_pitch)
        current_kind = _perfect_interval_kind(pitch, other_pitch)
        if previous_kind is not None and previous_kind == current_kind:
            return True
    return False


def _mel_candidates(vocab: Mapping[str, int]) -> list[tuple[str, int, int]]:
    candidates: list[tuple[str, int, int]] = []
    for token, token_id in vocab.items():
        if not token.startswith("MEL_INT12_"):
            continue
        try:
            interval = int(token.split("_")[-1])
        except ValueError:
            continue
        candidates.append((token, token_id, interval))
    return candidates


def _pending_melody_voice(tokens: Sequence[str]) -> int | None:
    if len(tokens) >= 2 and tokens[-2].startswith("VOICE_") and tokens[-1].startswith("DUR_"):
        return _parse_prefixed_int(tokens[-2], "VOICE_")
    if (
        len(tokens) >= 3
        and tokens[-3].startswith("VOICE_")
        and tokens[-2].startswith("DUR_")
        and tokens[-1].startswith("DUP_")
    ):
        return _parse_prefixed_int(tokens[-3], "VOICE_")
    return None


def _sonority_at_tick(events: Sequence[_PitchedEvent], tick: int | None) -> dict[int, int]:
    if tick is None:
        return {}
    active: dict[int, _PitchedEvent] = {}
    for event in events:
        if event.start_tick <= tick < event.end_tick:
            current = active.get(event.voice)
            if current is None or event.start_tick >= current.start_tick:
                active[event.voice] = event
    return {voice: event.pitch for voice, event in active.items()}


def _previous_event_tick(events: Sequence[_PitchedEvent], current_tick: int | None) -> int | None:
    if current_tick is None:
        return None
    previous_ticks = [event.start_tick for event in events if event.start_tick < current_tick]
    return max(previous_ticks) if previous_ticks else None


def _parse_abs_voice_token(token: str) -> tuple[int, int]:
    parts = token.split("_")
    if len(parts) != 4:
        raise ValueError(f"bad ABS_VOICE token: {token}")
    return int(parts[2]), int(parts[3])


def _parse_time_sig_token(token: str) -> tuple[int, int]:
    parts = token.split("_")
    if len(parts) != 4:
        raise ValueError(f"bad TIME_SIG token: {token}")
    return int(parts[2]), int(parts[3])


def _parse_prefixed_int(token: str, prefix: str) -> int | None:
    if not token.startswith(prefix):
        return None
    try:
        return int(token[len(prefix) :])
    except ValueError:
        return None


def _max_spacing(lower_voice: int, upper_voice: int) -> int:
    if lower_voice == 0 and upper_voice == 1:
        return BASS_TENOR_SPACING
    return DEFAULT_ADJACENT_SPACING


def _perfect_interval_kind(left_pitch: int, right_pitch: int) -> str | None:
    semitones = abs(right_pitch - left_pitch)
    interval_class = semitones % 12
    if semitones > 0 and interval_class == 0:
        return "octave"
    if interval_class == 7:
        return "fifth"
    return None


def _same_nonzero_direction(left_motion: int, right_motion: int) -> bool:
    return (left_motion > 0 and right_motion > 0) or (left_motion < 0 and right_motion < 0)
