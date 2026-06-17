from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import mido

from src.music.counterpoint import PitchedEvent, pitched_events_from_tokens

TPQ = 24
DEFAULT_BAR_LEN_TICKS = TPQ * 4
SATB_NAMES = ("BASS", "TENOR", "ALTO", "SOP")

_POS_RE = re.compile(r"^POS_(\d+)$")
_DUR_RE = re.compile(r"^DUR_(\d+)$")
_PITCH_RE = {
    name: re.compile(rf"^{name}_(\d+)$")
    for name in SATB_NAMES
}


@dataclass(frozen=True)
class VerticalSlice:
    bar_index: int
    pos_tick: int
    dur_tick: int
    pitches: tuple[int, int, int, int]

    @property
    def start_tick(self) -> int:
        return self.bar_index * DEFAULT_BAR_LEN_TICKS + self.pos_tick


@dataclass(frozen=True)
class V2Bar:
    piece_id: str
    bar_index: int
    tokens: list[str]
    plan_json: str
    bar_len_ticks: int
    source_path: str | None = None
    source_sha256: str | None = None


def split_tokens(value: object) -> list[str]:
    if isinstance(value, str):
        return [token for token in value.split() if token]
    if isinstance(value, Sequence):
        return [str(token) for token in value if token]
    raise TypeError(f"unsupported token value: {type(value)}")


def build_vocab(token_values: Iterable[object], *, special_tokens: Sequence[str] = ("<pad>", "<unk>")) -> dict[str, int]:
    seen: OrderedDict[str, None] = OrderedDict()
    for token in special_tokens:
        seen[str(token)] = None
    for value in token_values:
        for token in split_tokens(value):
            seen.setdefault(token, None)
    return {token: idx for idx, token in enumerate(seen)}


def build_v2_bars_from_v1_rows(rows: Sequence[Mapping[str, object]]) -> list[V2Bar]:
    sorted_rows = sorted(rows, key=lambda row: int(row.get("bar_index", 0)))
    if not sorted_rows:
        return []

    flat_tokens: list[str] = []
    for row in sorted_rows:
        flat_tokens.extend(split_tokens(row["tokens"]))

    events = pitched_events_from_tokens(flat_tokens, tpq=TPQ)
    piece_id = str(sorted_rows[0].get("piece_id", "piece"))
    result: list[V2Bar] = []

    bar_start = 0
    for row in sorted_rows:
        bar_index = int(row.get("bar_index", len(result)))
        bar_len = int(row.get("bar_len_ticks") or DEFAULT_BAR_LEN_TICKS)
        plan = _parse_plan(row.get("plan_json"), bar_index)
        row_tokens = split_tokens(row["tokens"])
        key = str(plan.get("key") or _first_token_suffix(row_tokens, "KEY_") or "C")
        time_sig = str(plan.get("time_sig") or _time_sig_from_tokens(row_tokens) or "4/4")

        slices = _vertical_slices_for_bar(
            events,
            bar_index=bar_index,
            bar_start_tick=bar_start,
            bar_len_ticks=bar_len,
        )
        if not slices:
            bar_start += bar_len
            continue

        tokens = _bar_tokens(slices, key=key, time_sig=time_sig)
        plan_json = json.dumps(
            {
                "bar_index": bar_index,
                "time_sig": time_sig,
                "key": key,
                "density_bucket": "DENSITY_V2",
                "pitch_range": max(max(s.pitches) - min(s.pitches) for s in slices),
                "polyphony_max": 4,
            }
        )
        result.append(
            V2Bar(
                piece_id=piece_id,
                bar_index=bar_index,
                tokens=tokens,
                plan_json=plan_json,
                bar_len_ticks=bar_len,
                source_path=_optional_str(row.get("source_path")),
                source_sha256=_optional_str(row.get("source_sha256")),
            )
        )
        bar_start += bar_len

    return result


def parse_v2_slices(tokens: Sequence[str]) -> list[VerticalSlice]:
    slices: list[VerticalSlice] = []
    bar_index = -1
    idx = 0

    while idx < len(tokens):
        token = tokens[idx]
        if token == "BAR":
            bar_index += 1
            idx += 1
            continue

        pos_match = _POS_RE.match(token)
        if not pos_match:
            idx += 1
            continue

        if idx + 5 >= len(tokens):
            idx += 1
            continue

        pos_tick = int(pos_match.group(1))
        pitches: list[int] = []
        valid = True
        for offset, name in enumerate(SATB_NAMES, start=1):
            match = _PITCH_RE[name].match(tokens[idx + offset])
            if not match:
                valid = False
                break
            pitches.append(int(match.group(1)))

        dur_match = _DUR_RE.match(tokens[idx + 5])
        if valid and dur_match and bar_index >= 0:
            slices.append(
                VerticalSlice(
                    bar_index=bar_index,
                    pos_tick=pos_tick,
                    dur_tick=int(dur_match.group(1)),
                    pitches=(pitches[0], pitches[1], pitches[2], pitches[3]),
                )
            )
            idx += 6
            continue

        idx += 1

    return slices


def v2_repetition_metrics(tokens: Sequence[str]) -> dict[str, float | int]:
    slices = parse_v2_slices(tokens)
    if not slices:
        return {
            "slice_count": 0,
            "unique_sonority_count": 0,
            "unique_sonority_rate": 0.0,
            "adjacent_repeat_count": 0,
            "adjacent_repeat_rate": 0.0,
            "longest_sonority_run": 0,
            "duplicate_bar_count": 0,
            "duplicate_bar_rate": 0.0,
        }

    sonorities = [item.pitches for item in slices]
    adjacent_repeat_count = sum(
        1 for prev, current in zip(sonorities, sonorities[1:]) if prev == current
    )
    bar_patterns: list[tuple[tuple[int, tuple[int, int, int, int]], ...]] = []
    for bar_index in sorted({item.bar_index for item in slices}):
        bar_patterns.append(
            tuple(
                (item.pos_tick, item.pitches)
                for item in slices
                if item.bar_index == bar_index
            )
        )
    duplicate_bar_count = len(bar_patterns) - len(set(bar_patterns))

    return {
        "slice_count": len(slices),
        "unique_sonority_count": len(set(sonorities)),
        "unique_sonority_rate": round(len(set(sonorities)) / len(sonorities), 6),
        "adjacent_repeat_count": adjacent_repeat_count,
        "adjacent_repeat_rate": round(adjacent_repeat_count / max(1, len(sonorities) - 1), 6),
        "longest_sonority_run": _longest_equal_run(sonorities),
        "duplicate_bar_count": duplicate_bar_count,
        "duplicate_bar_rate": round(duplicate_bar_count / max(1, len(bar_patterns)), 6),
    }


def render_v2_tokens_to_midi(tokens: Sequence[str], path: str | Path, *, tpq: int = TPQ, tempo_bpm: int = 84) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=tpq)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(meta)

    by_voice: list[list[tuple[int, int, int]]] = [[] for _ in SATB_NAMES]
    for item in parse_v2_slices(tokens):
        start = item.bar_index * DEFAULT_BAR_LEN_TICKS + item.pos_tick
        for voice, pitch in enumerate(item.pitches):
            by_voice[voice].append((start, item.dur_tick, pitch))

    for voice, notes in enumerate(by_voice):
        track = mido.MidiTrack()
        track.append(mido.Message("program_change", program=0, channel=voice, time=0))
        events: list[tuple[int, int, mido.Message]] = []
        for start, dur, pitch in notes:
            velocity = 72 if voice < 3 else 78
            events.append((start, 1, mido.Message("note_on", note=pitch, velocity=velocity, channel=voice, time=0)))
            events.append((start + max(1, dur), 0, mido.Message("note_off", note=pitch, velocity=0, channel=voice, time=0)))
        _append_delta_events(track, events)
        midi.tracks.append(track)

    midi.save(path)


def _vertical_slices_for_bar(
    events: Sequence[PitchedEvent],
    *,
    bar_index: int,
    bar_start_tick: int,
    bar_len_ticks: int,
) -> list[VerticalSlice]:
    bar_end = bar_start_tick + bar_len_ticks
    positions = {0}
    for event in events:
        if bar_start_tick <= event.start_tick < bar_end:
            positions.add(event.start_tick - bar_start_tick)

    sorted_positions = sorted(pos for pos in positions if 0 <= pos < bar_len_ticks)
    slices: list[VerticalSlice] = []
    for idx, pos in enumerate(sorted_positions):
        abs_tick = bar_start_tick + pos
        next_pos = sorted_positions[idx + 1] if idx + 1 < len(sorted_positions) else bar_len_ticks
        dur = next_pos - pos
        if dur <= 0:
            continue
        pitches = _active_satb_pitches(events, abs_tick)
        if pitches is None:
            continue
        slices.append(VerticalSlice(bar_index=bar_index, pos_tick=pos, dur_tick=dur, pitches=pitches))
    return slices


def _active_satb_pitches(events: Sequence[PitchedEvent], abs_tick: int) -> tuple[int, int, int, int] | None:
    active: dict[int, PitchedEvent] = {}
    for event in events:
        if event.voice not in range(4):
            continue
        if event.start_tick <= abs_tick < event.end_tick:
            previous = active.get(event.voice)
            if previous is None or event.start_tick >= previous.start_tick:
                active[event.voice] = event
    if any(voice not in active for voice in range(4)):
        return None
    # Voice indices in source data are not guaranteed to be SATB identities.
    # For v2 we enforce stable vertical ordering directly by pitch.
    ordered = sorted(active[voice].pitch for voice in range(4))
    return ordered[0], ordered[1], ordered[2], ordered[3]


def _bar_tokens(slices: Sequence[VerticalSlice], *, key: str, time_sig: str) -> list[str]:
    tokens = ["BAR", "STYLE_CHORALE", f"KEY_{key}", f"TIME_{time_sig.replace('/', '_')}", "TEXTURE_4"]
    for item in slices:
        tokens.append(f"POS_{item.pos_tick}")
        for name, pitch in zip(SATB_NAMES, item.pitches):
            tokens.append(f"{name}_{pitch}")
        tokens.append(f"DUR_{item.dur_tick}")
    return tokens


def _append_delta_events(track: mido.MidiTrack, events: Sequence[tuple[int, int, mido.Message]]) -> None:
    current = 0
    for abs_tick, _order, message in sorted(events, key=lambda item: (item[0], item[1])):
        message.time = max(0, abs_tick - current)
        track.append(message)
        current = abs_tick
    track.append(mido.MetaMessage("end_of_track", time=0))


def _longest_equal_run(values: Sequence[object]) -> int:
    longest = 0
    current = 0
    previous: object = object()
    for value in values:
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        longest = max(longest, current)
    return longest


def _parse_plan(value: object, bar_index: int) -> dict[str, object]:
    if value is None:
        return {"bar_index": bar_index}
    try:
        if isinstance(value, str):
            data = json.loads(value)
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            data = {}
    except (TypeError, json.JSONDecodeError):
        data = {}
    data.setdefault("bar_index", bar_index)
    return data


def _first_token_suffix(tokens: Sequence[str], prefix: str) -> str | None:
    for token in tokens:
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _time_sig_from_tokens(tokens: Sequence[str]) -> str | None:
    value = _first_token_suffix(tokens, "TIME_SIG_")
    if not value:
        return None
    parts = value.split("_")
    if len(parts) != 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
