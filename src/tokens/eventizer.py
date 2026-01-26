from dataclasses import dataclass
from pathlib import Path
import re
import warnings
from statistics import median
from typing import Dict, List, Optional, Tuple, Union

import music21
from music21.musicxml.xmlToM21 import MusicXMLWarning

DEFAULT_MAX_VOICES = 8
DEFAULT_MAX_OCTAVES_PER_PITCH_CLASS = 3
_NORMAL_NOTES_FRACTION_RE = re.compile(
    r"(<actual-notes>)([^<]+)(</actual-notes>\s*<normal-notes>)([^<]+)(</normal-notes>)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class NoteEvent:
    onset_tick: int
    duration_tick: int
    pitch: int


@dataclass(frozen=True)
class ScoreMeta:
    time_sig_token: str
    key_token: Optional[str]
    mapping_note: str


def ql_to_ticks(ql: float, tpq: int) -> int:
    return int(round(ql * tpq))


def format_signed_int(value: int) -> str:
    if value > 0:
        return f"+{value}"
    if value < 0:
        return str(value)
    return "0"


def _normalize_part_name(name: object) -> str:
    if not isinstance(name, str):
        name = "" if name is None else str(name)
    return name.strip().lower().replace(".", "").replace("_", " ")


def _median_pitch(part: music21.stream.Part) -> Optional[float]:
    pitches: List[int] = []
    for el in part.recurse().notes:
        if el.isChord:
            pitches.append(max(p.midi for p in el.pitches))
        else:
            pitches.append(el.pitch.midi)
    if not pitches:
        return None
    return median(pitches)


def _map_parts_by_name(
    parts: List[music21.stream.Part],
) -> Optional[Dict[int, music21.stream.Part]]:
    mapping: Dict[int, music21.stream.Part] = {}
    used = set()

    for part in parts:
        name = _normalize_part_name(part.partName or part.id or "")
        if not name:
            continue
        if "soprano" in name or name in {"s", "sop"}:
            mapping[3] = part
            used.add(part)
        elif "alto" in name or name in {"a", "alt"}:
            mapping[2] = part
            used.add(part)
        elif "tenor" in name or name in {"t", "ten"}:
            mapping[1] = part
            used.add(part)
        elif "bass" in name or name in {"b", "bas"}:
            mapping[0] = part
            used.add(part)

    if len(mapping) == 4:
        return mapping
    return None


def _group_parts_by_pitch(
    parts: List[music21.stream.Part],
    max_voices: int,
) -> List[List[music21.stream.Part]]:
    scored: List[Tuple[float, music21.stream.Part]] = []
    for part in parts:
        med = _median_pitch(part)
        if med is None:
            med = 0.0
        scored.append((med, part))
    scored.sort(key=lambda item: item[0])

    total = len(scored)
    groups: List[List[music21.stream.Part]] = []
    for i in range(max_voices):
        start = i * total // max_voices
        end = (i + 1) * total // max_voices
        groups.append([part for _, part in scored[start:end]])
    return groups


def map_parts_to_voices(
    score: music21.stream.Score,
    *,
    max_voices: int = DEFAULT_MAX_VOICES,
    voice_mode: str = "auto",
) -> Tuple[Dict[int, Union[music21.stream.Part, List[music21.stream.Part]]], str]:
    parts = list(score.parts)
    if max_voices < 1:
        raise ValueError("max_voices must be >= 1")
    if len(parts) > max_voices:
        grouped = _group_parts_by_pitch(parts, max_voices)
        mapping = {i: group for i, group in enumerate(grouped)}
        note = f"collapsed {len(parts)} parts into {max_voices} voices by pitch bins"
        return mapping, note

    if voice_mode not in {"auto", "parts", "pitch", "events"}:
        raise ValueError(f"unsupported voice_mode: {voice_mode}")

    if voice_mode == "parts":
        mapping = {i: part for i, part in enumerate(parts)}
        note = f"mapped {len(parts)} parts by order"
        return mapping, note

    if voice_mode == "pitch":
        mapping = _map_parts_by_pitch(parts)
        note = f"mapped {len(parts)} parts by median pitch (low->high)"
        return mapping, note

    if len(parts) == 4 and max_voices >= 4:
        mapping = _map_parts_by_name(parts)
        if mapping is not None:
            note = "mapped by part name (S/A/T/B keywords)"
            return mapping, note

    mapping = _map_parts_by_pitch(parts)
    note = f"mapped {len(parts)} parts by median pitch (low->high)"
    return mapping, note


def _map_parts_by_pitch(parts: List[music21.stream.Part]) -> Dict[int, music21.stream.Part]:
    scored: List[Tuple[float, music21.stream.Part]] = []
    for part in parts:
        med = _median_pitch(part)
        if med is None:
            med = 0.0
        scored.append((med, part))
    scored.sort(key=lambda item: item[0])
    
    mapping = {}
    for i, (med, part) in enumerate(scored):
        mapping[i] = part
    return mapping


def detect_time_signature(score: music21.stream.Score) -> Tuple[int, int]:
    ts = score.recurse().getElementsByClass(music21.meter.TimeSignature)
    if ts:
        return ts[0].numerator, ts[0].denominator
    return 4, 4


def detect_pickup_ticks(
    score: music21.stream.Score, tpq: int, bar_len_ticks: int
) -> int:
    bar_len_ql = bar_len_ticks / tpq
    pickup_ql = 0.0

    if hasattr(score, "anacrusis"):
        try:
            pickup_ql = score.anacrusis() or 0.0
        except Exception:
            pickup_ql = 0.0

    if pickup_ql <= 0.0:
        try:
            parts = list(score.parts)
            if parts:
                measures = list(parts[0].getElementsByClass(music21.stream.Measure))
                if len(measures) >= 2:
                    first_off = measures[0].offset
                    second_off = measures[1].offset
                    candidate = second_off - first_off
                    if 0.0 < candidate < (bar_len_ql - 1e-6):
                        pickup_ql = candidate
        except Exception:
            pickup_ql = 0.0

    if pickup_ql <= 0.0 or pickup_ql >= (bar_len_ql - 1e-6):
        return 0

    return ql_to_ticks(pickup_ql, tpq)


def detect_key_token(score: music21.stream.Score, override: Optional[str]) -> Optional[str]:
    if override:
        return f"KEY_{override}"

    keys = score.recurse().getElementsByClass(music21.key.Key)
    if keys:
        return _format_key_token(keys[0])

    key_sigs = score.recurse().getElementsByClass(music21.key.KeySignature)
    if key_sigs:
        return _format_key_token(key_sigs[0].asKey())

    return None


def _format_key_token(k: music21.key.Key) -> str:
    tonic = k.tonic.name.replace("-", "b")
    mode = k.mode
    if mode == "minor":
        return f"KEY_{tonic}m"
    return f"KEY_{tonic}"


def _compute_bar_length_ticks(time_sig: Tuple[int, int], tpq: int) -> int:
    numerator, denominator = time_sig
    bar_ql = numerator * (4.0 / denominator)
    return ql_to_ticks(bar_ql, tpq)


def _extract_events(part: music21.stream.Part, tpq: int, shift_ticks: int = 0) -> List[NoteEvent]:
    if hasattr(part, "stripTies"):
        part = part.stripTies(inPlace=False)
    
    # Use flatten() to ensure offsets are absolute relative to the start of the piece
    # (Fixes issue where .recurse() returned relative offsets per measure)
    flat_part = part.flatten()

    events: List[NoteEvent] = []
    for el in flat_part.notesAndRests:
        if el.isRest:
            continue
        if el.isChord:
            pitch = max(p.midi for p in el.pitches)
        else:
            pitch = el.pitch.midi
        onset = ql_to_ticks(el.offset, tpq) + shift_ticks
        dur = ql_to_ticks(el.duration.quarterLength, tpq)
        if dur <= 0:
            continue
        events.append(NoteEvent(onset_tick=onset, duration_tick=dur, pitch=pitch))
    events.sort(key=lambda ev: ev.onset_tick)
    return events


def _collapse_events(events: List[NoteEvent]) -> List[NoteEvent]:
    if not events:
        return []
    by_onset: Dict[int, NoteEvent] = {}
    for ev in events:
        existing = by_onset.get(ev.onset_tick)
        if existing is None:
            by_onset[ev.onset_tick] = ev
            continue
        if ev.pitch > existing.pitch:
            by_onset[ev.onset_tick] = ev
        elif ev.pitch == existing.pitch and ev.duration_tick > existing.duration_tick:
            by_onset[ev.onset_tick] = NoteEvent(
                onset_tick=ev.onset_tick,
                duration_tick=ev.duration_tick,
                pitch=ev.pitch,
            )
    return sorted(by_onset.values(), key=lambda ev: ev.onset_tick)


def _sanitize_musicxml_text(xml_text: str) -> str:
    def repl(match: re.Match) -> str:
        actual_raw = match.group(2).strip()
        normal_raw = match.group(4).strip()
        if "/" not in normal_raw:
            return match.group(0)
        parts = normal_raw.split("/", 1)
        try:
            num = int(parts[0].strip())
            den = int(parts[1].strip())
        except ValueError:
            return match.group(0)
        try:
            actual = int(actual_raw)
        except ValueError:
            new_actual = actual_raw
        else:
            new_actual = str(actual * den)
        return f"{match.group(1)}{new_actual}{match.group(3)}{num}{match.group(5)}"

    return _NORMAL_NOTES_FRACTION_RE.sub(repl, xml_text)


def _needs_rebar(warnings_list: List[warnings.WarningMessage]) -> bool:
    for item in warnings_list:
        message = str(item.message).lower()
        if "overfull" in message or "underfull" in message:
            return True
    return False


def _select_spread_indices(count: int, keep: int) -> List[int]:
    if keep >= count:
        return list(range(count))
    if keep <= 0:
        return []
    if keep == 1:
        return [count // 2]
    step = (count - 1) / (keep - 1)
    idxs = [int(round(i * step)) for i in range(keep)]
    idxs = sorted(set(idxs))
    if len(idxs) < keep:
        for idx in range(count):
            if idx not in idxs:
                idxs.append(idx)
                if len(idxs) == keep:
                    break
        idxs.sort()
    return idxs


def _collapse_unison_octaves(
    events: List[NoteEvent],
    *,
    max_octaves_per_pitch_class: int,
) -> Dict[int, List[NoteEvent]]:
    by_tick: Dict[int, List[NoteEvent]] = {}
    for ev in events:
        by_tick.setdefault(ev.onset_tick, []).append(ev)

    collapsed: Dict[int, List[NoteEvent]] = {}
    for onset, evs in by_tick.items():
        by_pc: Dict[int, Dict[int, int]] = {}
        for ev in evs:
            pc = ev.pitch % 12
            pitch_map = by_pc.setdefault(pc, {})
            pitch_map[ev.pitch] = max(pitch_map.get(ev.pitch, 0), ev.duration_tick)

        kept: List[NoteEvent] = []
        for pitch_map in by_pc.values():
            pitches = sorted(pitch_map.keys())
            if len(pitches) > max_octaves_per_pitch_class:
                idxs = _select_spread_indices(len(pitches), max_octaves_per_pitch_class)
                pitches = [pitches[i] for i in idxs]
            for pitch in pitches:
                kept.append(
                    NoteEvent(
                        onset_tick=onset,
                        duration_tick=pitch_map[pitch],
                        pitch=pitch,
                    )
                )
        kept.sort(key=lambda ev: ev.pitch)
        collapsed[onset] = kept
    return collapsed


def _assign_events_by_continuity(
    events_by_tick: Dict[int, List[NoteEvent]],
    *,
    max_voices: int,
) -> Dict[int, List[NoteEvent]]:
    events_by_voice: Dict[int, List[NoteEvent]] = {v: [] for v in range(max_voices)}
    prev_pitch: List[Optional[int]] = [None] * max_voices
    active_until: List[int] = [0] * max_voices

    for onset in sorted(events_by_tick.keys()):
        events = sorted(events_by_tick[onset], key=lambda ev: ev.pitch)

        active_count = sum(1 for v in range(max_voices) if active_until[v] > onset)
        free_voices = [v for v in range(max_voices) if active_until[v] <= onset]
        max_new = max_voices - active_count
        if max_new <= 0 or not free_voices:
            continue
        if len(events) > max_new:
            idxs = _select_spread_indices(len(events), max_new)
            events = [events[i] for i in idxs]

        available = set(free_voices)
        for ev in events:
            best_voice = None
            best_cost = None
            for v in sorted(available):
                if prev_pitch[v] is None:
                    cost = 1_000_000
                else:
                    cost = abs(ev.pitch - prev_pitch[v])
                if best_cost is None or cost < best_cost or (cost == best_cost and v < best_voice):
                    best_cost = cost
                    best_voice = v
            if best_voice is None:
                continue
            events_by_voice[best_voice].append(ev)
            prev_pitch[best_voice] = ev.pitch
            active_until[best_voice] = onset + ev.duration_tick
            available.remove(best_voice)

    last_used = max((v for v, evs in events_by_voice.items() if evs), default=-1)
    if last_used < 0:
        return {}
    return {v: events_by_voice[v] for v in range(last_used + 1)}


def eventize_musicxml(
    path: str,
    *,
    tpq: int = 24,
    reentry_ticks: int = 48,
    mel_range: int = 24,
    anchor_large_leaps: bool = False,
    align_pickup: bool = True,
    key_override: Optional[str] = None,
    max_voices: int = DEFAULT_MAX_VOICES,
    voice_mode: str = "auto",
    max_octaves_per_pitch_class: int = DEFAULT_MAX_OCTAVES_PER_PITCH_CLASS,
) -> Tuple[List[str], ScoreMeta]:
    warnings_list: List[warnings.WarningMessage]
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", MusicXMLWarning)
            score = music21.converter.parse(path)
            warnings_list = list(w)
    except Exception:
        xml_path = Path(path)
        try:
            raw_text = xml_path.read_text(encoding="utf-8")
        except Exception:
            raw_text = xml_path.read_text(errors="ignore")
        if "<normal-notes>" not in raw_text or "/" not in raw_text:
            raise
        sanitized = _sanitize_musicxml_text(raw_text)
        if sanitized == raw_text:
            raise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", MusicXMLWarning)
            score = music21.converter.parseData(sanitized, format="musicxml")
            warnings_list = list(w)
    if _needs_rebar(warnings_list):
        try:
            score = score.makeMeasures(inPlace=False)
            score = score.makeNotation(inPlace=False)
        except Exception:
            pass
    if voice_mode not in {"auto", "parts", "pitch", "events"}:
        raise ValueError(f"unsupported voice_mode: {voice_mode}")
    parts = list(score.parts)

    time_sig = detect_time_signature(score)
    time_sig_token = f"TIME_SIG_{time_sig[0]}_{time_sig[1]}"
    key_token = detect_key_token(score, key_override)
    bar_len_ticks = _compute_bar_length_ticks(time_sig, tpq)

    pickup_ticks = (
        detect_pickup_ticks(score, tpq, bar_len_ticks) if align_pickup else 0
    )
    shift_ticks = bar_len_ticks - pickup_ticks if pickup_ticks > 0 else 0

    if voice_mode in {"auto", "events"}:
        merged_events: List[NoteEvent] = []
        for part in parts:
            merged_events.extend(_extract_events(part, tpq, shift_ticks=shift_ticks))
        events_by_tick = _collapse_unison_octaves(
            merged_events,
            max_octaves_per_pitch_class=max_octaves_per_pitch_class,
        )
        events_by_voice = _assign_events_by_continuity(
            events_by_tick,
            max_voices=max_voices,
        )
        mapping_note = (
            f"collapsed unison/octaves (<= {max_octaves_per_pitch_class} per pitch class) "
            f"and assigned {len(events_by_voice)} voices by continuity"
        )
    else:
        parts_by_voice, mapping_note = map_parts_to_voices(
            score, max_voices=max_voices, voice_mode=voice_mode
        )
        events_by_voice: Dict[int, List[NoteEvent]] = {}
        for v, part in parts_by_voice.items():
            if isinstance(part, list):
                merged: List[NoteEvent] = []
                for sub_part in part:
                    merged.extend(_extract_events(sub_part, tpq, shift_ticks=shift_ticks))
                events_by_voice[v] = _collapse_events(merged)
            else:
                events_by_voice[v] = _extract_events(part, tpq, shift_ticks=shift_ticks)

    events_by_tick: Dict[int, Dict[int, NoteEvent]] = {}
    positions = set()
    max_end = 0
    for v, events in events_by_voice.items():
        for ev in events:
            positions.add(ev.onset_tick)
            max_end = max(max_end, ev.onset_tick + ev.duration_tick)
            if ev.onset_tick not in events_by_tick:
                events_by_tick[ev.onset_tick] = {}
            events_by_tick[ev.onset_tick][v] = ev

    if bar_len_ticks <= 0:
        raise ValueError("bar length in ticks must be positive")

    num_bars = max(1, int((max_end + bar_len_ticks - 1) // bar_len_ticks))

    tokens: List[str] = []
    voice_count = max(events_by_voice.keys(), default=-1) + 1
    prev_pitch: List[Optional[int]] = [None] * voice_count
    active_until: List[int] = [0] * voice_count
    last_end: List[Optional[int]] = [None] * voice_count

    for bar_idx in range(num_bars):
        bar_start = bar_idx * bar_len_ticks
        bar_end = bar_start + bar_len_ticks

        tokens.append("BAR")
        tokens.append(time_sig_token)
        if key_token:
            tokens.append(key_token)

        bar_anchor_emitted = [False] * voice_count
        events_at_bar = events_by_tick.get(bar_start, {})
        for v in range(voice_count):
            onset_ev = events_at_bar.get(v)
            if onset_ev is not None:
                anchor_pitch = onset_ev.pitch
            elif active_until[v] > bar_start and prev_pitch[v] is not None:
                anchor_pitch = prev_pitch[v]
            else:
                continue
            tokens.append(f"ABS_VOICE_{v}_{anchor_pitch}")
            prev_pitch[v] = anchor_pitch
            bar_anchor_emitted[v] = True

        bar_positions = sorted(t for t in positions if bar_start <= t < bar_end)
        for t in bar_positions:
            tokens.append(f"POS_{t - bar_start}")
            events_at_t = events_by_tick.get(t, {})

            active_pitches = []
            for v in range(voice_count):
                if active_until[v] > t and prev_pitch[v] is not None:
                    active_pitches.append(prev_pitch[v])
            for ev in events_at_t.values():
                if ev.pitch is not None:
                    active_pitches.append(ev.pitch)
            ref_pitch = min(active_pitches) if active_pitches else None

            for v in range(voice_count):
                ev = events_at_t.get(v)
                if ev is None:
                    continue

                anchor_needed = False
                if prev_pitch[v] is None:
                    anchor_needed = True
                elif last_end[v] is not None and (t - last_end[v]) >= reentry_ticks:
                    anchor_needed = True
                elif anchor_large_leaps and abs(ev.pitch - prev_pitch[v]) > mel_range:
                    anchor_needed = True

                if anchor_needed:
                    if not (bar_anchor_emitted[v] and t == bar_start):
                        tokens.append(f"ABS_VOICE_{v}_{ev.pitch}")
                    prev_pitch[v] = ev.pitch

                tokens.append(f"VOICE_{v}")
                tokens.append(f"DUR_{ev.duration_tick}")
                base_pitch = prev_pitch[v] if prev_pitch[v] is not None else ev.pitch
                mel_int = ev.pitch - base_pitch
                tokens.append(f"MEL_INT12_{format_signed_int(mel_int)}")

                if ref_pitch is None:
                    tokens.append("HARM_OCT_NA")
                    tokens.append("HARM_CLASS_NA")
                else:
                    diff = ev.pitch - ref_pitch
                    octv, klass = divmod(diff, 12)
                    tokens.append(f"HARM_OCT_{octv}")
                    tokens.append(f"HARM_CLASS_{klass}")

                prev_pitch[v] = ev.pitch
                active_until[v] = t + ev.duration_tick
                last_end[v] = t + ev.duration_tick

    meta = ScoreMeta(
        time_sig_token=time_sig_token,
        key_token=key_token,
        mapping_note=mapping_note,
    )
    return tokens, meta
