from dataclasses import dataclass, field, replace
from uuid import NAMESPACE_URL, uuid5

from src.api.canonical.types import (
    CanonicalScore,
    Event,
    GuitarFingering,
    Measure,
    Part,
    PartInfo,
    ScoreHeader,
)
from src.tokens.roundtrip import (
    parse_abs_voice_token,
    parse_last_token_int,
    parse_time_sig_token,
    parse_token_int,
)
from src.tokens.tokenizer import parse_voice_event

DEFAULT_GUITAR_TUNING = [40, 45, 50, 55, 59, 64]


@dataclass(frozen=True)
class ParseIssue:
    kind: str
    bar_index: int
    token_index: int
    token: str
    message: str


@dataclass
class ParseDiagnostics:
    skipped_invalid_voice_events: int = 0
    skipped_voice_before_pos: int = 0
    skipped_missing_anchor: int = 0
    parsed_pitched_events: int = 0
    parsed_rest_events: int = 0
    issues: list[ParseIssue] = field(default_factory=list)
    max_issues: int = 8

    def add_issue(
        self,
        *,
        kind: str,
        bar_index: int,
        token_index: int,
        token: str,
        message: str,
    ) -> None:
        if len(self.issues) >= self.max_issues:
            return
        self.issues.append(
            ParseIssue(
                kind=kind,
                bar_index=bar_index,
                token_index=token_index,
                token=token,
                message=message,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "skipped_invalid_voice_events": self.skipped_invalid_voice_events,
            "skipped_voice_before_pos": self.skipped_voice_before_pos,
            "skipped_missing_anchor": self.skipped_missing_anchor,
            "parsed_pitched_events": self.parsed_pitched_events,
            "parsed_rest_events": self.parsed_rest_events,
            "issues": [
                {
                    "kind": issue.kind,
                    "bar_index": issue.bar_index,
                    "token_index": issue.token_index,
                    "token": issue.token,
                    "message": issue.message,
                }
                for issue in self.issues
            ],
        }


def tokens_to_canonical_score(
    tokens: list[str],
    tpq: int = 24,
    part_info: PartInfo | None = None,
    *,
    ignore_invalid_events: bool = False,
    diagnostics: ParseDiagnostics | None = None,
) -> CanonicalScore:
    bars = split_bars(tokens)
    if not bars:
        raise ValueError("token stream must contain at least one BAR")
    voice_id_map = _canonical_voice_id_map(tokens)

    if part_info is None:
        part_info = PartInfo(
            id="part-0",
            instrument="classical_guitar",
            tuning=DEFAULT_GUITAR_TUNING,
            midi_program=24,
        )

    measures: list[Measure] = []
    events: list[Event] = []
    key_sig_map: dict[int, str] = {}
    time_sig_map: dict[int, str] = {}

    prev_pitch: dict[int, int | None] = {}
    current_time_sig: str | None = None
    current_key_sig: str | None = None
    bar_start_tick = 0

    for bar_index, bar_tokens in enumerate(bars):
        current_time_sig = _bar_time_sig(bar_tokens, current_time_sig)
        if current_time_sig is None:
            raise ValueError(f"missing TIME_SIG token for bar {bar_index}")

        current_key_sig = _bar_key_sig(bar_tokens, current_key_sig)
        if bar_start_tick not in time_sig_map or time_sig_map[bar_start_tick] != current_time_sig:
            time_sig_map[bar_start_tick] = current_time_sig
        if current_key_sig is not None and key_sig_map.get(bar_start_tick) != current_key_sig:
            key_sig_map[bar_start_tick] = current_key_sig

        measure = Measure(
            id=_stable_measure_id(bar_index, bar_start_tick),
            index=bar_index,
            start_tick=bar_start_tick,
            length_ticks=_bar_length_ticks(current_time_sig, tpq),
        )
        measures.append(measure)
        events.extend(
            _events_from_bar(
                bar_tokens=bar_tokens,
                bar_index=bar_index,
                bar_start_tick=bar_start_tick,
                part_info=part_info,
                prev_pitch=prev_pitch,
                voice_id_map=voice_id_map,
                ignore_invalid_events=ignore_invalid_events,
                diagnostics=diagnostics,
            )
        )
        bar_start_tick = measure.end_tick

    header = ScoreHeader(
        tpq=tpq,
        key_sig_map=key_sig_map,
        time_sig_map=time_sig_map,
    )
    events.sort(key=lambda e: e.start_tick)
    events = _compact_event_voice_ids(events)
    if ignore_invalid_events:
        events = _ensure_unique_event_ids(events, part_info.id)

    part = Part(info=part_info, events=events)
    return CanonicalScore(header=header, measures=measures, parts=[part])


def split_bars(tokens: list[str]) -> list[list[str]]:
    bars: list[list[str]] = []
    current_bar: list[str] = []
    for token in tokens:
        if token == "BAR":
            if current_bar:
                bars.append(current_bar)
            current_bar = ["BAR"]
            continue
        if current_bar:
            current_bar.append(token)
    if current_bar:
        bars.append(current_bar)
    return bars


def _bar_time_sig(bar_tokens: list[str], fallback: str | None) -> str | None:
    for token in bar_tokens:
        if token.startswith("TIME_SIG_"):
            numerator, denominator = parse_time_sig_token(token)
            return f"{numerator}/{denominator}"
    return fallback


def _bar_key_sig(bar_tokens: list[str], fallback: str | None) -> str | None:
    for token in bar_tokens:
        if token.startswith("KEY_"):
            return token[len("KEY_") :]
    return fallback


def _bar_length_ticks(time_sig: str, tpq: int) -> int:
    numerator, denominator = time_sig.split("/", 1)
    return int(round(int(numerator) * (4.0 / int(denominator)) * tpq))


def _events_from_bar(
    bar_tokens: list[str],
    bar_index: int,
    bar_start_tick: int,
    part_info: PartInfo,
    prev_pitch: dict[int, int | None],
    voice_id_map: dict[int, int],
    *,
    ignore_invalid_events: bool,
    diagnostics: ParseDiagnostics | None,
) -> list[Event]:
    events: list[Event] = []
    event_counts: dict[tuple[int, int], int] = {}
    current_pos_tick: int | None = None

    idx = 0
    while idx < len(bar_tokens):
        token = bar_tokens[idx]

        if token == "BAR" or token.startswith("TIME_SIG_") or token.startswith("KEY_"):
            idx += 1
            continue
        if token.startswith("POS_"):
            current_pos_tick = bar_start_tick + parse_token_int(token)
            idx += 1
            continue
        if token.startswith("ABS_VOICE_"):
            voice, pitch = parse_abs_voice_token(token)
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
        if token.startswith("ABS_LOW_") or token.startswith("ABS_HIGH_") or token.startswith("REF_VOICE_"):
            idx += 1
            continue
        if token.startswith("VOICE_"):
            try:
                voice_event, next_idx = parse_voice_event(bar_tokens, idx)
            except ValueError as exc:
                if ignore_invalid_events:
                    if diagnostics is not None:
                        diagnostics.skipped_invalid_voice_events += 1
                        diagnostics.add_issue(
                            kind="invalid_voice_event",
                            bar_index=bar_index,
                            token_index=idx,
                            token=token,
                            message=str(exc),
                        )
                    idx = _skip_invalid_voice_event(bar_tokens, idx)
                    continue
                raise

            if current_pos_tick is None:
                if ignore_invalid_events:
                    if diagnostics is not None:
                        diagnostics.skipped_voice_before_pos += 1
                        diagnostics.add_issue(
                            kind="voice_before_pos",
                            bar_index=bar_index,
                            token_index=idx,
                            token=token,
                            message=f"VOICE token before POS in bar starting at tick {bar_start_tick}",
                        )
                    idx = next_idx
                    continue
                raise ValueError(f"VOICE token before POS in bar starting at tick {bar_start_tick}")

            canonical_voice_id = voice_id_map[voice_event.voice]
            event_ordinal_key = (canonical_voice_id, current_pos_tick)
            ordinal = event_counts.get(event_ordinal_key, 0)
            event_counts[event_ordinal_key] = ordinal + 1

            if voice_event.is_rest:
                events.append(
                    Event(
                        id=_stable_event_id(part_info.id, canonical_voice_id, current_pos_tick, ordinal),
                        start_tick=current_pos_tick,
                        dur_tick=voice_event.rest_ticks,
                        pitch_midi=None,
                        voice_id=canonical_voice_id,
                    )
                )
                if diagnostics is not None:
                    diagnostics.parsed_rest_events += 1
                idx = next_idx
                continue

            previous_pitch = prev_pitch.get(voice_event.voice)
            if previous_pitch is None:
                if ignore_invalid_events:
                    if diagnostics is not None:
                        diagnostics.skipped_missing_anchor += 1
                        diagnostics.add_issue(
                            kind="missing_anchor",
                            bar_index=bar_index,
                            token_index=idx,
                            token=token,
                            message=f"missing anchor before VOICE_{voice_event.voice} at tick {current_pos_tick}",
                        )
                    idx = next_idx
                    continue
                raise ValueError(f"missing anchor before VOICE_{voice_event.voice} at tick {current_pos_tick}")

            pitch_midi = previous_pitch + voice_event.mel_int
            prev_pitch[voice_event.voice] = pitch_midi
            events.append(
                Event(
                    id=_stable_event_id(part_info.id, canonical_voice_id, current_pos_tick, ordinal),
                    start_tick=current_pos_tick,
                    dur_tick=voice_event.duration_ticks,
                    pitch_midi=pitch_midi,
                    voice_id=canonical_voice_id,
                    fingering=_to_fingering(voice_event.string, voice_event.fret, len(part_info.tuning)),
                )
            )
            if diagnostics is not None:
                diagnostics.parsed_pitched_events += 1
            idx = next_idx
            continue

        idx += 1

    return events


def _skip_invalid_voice_event(bar_tokens: list[str], idx: int) -> int:
    next_idx = idx + 1
    while next_idx < len(bar_tokens):
        token = bar_tokens[next_idx]
        if token == "BAR" or token.startswith("POS_") or token.startswith("VOICE_"):
            break
        next_idx += 1
    return next_idx


def _ensure_unique_event_ids(events: list[Event], part_id: str) -> list[Event]:
    seen_ids: set[str] = set()
    deduped: list[Event] = []
    for idx, event in enumerate(events):
        event_id = event.id
        if event_id in seen_ids:
            event_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"bach-gen:event-dedupe:{part_id}:{idx}:{event.voice_id}:{event.start_tick}",
                )
            )
            event = replace(event, id=event_id)
        seen_ids.add(event_id)
        deduped.append(event)
    return deduped


def _compact_event_voice_ids(events: list[Event]) -> list[Event]:
    used_voice_ids = sorted({event.voice_id for event in events})
    if used_voice_ids == list(range(len(used_voice_ids))):
        return events

    remapped_voice_ids = {
        original_voice_id: compact_voice_id
        for compact_voice_id, original_voice_id in enumerate(used_voice_ids)
    }
    return [
        replace(event, voice_id=remapped_voice_ids[event.voice_id])
        for event in events
    ]


def _to_fingering(string_number: int | None, fret: int | None, string_count: int) -> GuitarFingering | None:
    if string_number is None or fret is None:
        return None
    if string_count <= 0:
        raise ValueError("part tuning must define at least one string when tab data is present")
    if not 1 <= string_number <= string_count:
        raise ValueError(f"string number {string_number} out of range for tuning with {string_count} strings")
    return GuitarFingering(string_index=string_count - string_number, fret=fret)


def _canonical_voice_id_map(tokens: list[str]) -> dict[int, int]:
    raw_voice_ids = sorted(
        {
            parse_token_int(token)
            for token in tokens
            if token.startswith("VOICE_")
        }
    )
    return {raw_voice_id: canonical_voice_id for canonical_voice_id, raw_voice_id in enumerate(raw_voice_ids)}


def _stable_measure_id(index: int, start_tick: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"bach-gen:measure:{index}:{start_tick}"))


def _stable_event_id(part_id: str, voice_id: int, start_tick: int, ordinal: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"bach-gen:event:{part_id}:{voice_id}:{start_tick}:{ordinal}"))
