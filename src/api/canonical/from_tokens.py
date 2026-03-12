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


def tokens_to_canonical_score(
    tokens: list[str],
    tpq: int = 24,
    part_info: PartInfo | None = None,
) -> CanonicalScore:
    bars = split_bars(tokens)
    if not bars:
        raise ValueError("token stream must contain at least one BAR")

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
                bar_start_tick=bar_start_tick,
                part_info=part_info,
                prev_pitch=prev_pitch,
            )
        )
        bar_start_tick = measure.end_tick

    header = ScoreHeader(
        tpq=tpq,
        key_sig_map=key_sig_map,
        time_sig_map=time_sig_map,
    )
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
    bar_start_tick: int,
    part_info: PartInfo,
    prev_pitch: dict[int, int | None],
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
            if current_pos_tick is None:
                raise ValueError(f"VOICE token before POS in bar starting at tick {bar_start_tick}")

            voice_event, next_idx = parse_voice_event(bar_tokens, idx)
            event_ordinal_key = (voice_event.voice, current_pos_tick)
            ordinal = event_counts.get(event_ordinal_key, 0)
            event_counts[event_ordinal_key] = ordinal + 1

            if voice_event.is_rest:
                events.append(
                    Event(
                        id=_stable_event_id(part_info.id, voice_event.voice, current_pos_tick, ordinal),
                        start_tick=current_pos_tick,
                        dur_tick=voice_event.rest_ticks,
                        pitch_midi=None,
                        voice_id=voice_event.voice,
                    )
                )
                idx = next_idx
                continue

            previous_pitch = prev_pitch.get(voice_event.voice)
            if previous_pitch is None:
                raise ValueError(f"missing anchor before VOICE_{voice_event.voice} at tick {current_pos_tick}")

            pitch_midi = previous_pitch + voice_event.mel_int
            prev_pitch[voice_event.voice] = pitch_midi
            events.append(
                Event(
                    id=_stable_event_id(part_info.id, voice_event.voice, current_pos_tick, ordinal),
                    start_tick=current_pos_tick,
                    dur_tick=voice_event.duration_ticks,
                    pitch_midi=pitch_midi,
                    voice_id=voice_event.voice,
                    fingering=_to_fingering(voice_event.string, voice_event.fret, len(part_info.tuning)),
                )
            )
            idx = next_idx
            continue

        idx += 1

    return events


def _to_fingering(string_number: int | None, fret: int | None, string_count: int) -> GuitarFingering | None:
    if string_number is None or fret is None:
        return None
    if string_count <= 0:
        raise ValueError("part tuning must define at least one string when tab data is present")
    if not 1 <= string_number <= string_count:
        raise ValueError(f"string number {string_number} out of range for tuning with {string_count} strings")
    return GuitarFingering(string_index=string_count - string_number, fret=fret)


def _stable_measure_id(index: int, start_tick: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"bach-gen:measure:{index}:{start_tick}"))


def _stable_event_id(part_id: str, voice_id: int, start_tick: int, ordinal: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"bach-gen:event:{part_id}:{voice_id}:{start_tick}:{ordinal}"))
