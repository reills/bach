from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

from src.tokens.intervals import format_signed_interval


@dataclass(frozen=True)
class VoiceEvent:
    voice: int
    duration_ticks: int = 0
    rest_ticks: int = 0
    mel_int: int = 0
    harm_oct: Union[int, None] = None
    harm_class: Union[int, None] = None
    dup_count: Union[int, None] = None
    string: Union[int, None] = None
    fret: Union[int, None] = None

    def __post_init__(self) -> None:
        if self.voice < 0:
            raise ValueError("voice must be >= 0")
        if self.duration_ticks < 0:
            raise ValueError("duration_ticks must be >= 0")
        if self.rest_ticks < 0:
            raise ValueError("rest_ticks must be >= 0")
        if self.dup_count is not None and self.dup_count < 1:
            raise ValueError("dup_count must be >= 1 when set")
        if (self.string is None) ^ (self.fret is None):
            raise ValueError("string and fret must both be set or both be None")

        is_rest = self.rest_ticks > 0
        is_pitched = self.duration_ticks > 0
        if is_rest == is_pitched:
            raise ValueError("voice event must be exactly one of rest or pitched")

        if is_rest:
            if self.dup_count is not None:
                raise ValueError("rest event cannot set dup_count")
            if self.harm_oct is not None or self.harm_class is not None:
                raise ValueError("rest event cannot set harmonic interval fields")
            if self.string is not None or self.fret is not None:
                raise ValueError("rest event cannot set tab fields")
            return

        if (self.harm_oct is None) ^ (self.harm_class is None):
            raise ValueError("harm_oct and harm_class must both be set or both be None")
        if self.harm_class is not None and not (0 <= self.harm_class <= 11):
            raise ValueError("harm_class must be in [0, 11]")

    @property
    def is_rest(self) -> bool:
        return self.rest_ticks > 0


def tokenize_event_text(text: str) -> List[str]:
    raw = text.replace("\n", ",")
    tokens = [tok.strip() for tok in raw.split(",")]
    return [tok for tok in tokens if tok]


def parse_voice_event(tokens: Sequence[str], start_idx: int) -> Tuple[VoiceEvent, int]:
    if start_idx >= len(tokens):
        raise ValueError("start_idx out of range")

    voice_tok = tokens[start_idx]
    if not voice_tok.startswith("VOICE_"):
        raise ValueError(f"expected VOICE token at index {start_idx}, got {voice_tok!r}")

    voice = _parse_prefixed_int(voice_tok, "VOICE_", start_idx)
    next_idx = start_idx + 1
    if next_idx >= len(tokens):
        raise ValueError(f"truncated VOICE event at index {start_idx}")

    first_tok = tokens[next_idx]
    if first_tok.startswith("REST_"):
        rest_ticks = _parse_prefixed_int(first_tok, "REST_", next_idx)
        return VoiceEvent(voice=voice, rest_ticks=rest_ticks), next_idx + 1

    duration_ticks = _parse_prefixed_int(first_tok, "DUR_", next_idx)
    next_idx += 1

    dup_count = None
    if next_idx < len(tokens) and tokens[next_idx].startswith("DUP_"):
        dup_count = _parse_prefixed_int(tokens[next_idx], "DUP_", next_idx)
        next_idx += 1

    if next_idx + 2 >= len(tokens):
        raise ValueError(f"truncated pitched VOICE event at index {start_idx}")

    mel_int = _parse_signed_mel(tokens[next_idx], next_idx)
    harm_oct = _parse_harm_oct(tokens[next_idx + 1], next_idx + 1)
    harm_class = _parse_harm_class(tokens[next_idx + 2], next_idx + 2)
    next_idx += 3

    string = None
    fret = None
    if next_idx < len(tokens) and tokens[next_idx].startswith("STR_"):
        string = _parse_prefixed_int(tokens[next_idx], "STR_", next_idx)
        next_idx += 1
        if next_idx >= len(tokens) or not tokens[next_idx].startswith("FRET_"):
            raise ValueError(f"missing FRET token after STR at index {next_idx}")
        fret = _parse_prefixed_int(tokens[next_idx], "FRET_", next_idx)
        next_idx += 1
    elif next_idx < len(tokens) and tokens[next_idx].startswith("FRET_"):
        raise ValueError(f"FRET token must follow STR token at index {next_idx}")

    return (
        VoiceEvent(
            voice=voice,
            duration_ticks=duration_ticks,
            mel_int=mel_int,
            harm_oct=harm_oct,
            harm_class=harm_class,
            dup_count=dup_count,
            string=string,
            fret=fret,
        ),
        next_idx,
    )


def serialize_voice_event(event: VoiceEvent) -> List[str]:
    tokens = [f"VOICE_{event.voice}"]
    if event.is_rest:
        tokens.append(f"REST_{event.rest_ticks}")
        return tokens

    tokens.append(f"DUR_{event.duration_ticks}")
    if event.dup_count is not None:
        tokens.append(f"DUP_{event.dup_count}")

    tokens.append(f"MEL_INT12_{format_signed_interval(event.mel_int)}")
    if event.harm_oct is None:
        tokens.append("HARM_OCT_NA")
        tokens.append("HARM_CLASS_NA")
    else:
        tokens.append(f"HARM_OCT_{event.harm_oct}")
        tokens.append(f"HARM_CLASS_{event.harm_class}")

    if event.string is not None:
        tokens.append(f"STR_{event.string}")
        tokens.append(f"FRET_{event.fret}")

    return tokens


def parse_event_stream(tokens: Sequence[str]) -> List[Union[str, VoiceEvent]]:
    parsed: List[Union[str, VoiceEvent]] = []
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok.startswith("VOICE_"):
            event, idx = parse_voice_event(tokens, idx)
            parsed.append(event)
            continue
        parsed.append(tok)
        idx += 1
    return parsed


def serialize_event_stream(items: Sequence[Union[str, VoiceEvent]]) -> List[str]:
    tokens: List[str] = []
    for item in items:
        if isinstance(item, VoiceEvent):
            tokens.extend(serialize_voice_event(item))
        else:
            tokens.append(item)
    return tokens


def canonicalize_event_stream(tokens: Sequence[str]) -> List[str]:
    return serialize_event_stream(parse_event_stream(tokens))


def _parse_prefixed_int(token: str, prefix: str, idx: int) -> int:
    if not token.startswith(prefix):
        raise ValueError(f"expected {prefix} token at index {idx}, got {token!r}")
    value = token[len(prefix) :]
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"bad integer token at index {idx}: {token!r}") from exc


def _parse_signed_mel(token: str, idx: int) -> int:
    if not token.startswith("MEL_INT12_"):
        raise ValueError(f"expected MEL_INT12 token at index {idx}, got {token!r}")
    value = token[len("MEL_INT12_") :]
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"bad MEL_INT12 token at index {idx}: {token!r}") from exc


def _parse_harm_oct(token: str, idx: int) -> Union[int, None]:
    if not token.startswith("HARM_OCT_"):
        raise ValueError(f"expected HARM_OCT token at index {idx}, got {token!r}")
    value = token[len("HARM_OCT_") :]
    if value == "NA":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"bad HARM_OCT token at index {idx}: {token!r}") from exc


def _parse_harm_class(token: str, idx: int) -> Union[int, None]:
    if not token.startswith("HARM_CLASS_"):
        raise ValueError(f"expected HARM_CLASS token at index {idx}, got {token!r}")
    value = token[len("HARM_CLASS_") :]
    if value == "NA":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"bad HARM_CLASS token at index {idx}: {token!r}") from exc
    if not (0 <= parsed <= 11):
        raise ValueError(f"HARM_CLASS out of range at index {idx}: {token!r}")
    return parsed
