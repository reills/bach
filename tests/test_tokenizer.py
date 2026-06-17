import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.tokens.tokenizer import (
    VoiceEvent,
    canonicalize_event_stream,
    parse_event_stream,
    parse_voice_event,
    serialize_event_stream,
    serialize_voice_event,
    tokenize_event_text,
)


def test_parse_pitched_voice_event_with_dup_and_tab():
    tokens = [
        "VOICE_2",
        "DUR_12",
        "DUP_3",
        "MEL_INT12_+7",
        "HARM_OCT_1",
        "HARM_CLASS_7",
        "STR_4",
        "FRET_2",
        "POS_6",
    ]

    event, next_idx = parse_voice_event(tokens, 0)
    assert next_idx == 8
    assert event.voice == 2
    assert event.duration_ticks == 12
    assert event.dup_count == 3
    assert event.mel_int == 7
    assert event.harm_oct == 1
    assert event.harm_class == 7
    assert event.string == 4
    assert event.fret == 2


def test_parse_rest_voice_event():
    tokens = ["VOICE_1", "REST_18", "BAR"]
    event, next_idx = parse_voice_event(tokens, 0)
    assert next_idx == 2
    assert event.voice == 1
    assert event.is_rest
    assert event.rest_ticks == 18


def test_serialize_voice_event_uses_canonical_order():
    event = VoiceEvent(
        voice=3,
        duration_ticks=24,
        dup_count=2,
        mel_int=11,
        harm_oct=0,
        harm_class=11,
    )
    assert serialize_voice_event(event) == [
        "VOICE_3",
        "DUR_24",
        "DUP_2",
        "MEL_INT12_+11",
        "HARM_OCT_0",
        "HARM_CLASS_11",
    ]


def test_canonicalize_event_stream_normalizes_mel_sign():
    tokens = [
        "BAR",
        "POS_0",
        "VOICE_0",
        "DUR_6",
        "MEL_INT12_2",
        "HARM_OCT_0",
        "HARM_CLASS_2",
        "VOICE_1",
        "REST_3",
    ]
    assert canonicalize_event_stream(tokens) == [
        "BAR",
        "POS_0",
        "VOICE_0",
        "DUR_6",
        "MEL_INT12_+2",
        "HARM_OCT_0",
        "HARM_CLASS_2",
        "VOICE_1",
        "REST_3",
    ]


def test_parse_event_stream_roundtrip_and_text_tokenization():
    text = "BAR, POS_0, VOICE_0, DUR_4, MEL_INT12_0, HARM_OCT_NA, HARM_CLASS_NA"
    tokens = tokenize_event_text(text)
    parsed = parse_event_stream(tokens)
    assert serialize_event_stream(parsed) == tokens


def test_parse_rejects_mixed_harm_na_state():
    tokens = ["VOICE_0", "DUR_8", "MEL_INT12_0", "HARM_OCT_NA", "HARM_CLASS_2"]
    with pytest.raises(ValueError, match="harm_oct and harm_class must both be set or both be None"):
        parse_voice_event(tokens, 0)
