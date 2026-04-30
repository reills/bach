import json
from pathlib import Path

from music21.midi import MidiFile

from src.chorale_v2 import (
    build_v2_bars_from_v1_rows,
    parse_v2_slices,
    render_v2_tokens_to_midi,
    v2_repetition_metrics,
)


def _v1_rows():
    return [
        {
            "piece_id": "chorale",
            "source_path": "chorale.xml",
            "source_sha256": "abc",
            "bar_index": 0,
            "bar_len_ticks": 96,
            "plan_json": json.dumps(
                {
                    "bar_index": 0,
                    "time_sig": "4/4",
                    "key": "C",
                    "density_bucket": "DENSITY_HIGH",
                    "pitch_range": 24,
                    "polyphony_max": 4,
                }
            ),
            "tokens": (
                "BAR TIME_SIG_4_4 KEY_C "
                "ABS_VOICE_0_48 ABS_VOICE_1_55 ABS_VOICE_2_64 ABS_VOICE_3_72 "
                "POS_0 VOICE_0 DUR_24 MEL_INT12_0 HARM_OCT_0 HARM_CLASS_0 "
                "VOICE_1 DUR_24 MEL_INT12_0 HARM_OCT_0 HARM_CLASS_7 "
                "VOICE_2 DUR_24 MEL_INT12_0 HARM_OCT_1 HARM_CLASS_4 "
                "VOICE_3 DUR_24 MEL_INT12_0 HARM_OCT_2 HARM_CLASS_0 "
                "POS_24 VOICE_0 DUR_72 MEL_INT12_+2 HARM_OCT_0 HARM_CLASS_2 "
                "VOICE_1 DUR_72 MEL_INT12_+2 HARM_OCT_0 HARM_CLASS_9 "
                "VOICE_2 DUR_72 MEL_INT12_+1 HARM_OCT_1 HARM_CLASS_5 "
                "VOICE_3 DUR_72 MEL_INT12_0 HARM_OCT_2 HARM_CLASS_0"
            ),
        }
    ]


def test_chorale_v2_tokenization_creates_vertical_satb_tokens():
    bars = build_v2_bars_from_v1_rows(_v1_rows())

    assert len(bars) == 1
    tokens = bars[0].tokens
    assert "BAR" in tokens
    assert "POS_0" in tokens
    assert "BASS_48" in tokens
    assert "TENOR_55" in tokens
    assert "ALTO_64" in tokens
    assert "SOP_72" in tokens
    assert "DUR_24" in tokens


def test_chorale_v2_voices_stay_in_fixed_satb_order():
    tokens = build_v2_bars_from_v1_rows(_v1_rows())[0].tokens
    pos = tokens.index("POS_0")

    assert [token.split("_", 1)[0] for token in tokens[pos + 1 : pos + 5]] == [
        "BASS",
        "TENOR",
        "ALTO",
        "SOP",
    ]


def test_chorale_v2_does_not_require_harm_tokens():
    tokens = build_v2_bars_from_v1_rows(_v1_rows())[0].tokens

    assert not any(token.startswith("HARM_") for token in tokens)


def test_chorale_v2_tokens_render_to_midi(tmp_path: Path):
    tokens = build_v2_bars_from_v1_rows(_v1_rows())[0].tokens
    out = tmp_path / "chorale_v2.mid"

    render_v2_tokens_to_midi(tokens, out)

    assert parse_v2_slices(tokens)
    midi = MidiFile()
    midi.open(str(out))
    midi.read()
    midi.close()
    assert midi.ticksPerQuarterNote == 24


def test_chorale_v2_enforces_satb_by_pitch_order_not_voice_id():
    rows = _v1_rows()
    rows[0]["tokens"] = (
        "BAR TIME_SIG_4_4 KEY_C "
        "ABS_VOICE_0_72 ABS_VOICE_1_48 ABS_VOICE_2_64 ABS_VOICE_3_55 "
        "POS_0 VOICE_0 DUR_24 MEL_INT12_0 HARM_OCT_0 HARM_CLASS_0 "
        "VOICE_1 DUR_24 MEL_INT12_0 HARM_OCT_0 HARM_CLASS_7 "
        "VOICE_2 DUR_24 MEL_INT12_0 HARM_OCT_1 HARM_CLASS_4 "
        "VOICE_3 DUR_24 MEL_INT12_0 HARM_OCT_2 HARM_CLASS_0"
    )
    tokens = build_v2_bars_from_v1_rows(rows)[0].tokens
    pos = tokens.index("POS_0")
    assert tokens[pos + 1 : pos + 5] == ["BASS_48", "TENOR_55", "ALTO_64", "SOP_72"]


def test_chorale_v2_repetition_metrics_detect_collapsed_sonority_loop():
    tokens = (
        "BAR STYLE_CHORALE KEY_C TIME_4_4 TEXTURE_4 "
        "POS_0 BASS_48 TENOR_55 ALTO_64 SOP_72 DUR_24 "
        "POS_24 BASS_48 TENOR_55 ALTO_64 SOP_72 DUR_24 "
        "BAR STYLE_CHORALE KEY_C TIME_4_4 TEXTURE_4 "
        "POS_0 BASS_48 TENOR_55 ALTO_64 SOP_72 DUR_24 "
        "POS_24 BASS_48 TENOR_55 ALTO_64 SOP_72 DUR_24"
    ).split()

    metrics = v2_repetition_metrics(tokens)

    assert metrics["slice_count"] == 4
    assert metrics["unique_sonority_count"] == 1
    assert metrics["adjacent_repeat_rate"] == 1.0
    assert metrics["longest_sonority_run"] == 4
    assert metrics["duplicate_bar_rate"] == 0.5
