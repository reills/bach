import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.dataio.descriptors import (
    _classify_density,
    _compute_pitch_range,
    _extract_key,
    _extract_time_sig,
    compute_bar_plan,
)


def _voice_event(voice, duration, mel):
    return [
        f"VOICE_{voice}",
        f"DUR_{duration}",
        f"MEL_INT12_{mel}",
        "HARM_OCT_0",
        "HARM_CLASS_0",
    ]


def _split_bars(tokens):
    bars = []
    current = []
    for tok in tokens:
        if tok == "BAR":
            if current:
                bars.append(current)
            current = ["BAR"]
        else:
            current.append(tok)
    if current:
        bars.append(current)
    return bars


def test_extract_time_sig():
    assert _extract_time_sig(["BAR", "TIME_SIG_4_4"], None) == "4/4"
    assert _extract_time_sig(["BAR", "TIME_SIG_3_4"], None) == "3/4"
    assert _extract_time_sig(["BAR", "TIME_SIG_6_8"], None) == "6/8"


def test_extract_time_sig_inheritance():
    assert _extract_time_sig(["BAR", "KEY_C"], "4/4") == "4/4"


def test_extract_key():
    assert _extract_key(["BAR", "KEY_C"], None) == "C"
    assert _extract_key(["BAR", "KEY_Am"], None) == "Am"
    assert _extract_key(["BAR", "KEY_F#m"], None) == "F#m"


def test_extract_key_inheritance():
    assert _extract_key(["BAR", "TIME_SIG_4_4"], "C") == "C"


def test_density_classification():
    assert _classify_density(0) == "DENSITY_LOW"
    assert _classify_density(3) == "DENSITY_LOW"
    assert _classify_density(4) == "DENSITY_MED"
    assert _classify_density(8) == "DENSITY_MED"
    assert _classify_density(9) == "DENSITY_HIGH"


def test_compute_pitch_range():
    assert _compute_pitch_range([], []) is None
    assert _compute_pitch_range([60], []) == 0
    assert _compute_pitch_range([60, 72], []) == 12
    assert _compute_pitch_range([60], [48]) == 12


def test_pitch_reconstruction_simple():
    tokens = [
        "BAR",
        "TIME_SIG_4_4",
        "KEY_C",
        "POS_0",
        "ABS_VOICE_0_60",
    ] + _voice_event(0, 24, "0")

    plan, _ = compute_bar_plan(tokens, bar_index=0, tpq=24)
    assert plan.density_bucket == "DENSITY_LOW"
    assert plan.pitch_range == 0
    assert plan.polyphony_max == 1


def test_pitch_reconstruction_polyphonic():
    tokens = [
        "BAR",
        "TIME_SIG_4_4",
        "KEY_C",
        "ABS_BASS_40",
        "ABS_SOP_67",
        "POS_0",
        "ABS_VOICE_1_52",
        "ABS_VOICE_2_55",
    ]
    tokens += _voice_event(0, 24, "0")
    tokens += _voice_event(1, 24, "0")
    tokens += _voice_event(2, 24, "0")
    tokens += _voice_event(3, 24, "0")

    plan, _ = compute_bar_plan(tokens, bar_index=0, tpq=24)
    assert plan.density_bucket == "DENSITY_MED"
    assert plan.pitch_range == 27
    assert plan.polyphony_max == 4


def test_state_carry_across_bars():
    bar1 = [
        "BAR",
        "TIME_SIG_1_4",
        "KEY_C",
        "POS_0",
        "ABS_VOICE_0_60",
    ] + _voice_event(0, 4, "0")

    bar2 = [
        "BAR",
        "TIME_SIG_1_4",
        "KEY_C",
        "POS_0",
    ] + _voice_event(0, 4, "+2")

    state = None
    _, state = compute_bar_plan(bar1, bar_index=0, running_state=state, tpq=4)
    plan2, _ = compute_bar_plan(bar2, bar_index=1, running_state=state, tpq=4)
    assert plan2.pitch_range == 0


def test_empty_bar():
    tokens = ["BAR", "TIME_SIG_4_4", "KEY_C"]
    plan, _ = compute_bar_plan(tokens, bar_index=0, tpq=24)
    assert plan.density_bucket == "DENSITY_LOW"
    assert plan.pitch_range is None
    assert plan.polyphony_max == 0


def test_single_pitch_bar():
    tokens = [
        "BAR",
        "TIME_SIG_4_4",
        "KEY_C",
        "POS_0",
        "ABS_VOICE_0_60",
    ] + _voice_event(0, 24, "0")
    plan, _ = compute_bar_plan(tokens, bar_index=0, tpq=24)
    assert plan.pitch_range == 0
    assert plan.polyphony_max == 1


def test_pitch_range_includes_held_notes():
    bar1 = [
        "BAR",
        "TIME_SIG_1_4",
        "KEY_C",
        "POS_0",
        "ABS_VOICE_0_60",
    ] + _voice_event(0, 8, "0")

    bar2 = ["BAR", "TIME_SIG_1_4", "KEY_C"]

    state = None
    _, state = compute_bar_plan(bar1, bar_index=0, running_state=state, tpq=4)
    plan2, _ = compute_bar_plan(bar2, bar_index=1, running_state=state, tpq=4)
    assert plan2.pitch_range == 0


def test_missing_time_sig_error():
    with pytest.raises(ValueError):
        compute_bar_plan(["BAR", "KEY_C"], bar_index=0, tpq=24)


def test_missing_key_error():
    with pytest.raises(ValueError):
        compute_bar_plan(["BAR", "TIME_SIG_4_4"], bar_index=0, tpq=24)


def test_missing_anchor_error():
    tokens = [
        "BAR",
        "TIME_SIG_4_4",
        "KEY_C",
        "POS_0",
    ] + _voice_event(0, 24, "0")
    with pytest.raises(ValueError):
        compute_bar_plan(tokens, bar_index=0, tpq=24)


def test_integration_bwv66_tokens():
    tokens_path = Path("out/bwv66.6.tokens.txt")
    if not tokens_path.exists():
        pytest.skip("Missing out/bwv66.6.tokens.txt")

    raw = tokens_path.read_text(encoding="utf-8").replace("\n", ",")
    tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    bars = _split_bars(tokens)

    state = None
    for idx, bar_tokens in enumerate(bars):
        plan, state = compute_bar_plan(bar_tokens, bar_index=idx, running_state=state)
        assert plan.time_sig
        assert plan.key
        assert plan.density_bucket.startswith("DENSITY_")
        assert plan.polyphony_max is not None
        if plan.pitch_range is not None:
            assert plan.pitch_range >= 0
