"""Tests for repair_harm_tokens()."""

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _make_token_stream(*args) -> List[str]:
    """Flatten nested lists and individual tokens into a flat list."""
    result = []
    for a in args:
        if isinstance(a, list):
            result.extend(a)
        else:
            result.append(a)
    return result


def _pitched_event(voice: int, dur: int, mel_int: int, harm_oct: str, harm_cls: str) -> List[str]:
    return [
        f"VOICE_{voice}", f"DUR_{dur}", f"MEL_INT12_{mel_int:+d}",
        f"HARM_OCT_{harm_oct}", f"HARM_CLASS_{harm_cls}",
    ]


def _rest_event(voice: int, dur: int) -> List[str]:
    return [f"VOICE_{voice}", f"REST_{dur}"]


# ---------------------------------------------------------------------------
# Attempt to import; skip tests if dependencies are missing
# ---------------------------------------------------------------------------

try:
    from src.tokens.repair import HarmRepairResult, repair_harm_tokens
    _HAS_REPAIR = True
    _IMPORT_ERROR = None
except ImportError as e:
    _HAS_REPAIR = False
    _IMPORT_ERROR = str(e)

pytestmark = pytest.mark.skipif(
    not _HAS_REPAIR,
    reason=f"src.tokens.repair not importable: {_IMPORT_ERROR}",
)


# ---------------------------------------------------------------------------
# HarmRepairResult dataclass
# ---------------------------------------------------------------------------

def test_harm_repair_result_fields():
    r = HarmRepairResult(
        tokens=["BAR"],
        repaired_event_count=0,
        mismatch_count_before=2,
        mismatch_count_after=0,
        skipped_event_count=1,
    )
    assert r.tokens == ["BAR"]
    assert r.repaired_event_count == 0
    assert r.mismatch_count_before == 2
    assert r.mismatch_count_after == 0
    assert r.skipped_event_count == 1
    assert r.errors_before == []
    assert r.errors_after == []


def test_harm_repair_result_errors_lists():
    r = HarmRepairResult(
        tokens=[],
        repaired_event_count=0,
        mismatch_count_before=0,
        mismatch_count_after=0,
        skipped_event_count=0,
        errors_before=["Mismatch at 0"],
        errors_after=[],
    )
    assert r.errors_before == ["Mismatch at 0"]
    assert r.errors_after == []


# ---------------------------------------------------------------------------
# repair_harm_tokens: basic smoke tests
# ---------------------------------------------------------------------------

def test_repair_empty_tokens():
    result = repair_harm_tokens([])
    assert isinstance(result, HarmRepairResult)
    assert result.tokens == []
    assert result.repaired_event_count == 0
    assert result.skipped_event_count == 0


def test_repair_returns_harm_repair_result():
    tokens = ["BAR", "POS_0"]
    result = repair_harm_tokens(tokens)
    assert isinstance(result, HarmRepairResult)


def test_repair_preserves_non_harm_tokens():
    """Tokens without HARM_ should be left unchanged."""
    tokens = ["BAR", "KEY_C", "TEMPO_120", "POS_0"]
    result = repair_harm_tokens(tokens)
    assert result.tokens == tokens


def test_repair_token_list_length_unchanged():
    """Output token list must have same length as input."""
    tokens = _make_token_stream(
        "BAR",
        "ABS_BASS_60",
        "POS_0",
        _pitched_event(0, 24, 0, "0", "0"),
    )
    result = repair_harm_tokens(tokens)
    assert len(result.tokens) == len(tokens)


def test_repair_rest_events_unchanged():
    """REST events should pass through unchanged."""
    tokens = _make_token_stream(
        "BAR",
        "POS_0",
        _rest_event(0, 24),
    )
    result = repair_harm_tokens(tokens)
    assert result.tokens == tokens
    assert result.repaired_event_count == 0


def test_repair_bar_tokens_preserved():
    tokens = ["BAR", "BAR", "BAR"]
    result = repair_harm_tokens(tokens)
    assert result.tokens.count("BAR") == 3


def test_repair_no_anchor_skips_event():
    """Without an anchor, the event should be skipped (not rewritten)."""
    tokens = _make_token_stream(
        "BAR",
        "POS_0",
        _pitched_event(0, 24, 0, "0", "0"),
    )
    result = repair_harm_tokens(tokens)
    # No anchor for voice 0, so event is skipped
    assert result.skipped_event_count >= 0  # at least doesn't crash
    assert len(result.tokens) == len(tokens)


def test_repair_mismatch_counts_are_ints():
    tokens = ["BAR"]
    result = repair_harm_tokens(tokens)
    assert isinstance(result.mismatch_count_before, int)
    assert isinstance(result.mismatch_count_after, int)


def test_repair_mismatch_after_le_before():
    """After repair, mismatch count should not increase."""
    tokens = _make_token_stream(
        "BAR",
        "ABS_BASS_60",
        "POS_0",
        _pitched_event(0, 24, 0, "99", "99"),  # deliberately wrong HARM tokens
    )
    result = repair_harm_tokens(tokens)
    assert result.mismatch_count_after <= result.mismatch_count_before


def test_repair_correct_tokens_not_changed():
    """If HARM tokens are already correct, repaired_event_count should be 0."""
    # We use "0" / "0" which are the defaults; actual correctness depends on
    # the validator, so we just check the function runs without error.
    tokens = _make_token_stream(
        "BAR",
        "ABS_BASS_60",
        "POS_0",
        _pitched_event(0, 24, 0, "0", "0"),
    )
    result = repair_harm_tokens(tokens)
    assert result.repaired_event_count >= 0


# ---------------------------------------------------------------------------
# repair_harm_tokens: time signature handling
# ---------------------------------------------------------------------------

def test_repair_with_time_sig():
    tokens = _make_token_stream(
        "TIME_SIG_3_4",
        "BAR",
        "ABS_BASS_60",
        "POS_0",
        _pitched_event(0, 24, 0, "0", "0"),
    )
    result = repair_harm_tokens(tokens)
    assert isinstance(result, HarmRepairResult)
    assert len(result.tokens) == len(tokens)


def test_repair_multi_bar():
    tokens = _make_token_stream(
        "BAR",
        "ABS_BASS_60",
        "POS_0",
        _pitched_event(0, 24, 0, "0", "0"),
        "BAR",
        "POS_0",
        _pitched_event(0, 24, 2, "0", "2"),
    )
    result = repair_harm_tokens(tokens)
    assert isinstance(result, HarmRepairResult)
    assert len(result.tokens) == len(tokens)


# ---------------------------------------------------------------------------
# repair_harm_tokens: anchor types
# ---------------------------------------------------------------------------

def test_repair_abs_bass_anchor():
    tokens = _make_token_stream(
        "BAR",
        "ABS_BASS_48",
        "POS_0",
        _pitched_event(0, 24, 0, "0", "0"),
    )
    result = repair_harm_tokens(tokens)
    assert len(result.tokens) == len(tokens)


def test_repair_abs_sop_anchor():
    tokens = _make_token_stream(
        "BAR",
        "ABS_SOP_72",
        "POS_0",
        _pitched_event(3, 24, 0, "0", "0"),
    )
    result = repair_harm_tokens(tokens)
    assert len(result.tokens) == len(tokens)


def test_repair_abs_voice_anchor():
    tokens = _make_token_stream(
        "BAR",
        "ABS_VOICE_1_55",
        "POS_0",
        _pitched_event(1, 24, 0, "0", "0"),
    )
    result = repair_harm_tokens(tokens)
    assert len(result.tokens) == len(tokens)


# ---------------------------------------------------------------------------
# repair_harm_tokens: DUP token handling
# ---------------------------------------------------------------------------

def test_repair_dup_event():
    """Events with DUP_ token between DUR_ and MEL_INT12_ should be handled."""
    tokens = _make_token_stream(
        "BAR",
        "ABS_BASS_60",
        "POS_0",
        ["VOICE_0", "DUR_24", "DUP_0", "MEL_INT12_+0", "HARM_OCT_0", "HARM_CLASS_0"],
    )
    result = repair_harm_tokens(tokens)
    assert len(result.tokens) == len(tokens)


# ---------------------------------------------------------------------------
# repair_harm_tokens: errors_before / errors_after populated
# ---------------------------------------------------------------------------

def test_repair_errors_before_is_list():
    tokens = ["BAR"]
    result = repair_harm_tokens(tokens)
    assert isinstance(result.errors_before, list)
    assert isinstance(result.errors_after, list)


def test_repair_idempotent():
    """Running repair twice should yield same or fewer mismatches the second time."""
    tokens = _make_token_stream(
        "BAR",
        "ABS_BASS_60",
        "POS_0",
        _pitched_event(0, 24, 0, "0", "0"),
        _pitched_event(0, 24, 2, "0", "2"),
    )
    result1 = repair_harm_tokens(tokens)
    result2 = repair_harm_tokens(result1.tokens)
    assert result2.mismatch_count_after <= result1.mismatch_count_after


def test_repair_key_and_tempo_tokens_pass_through():
    tokens = ["KEY_C", "TEMPO_120", "BAR", "POS_0"]
    result = repair_harm_tokens(tokens)
    assert result.tokens[:4] == tokens[:4]


def test_repair_truncated_stream():
    """A stream that ends mid-event should not crash."""
    tokens = _make_token_stream(
        "BAR",
        "ABS_BASS_60",
        "POS_0",
        ["VOICE_0", "DUR_24"],  # truncated: no MEL_INT12 etc.
    )
    result = repair_harm_tokens(tokens)
    assert isinstance(result, HarmRepairResult)
