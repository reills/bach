"""Tests for scripts/eval_basic.py — CLI and core evaluate() function."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import the module directly (scripts/ is not a package, so use importlib)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "eval_basic", ROOT / "scripts" / "eval_basic.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

evaluate = _mod.evaluate
main = _mod.main


# ---------------------------------------------------------------------------
# Tiny fixture: a minimal token stream (2 bars, with and without tab)
# ---------------------------------------------------------------------------

MINIMAL_TOKENS_NO_TAB = [
    "BAR",
    "POS_0",
    "VOICE_0", "DUR_24", "MEL_INT12_+2", "HARM_OCT_0", "HARM_CLASS_4",
    "VOICE_0", "DUR_12", "MEL_INT12_-3", "HARM_OCT_NA", "HARM_CLASS_NA",
    "BAR",
    "POS_0",
    "VOICE_0", "REST_24",
    "VOICE_0", "DUR_24", "MEL_INT12_+7", "HARM_OCT_0", "HARM_CLASS_7",
]

MINIMAL_TOKENS_WITH_TAB = MINIMAL_TOKENS_NO_TAB + [
    "STR_2", "FRET_3",
    "STR_1", "FRET_0",
    "STR_3", "FRET_5",
]

TOKENS_BAD_INTERVAL = [
    "BAR",
    "VOICE_0", "DUR_24", "MEL_INT12_+30",   # out of [-24, 24]
]

# C major pitch-class set: {0, 2, 4, 5, 7, 9, 11}
# ABS_VOICE_0_60 anchors voice 0 at MIDI 60 (C4, PC=0).
# +2 -> D4 (62, PC=2, in key), +3 -> F4 (65, PC=5, in key),
# +1 -> F#4 (66, PC=6, NOT in key), -3 -> Eb4 (63, PC=3, NOT in key) -> rate = 2/4 = 0.5
OFF_KEY_TOKENS = [
    "BAR",
    "ABS_VOICE_0_60",
    "POS_0",
    "VOICE_0", "DUR_24", "MEL_INT12_+2", "HARM_OCT_NA", "HARM_CLASS_NA",
    "VOICE_0", "DUR_24", "MEL_INT12_+3", "HARM_OCT_NA", "HARM_CLASS_NA",
    "VOICE_0", "DUR_24", "MEL_INT12_+1", "HARM_OCT_NA", "HARM_CLASS_NA",
    "VOICE_0", "DUR_24", "MEL_INT12_-3", "HARM_OCT_NA", "HARM_CLASS_NA",
]

# Stream that declares its own key via KEY_ token — no --key arg needed.
# KEY_C -> C major; same pitch sequence as OFF_KEY_TOKENS -> rate = 0.5
OFF_KEY_STREAM_KEY_TOKENS = ["KEY_C"] + OFF_KEY_TOKENS

# All four pitches land in C major: D4(+2->62), F4(+3->65), G4(+2->67), A4(+2->69)
IN_KEY_TOKENS = [
    "BAR",
    "ABS_VOICE_0_60",
    "POS_0",
    "VOICE_0", "DUR_24", "MEL_INT12_+2", "HARM_OCT_NA", "HARM_CLASS_NA",
    "VOICE_0", "DUR_24", "MEL_INT12_+3", "HARM_OCT_NA", "HARM_CLASS_NA",
    "VOICE_0", "DUR_24", "MEL_INT12_+2", "HARM_OCT_NA", "HARM_CLASS_NA",
    "VOICE_0", "DUR_24", "MEL_INT12_+2", "HARM_OCT_NA", "HARM_CLASS_NA",
]

# HARM tokens that are intentionally wrong.
# Solo voice, ABS anchor at 60. ref_pitch = 60, diff = 0 -> expected HARM_OCT_0, HARM_CLASS_0.
# Stream writes HARM_CLASS_4 -> 1 mismatch.
HARM_MISMATCH_TOKENS = [
    "BAR",
    "ABS_VOICE_0_60",
    "POS_0",
    "VOICE_0", "DUR_24", "MEL_INT12_0", "HARM_OCT_0", "HARM_CLASS_4",
]

# Correct HARM tokens for a solo note: ref = pitch, diff = 0 -> HARM_OCT_0, HARM_CLASS_0.
HARM_VALID_TOKENS = [
    "BAR",
    "ABS_VOICE_0_60",
    "POS_0",
    "VOICE_0", "DUR_24", "MEL_INT12_0", "HARM_OCT_0", "HARM_CLASS_0",
]

# 3 bars: first two identical, third different -> duplicate_bar_rate = 1/3
DUPLICATE_BAR_TOKENS = [
    "BAR", "POS_0", "VOICE_0", "REST_24",
    "BAR", "POS_0", "VOICE_0", "REST_24",
    "BAR", "POS_0", "VOICE_0", "REST_48",
]

# 2 keyed bars:
# Bar 1 final onset lands on G in C major -> cadence proxy hit
# Bar 2 final onset lands on E in C major -> not a cadence proxy hit
CADENCE_TOKENS = [
    "KEY_C",
    "BAR",
    "ABS_VOICE_0_62",
    "POS_0",
    "VOICE_0", "DUR_24", "MEL_INT12_+5", "HARM_OCT_NA", "HARM_CLASS_NA",
    "BAR",
    "ABS_VOICE_0_60",
    "POS_0",
    "VOICE_0", "DUR_24", "MEL_INT12_+4", "HARM_OCT_NA", "HARM_CLASS_NA",
]

# Key changes should affect off-key scoring at the point of the change.
KEY_CHANGE_TOKENS = [
    "BAR", "KEY_C", "ABS_VOICE_0_60", "POS_0",
    "VOICE_0", "DUR_24", "MEL_INT12_0", "HARM_OCT_0", "HARM_CLASS_0",
    "BAR", "KEY_G", "ABS_VOICE_0_66", "POS_0",
    "VOICE_0", "DUR_24", "MEL_INT12_0", "HARM_OCT_0", "HARM_CLASS_0",
]

# Grammar violations are only malformed VOICE_* events, not random standalone tokens.
GRAMMAR_VIOLATION_TOKENS = [
    "BAR",
    "POS_0",
    "VOICE_0", "DUR_24", "HARM_OCT_0", "HARM_CLASS_0",
]

GRAMMAR_TAB_VIOLATION_TOKENS = [
    "BAR",
    "ABS_VOICE_0_60",
    "POS_0",
    "VOICE_0", "DUR_24", "MEL_INT12_0", "HARM_OCT_0", "HARM_CLASS_0", "STR_2",
]


# ---------------------------------------------------------------------------
# Unit tests for evaluate()
# ---------------------------------------------------------------------------

def test_bar_count():
    m = evaluate(MINIMAL_TOKENS_NO_TAB)
    assert m["bar_count"] == 2


def test_voice_event_and_rest_counts():
    m = evaluate(MINIMAL_TOKENS_NO_TAB)
    # 3 pitched DUR events, 1 REST
    assert m["voice_event_count"] == 3
    assert m["rest_event_count"] == 1


def test_interval_range_ok_within_bounds():
    m = evaluate(MINIMAL_TOKENS_NO_TAB)
    assert m["interval_range_ok"] is True
    assert m["mel_int_range"] == [-3, 7]


def test_interval_range_fails_out_of_bounds():
    m = evaluate(TOKENS_BAD_INTERVAL)
    assert m["interval_range_ok"] is False
    assert m["mel_int_range"][1] == 30


def test_no_tab_tokens():
    m = evaluate(MINIMAL_TOKENS_NO_TAB)
    assert m["tab_present"] is False
    assert "tab_fret_max" not in m


def test_tab_metrics_present():
    m = evaluate(MINIMAL_TOKENS_WITH_TAB)
    assert m["tab_present"] is True
    assert m["tab_fret_max"] == 5
    # one open string (FRET_0) out of 3
    assert abs(m["tab_open_string_pct"] - (1 / 3)) < 0.001


def test_token_validity_all_known():
    vocab = {"BAR", "POS_0", "VOICE_0", "DUR_24", "DUR_12", "REST_24",
             "MEL_INT12_+2", "MEL_INT12_-3", "MEL_INT12_+7",
             "HARM_OCT_0", "HARM_CLASS_4", "HARM_OCT_NA", "HARM_CLASS_NA",
             "HARM_CLASS_7"}
    m = evaluate(MINIMAL_TOKENS_NO_TAB, vocab=vocab)
    assert m["token_validity"] == 1.0


def test_token_validity_some_unknown():
    vocab = {"BAR"}  # only BAR is known
    m = evaluate(MINIMAL_TOKENS_NO_TAB, vocab=vocab)
    assert m["token_validity"] < 1.0


def test_empty_stream():
    m = evaluate([])
    assert m["bar_count"] == 0
    assert m["token_count"] == 0
    assert m["tab_present"] is False
    assert m["mel_int_range"] is None
    assert m["interval_range_ok"] is True  # vacuously true


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_cli_token_file(tmp_path):
    tok_file = tmp_path / "stream.txt"
    tok_file.write_text(" ".join(MINIMAL_TOKENS_NO_TAB), encoding="utf-8")
    out = tmp_path / "metrics.json"
    rc = main(["--token-file", str(tok_file), "--output-json", str(out), "--quiet"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["bar_count"] == 2
    assert data["interval_range_ok"] is True


def test_cli_token_file_with_vocab(tmp_path):
    tok_file = tmp_path / "stream.txt"
    tok_file.write_text(" ".join(MINIMAL_TOKENS_NO_TAB), encoding="utf-8")
    vocab_file = tmp_path / "vocab.json"
    # Only a subset: some tokens will be unknown
    vocab_file.write_text(json.dumps({"BAR": 0, "POS_0": 1}), encoding="utf-8")
    out = tmp_path / "metrics.json"
    rc = main(["--token-file", str(tok_file),
               "--vocab", str(vocab_file),
               "--output-json", str(out),
               "--quiet"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["token_validity"] is not None
    assert data["token_validity"] < 1.0


def test_cli_token_file_newline_separated(tmp_path):
    tok_file = tmp_path / "stream.txt"
    tok_file.write_text("\n".join(MINIMAL_TOKENS_WITH_TAB), encoding="utf-8")
    out = tmp_path / "metrics.json"
    rc = main(["--token-file", str(tok_file), "--output-json", str(out), "--quiet"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["tab_present"] is True
    assert data["tab_fret_max"] == 5


def test_cli_parquet(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    rows = [
        {"tokens": " ".join(MINIMAL_TOKENS_NO_TAB[:10])},
        {"tokens": " ".join(MINIMAL_TOKENS_NO_TAB[10:])},
    ]
    df = pd.DataFrame(rows)
    parquet_file = tmp_path / "events.parquet"
    df.to_parquet(parquet_file, index=False)
    out = tmp_path / "metrics.json"
    rc = main(["--parquet", str(parquet_file), "--output-json", str(out), "--quiet"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["token_count"] == len(MINIMAL_TOKENS_NO_TAB)


def test_cli_json_output_keys(tmp_path):
    tok_file = tmp_path / "stream.txt"
    tok_file.write_text(" ".join(MINIMAL_TOKENS_NO_TAB), encoding="utf-8")
    out = tmp_path / "metrics.json"
    main(["--token-file", str(tok_file), "--output-json", str(out), "--quiet"])
    data = json.loads(out.read_text())
    for key in ("token_count", "bar_count", "interval_range_ok",
                "voice_event_count", "rest_event_count", "tab_present"):
        assert key in data, f"missing key: {key}"


def test_evaluate_includes_counterpoint_metrics():
    m = evaluate(HARM_VALID_TOKENS)
    assert "counterpoint_parallel_fifths" in m
    assert "counterpoint_avg_active_voices" in m
    assert "counterpoint_harmonic_metadata_mismatches" in m


# ---------------------------------------------------------------------------
# off_key_rate
# ---------------------------------------------------------------------------

def test_off_key_rate_no_key_returns_none():
    # No --key and no KEY_* in stream -> cannot compute, should be None
    m = evaluate(MINIMAL_TOKENS_NO_TAB)
    assert m["off_key_rate"] is None


def test_off_key_rate_all_in_key():
    m = evaluate(IN_KEY_TOKENS, key="C")
    assert m["off_key_rate"] == 0.0


def test_off_key_rate_partial():
    m = evaluate(OFF_KEY_TOKENS, key="C")
    assert m["off_key_rate"] is not None
    assert m["off_key_count"] == 2
    assert m["pitched_onset_count"] == 4
    assert abs(m["off_key_rate"] - 0.5) < 0.001


def test_off_key_rate_from_stream_key_token():
    # KEY_C is prepended to stream; no explicit key arg
    m = evaluate(OFF_KEY_STREAM_KEY_TOKENS)
    assert m["off_key_rate"] is not None
    assert abs(m["off_key_rate"] - 0.5) < 0.001


def test_off_key_rate_explicit_key_overrides_nothing():
    # Explicit key arg is used; stream has no KEY_ token
    m = evaluate(OFF_KEY_TOKENS, key="C")
    assert m["off_key_rate"] is not None
    # Same fixture -> same result
    assert abs(m["off_key_rate"] - 0.5) < 0.001


def test_off_key_rate_minor_key():
    # A minor natural: {0, 2, 3, 5, 7, 8, 10}
    # ABS_VOICE_0_69 = A4 (MIDI 69, PC=9 – wait that's A, need tonic A minor)
    # Anchor at 69 (A4). MEL_INT12_+2 -> 71 (B4, PC=11, in Am). MEL_INT12_+1 -> 72 (C5, PC=0, in Am).
    tokens = [
        "BAR", "ABS_VOICE_0_69", "POS_0",
        "VOICE_0", "DUR_24", "MEL_INT12_+2", "HARM_OCT_NA", "HARM_CLASS_NA",
        "VOICE_0", "DUR_24", "MEL_INT12_+1", "HARM_OCT_NA", "HARM_CLASS_NA",
    ]
    m = evaluate(tokens, key="Am")
    assert m["off_key_count"] == 0
    assert m["pitched_onset_count"] == 2
    assert m["off_key_rate"] == 0.0


def test_off_key_rate_uses_latest_stream_key():
    m = evaluate(KEY_CHANGE_TOKENS)
    assert m["off_key_count"] == 0
    assert m["pitched_onset_count"] == 2
    assert m["off_key_rate"] == 0.0


def test_off_key_rate_cli_key_flag(tmp_path):
    tok_file = tmp_path / "stream.txt"
    tok_file.write_text(" ".join(OFF_KEY_TOKENS), encoding="utf-8")
    out = tmp_path / "metrics.json"
    rc = main(["--token-file", str(tok_file), "--key", "C", "--output-json", str(out), "--quiet"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["off_key_rate"] is not None
    assert abs(data["off_key_rate"] - 0.5) < 0.001


# ---------------------------------------------------------------------------
# harm_mismatch_count
# ---------------------------------------------------------------------------

def test_harm_mismatch_count_present_in_output():
    m = evaluate(MINIMAL_TOKENS_NO_TAB)
    assert "harm_mismatch_count" in m


def test_harm_mismatch_count_valid_stream():
    m = evaluate(HARM_VALID_TOKENS)
    if m["harm_mismatch_count"] is None:
        pytest.skip("validator not available")
    assert m["harm_mismatch_count"] == 0


def test_harm_mismatch_count_detects_mismatch():
    m = evaluate(HARM_MISMATCH_TOKENS)
    if m["harm_mismatch_count"] is None:
        pytest.skip("validator not available")
    assert m["harm_mismatch_count"] >= 1


# ---------------------------------------------------------------------------
# duplicate_bar_rate
# ---------------------------------------------------------------------------

def test_duplicate_bar_rate_no_duplicates():
    m = evaluate(MINIMAL_TOKENS_NO_TAB)
    assert m["duplicate_bar_rate"] == 0.0


def test_duplicate_bar_rate_with_duplicate():
    m = evaluate(DUPLICATE_BAR_TOKENS)
    assert m["duplicate_bar_count"] == 1
    assert m["duplicate_bar_rate"] is not None
    # 3 bars, 1 duplicate -> 1/3
    assert abs(m["duplicate_bar_rate"] - (1 / 3)) < 0.001


def test_duplicate_bar_rate_empty_stream():
    m = evaluate([])
    assert m["duplicate_bar_rate"] is None


def test_duplicate_bar_rate_all_duplicate():
    tokens = [
        "BAR", "POS_0", "VOICE_0", "REST_24",
        "BAR", "POS_0", "VOICE_0", "REST_24",
        "BAR", "POS_0", "VOICE_0", "REST_24",
    ]
    m = evaluate(tokens)
    # bars: [same, same, same] -> 2 duplicates / 3 total = 0.6667
    assert m["duplicate_bar_count"] == 2
    assert m["duplicate_bar_rate"] is not None
    assert abs(m["duplicate_bar_rate"] - (2 / 3)) < 0.001


# ---------------------------------------------------------------------------
# cadence_proxy_rate
# ---------------------------------------------------------------------------

def test_cadence_proxy_rate_present():
    m = evaluate(MINIMAL_TOKENS_NO_TAB)
    assert "cadence_proxy_rate" in m


def test_cadence_proxy_rate_correct():
    m = evaluate(CADENCE_TOKENS)
    assert m["cadence_proxy_hits"] == 1
    assert m["cadence_proxy_eligible_bars"] == 2
    assert m["cadence_proxy_rate"] is not None
    assert abs(m["cadence_proxy_rate"] - 0.5) < 0.001


def test_cadence_proxy_rate_all_cadence():
    tokens = [
        "KEY_C",
        "BAR", "ABS_VOICE_0_60", "POS_0",
        "VOICE_0", "DUR_24", "MEL_INT12_0", "HARM_OCT_0", "HARM_CLASS_0",
        "BAR", "ABS_VOICE_0_60", "POS_0",
        "VOICE_0", "DUR_24", "MEL_INT12_+7", "HARM_OCT_0", "HARM_CLASS_7",
    ]
    m = evaluate(tokens)
    assert m["cadence_proxy_hits"] == 2
    assert m["cadence_proxy_eligible_bars"] == 2
    assert m["cadence_proxy_rate"] == 1.0


def test_cadence_proxy_rate_no_mel_events():
    tokens = ["KEY_C", "BAR", "POS_0", "VOICE_0", "REST_24", "BAR", "POS_0", "VOICE_0", "REST_48"]
    m = evaluate(tokens)
    assert m["cadence_proxy_hits"] == 0
    assert m["cadence_proxy_eligible_bars"] == 0
    assert m["cadence_proxy_rate"] is None


def test_cadence_proxy_rate_without_key_is_none():
    tokens = [
        "BAR", "ABS_VOICE_0_60", "POS_0",
        "VOICE_0", "DUR_24", "MEL_INT12_0", "HARM_OCT_0", "HARM_CLASS_0",
    ]
    m = evaluate(tokens)
    assert m["cadence_proxy_eligible_bars"] == 0
    assert m["cadence_proxy_rate"] is None


# ---------------------------------------------------------------------------
# token_grammar_violations
# ---------------------------------------------------------------------------

def test_grammar_violations_zero_on_valid_stream():
    m = evaluate(MINIMAL_TOKENS_NO_TAB)
    assert m["token_grammar_violations"] == 0


def test_grammar_violations_zero_on_valid_with_tab():
    m = evaluate(MINIMAL_TOKENS_WITH_TAB)
    assert m["token_grammar_violations"] == 0


def test_grammar_violations_dur_without_voice():
    m = evaluate(GRAMMAR_VIOLATION_TOKENS)
    assert m["token_grammar_violations"] >= 1


def test_grammar_violations_missing_fret_after_str():
    m = evaluate(GRAMMAR_TAB_VIOLATION_TOKENS)
    assert m["token_grammar_violations"] >= 1


def test_grammar_violations_ignore_non_voice_standalone_tokens():
    tokens = ["BAR", "POS_0", "HARM_OCT_0", "HARM_CLASS_0"]
    m = evaluate(tokens)
    assert m["token_grammar_violations"] == 0


def test_grammar_violations_empty_stream():
    m = evaluate([])
    assert m["token_grammar_violations"] == 0
