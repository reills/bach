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
