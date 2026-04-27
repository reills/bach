import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "audit_dataset", ROOT / "scripts" / "audit_dataset.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

audit_dataset = _mod.audit_dataset
main = _mod.main


VALID_BAR_TOKENS = [
    "BAR",
    "TIME_SIG_4_4",
    "KEY_C",
    "ABS_VOICE_0_60",
    "POS_0",
    "VOICE_0",
    "DUR_24",
    "MEL_INT12_0",
    "HARM_OCT_0",
    "HARM_CLASS_0",
]


def _write_events(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    pytest.importorskip("pandas")
    import pandas as pd

    events_path = tmp_path / "events.parquet"
    pd.DataFrame(rows).to_parquet(events_path, index=False)
    return events_path


def test_audit_dataset_accepts_clean_parquet(tmp_path):
    events_path = _write_events(
        tmp_path,
        [
            {
                "piece_id": "piece-a",
                "bar_index": 0,
                "tokens": " ".join(VALID_BAR_TOKENS),
                "bar_len_ticks": 96,
            }
        ],
    )

    report = audit_dataset(events_path)

    assert report["ok"] is True
    assert report["failures"] == []
    assert report["total_bars"] == 1
    assert report["total_pieces"] == 1
    assert report["mel_int_out_of_range_count"] == 0
    assert report["malformed_voice_event_count"] == 0
    assert report["harmonic_metadata_mismatch_count"] == 0
    assert report["metrics"]["counterpoint_harmonic_metadata_mismatches"] == 0


def test_audit_dataset_reports_training_blockers(tmp_path):
    bad_tokens = [
        "BAR",
        "TIME_SIG_4_4",
        "ABS_VOICE_0_60",
        "POS_0",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_+30",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_0",
        "DUR_24",
        "HARM_OCT_0",
        "HARM_CLASS_0",
    ]
    events_path = _write_events(
        tmp_path,
        [
            {
                "piece_id": "piece-a",
                "bar_index": 0,
                "tokens": " ".join(bad_tokens),
                "bar_len_ticks": 96,
            }
        ],
    )

    report = audit_dataset(events_path)

    assert report["ok"] is False
    assert "mel_int_out_of_range" in report["failures"]
    assert "malformed_voice_events" in report["failures"]
    assert report["mel_int_out_of_range_count"] == 1
    assert report["malformed_voice_event_count"] == 1


def test_audit_dataset_checks_vocab_unknown_tokens(tmp_path):
    events_path = _write_events(
        tmp_path,
        [
            {
                "piece_id": "piece-a",
                "bar_index": 0,
                "tokens": " ".join(VALID_BAR_TOKENS),
                "bar_len_ticks": 96,
            }
        ],
    )
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"BAR": 0}), encoding="utf-8")

    report = audit_dataset(events_path, vocab_path=vocab_path)

    assert report["ok"] is False
    assert "unknown_tokens" in report["failures"]
    assert report["unknown_token_count"] == len(VALID_BAR_TOKENS) - 1


def test_audit_dataset_cli_writes_stats_json_and_fails_strict(tmp_path):
    bad_tokens = ["BAR", "POS_0", "VOICE_0", "DUR_24", "MEL_INT12_+30"]
    events_path = _write_events(
        tmp_path,
        [
            {
                "piece_id": "piece-a",
                "bar_index": 0,
                "tokens": " ".join(bad_tokens),
                "bar_len_ticks": 96,
            }
        ],
    )

    rc = main(["--events", str(events_path), "--quiet"])

    assert rc == 1
    stats_path = tmp_path / "stats.json"
    assert stats_path.exists()
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    assert data["ok"] is False


def test_audit_dataset_cli_warn_only_exits_zero(tmp_path):
    events_path = _write_events(
        tmp_path,
        [
            {
                "piece_id": "piece-a",
                "bar_index": 0,
                "tokens": "BAR POS_0 VOICE_0 DUR_24 MEL_INT12_+30",
                "bar_len_ticks": 96,
            }
        ],
    )
    out_path = tmp_path / "audit.json"

    rc = main(
        [
            "--events",
            str(events_path),
            "--output-json",
            str(out_path),
            "--warn-only",
            "--quiet",
        ]
    )

    assert rc == 0
    assert json.loads(out_path.read_text(encoding="utf-8"))["ok"] is False
