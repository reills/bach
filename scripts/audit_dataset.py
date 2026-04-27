"""
Audit a processed Bach token dataset before training.

The audit is intentionally strict by default. It writes a JSON report and exits
nonzero if the dataset contains issues that should block training:

* MEL_INT12 tokens outside [-24, 24]
* malformed VOICE_* events
* harmonic metadata mismatches
* unknown tokens when --vocab is provided
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_basic import evaluate  # noqa: E402


_MEL_INT_RE = re.compile(r"^MEL_INT12_([+-]?\d+)$")


def _iter_tokens(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield from (token for token in value.split() if token)
        return
    if isinstance(value, (list, tuple)):
        for token in value:
            if token:
                yield str(token)
        return
    raise ValueError(f"unsupported tokens value type: {type(value)}")


def _load_events(path: Path, *, tokens_col: str) -> tuple[list[dict[str, object]], list[str]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to audit a dataset") from exc

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if tokens_col not in df.columns:
        raise RuntimeError(f"Column '{tokens_col}' not found in {path}; found {list(df.columns)}")

    records = df.to_dict(orient="records")
    tokens: list[str] = []
    for value in df[tokens_col].tolist():
        tokens.extend(_iter_tokens(value))
    return records, tokens


def _load_vocab(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(key) for key in data}
    if isinstance(data, list):
        return {str(token) for token in data}
    raise RuntimeError(f"vocab must be a dict or list: {path}")


def _mel_interval_stats(tokens: Sequence[str]) -> dict[str, object]:
    values: list[int] = []
    out_of_range: list[dict[str, object]] = []
    for index, token in enumerate(tokens):
        match = _MEL_INT_RE.match(token)
        if match is None:
            continue
        value = int(match.group(1))
        values.append(value)
        if value < -24 or value > 24:
            out_of_range.append({"index": index, "token": token, "value": value})

    return {
        "mel_int_count": len(values),
        "mel_int_min": min(values) if values else None,
        "mel_int_max": max(values) if values else None,
        "mel_int_out_of_range_count": len(out_of_range),
        "mel_int_out_of_range_examples": out_of_range[:10],
    }


def _piece_bar_stats(records: Sequence[dict[str, object]]) -> dict[str, object]:
    piece_counter: Counter[str] = Counter()
    bar_count = len(records)
    bar_len_values: list[float] = []
    missing_piece_id = 0
    duplicate_bar_keys = 0
    seen_bar_keys: set[tuple[str, int]] = set()

    for row in records:
        raw_piece = row.get("piece_id")
        piece_id = str(raw_piece) if raw_piece is not None else ""
        if not piece_id:
            missing_piece_id += 1
        else:
            piece_counter[piece_id] += 1

        raw_bar_index = row.get("bar_index")
        if piece_id and raw_bar_index is not None:
            try:
                key = (piece_id, int(raw_bar_index))
            except (TypeError, ValueError):
                key = (piece_id, -1)
            if key in seen_bar_keys:
                duplicate_bar_keys += 1
            seen_bar_keys.add(key)

        raw_bar_len = row.get("bar_len_ticks")
        if raw_bar_len is not None:
            try:
                bar_len_values.append(float(raw_bar_len))
            except (TypeError, ValueError):
                pass

    return {
        "total_bars": bar_count,
        "total_pieces": len(piece_counter),
        "missing_piece_id_rows": missing_piece_id,
        "duplicate_piece_bar_index_rows": duplicate_bar_keys,
        "avg_bar_len_ticks": round(sum(bar_len_values) / len(bar_len_values), 3)
        if bar_len_values
        else None,
        "min_bars_per_piece": min(piece_counter.values()) if piece_counter else None,
        "max_bars_per_piece": max(piece_counter.values()) if piece_counter else None,
    }


def audit_dataset(
    events_path: Path,
    *,
    vocab_path: Path | None = None,
    tokens_col: str = "tokens",
) -> dict[str, object]:
    records, tokens = _load_events(events_path, tokens_col=tokens_col)
    vocab = _load_vocab(vocab_path) if vocab_path is not None else None
    eval_metrics = evaluate(tokens, vocab=vocab)
    mel_stats = _mel_interval_stats(tokens)
    piece_stats = _piece_bar_stats(records)

    token_grammar_violations = int(eval_metrics.get("token_grammar_violations") or 0)
    harmonic_mismatches = eval_metrics.get("harm_mismatch_count")
    token_validity = eval_metrics.get("token_validity")
    unknown_token_count = None
    if vocab is not None:
        unknown_token_count = sum(1 for token in tokens if token not in vocab)

    failures: list[str] = []
    if mel_stats["mel_int_out_of_range_count"]:
        failures.append("mel_int_out_of_range")
    if token_grammar_violations:
        failures.append("malformed_voice_events")
    if isinstance(harmonic_mismatches, int) and harmonic_mismatches:
        failures.append("harmonic_metadata_mismatches")
    if unknown_token_count:
        failures.append("unknown_tokens")
    if piece_stats["missing_piece_id_rows"]:
        failures.append("missing_piece_id")
    if piece_stats["duplicate_piece_bar_index_rows"]:
        failures.append("duplicate_piece_bar_index")

    return {
        "ok": not failures,
        "failures": failures,
        "events_path": str(events_path),
        "vocab_path": str(vocab_path) if vocab_path is not None else None,
        "tokens_column": tokens_col,
        "token_count": len(tokens),
        "unknown_token_count": unknown_token_count,
        "token_validity": token_validity,
        **piece_stats,
        **mel_stats,
        "malformed_voice_event_count": token_grammar_violations,
        "harmonic_metadata_mismatch_count": harmonic_mismatches,
        "metrics": eval_metrics,
    }


def _print_summary(report: dict[str, object]) -> None:
    print("=" * 58)
    print("dataset audit")
    print("=" * 58)
    for key in (
        "ok",
        "total_bars",
        "total_pieces",
        "token_count",
        "mel_int_out_of_range_count",
        "malformed_voice_event_count",
        "harmonic_metadata_mismatch_count",
        "unknown_token_count",
    ):
        print(f"  {key:<34}: {report.get(key)}")
    failures = report.get("failures")
    if failures:
        print(f"  failures{'':<26}: {', '.join(str(item) for item in failures)}")
    print("=" * 58)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a processed events parquet/csv before training.")
    parser.add_argument("--events", type=Path, default=Path("data/processed/events.parquet"))
    parser.add_argument("--vocab", type=Path, help="Optional vocab JSON to check token validity.")
    parser.add_argument("--tokens-column", default="tokens")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output report path. Defaults to stats.json next to --events.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the text summary.")
    parser.add_argument("--warn-only", action="store_true", help="Always exit 0 after writing the report.")
    args = parser.parse_args(argv)

    events_path = args.events
    if not events_path.exists():
        raise SystemExit(f"Events file not found: {events_path}")
    if args.vocab is not None and not args.vocab.exists():
        raise SystemExit(f"Vocab file not found: {args.vocab}")

    output_json = args.output_json or events_path.parent / "stats.json"
    report = audit_dataset(events_path, vocab_path=args.vocab, tokens_col=args.tokens_column)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.quiet:
        _print_summary(report)
        print(f"wrote {output_json}")

    if not report["ok"] and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
