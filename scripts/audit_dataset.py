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

from src.tokens.tokenizer import parse_voice_event  # noqa: E402
from src.tokens.validator import validate_harm_tokens  # noqa: E402


_MEL_INT_RE = re.compile(r"^MEL_INT12_([+-]?\d+)$")
_DUR_RE = re.compile(r"^DUR_(\d+)$")
_REST_RE = re.compile(r"^REST_(\d+)$")
_VOICE_RE = re.compile(r"^VOICE_(\d+)$")
_POS_RE = re.compile(r"^POS_(\d+)$")


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


def _tokens_from_row(row: dict[str, object], *, tokens_col: str) -> list[str]:
    return list(_iter_tokens(row[tokens_col]))


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


def _basic_token_metrics(tokens: Sequence[str], *, vocab: set[str] | None) -> dict[str, object]:
    bar_count = 0
    dur_count = 0
    rest_count = 0
    unknown_count = 0
    total = len(tokens)

    for token in tokens:
        if vocab is not None and token not in vocab:
            unknown_count += 1
        if token == "BAR":
            bar_count += 1
        elif _DUR_RE.match(token):
            dur_count += 1
        elif _REST_RE.match(token):
            rest_count += 1

    return {
        "token_count": total,
        "bar_count": bar_count,
        "token_validity": round((total - unknown_count) / total, 4)
        if vocab is not None and total > 0
        else None,
        "unknown_token_count": unknown_count if vocab is not None else None,
        "voice_event_count": dur_count,
        "rest_event_count": rest_count,
    }


def _count_grammar_violations(tokens: Sequence[str]) -> int:
    violations = 0
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("VOICE_"):
            try:
                _, idx = parse_voice_event(tokens, idx)
            except ValueError:
                violations += 1
                idx += 1
            continue
        idx += 1
    return violations


def _polyphony_stats_from_rows(records: Sequence[dict[str, object]], *, tokens_col: str) -> dict[str, object]:
    voices_per_bar: list[int] = []
    notes_per_onset: list[int] = []

    for row in records:
        bar_tokens = _tokens_from_row(row, tokens_col=tokens_col)
        voices_in_bar: set[int] = set()
        notes_at_current_pos = 0
        saw_pos = False
        for token in bar_tokens:
            voice_match = _VOICE_RE.match(token)
            if voice_match:
                voices_in_bar.add(int(voice_match.group(1)))
                notes_at_current_pos += 1
            elif _POS_RE.match(token):
                if saw_pos and notes_at_current_pos > 0:
                    notes_per_onset.append(notes_at_current_pos)
                notes_at_current_pos = 0
                saw_pos = True
        if saw_pos and notes_at_current_pos > 0:
            notes_per_onset.append(notes_at_current_pos)
        voices_per_bar.append(len(voices_in_bar))

    total_bars = len(voices_per_bar)
    if total_bars == 0:
        return {
            "avg_voices_per_bar": None,
            "avg_notes_per_onset": None,
            "pct_bars_2plus_voices": None,
            "pct_bars_3plus_voices": None,
        }

    bars_2plus = sum(1 for count in voices_per_bar if count >= 2)
    bars_3plus = sum(1 for count in voices_per_bar if count >= 3)
    return {
        "avg_voices_per_bar": round(sum(voices_per_bar) / total_bars, 3),
        "avg_notes_per_onset": round(sum(notes_per_onset) / len(notes_per_onset), 3)
        if notes_per_onset
        else 0.0,
        "pct_bars_2plus_voices": round(100.0 * bars_2plus / total_bars, 2),
        "pct_bars_3plus_voices": round(100.0 * bars_3plus / total_bars, 2),
    }


def _harmonic_mismatch_stats(
    records: Sequence[dict[str, object]],
    *,
    tokens_col: str,
    progress_every: int = 100,
) -> dict[str, object]:
    rows_by_piece: dict[str, list[dict[str, object]]] = {}
    for row in records:
        piece_id = str(row.get("piece_id") or "")
        rows_by_piece.setdefault(piece_id, []).append(row)

    mismatch_count = 0
    missing_anchor_count = 0
    examples: list[str] = []
    total_pieces = len(rows_by_piece)

    for piece_idx, (piece_id, rows) in enumerate(sorted(rows_by_piece.items()), start=1):
        rows = sorted(rows, key=lambda row: int(row.get("bar_index") or 0))
        piece_tokens: list[str] = []
        for row in rows:
            piece_tokens.extend(_tokens_from_row(row, tokens_col=tokens_col))

        errors = validate_harm_tokens(piece_tokens)
        mismatch_count += len(errors)
        missing_anchor_count += sum("has no previous pitch anchor" in error for error in errors)
        for error in errors[: max(0, 10 - len(examples))]:
            examples.append(f"{piece_id}: {error}")

        if progress_every > 0 and (piece_idx % progress_every == 0 or piece_idx == total_pieces):
            print(
                f"audited harmonic metadata for {piece_idx}/{total_pieces} pieces "
                f"(mismatches={mismatch_count})",
                flush=True,
            )

    return {
        "harmonic_metadata_mismatch_count": mismatch_count,
        "harmonic_missing_anchor_count": missing_anchor_count,
        "harmonic_metadata_mismatch_examples": examples,
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
    progress_every: int = 100,
) -> dict[str, object]:
    print(f"loading events: {events_path}", flush=True)
    records, tokens = _load_events(events_path, tokens_col=tokens_col)
    vocab = _load_vocab(vocab_path) if vocab_path is not None else None
    print(f"loaded {len(records)} rows and {len(tokens)} tokens", flush=True)

    basic_metrics = _basic_token_metrics(tokens, vocab=vocab)
    mel_stats = _mel_interval_stats(tokens)
    piece_stats = _piece_bar_stats(records)
    polyphony_stats = _polyphony_stats_from_rows(records, tokens_col=tokens_col)
    print("finished interval, vocab, and polyphony checks", flush=True)

    token_grammar_violations = _count_grammar_violations(tokens)
    print(f"finished grammar checks (violations={token_grammar_violations})", flush=True)

    harm_stats = _harmonic_mismatch_stats(
        records,
        tokens_col=tokens_col,
        progress_every=progress_every,
    )
    harmonic_mismatches = harm_stats["harmonic_metadata_mismatch_count"]
    unknown_token_count = basic_metrics["unknown_token_count"]

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
        **basic_metrics,
        **piece_stats,
        **mel_stats,
        "malformed_voice_event_count": token_grammar_violations,
        **harm_stats,
        **polyphony_stats,
        "metrics": {
            **basic_metrics,
            **mel_stats,
            **polyphony_stats,
            "interval_range_ok": mel_stats["mel_int_out_of_range_count"] == 0,
            "mel_int_range": [
                mel_stats["mel_int_min"],
                mel_stats["mel_int_max"],
            ]
            if mel_stats["mel_int_min"] is not None
            else None,
            "token_grammar_violations": token_grammar_violations,
            "harm_mismatch_count": harmonic_mismatches,
        },
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
    parser.add_argument("--progress-every", type=int, default=100)
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
    report = audit_dataset(
        events_path,
        vocab_path=args.vocab,
        tokens_col=args.tokens_column,
        progress_every=args.progress_every,
    )
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
