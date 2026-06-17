from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chorale_v2 import build_v2_bars_from_v1_rows, build_vocab


def build_dataset(args: argparse.Namespace) -> dict[str, object]:
    df = pd.read_parquet(args.events)
    required = {"piece_id", "bar_index", "tokens"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing required columns in {args.events}: {sorted(missing)}")
    if args.source_contains:
        if "source_path" not in df.columns:
            raise SystemExit("--source-contains requires source_path column in events")
        df = df[df["source_path"].astype(str).str.contains(args.source_contains, regex=False, na=False)].copy()
    if args.piece_contains:
        df = df[df["piece_id"].astype(str).str.contains(args.piece_contains, regex=False, na=False)].copy()
    if df.empty:
        raise SystemExit("no rows left after applying filters")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    skipped: list[str] = []

    grouped = df.sort_values(["piece_id", "bar_index"]).groupby("piece_id", sort=True)
    groups = [(str(piece_id), group) for piece_id, group in grouped]
    if args.shuffle_pieces:
        rng = random.Random(args.seed)
        rng.shuffle(groups)

    produced_piece_count = 0
    for piece_id, group in groups:
        if args.limit_pieces and produced_piece_count >= args.limit_pieces:
            break
        bars = build_v2_bars_from_v1_rows(list(group.to_dict("records")))
        if not bars:
            skipped.append(str(piece_id))
            continue
        if args.min_bars and len(bars) < args.min_bars:
            skipped.append(str(piece_id))
            continue
        produced_piece_count += 1
        rows.extend(
            {
                "piece_id": bar.piece_id,
                "source_path": bar.source_path,
                "source_sha256": bar.source_sha256,
                "bar_index": bar.bar_index,
                "tokens": " ".join(bar.tokens),
                "plan_json": bar.plan_json,
                "bar_len_ticks": bar.bar_len_ticks,
            }
            for bar in bars
        )

    if not rows:
        raise SystemExit("no v2 bars were produced")

    out_df = pd.DataFrame(rows)
    events_out = args.output_dir / "events.parquet"
    vocab_out = args.output_dir / "vocab.json"
    summary_out = args.output_dir / "summary.json"
    out_df.to_parquet(events_out, index=False)

    vocab = build_vocab(out_df["tokens"].tolist())
    vocab_out.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    summary = {
        "source_events": str(args.events),
        "events": str(events_out),
        "vocab": str(vocab_out),
        "piece_count": int(out_df["piece_id"].nunique()),
        "bar_count": int(len(out_df)),
        "vocab_size": len(vocab),
        "skipped_piece_count": len(skipped),
        "skipped_pieces": skipped,
        "contains_harm_tokens": any(token.startswith("HARM_") for token in vocab),
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build experimental chorale-v2 vertical SATB dataset.")
    parser.add_argument("--events", type=Path, default=Path("data/overfit_20_chorales/events.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/chorale_v2_overfit_20"))
    parser.add_argument("--limit-pieces", type=int, default=20)
    parser.add_argument("--min-bars", type=int, default=0)
    parser.add_argument("--source-contains", default=None,
                        help="Optional substring filter on source_path before selecting pieces.")
    parser.add_argument("--piece-contains", default=None,
                        help="Optional substring filter on piece_id before selecting pieces.")
    parser.add_argument("--shuffle-pieces", action="store_true",
                        help="Shuffle eligible pieces before applying --limit-pieces.")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
