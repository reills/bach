"""Prepare a clean 20-piece four-part chorale subset for overfit tests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


CHORALE_SOURCE_FRAGMENT = "vocal-works/chorales/BWV 253-438 Chorales for four voices"

_VOICE_RE = re.compile(r"^VOICE_(\d+)$")
_TAB_PREFIXES = ("STR_", "FRET_")


def _split_tokens(value: object) -> list[str]:
    if isinstance(value, str):
        return [token for token in value.split() if token]
    if isinstance(value, Sequence):
        return [str(token) for token in value if token]
    raise TypeError(f"unsupported tokens value: {type(value)}")


def _bar_voice_count(tokens: Sequence[str]) -> int:
    voices: set[int] = set()
    for token in tokens:
        match = _VOICE_RE.match(token)
        if match:
            voices.add(int(match.group(1)))
    return len(voices)


def _has_tab_tokens(tokens: Sequence[str]) -> bool:
    return any(token.startswith(_TAB_PREFIXES) for token in tokens)


def _piece_stats(group: pd.DataFrame) -> dict[str, Any]:
    voice_counts: list[int] = []
    tab_bars = 0
    token_count = 0
    for value in group["tokens"]:
        tokens = _split_tokens(value)
        voice_counts.append(_bar_voice_count(tokens))
        token_count += len(tokens)
        if _has_tab_tokens(tokens):
            tab_bars += 1

    bar_count = len(voice_counts)
    return {
        "bars": bar_count,
        "tokens": token_count,
        "avg_voices_per_bar": round(sum(voice_counts) / bar_count, 6) if bar_count else 0.0,
        "pct_bars_3plus_voices": round(
            sum(count >= 3 for count in voice_counts) / bar_count, 6
        ) if bar_count else 0.0,
        "pct_bars_4plus_voices": round(
            sum(count >= 4 for count in voice_counts) / bar_count, 6
        ) if bar_count else 0.0,
        "tab_bars": tab_bars,
    }


def select_clean_pieces(
    events: pd.DataFrame,
    *,
    limit: int,
    min_bars: int,
    min_pct_4plus: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    chorales = events[
        events["source_path"].astype(str).str.contains(
            CHORALE_SOURCE_FRAGMENT,
            regex=False,
            na=False,
        )
    ].copy()
    if chorales.empty:
        raise SystemExit(
            f"no rows found under source fragment: {CHORALE_SOURCE_FRAGMENT!r}"
        )

    rows: list[dict[str, Any]] = []
    grouped = chorales.groupby(["piece_id", "source_path"], sort=True)
    for (piece_id, source_path), group in grouped:
        stats = _piece_stats(group)
        rows.append(
            {
                "piece_id": str(piece_id),
                "source_path": str(source_path),
                **stats,
            }
        )

    candidates = pd.DataFrame(rows)
    candidates = candidates[
        (candidates["bars"] >= min_bars)
        & (candidates["tab_bars"] == 0)
        & (candidates["pct_bars_4plus_voices"] >= min_pct_4plus)
    ].copy()
    if len(candidates) < limit:
        raise SystemExit(
            "not enough clean chorale candidates: "
            f"need {limit}, found {len(candidates)}"
        )

    candidates = candidates.sort_values(
        [
            "pct_bars_4plus_voices",
            "pct_bars_3plus_voices",
            "avg_voices_per_bar",
            "bars",
            "piece_id",
        ],
        ascending=[False, False, False, False, True],
    )
    selected_rows = candidates.head(limit).to_dict(orient="records")
    selected_ids = {row["piece_id"] for row in selected_rows}
    selected_events = chorales[chorales["piece_id"].isin(selected_ids)].copy()
    selected_events = selected_events.sort_values(["piece_id", "bar_index"])
    return selected_events, selected_rows


def _write_optional_barplans(
    source_barplans: Path | None,
    output_dir: Path,
    selected_piece_ids: set[str],
) -> None:
    if source_barplans is None or not source_barplans.exists():
        return
    barplans = pd.read_parquet(source_barplans)
    if "piece_id" not in barplans.columns:
        return
    selected = barplans[barplans["piece_id"].isin(selected_piece_ids)].copy()
    selected = selected.sort_values(["piece_id", "bar_index"])
    selected.to_parquet(output_dir / "barplans.parquet", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a clean 20-piece four-part chorale overfit subset."
    )
    parser.add_argument("--events", type=Path, default=Path("data/processed_rebuilt/events.parquet"))
    parser.add_argument("--barplans", type=Path, default=Path("data/processed_rebuilt/barplans.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/overfit_20_chorales"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-bars", type=int, default=8)
    parser.add_argument("--min-pct-4plus", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    events = pd.read_parquet(args.events)
    required = {"piece_id", "source_path", "bar_index", "tokens"}
    missing = required - set(events.columns)
    if missing:
        raise SystemExit(f"events file missing columns: {sorted(missing)}")

    selected_events, selected_rows = select_clean_pieces(
        events,
        limit=args.limit,
        min_bars=args.min_bars,
        min_pct_4plus=args.min_pct_4plus,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    selected_events.to_parquet(args.output / "events.parquet", index=False)
    selected_piece_ids = {row["piece_id"] for row in selected_rows}
    _write_optional_barplans(args.barplans, args.output, selected_piece_ids)

    stats = {
        "source_events": str(args.events),
        "source_fragment": CHORALE_SOURCE_FRAGMENT,
        "total_pieces": len(selected_piece_ids),
        "total_bars": int(len(selected_events)),
        "total_tokens": int(sum(len(_split_tokens(value)) for value in selected_events["tokens"])),
        "selection": {
            "limit": args.limit,
            "min_bars": args.min_bars,
            "min_pct_4plus": args.min_pct_4plus,
        },
        "pieces": selected_rows,
    }
    (args.output / "selected_pieces.json").write_text(
        json.dumps(selected_rows, indent=2),
        encoding="utf-8",
    )
    (args.output / "stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    print(
        f"wrote {len(selected_piece_ids)} pieces, {len(selected_events)} bars "
        f"to {args.output / 'events.parquet'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
