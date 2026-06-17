#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instrumental_v3.data import save_dataset
from src.instrumental_v3.metrics import evaluate_slices
from src.instrumental_v3.representation import parse_musicxml_to_piece

DEFAULT_INVENTIONS = ROOT / "data/tobis_xml/instrumental-works/keyboard-works/BWV 772-786 Inventions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build instrumental_v3 fixed-grid invention dataset.")
    parser.add_argument("--source-dir", default=str(DEFAULT_INVENTIONS))
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=None,
        help="One or more source directories. Overrides --source-dir.",
    )
    parser.add_argument("--output", default="data/instrumental_v3/inventions_tiny.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-bars", type=int, default=32)
    parser.add_argument("--min-slices", type=int, default=64)
    parser.add_argument("--max-empty-rate", type=float, default=0.10)
    parser.add_argument("--max-crossing-rate", type=float, default=0.08)
    parser.add_argument(
        "--normalize-key",
        action="store_true",
        help="Transpose each piece to C/C minor before slicing so patterns share a tonal frame.",
    )
    parser.add_argument("--summary", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dirs = [Path(path) for path in (args.source_dirs or [args.source_dir])]
    paths = []
    for source_dir in source_dirs:
        paths.extend(source_dir.glob("**/*.xml"))
        paths.extend(source_dir.glob("**/*.musicxml"))
    paths = sorted(set(paths))
    if args.limit > 0:
        paths = paths[: args.limit]
    pieces = []
    skipped = []
    for path in paths:
        try:
            piece = parse_musicxml_to_piece(
                path,
                max_bars=args.max_bars,
                normalize_key=args.normalize_key,
            )
        except Exception as exc:
            skipped.append({"path": str(path), "reason": "parse_error", "error": str(exc)})
            continue
        report = evaluate_slices(piece.slices)
        if len(piece.slices) < args.min_slices:
            skipped.append({"path": str(path), "reason": "too_short", "slices": len(piece.slices)})
            continue
        if report.empty_slice_rate > args.max_empty_rate:
            skipped.append(
                {
                    "path": str(path),
                    "reason": "empty_rate",
                    "empty_rate": report.empty_slice_rate,
                }
            )
            continue
        if report.voice_crossing_rate > args.max_crossing_rate:
            skipped.append(
                {
                    "path": str(path),
                    "reason": "crossing_rate",
                    "crossing_rate": report.voice_crossing_rate,
                }
            )
            continue
        pieces.append(piece)
        print(
            f"ok {piece.piece_id}: slices={len(piece.slices)} key={piece.key} "
            f"empty={report.empty_slice_rate:.3f} crossing={report.voice_crossing_rate:.3f}"
        )
    if not pieces:
        raise SystemExit("no pieces were parsed")

    meta = {
        "source_dirs": [str(path) for path in source_dirs],
        "piece_count": len(pieces),
        "normalize_key": bool(args.normalize_key),
        "skipped": skipped,
        "target": "Bach two-part inventions, instrumental_v3 continuation-first",
    }
    save_dataset(args.output, pieces, meta=meta)
    summary_path = Path(args.summary) if args.summary else Path(args.output).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({**meta, "pieces": [p.piece_id for p in pieces]}, f, indent=2)
    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")
    if skipped:
        print(f"skipped {len(skipped)} files")


if __name__ == "__main__":
    main()
