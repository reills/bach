#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emi.fragments import fragment_to_jsonl
from src.emi.v6_fragments import extract_v6_fragments, summarize_v6_fragments
from src.instrumental_v6.data import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build EMI-style signature fragments from an instrumental_v6 dataset."
    )
    parser.add_argument("--data-dir", default="data/instrumental_v6/clean_bach_long_v1")
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--length-slices", type=int, default=8)
    parser.add_argument("--hop-slices", type=int, default=4)
    parser.add_argument("--min-notes", type=int, default=2)
    parser.add_argument("--limit-pieces", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    pieces, metadata = load_dataset(data_dir / "pieces.json")
    if args.limit_pieces > 0:
        pieces = pieces[: args.limit_pieces]

    fragments = []
    for piece in pieces:
        fragments.extend(
            extract_v6_fragments(
                piece,
                length_slices=args.length_slices,
                hop_slices=args.hop_slices,
                min_notes=args.min_notes,
            )
        )

    output = (
        Path(args.output)
        if args.output
        else data_dir / "emi_v6_fragments.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for fragment in fragments:
            handle.write(fragment_to_jsonl(fragment) + "\n")

    summary = {
        "data_dir": str(data_dir),
        "dataset_piece_count": int(metadata.get("piece_count", len(pieces))),
        "used_piece_count": len(pieces),
        "length_slices": args.length_slices,
        "hop_slices": args.hop_slices,
        "min_notes": args.min_notes,
        **summarize_v6_fragments(fragments),
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
