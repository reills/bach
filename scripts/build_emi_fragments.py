#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emi.fragments import extract_fragments, fragment_to_jsonl, summarize_fragments
from src.instrumental_v3.data import load_dataset as load_v3_dataset
from src.instrumental_v3.representation import FIELD_NAMES, InstrumentalV3Piece, SliceEvent
from src.instrumental_v4.data import load_v4_dataset
from src.instrumental_v4.representation import V4Piece


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EMI-style fragment/signature JSONL from instrumental datasets.")
    parser.add_argument("--dataset", default="data/instrumental_v4/keyboard_overture_cnorm_outer2_v4.json")
    parser.add_argument("--format", choices=["v3", "v4"], default="v4")
    parser.add_argument("--output", default="data/emi_fragments/keyboard_overture_cnorm_outer2.fragments.jsonl")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--length-slices", type=int, default=8, help="Fragment window length in fixed-grid slices.")
    parser.add_argument("--hop-slices", type=int, default=4, help="Sliding-window hop in fixed-grid slices.")
    parser.add_argument("--min-notes", type=int, default=2)
    parser.add_argument("--limit-pieces", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pieces = _load_pieces(args.dataset, args.format)
    if args.limit_pieces > 0:
        pieces = pieces[: args.limit_pieces]

    fragments = []
    for piece in pieces:
        fragments.extend(
            extract_fragments(
                piece,
                length_slices=args.length_slices,
                hop_slices=args.hop_slices,
                min_notes=args.min_notes,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for fragment in fragments:
            f.write(fragment_to_jsonl(fragment) + "\n")

    summary = {
        "dataset": args.dataset,
        "format": args.format,
        "length_slices": args.length_slices,
        "hop_slices": args.hop_slices,
        "min_notes": args.min_notes,
        **summarize_fragments(fragments),
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"wrote {output}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _load_pieces(path: str, fmt: str) -> list[InstrumentalV3Piece]:
    if fmt == "v3":
        pieces, _ = load_v3_dataset(path)
        return pieces
    pieces, _ = load_v4_dataset(path)
    return [_v4_to_v3_piece(piece) for piece in pieces]


def _v4_to_v3_piece(piece: V4Piece) -> InstrumentalV3Piece:
    return InstrumentalV3Piece(
        piece_id=piece.piece_id,
        source_path=piece.source_path,
        tpq=piece.tpq,
        grid_ticks=piece.grid_ticks,
        time_signature=piece.time_signature,
        key=piece.key,
        key_pc=piece.key_pc,
        mode=piece.mode,
        bar_len_ticks=piece.bar_len_ticks,
        steps_per_bar=piece.steps_per_bar,
        slices=[SliceEvent(row[: len(FIELD_NAMES)]) for row in piece.rows],
    )


if __name__ == "__main__":
    main()
