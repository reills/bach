#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instrumental_v3.data import load_dataset as load_v3_dataset
from src.instrumental_v4.data import save_v4_dataset
from src.instrumental_v4.representation import build_v4_piece


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build instrumental_v4 planner/generator dataset from v3 slices.")
    parser.add_argument("--v3-dataset", default="data/instrumental_v3/keyboard_overture_cnorm_outer2.json")
    parser.add_argument("--output", default="data/instrumental_v4/keyboard_overture_cnorm_outer2_v4.json")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v3_pieces, v3_meta = load_v3_dataset(args.v3_dataset)
    if args.limit > 0:
        v3_pieces = v3_pieces[: args.limit]
    pieces = [build_v4_piece(piece) for piece in v3_pieces]
    meta = {
        "source_v3_dataset": args.v3_dataset,
        "source_v3_meta": v3_meta,
        "piece_count": len(pieces),
        "total_slice_rows": sum(len(piece.rows) for piece in pieces),
        "total_plan_rows": sum(len(piece.plans) for piece in pieces),
    }
    save_v4_dataset(args.output, pieces, meta=meta)
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({**meta, "pieces": [piece.piece_id for piece in pieces]}, f, indent=2)
    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")
    print(json.dumps({k: v for k, v in meta.items() if k != "source_v3_meta"}, indent=2))


if __name__ == "__main__":
    main()
