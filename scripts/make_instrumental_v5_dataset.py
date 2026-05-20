#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instrumental_v4.data import load_v4_dataset
from src.instrumental_v4.representation import V4Piece
from src.instrumental_v5.data import save_events, save_vocab, summarize_conditioning_coverage
from src.instrumental_v5.representation import (
    V5_EMI_FIELD_NAMES,
    V5_FEATURE_SPECS,
    V5_FIELD_NAMES,
    V5Piece,
    build_v5_piece,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Parquet-first instrumental_v5 EMI-conditioned dataset.")
    parser.add_argument("--v4-dataset", default="data/instrumental_v4/keyboard_overture_cnorm_outer2_v4.json")
    parser.add_argument("--output-dir", default="data/instrumental_v5/keyboard_overture_cnorm_outer2_v5")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--length-slices", type=int, default=8)
    parser.add_argument("--hop-slices", type=int, default=4)
    parser.add_argument("--min-notes", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2604)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pieces, v4_meta = load_v4_dataset(args.v4_dataset)
    if args.limit > 0:
        pieces = pieces[: args.limit]

    summary = build_v5_outputs(
        pieces,
        output_dir=Path(args.output_dir),
        source_dataset=args.v4_dataset,
        source_meta=v4_meta,
        length_slices=args.length_slices,
        hop_slices=args.hop_slices,
        min_notes=args.min_notes,
        val_split=args.val_split,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_v5_outputs(
    pieces: list[V4Piece],
    *,
    output_dir: Path,
    source_dataset: str,
    source_meta: dict[str, Any] | None = None,
    length_slices: int = 8,
    hop_slices: int = 4,
    min_notes: int = 2,
    val_split: float = 0.1,
    seed: int = 2604,
) -> dict[str, Any]:
    if not pieces:
        raise ValueError("no pieces supplied")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_by_piece = split_piece_ids([piece.piece_id for piece in pieces], val_split=val_split, seed=seed)
    v5_pieces = [
        build_v5_piece(
            piece,
            length_slices=length_slices,
            hop_slices=hop_slices,
            min_notes=min_notes,
        )
        for piece in pieces
    ]

    events = pieces_to_events_dataframe(v5_pieces, split_by_piece=split_by_piece)
    events_path = output_dir / "events.parquet"
    save_events(events_path, events)

    vocab_path = output_dir / "vocab.json"
    save_vocab(vocab_path)

    fragment_paths = write_fragment_jsonl(output_dir, v5_pieces, split_by_piece)

    train_piece_ids = sorted(piece_id for piece_id, split in split_by_piece.items() if split == "train")
    val_piece_ids = sorted(piece_id for piece_id, split in split_by_piece.items() if split == "val")
    summary: dict[str, Any] = {
        "source_v4_dataset": source_dataset,
        "source_v4_meta": source_meta or {},
        "piece_count": len(v5_pieces),
        "event_row_count": int(len(events)),
        "fragment_count": int(sum(len(piece.fragments) for piece in v5_pieces)),
        "length_slices": length_slices,
        "hop_slices": hop_slices,
        "min_notes": min_notes,
        "val_split": val_split,
        "seed": seed,
        "field_names": V5_FIELD_NAMES,
        "emi_field_names": V5_EMI_FIELD_NAMES,
        "feature_specs": V5_FEATURE_SPECS,
        "conditioning_coverage": summarize_conditioning_coverage(events),
        "train_piece_ids": train_piece_ids,
        "val_piece_ids": val_piece_ids,
        "events_path": str(events_path),
        "vocab_path": str(vocab_path),
        **{f"{name}_path": str(path) for name, path in fragment_paths.items()},
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    summary["metadata_path"] = str(metadata_path)
    return summary


def split_piece_ids(piece_ids: list[str], *, val_split: float, seed: int) -> dict[str, str]:
    unique_piece_ids = sorted(dict.fromkeys(piece_ids))
    if len(unique_piece_ids) != len(piece_ids):
        raise ValueError("piece ids must be unique for piece-level splitting")
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be >= 0 and < 1")
    if len(unique_piece_ids) < 2 or val_split == 0.0:
        return {piece_id: "train" for piece_id in unique_piece_ids}

    n_val = max(1, int(round(len(unique_piece_ids) * val_split)))
    n_val = min(n_val, len(unique_piece_ids) - 1)
    shuffled = unique_piece_ids[:]
    random.Random(seed).shuffle(shuffled)
    val_ids = set(shuffled[:n_val])
    return {piece_id: ("val" if piece_id in val_ids else "train") for piece_id in unique_piece_ids}


def pieces_to_events_dataframe(pieces: list[V5Piece], *, split_by_piece: dict[str, str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for piece in pieces:
        split = split_by_piece[piece.piece_id]
        for row_index, row in enumerate(piece.rows):
            record: dict[str, object] = {
                "piece_id": piece.piece_id,
                "split": split,
                "source_path": piece.source_path,
                "row_index": row_index,
                "tpq": piece.tpq,
                "grid_ticks": piece.grid_ticks,
                "time_signature": piece.time_signature,
                "key": piece.key,
                "bar_len_ticks": piece.bar_len_ticks,
                "steps_per_bar": piece.steps_per_bar,
            }
            record.update({name: int(value) for name, value in zip(V5_FIELD_NAMES, row)})
            records.append(record)
    if not records:
        return pd.DataFrame(columns=_event_columns())
    return pd.DataFrame.from_records(records, columns=_event_columns())


def write_fragment_jsonl(
    output_dir: Path,
    pieces: list[V5Piece],
    split_by_piece: dict[str, str],
) -> dict[str, Path]:
    all_path = output_dir / "emi_fragments.jsonl"
    train_path = output_dir / "train_emi_fragments.jsonl"
    val_path = output_dir / "val_emi_fragments.jsonl"
    split_handles = {
        "train": train_path.open("w", encoding="utf-8"),
        "val": val_path.open("w", encoding="utf-8"),
    }
    try:
        with all_path.open("w", encoding="utf-8") as all_f:
            for piece in pieces:
                split = split_by_piece[piece.piece_id]
                for fragment in piece.fragments:
                    row = fragment.to_dict()
                    row["split"] = split
                    line = json.dumps(row, sort_keys=True)
                    all_f.write(line + "\n")
                    split_handles[split].write(line + "\n")
    finally:
        for handle in split_handles.values():
            handle.close()
    return {
        "emi_fragments": all_path,
        "train_emi_fragments": train_path,
        "val_emi_fragments": val_path,
    }


def _event_columns() -> list[str]:
    return [
        "piece_id",
        "split",
        "source_path",
        "row_index",
        "tpq",
        "grid_ticks",
        "time_signature",
        "key",
        "bar_len_ticks",
        "steps_per_bar",
        *V5_FIELD_NAMES,
    ]


if __name__ == "__main__":
    main()
