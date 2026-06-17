#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instrumental_v5.tokenize import (
    build_tokenized_split,
    load_v5_vocab,
    save_tokenized_split,
    validate_events,
    write_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-tokenize v5 events.parquet into fixed-length tensor windows.")
    parser.add_argument("--events", required=True, help="Canonical v5 events.parquet path.")
    parser.add_argument("--vocab", required=True, help="v5 vocab.json path.")
    parser.add_argument("--output-dir", required=True, help="Output tokenized directory.")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=0, help="Window stride; default is seq_len // 2.")
    parser.add_argument("--drop-short", action="store_true", help="Drop pieces shorter than seq_len instead of padding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events_path = Path(args.events)
    vocab_path = Path(args.vocab)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_v5_vocab(vocab_path)
    events = pd.read_parquet(events_path)
    validate_events(events)

    stride = args.stride if args.stride > 0 else None
    train = build_tokenized_split(
        events,
        split="train",
        seq_len=args.seq_len,
        stride=stride,
        drop_short=args.drop_short,
    )
    val = None
    if "val" in set(events["split"]):
        try:
            val = build_tokenized_split(
                events,
                split="val",
                seq_len=args.seq_len,
                stride=stride,
                drop_short=args.drop_short,
            )
        except ValueError:
            val = None

    train_path = output_dir / "train.pt"
    save_tokenized_split(train_path, train)
    val_path = None
    if val is not None:
        val_path = output_dir / "val.pt"
        save_tokenized_split(val_path, val)

    metadata = {
        "events_path": str(events_path),
        "vocab_path": str(vocab_path),
        "seq_len": args.seq_len,
        "stride": args.stride if args.stride > 0 else max(1, args.seq_len // 2),
        "drop_short": bool(args.drop_short),
        "field_names": vocab["field_names"],
        "feature_specs": vocab["feature_specs"],
        "train_path": str(train_path),
        "train_windows": int(train.windows.shape[0]),
        "train_pieces": sorted(set(train.piece_ids)),
        "val_path": str(val_path) if val_path else None,
        "val_windows": int(val.windows.shape[0]) if val is not None else 0,
        "val_pieces": sorted(set(val.piece_ids)) if val is not None else [],
    }
    write_metadata(output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
