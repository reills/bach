import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List


def _iter_tokens(values: Iterable[object]) -> Iterable[str]:
    for value in values:
        if isinstance(value, str):
            for token in value.split():
                if token:
                    yield token
        elif isinstance(value, (list, tuple)):
            for token in value:
                if token:
                    yield str(token)


def _load_token_column(events_path: Path, column: str) -> List[object]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to read events; install pandas and a parquet engine."
        ) from exc

    if events_path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(events_path, columns=[column])
        except Exception as exc:
            raise RuntimeError(
                "Failed to read parquet. Install pyarrow/fastparquet or provide a CSV."
            ) from exc
    else:
        df = pd.read_csv(events_path, usecols=[column])

    if column not in df.columns:
        raise RuntimeError(f"Column '{column}' not found in {events_path}")

    return df[column].tolist()


def build_vocab(values: Iterable[object], keep_order: bool) -> List[str]:
    seen = OrderedDict()
    for token in _iter_tokens(values):
        if token not in seen:
            seen[token] = None
    tokens = list(seen.keys())
    if not keep_order:
        tokens = sorted(tokens)
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a vocabulary mapping from an events parquet/csv."
    )
    parser.add_argument(
        "--events",
        default="data/processed/events.parquet",
        help="Path to events.parquet or events.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output vocab.json (default: alongside the events file).",
    )
    parser.add_argument(
        "--column",
        default="tokens",
        help="Column containing token strings.",
    )
    parser.add_argument(
        "--keep-order",
        action="store_true",
        help="Preserve first-seen token order instead of sorting.",
    )
    parser.add_argument(
        "--special-tokens",
        default="",
        help="Comma-separated list of special tokens to prepend.",
    )
    args = parser.parse_args()

    events_path = Path(args.events)
    if not events_path.exists():
        raise SystemExit(f"Events file not found: {events_path}")

    output_path = Path(args.output) if args.output else events_path.parent / "vocab.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = _load_token_column(events_path, args.column)
    tokens = build_vocab(values, keep_order=args.keep_order)

    special_tokens = [t.strip() for t in args.special_tokens.split(",") if t.strip()]
    if special_tokens:
        special_set = set(special_tokens)
        tokens = special_tokens + [t for t in tokens if t not in special_set]

    vocab = {token: idx for idx, token in enumerate(tokens)}

    with output_path.open("w") as f:
        json.dump(vocab, f, indent=2)

    print(f"Saved vocab with {len(vocab)} tokens to {output_path}")


if __name__ == "__main__":
    main()
