from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chorale_v2 import render_v2_tokens_to_midi, v2_repetition_metrics
from src.inference.generate_v1 import GenerationConfig, _generate_from_loaded
from src.models.notelm import load_notelm_checkpoint


def split_tokens(value: object) -> list[str]:
    if isinstance(value, str):
        return [token for token in value.split() if token]
    if isinstance(value, Sequence):
        return [str(token) for token in value if token]
    raise TypeError(f"unsupported token value: {type(value)}")


def tokens_to_bars(tokens: Sequence[str]) -> list[list[str]]:
    bars: list[list[str]] = []
    current: list[str] | None = None
    for token in tokens:
        if token == "BAR":
            if current is not None:
                bars.append(current)
            current = ["BAR"]
        elif current is not None:
            current.append(token)
    if current is not None:
        bars.append(current)
    return bars


def flatten(bars: Sequence[Sequence[str]]) -> list[str]:
    return [token for bar in bars for token in bar]


def run_generation(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.events)
    grouped = df.sort_values(["piece_id", "bar_index"]).groupby("piece_id", sort=True)
    try:
        piece_id, group = next((pid, grp) for pid, grp in grouped if len(grp) >= args.prompt_bars + args.continuation_bars)
    except StopIteration as exc:
        raise SystemExit("no piece has enough bars for the requested continuation") from exc

    rows = list(group.itertuples(index=False))
    prompt_bars = [split_tokens(getattr(row, "tokens")) for row in rows[: args.prompt_bars]]
    reference_bars = [
        split_tokens(getattr(row, "tokens"))
        for row in rows[args.prompt_bars : args.prompt_bars + args.continuation_bars]
    ]
    prompt_tokens = flatten(prompt_bars)
    reference_tokens = flatten(reference_bars)

    loaded = load_notelm_checkpoint(args.checkpoint, vocab_path=args.vocab, device=args.device)
    seed_tokens = [token for token in prompt_tokens if token in loaded.vocab]
    config = GenerationConfig(
        max_length=min(args.max_length, len(seed_tokens) + max(args.min_new_tokens, len(reference_tokens) * args.max_new_multiplier)),
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        use_grammar_mask=False,
        use_voice_leading_mask=False,
        eos_token=args.eos_token,
        use_chorale_v2_mask=True,
        v2_max_sonority_repeats=args.max_sonority_repeats,
    )
    candidates = []
    for attempt in range(max(1, args.attempts)):
        result = _generate_from_loaded(loaded, seed_tokens=seed_tokens, generation_config=config)
        generated_bars = tokens_to_bars(result.tokens)
        generated_continuation_bars = generated_bars[args.prompt_bars : args.prompt_bars + args.continuation_bars]
        generated_continuation_tokens = flatten(generated_continuation_bars)
        metrics = v2_repetition_metrics(generated_continuation_tokens)
        candidates.append(
            {
                "attempt": attempt + 1,
                "result": result,
                "generated_continuation_bars": generated_continuation_bars,
                "generated_continuation_tokens": generated_continuation_tokens,
                "repetition_metrics": metrics,
                "repetition_score": _v2_repetition_score(metrics),
            }
        )

    selected = min(candidates, key=lambda item: float(item["repetition_score"]))
    result = selected["result"]
    generated_continuation_bars = selected["generated_continuation_bars"]
    generated_continuation_tokens = selected["generated_continuation_tokens"]

    prompt_mid = args.out_dir / "prompt.mid"
    generated_mid = args.out_dir / "generated_continuation.mid"
    reference_mid = args.out_dir / "reference_continuation.mid"
    render_v2_tokens_to_midi(prompt_tokens, prompt_mid)
    render_v2_tokens_to_midi(generated_continuation_tokens, generated_mid)
    render_v2_tokens_to_midi(reference_tokens, reference_mid)

    (args.out_dir / "prompt_tokens.txt").write_text(" ".join(prompt_tokens), encoding="utf-8")
    (args.out_dir / "generated_tokens.txt").write_text(" ".join(result.tokens), encoding="utf-8")
    (args.out_dir / "generated_continuation_tokens.txt").write_text(" ".join(generated_continuation_tokens), encoding="utf-8")
    (args.out_dir / "reference_continuation_tokens.txt").write_text(" ".join(reference_tokens), encoding="utf-8")

    summary = {
        "piece_id": str(piece_id),
        "checkpoint": str(args.checkpoint),
        "vocab": str(args.vocab),
        "events": str(args.events),
        "prompt_bars": args.prompt_bars,
        "continuation_bars": args.continuation_bars,
        "seed_token_count": len(seed_tokens),
        "generated_token_count": len(result.tokens),
        "generated_continuation_token_count": len(generated_continuation_tokens),
        "generated_continuation_bar_count": len(generated_continuation_bars),
        "stopped_on_eos": result.stopped_on_eos,
        "attempts": len(candidates),
        "selected_attempt": selected["attempt"],
        "repetition_score": selected["repetition_score"],
        "repetition_metrics": selected["repetition_metrics"],
        "paths": {
            "prompt_midi": str(prompt_mid),
            "generated_continuation_midi": str(generated_mid),
            "reference_continuation_midi": str(reference_mid),
            "prompt_tokens": str(args.out_dir / "prompt_tokens.txt"),
            "generated_tokens": str(args.out_dir / "generated_tokens.txt"),
            "generated_continuation_tokens": str(args.out_dir / "generated_continuation_tokens.txt"),
            "reference_continuation_tokens": str(args.out_dir / "reference_continuation_tokens.txt"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _v2_repetition_score(metrics: dict[str, float | int]) -> float:
    if int(metrics.get("slice_count", 0)) == 0:
        return 1_000_000.0
    adjacent_rate = float(metrics.get("adjacent_repeat_rate", 0.0))
    duplicate_bar_rate = float(metrics.get("duplicate_bar_rate", 0.0))
    unique_rate = float(metrics.get("unique_sonority_rate", 0.0))
    longest_run = int(metrics.get("longest_sonority_run", 0))
    unique_count = int(metrics.get("unique_sonority_count", 0))
    return round(
        adjacent_rate * 100.0
        + duplicate_bar_rate * 80.0
        + max(0, longest_run - 2) * 20.0
        - unique_rate * 25.0
        - min(unique_count, 16),
        6,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a chorale-v2 continuation and MIDI exports.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=Path("data/chorale_v2_overfit_20/events.parquet"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt-bars", type=int, default=2)
    parser.add_argument("--continuation-bars", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=4,
                        help="Generate multiple candidates and keep the least repetitive v2 continuation.")
    parser.add_argument("--max-sonority-repeats", type=int, default=2,
                        help="Hard limit for consecutive identical SATB sonorities during v2 decoding.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-new-tokens", type=int, default=256)
    parser.add_argument("--max-new-multiplier", type=int, default=2)
    parser.add_argument("--eos-token", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_generation(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
