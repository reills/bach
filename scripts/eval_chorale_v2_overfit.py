from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_chorale_v2 import flatten, split_tokens, tokens_to_bars, _v2_repetition_score
from src.chorale_v2 import parse_v2_slices, render_v2_tokens_to_midi, v2_repetition_metrics
from src.inference.generate_v1 import GenerationConfig, _generate_from_loaded
from src.models.notelm import load_notelm_checkpoint


def run_eval(args: argparse.Namespace) -> dict[str, object]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.events)
    if args.source_contains:
        if "source_path" not in df.columns:
            raise SystemExit("--source-contains requires source_path column in events")
        df = df[df["source_path"].astype(str).str.contains(args.source_contains, regex=False, na=False)].copy()
    if args.piece_contains:
        df = df[df["piece_id"].astype(str).str.contains(args.piece_contains, regex=False, na=False)].copy()
    if df.empty:
        raise SystemExit("no rows left after applying filters")
    loaded = load_notelm_checkpoint(args.checkpoint, vocab_path=args.vocab, device=args.device)

    samples: list[dict[str, object]] = []
    grouped = df.sort_values(["piece_id", "bar_index"]).groupby("piece_id", sort=True)
    eligible: list[tuple[str, pd.DataFrame]] = []
    for piece_id, group in grouped:
        if len(group) < args.prompt_bars + args.continuation_bars:
            continue
        eligible.append((str(piece_id), group))
    if not eligible:
        raise SystemExit("no eligible pieces evaluated")
    if args.shuffle_pieces:
        rng = random.Random(args.seed)
        rng.shuffle(eligible)

    for piece_id, group in eligible:
        if args.samples and len(samples) >= args.samples:
            break

        rows = list(group.itertuples(index=False))
        prompt_bars = [split_tokens(getattr(row, "tokens")) for row in rows[: args.prompt_bars]]
        reference_bars = [
            split_tokens(getattr(row, "tokens"))
            for row in rows[args.prompt_bars : args.prompt_bars + args.continuation_bars]
        ]
        prompt_tokens = flatten(prompt_bars)
        reference_tokens = flatten(reference_bars)
        seed_tokens = [token for token in prompt_tokens if token in loaded.vocab]
        config = GenerationConfig(
            max_length=min(args.max_length, len(seed_tokens) + max(args.min_new_tokens, len(reference_tokens) * args.max_new_multiplier)),
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            use_grammar_mask=False,
            use_voice_leading_mask=False,
            use_chorale_v2_mask=True,
            v2_max_sonority_repeats=args.max_sonority_repeats,
        )
        candidates = []
        for attempt in range(max(1, args.attempts)):
            result = _generate_from_loaded(loaded, seed_tokens=seed_tokens, generation_config=config)
            generated_bars = tokens_to_bars(result.tokens)
            generated_continuation_tokens = flatten(
                generated_bars[args.prompt_bars : args.prompt_bars + args.continuation_bars]
            )
            metrics = v2_repetition_metrics(generated_continuation_tokens)
            candidates.append(
                {
                    "attempt": attempt + 1,
                    "tokens": generated_continuation_tokens,
                    "repetition_metrics": metrics,
                    "repetition_score": _v2_repetition_score(metrics),
                }
            )
        selected = min(candidates, key=lambda item: float(item["repetition_score"]))
        generated_continuation_tokens = selected["tokens"]
        sample_dir = args.out_dir / f"sample_{len(samples) + 1:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        render_v2_tokens_to_midi(prompt_tokens, sample_dir / "prompt.mid")
        render_v2_tokens_to_midi(generated_continuation_tokens, sample_dir / "generated_continuation.mid")
        render_v2_tokens_to_midi(reference_tokens, sample_dir / "reference_continuation.mid")
        (sample_dir / "generated_continuation_tokens.txt").write_text(
            " ".join(generated_continuation_tokens), encoding="utf-8"
        )
        (sample_dir / "reference_continuation_tokens.txt").write_text(
            " ".join(reference_tokens), encoding="utf-8"
        )

        match_rate = _token_match_rate(generated_continuation_tokens, reference_tokens)
        sample = {
            "index": len(samples) + 1,
            "piece_id": str(piece_id),
            "prompt_bars": args.prompt_bars,
            "continuation_bars": args.continuation_bars,
            "generated_continuation_bar_count": len(tokens_to_bars(generated_continuation_tokens)),
            "generated_slice_count": len(parse_v2_slices(generated_continuation_tokens)),
            "reference_slice_count": len(parse_v2_slices(reference_tokens)),
            "token_match_rate": match_rate,
            "attempts": len(candidates),
            "selected_attempt": selected["attempt"],
            "repetition_score": selected["repetition_score"],
            "repetition_metrics": selected["repetition_metrics"],
            "paths": {
                "sample_dir": str(sample_dir),
                "prompt_midi": str(sample_dir / "prompt.mid"),
                "generated_continuation_midi": str(sample_dir / "generated_continuation.mid"),
                "reference_continuation_midi": str(sample_dir / "reference_continuation.mid"),
                "generated_continuation_tokens": str(sample_dir / "generated_continuation_tokens.txt"),
                "reference_continuation_tokens": str(sample_dir / "reference_continuation_tokens.txt"),
            },
        }
        (sample_dir / "summary.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
        samples.append(sample)

    if not samples:
        raise SystemExit("no eligible pieces evaluated")

    summary = {
        "checkpoint": str(args.checkpoint),
        "vocab": str(args.vocab),
        "events": str(args.events),
        "sample_count": len(samples),
        "token_match_rate_avg": _mean([sample["token_match_rate"] for sample in samples]),
        "generated_slice_count_avg": _mean([sample["generated_slice_count"] for sample in samples]),
        "repetition_score_avg": _mean([sample["repetition_score"] for sample in samples]),
        "adjacent_repeat_rate_avg": _mean([
            sample["repetition_metrics"]["adjacent_repeat_rate"] for sample in samples
        ]),
        "duplicate_bar_rate_avg": _mean([
            sample["repetition_metrics"]["duplicate_bar_rate"] for sample in samples
        ]),
        "samples": samples,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _token_match_rate(generated: Sequence[str], reference: Sequence[str]) -> float | None:
    limit = min(len(generated), len(reference))
    if limit == 0:
        return None
    return round(sum(generated[i] == reference[i] for i in range(limit)) / limit, 6)


def _mean(values: Sequence[object]) -> float | None:
    nums = [float(value) for value in values if isinstance(value, (int, float))]
    return round(statistics.mean(nums), 6) if nums else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate chorale-v2 overfit continuations.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=Path("data/chorale_v2_overfit_20/events.parquet"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--source-contains", default=None,
                        help="Optional substring filter on source_path before sampling pieces.")
    parser.add_argument("--piece-contains", default=None,
                        help="Optional substring filter on piece_id before sampling pieces.")
    parser.add_argument("--shuffle-pieces", action="store_true",
                        help="Shuffle eligible pieces before taking --samples.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--prompt-bars", type=int, default=2)
    parser.add_argument("--continuation-bars", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=4,
                        help="Generate multiple candidates and keep the least repetitive v2 continuation.")
    parser.add_argument("--max-sonority-repeats", type=int, default=2,
                        help="Hard limit for consecutive identical SATB sonorities during v2 decoding.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-new-tokens", type=int, default=256)
    parser.add_argument("--max-new-multiplier", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_eval(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
