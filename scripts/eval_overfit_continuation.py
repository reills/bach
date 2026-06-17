"""Evaluate overfit continuation from the first bars of training chorales."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataio.collate_miditok import PrefixControlConfig, build_prefix_tokens
from src.inference.generate_v1 import GenerationConfig, _generate_from_loaded
from src.models.notelm import load_notelm_checkpoint
from src.tokens.schema import BarPlan


def _load_eval_basic():
    spec = importlib.util.spec_from_file_location(
        "eval_basic", ROOT / "scripts" / "eval_basic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_tokens(value: object) -> list[str]:
    if isinstance(value, str):
        return [token for token in value.split() if token]
    if isinstance(value, Sequence):
        return [str(token) for token in value if token]
    raise TypeError(f"unsupported tokens value: {type(value)}")


def _parse_plan(value: object, bar_index: int) -> BarPlan | None:
    if value is None or pd.isna(value):
        return None
    data = json.loads(value) if isinstance(value, str) else dict(value)
    data.setdefault("bar_index", bar_index)
    return BarPlan(**data)


def _bar_count(tokens: Sequence[str]) -> int:
    return sum(1 for token in tokens if token == "BAR")


def _tokens_to_bars(tokens: Sequence[str]) -> list[list[str]]:
    bars: list[list[str]] = []
    current: list[str] | None = None
    for token in tokens:
        if token == "BAR":
            if current is not None:
                bars.append(current)
            current = ["BAR"]
            continue
        if current is not None:
            current.append(token)
    if current is not None:
        bars.append(current)
    return bars


def _flatten(bars: Sequence[Sequence[str]]) -> list[str]:
    return [token for bar in bars for token in bar]


def _token_match_rate(generated: Sequence[str], reference: Sequence[str]) -> float | None:
    limit = min(len(generated), len(reference))
    if limit == 0:
        return None
    return round(sum(generated[i] == reference[i] for i in range(limit)) / limit, 6)


def _exact_bar_matches(generated_bars: Sequence[Sequence[str]], reference_bars: Sequence[Sequence[str]]) -> int:
    return sum(
        1
        for generated, reference in zip(generated_bars, reference_bars)
        if list(generated) == list(reference)
    )


def _piece_prefix_tokens(
    plan: BarPlan | None,
    total_bars: int,
    *,
    key_from_plan: bool,
    key: str | None,
    style: str | None,
    difficulty: str | None,
    measures_token_prefix: str,
) -> list[str]:
    return build_prefix_tokens(
        plan,
        total_bars,
        PrefixControlConfig(
            style=style,
            difficulty=difficulty,
            measures=total_bars,
            measures_token_prefix=measures_token_prefix,
            key_from_plan=key_from_plan,
            key_override=key,
        ),
    )


def _summarize(values: Sequence[Any]) -> dict[str, float | int | None]:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numeric:
        return {"avg": None, "min": None, "max": None}
    return {"avg": round(mean(numeric), 6), "min": min(numeric), "max": max(numeric)}


def _pass_flags(
    metrics: dict[str, Any],
    token_match_rate: float | None,
    *,
    generated_continuation_bar_count: int,
    requested_continuation_bars: int,
) -> dict[str, bool]:
    avg_voices = metrics.get("avg_voices_per_bar")
    pct_3plus = metrics.get("pct_bars_3plus_voices")
    duplicate_rate = metrics.get("duplicate_bar_rate")
    cadence_rate = metrics.get("cadence_proxy_rate")
    grammar_violations = metrics.get("token_grammar_violations")
    return {
        "generates_requested_bars": generated_continuation_bar_count >= requested_continuation_bars,
        "recognizable_token_overlap": bool(
            token_match_rate is not None
            and token_match_rate >= 0.25
            and generated_continuation_bar_count >= requested_continuation_bars
        ),
        "keeps_4_voice_texture": bool(
            isinstance(avg_voices, (int, float))
            and avg_voices >= 3.5
            and isinstance(pct_3plus, (int, float))
            and pct_3plus >= 90.0
        ),
        "cadence_proxy": bool(isinstance(cadence_rate, (int, float)) and cadence_rate >= 0.5),
        "avoids_obvious_repetition": bool(
            isinstance(duplicate_rate, (int, float))
            and duplicate_rate <= 0.25
            and grammar_violations == 0
        ),
    }


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.events)
    loaded = load_notelm_checkpoint(args.checkpoint, vocab_path=args.vocab, device=args.device)
    eval_basic = _load_eval_basic()
    vocab_set = set(loaded.vocab)

    base_generation_config = GenerationConfig(
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        use_scg=args.use_scg,
        use_grammar_mask=args.use_grammar_mask,
        target_texture=args.texture,
        alpha=args.alpha,
        gamma=args.gamma,
        eos_token=args.eos_token,
    )
    base_generation_config_data = asdict(base_generation_config)

    samples: list[dict[str, Any]] = []
    grouped = df.sort_values(["piece_id", "bar_index"]).groupby("piece_id", sort=True)
    for sample_index, (piece_id, group) in enumerate(grouped, start=1):
        if args.samples and sample_index > args.samples:
            break
        if len(group) < args.prompt_bars + args.continuation_bars:
            continue

        rows = list(group.itertuples(index=False))
        first = rows[0]
        first_plan = (
            _parse_plan(getattr(first, "plan_json"), int(getattr(first, "bar_index")))
            if hasattr(first, "plan_json")
            else None
        )
        total_bars = args.prompt_bars + args.continuation_bars
        prefix = _piece_prefix_tokens(
            first_plan,
            total_bars,
            key_from_plan=args.key_from_plan,
            key=args.key,
            style=args.style,
            difficulty=args.difficulty,
            measures_token_prefix=args.measures_token_prefix,
        )
        prompt_bars = [_split_tokens(getattr(row, "tokens")) for row in rows[: args.prompt_bars]]
        reference_bars = [
            _split_tokens(getattr(row, "tokens"))
            for row in rows[args.prompt_bars : args.prompt_bars + args.continuation_bars]
        ]
        prompt_tokens = _flatten(prompt_bars)
        reference_tokens = _flatten(reference_bars)

        seed_tokens = prefix + prompt_tokens
        max_length = min(
            args.max_length,
            len(seed_tokens) + max(args.min_new_tokens, int(len(reference_tokens) * args.max_new_multiplier)),
        )
        generation_config = GenerationConfig(
            **{**base_generation_config_data, "max_length": max_length}
        )
        result = _generate_from_loaded(
            loaded,
            seed_tokens=seed_tokens,
            generation_config=generation_config,
        )

        generated_piece_tokens = result.tokens[len(prefix) :]
        generated_new_tokens = result.tokens[len(seed_tokens) :]
        generated_bars = _tokens_to_bars(generated_piece_tokens)
        generated_continuation_bars = generated_bars[
            args.prompt_bars : args.prompt_bars + args.continuation_bars
        ]
        generated_continuation_tokens = _flatten(generated_continuation_bars)
        scored_tokens = prefix + _flatten(generated_bars[:total_bars])
        metrics = eval_basic.evaluate(scored_tokens, vocab=vocab_set)

        token_match_rate = _token_match_rate(
            generated_continuation_tokens or generated_new_tokens,
            reference_tokens,
        )
        exact_bar_matches = _exact_bar_matches(generated_continuation_bars, reference_bars)
        pass_flags = _pass_flags(
            metrics,
            token_match_rate,
            generated_continuation_bar_count=len(generated_continuation_bars),
            requested_continuation_bars=args.continuation_bars,
        )

        sample_dir = args.out_dir / f"sample_{sample_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "generated_tokens.txt").write_text(
            " ".join(scored_tokens),
            encoding="utf-8",
        )
        (sample_dir / "prompt_tokens.txt").write_text(
            " ".join(prefix + prompt_tokens),
            encoding="utf-8",
        )
        (sample_dir / "reference_tokens.txt").write_text(
            " ".join(prefix + prompt_tokens + reference_tokens),
            encoding="utf-8",
        )
        (sample_dir / "reference_continuation_tokens.txt").write_text(
            " ".join(reference_tokens),
            encoding="utf-8",
        )
        sample = {
            "index": sample_index,
            "piece_id": str(piece_id),
            "source_path": str(group["source_path"].iloc[0]) if "source_path" in group else None,
            "prompt_bars": args.prompt_bars,
            "continuation_bars": args.continuation_bars,
            "seed_token_count": len(seed_tokens),
            "generated_token_count": len(scored_tokens),
            "generated_bar_count": _bar_count(scored_tokens),
            "reference_continuation_token_count": len(reference_tokens),
            "generated_continuation_token_count": len(generated_continuation_tokens),
            "generated_continuation_bar_count": len(generated_continuation_bars),
            "token_match_rate": token_match_rate,
            "exact_continuation_bar_matches": exact_bar_matches,
            "pass_flags": pass_flags,
            "metrics": metrics,
            "paths": {
                "generated_tokens": str(sample_dir / "generated_tokens.txt"),
                "prompt_tokens": str(sample_dir / "prompt_tokens.txt"),
                "reference_tokens": str(sample_dir / "reference_tokens.txt"),
                "reference_continuation_tokens": str(sample_dir / "reference_continuation_tokens.txt"),
            },
        }
        (sample_dir / "metrics.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
        samples.append(sample)
        if not args.quiet:
            print(
                f"{sample_index:03d} {piece_id}: "
                f"match={token_match_rate} voices={metrics.get('avg_voices_per_bar')} "
                f"cadence={metrics.get('cadence_proxy_rate')} dup={metrics.get('duplicate_bar_rate')}",
                flush=True,
            )

    if not samples:
        raise SystemExit("no eligible pieces evaluated")

    flag_names = sorted(samples[0]["pass_flags"])
    summary = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "vocab": str(args.vocab),
            "events": str(args.events),
            "prompt_bars": args.prompt_bars,
            "continuation_bars": args.continuation_bars,
            "samples": args.samples,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_length": args.max_length,
            "use_grammar_mask": args.use_grammar_mask,
            "use_scg": args.use_scg,
            "texture": args.texture,
            "device": args.device,
        },
        "sample_count": len(samples),
        "token_match_rate": _summarize([sample["token_match_rate"] for sample in samples]),
        "exact_continuation_bar_matches": _summarize(
            [sample["exact_continuation_bar_matches"] for sample in samples]
        ),
        "generated_continuation_bar_count": _summarize(
            [sample["generated_continuation_bar_count"] for sample in samples]
        ),
        "pass_rates": {
            name: round(
                sum(1 for sample in samples if sample["pass_flags"][name]) / len(samples),
                6,
            )
            for name in flag_names
        },
        "metrics": {
            name: _summarize([sample["metrics"].get(name) for sample in samples])
            for name in [
                "avg_voices_per_bar",
                "pct_bars_3plus_voices",
                "cadence_proxy_rate",
                "duplicate_bar_rate",
                "token_grammar_violations",
                "counterpoint_avg_active_voices",
                "counterpoint_voice_crossings",
                "counterpoint_parallel_fifths",
                "counterpoint_parallel_octaves",
            ]
        },
        "samples": samples,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continue overfit chorales from first bars and summarize quality."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=Path("data/overfit_20_chorales/events.parquet"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--prompt-bars", type=int, default=2)
    parser.add_argument("--continuation-bars", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--use-grammar-mask", action="store_true")
    parser.add_argument("--use-scg", action="store_true")
    parser.add_argument("--texture", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--eos-token", default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-new-tokens", type=int, default=512)
    parser.add_argument("--max-new-multiplier", type=float, default=3.0)
    parser.add_argument("--key", default=None)
    parser.add_argument("--style", default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--measures-token-prefix", default="MEAS")
    parser.add_argument("--no-key-from-plan", dest="key_from_plan", action="store_false")
    parser.set_defaults(key_from_plan=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prompt_bars <= 0:
        raise SystemExit("--prompt-bars must be positive")
    if args.continuation_bars <= 0:
        raise SystemExit("--continuation-bars must be positive")
    summary = run_eval(args)
    if not args.quiet:
        print(f"summary: {args.out_dir / 'summary.json'}")
        print(f"token_match_rate: {summary['token_match_rate']}")
        print(f"pass_rates: {summary['pass_rates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
