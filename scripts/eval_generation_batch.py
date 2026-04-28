"""Run repeated checkpoint generations and summarize counterpoint quality."""

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

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.compose_service import ComposeDiagnosticsError, compose_baseline
from src.inference.controls import ComposeControls, build_compose_seed_tokens
from src.inference.generate_v1 import GenerationConfig, GenerationResult, _generate_from_loaded
from src.models.notelm import load_notelm_checkpoint

QUALITY_METRICS = [
    "counterpoint_avg_active_voices",
    "counterpoint_monophonic_position_rate",
    "counterpoint_voice_crossings",
    "counterpoint_spacing_violations",
    "counterpoint_parallel_fifths",
    "counterpoint_parallel_octaves",
    "counterpoint_dissonance_on_strong_beat",
    "counterpoint_unresolved_dissonances",
    "harm_mismatch_count",
    "off_key_rate",
    "duplicate_bar_rate",
]


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _summarize_metric(values: Sequence[Any]) -> dict[str, float | int | None]:
    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numeric_values:
        return {"avg": None, "min": None, "max": None}
    return {
        "avg": round(mean(numeric_values), 6),
        "min": min(numeric_values),
        "max": max(numeric_values),
    }


def summarize_samples(samples: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    successful = [sample for sample in samples if sample.get("ok")]
    return {
        metric: _summarize_metric(
            [sample.get("metrics", {}).get(metric) for sample in successful]
        )
        for metric in QUALITY_METRICS
    }


def _write_sample_outputs(sample_dir: Path, result, metrics: dict[str, Any]) -> dict[str, str]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "tokens": sample_dir / "tokens.txt",
        "metrics": sample_dir / "metrics.json",
        "musicxml": sample_dir / "example.musicxml",
        "midi": sample_dir / "example.mid",
    }
    paths["tokens"].write_text(" ".join(result.generation.tokens), encoding="utf-8")
    paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    paths["musicxml"].write_text(result.score_xml, encoding="utf-8")
    paths["midi"].write_bytes(result.midi)
    return {name: str(path) for name, path in paths.items()}


def _write_failure(sample_dir: Path, exc: Exception) -> dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ok": False,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    if isinstance(exc, ComposeDiagnosticsError):
        payload["stage"] = exc.stage
        payload["report_path"] = str(exc.report_path) if exc.report_path is not None else None
    (sample_dir / "failure.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    eval_basic = _load_eval_basic()
    loaded = load_notelm_checkpoint(
        args.checkpoint,
        vocab_path=args.vocab,
        device=args.device,
    )

    controls = ComposeControls(
        key=args.key,
        style=args.style,
        difficulty=args.difficulty,
        measures=args.measures,
        texture=args.texture,
    )
    seed_tokens = build_compose_seed_tokens(controls)
    generation_config = GenerationConfig(
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

    def generator(
        checkpoint_path: str | Path,
        *,
        seed_tokens: list[str | int],
        generation_config: GenerationConfig,
        vocab_path: str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> GenerationResult:
        del checkpoint_path, vocab_path, device
        return _generate_from_loaded(
            loaded,
            seed_tokens=seed_tokens,
            generation_config=generation_config,
        )

    sample_summaries: list[dict[str, Any]] = []
    for sample_index in range(1, args.samples + 1):
        sample_dir = args.out_dir / f"sample_{sample_index:03d}"
        if not args.quiet:
            print(f"sample {sample_index:03d}/{args.samples}: generating", flush=True)
        try:
            result = compose_baseline(
                args.checkpoint,
                seed_tokens=seed_tokens,
                generation_config=generation_config,
                vocab_path=loaded.vocab_path,
                device=args.device,
                render_mode=args.render_mode,
                generator=generator,
            )
            metrics = eval_basic.evaluate(result.generation.tokens)
            paths = _write_sample_outputs(sample_dir, result, metrics)
            sample_summaries.append(
                {
                    "index": sample_index,
                    "ok": True,
                    "render_mode": result.render_mode,
                    "paths": paths,
                    "metrics": {metric: metrics.get(metric) for metric in QUALITY_METRICS},
                }
            )
        except Exception as exc:
            failure = _write_failure(sample_dir, exc)
            sample_summaries.append({"index": sample_index, **failure})
            if args.fail_fast:
                raise

    summary = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "vocab": str(args.vocab) if args.vocab is not None else None,
            "samples": args.samples,
            "seed": args.seed,
            "key": args.key,
            "style": args.style,
            "difficulty": args.difficulty,
            "measures": args.measures,
            "texture": args.texture,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_length": args.max_length,
            "use_grammar_mask": args.use_grammar_mask,
            "use_scg": args.use_scg,
            "render_mode": args.render_mode,
            "device": args.device,
            "seed_tokens": [str(token) for token in seed_tokens],
            "generation_config": _jsonable(asdict(generation_config)),
        },
        "sample_count": args.samples,
        "successful_count": sum(1 for sample in sample_summaries if sample.get("ok")),
        "failed_count": sum(1 for sample in sample_summaries if not sample.get("ok")),
        "metrics": summarize_samples(sample_summaries),
        "samples": sample_summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeated NoteLM generations and summarize quality metrics."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--texture", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--use-grammar-mask", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--key", default="C")
    parser.add_argument("--style", default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--measures", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--use-scg", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--eos-token", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--render-mode", choices=["guitar", "piano"], default="piano")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.samples <= 0:
        parser.error("--samples must be positive")

    summary = run_batch(args)
    if not args.quiet:
        print(f"summary: {args.out_dir / 'summary.json'}")
        print(
            f"samples: {summary['successful_count']} ok, "
            f"{summary['failed_count']} failed"
        )
        for metric, stats in summary["metrics"].items():
            print(
                f"{metric}: avg={stats['avg']} min={stats['min']} max={stats['max']}"
            )
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
