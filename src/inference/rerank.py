from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from src.inference.generate_v1 import GenerationConfig, GenerationResult
from src.music.counterpoint import evaluate_counterpoint_tokens, pitched_events_from_tokens
from src.tokens.repair import repair_harm_tokens
from src.tokens.tokenizer import parse_voice_event
from src.tokens.validator import validate_harm_tokens

QUALITY_PASSES_DEFAULT = 4
QUALITY_PASSES_MAX = 16

QUALITY_SCORE_WEIGHTS = {
    "harm_mismatch_count": 1000,
    "token_grammar_violations": 100,
    "counterpoint_parallel_octaves": 80,
    "counterpoint_parallel_fifths": 80,
    "counterpoint_voice_crossings": 30,
    "counterpoint_spacing_violations": 15,
    "counterpoint_unresolved_dissonances": 10,
    "counterpoint_dissonance_on_strong_beat": 6,
    "duplicate_bar_rate": 80,
    "counterpoint_static_voice_rate": 40,
    "repeated_pitch_rate": 20,
    "repeated_interval_rate": 20,
    "counterpoint_monophonic_position_rate": 100,
    "counterpoint_avg_active_voices": -5,
}

GeneratorFn = Callable[..., GenerationResult]
PostprocessFn = Callable[[GenerationResult], GenerationResult]
EvaluateFn = Callable[[Sequence[str]], dict[str, int | float | None]]


@dataclass(frozen=True)
class RerankCandidate:
    index: int
    raw_generation: GenerationResult
    generation: GenerationResult
    metrics: dict[str, int | float | None]
    score: float


@dataclass(frozen=True)
class RerankResult:
    best: RerankCandidate
    candidates: list[RerankCandidate]


def normalize_quality_passes(value: int | None, *, default: int = QUALITY_PASSES_DEFAULT) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("qualityPasses must be an integer")
    return min(max(value, 1), QUALITY_PASSES_MAX)


def rerank_generations(
    checkpoint_path: str | Path,
    *,
    seed_tokens: Sequence[str | int],
    generation_config: GenerationConfig,
    vocab_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    generator: GeneratorFn,
    quality_passes: int = QUALITY_PASSES_DEFAULT,
    postprocess_generation: PostprocessFn | None = None,
    evaluate_fn: EvaluateFn | None = None,
) -> RerankResult:
    passes = normalize_quality_passes(quality_passes)
    evaluator = evaluate_fn or evaluate_quality_metrics
    candidates: list[RerankCandidate] = []

    for index in range(passes):
        raw_generation = generator(
            checkpoint_path,
            seed_tokens=list(seed_tokens),
            generation_config=generation_config,
            vocab_path=vocab_path,
            device=device,
        )
        generation = (
            postprocess_generation(raw_generation)
            if postprocess_generation is not None
            else repair_generation_harmonic_metadata(raw_generation, vocab_path=vocab_path)
        )
        metrics = evaluator(generation.tokens)
        candidates.append(
            RerankCandidate(
                index=index,
                raw_generation=raw_generation,
                generation=generation,
                metrics=metrics,
                score=score_quality_metrics(metrics),
            )
        )

    if not candidates:
        raise ValueError("qualityPasses must be positive")
    return RerankResult(
        best=min(candidates, key=lambda candidate: (candidate.score, candidate.index)),
        candidates=candidates,
    )


def repair_generation_harmonic_metadata(
    generation: GenerationResult,
    *,
    vocab_path: str | Path | None = None,
    vocab: dict[str, int] | None = None,
    tpq: int = 24,
) -> GenerationResult:
    repair_result = repair_harm_tokens(list(generation.tokens), tpq=tpq)
    if repair_result.tokens == generation.tokens:
        return generation

    repaired_ids = _ids_for_tokens(repair_result.tokens, vocab_path=vocab_path, vocab=vocab)
    return replace(
        generation,
        ids=repaired_ids if repaired_ids is not None else generation.ids,
        tokens=repair_result.tokens,
    )


def evaluate_quality_metrics(tokens: Sequence[str]) -> dict[str, int | float | None]:
    counterpoint = evaluate_counterpoint_tokens(tokens).to_dict()
    return {
        "harm_mismatch_count": _harm_mismatch_count(tokens),
        "token_grammar_violations": _count_grammar_violations(tokens),
        "duplicate_bar_rate": _duplicate_bar_rate(tokens),
        "repeated_pitch_rate": _repeated_pitch_rate(tokens),
        "repeated_interval_rate": _repeated_interval_rate(tokens),
        **_polyphony_metrics(tokens),
        "counterpoint_parallel_octaves": counterpoint.get("parallel_octaves"),
        "counterpoint_parallel_fifths": counterpoint.get("parallel_fifths"),
        "counterpoint_voice_crossings": counterpoint.get("voice_crossings"),
        "counterpoint_spacing_violations": counterpoint.get("spacing_violations"),
        "counterpoint_unresolved_dissonances": counterpoint.get("unresolved_dissonances"),
        "counterpoint_dissonance_on_strong_beat": counterpoint.get("dissonance_on_strong_beat"),
        "counterpoint_monophonic_position_rate": counterpoint.get("monophonic_position_rate"),
        "counterpoint_avg_active_voices": counterpoint.get("avg_active_voices"),
        "counterpoint_static_voice_rate": counterpoint.get("static_voice_rate"),
    }


def score_quality_metrics(metrics: dict[str, Any]) -> float:
    score = 0.0
    for metric, weight in QUALITY_SCORE_WEIGHTS.items():
        score += weight * _numeric_metric(metrics.get(metric))
    if metrics.get("counterpoint_avg_active_voices") is not None:
        score += 100 * max(0.0, 3.5 - _numeric_metric(metrics.get("counterpoint_avg_active_voices")))
    if metrics.get("pct_bars_3plus_voices") is not None:
        score += 50 * max(0.0, 0.8 - (_numeric_metric(metrics.get("pct_bars_3plus_voices")) / 100.0))
    return round(score, 6)


def _harm_mismatch_count(tokens: Sequence[str]) -> int | None:
    try:
        return len(validate_harm_tokens(list(tokens)))
    except Exception:
        return None


def _count_grammar_violations(tokens: Sequence[str]) -> int:
    violations = 0
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("VOICE_"):
            try:
                _, next_idx = parse_voice_event(tokens, idx)
            except ValueError:
                violations += 1
                idx += 1
                continue
            idx = next_idx
            continue
        idx += 1
    return violations


def _duplicate_bar_rate(tokens: Sequence[str]) -> float | None:
    bars = _split_bars(tokens)
    if not bars:
        return None
    seen: set[tuple[str, ...]] = set()
    duplicate_count = 0
    for bar in bars:
        key = tuple(bar)
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)
    return round(duplicate_count / len(bars), 4)


def _polyphony_metrics(tokens: Sequence[str]) -> dict[str, float | None]:
    bars = _split_bars(tokens)
    if not bars:
        return {
            "pct_bars_2plus_voices": None,
            "pct_bars_3plus_voices": None,
        }
    voice_counts = [_pitched_voice_count(bar) for bar in bars]
    return {
        "pct_bars_2plus_voices": round(100 * sum(count >= 2 for count in voice_counts) / len(voice_counts), 2),
        "pct_bars_3plus_voices": round(100 * sum(count >= 3 for count in voice_counts) / len(voice_counts), 2),
    }


def _repeated_pitch_rate(tokens: Sequence[str]) -> float | None:
    events_by_voice = _pitched_events_by_voice(tokens)
    repeated = 0
    transitions = 0
    for events in events_by_voice.values():
        ordered = sorted(events, key=lambda event: (event.start_tick, event.pitch))
        for previous, current in zip(ordered, ordered[1:]):
            transitions += 1
            if previous.pitch == current.pitch:
                repeated += 1
    if transitions == 0:
        return None
    return round(repeated / transitions, 4)


def _repeated_interval_rate(tokens: Sequence[str]) -> float | None:
    events_by_voice = _pitched_events_by_voice(tokens)
    repeated = 0
    transitions = 0
    for events in events_by_voice.values():
        ordered = sorted(events, key=lambda event: (event.start_tick, event.pitch))
        intervals = [
            current.pitch - previous.pitch
            for previous, current in zip(ordered, ordered[1:])
        ]
        for previous_interval, current_interval in zip(intervals, intervals[1:]):
            transitions += 1
            if previous_interval == current_interval:
                repeated += 1
    if transitions == 0:
        return None
    return round(repeated / transitions, 4)


def _pitched_events_by_voice(tokens: Sequence[str]):
    events_by_voice = {}
    for event in pitched_events_from_tokens(tokens):
        events_by_voice.setdefault(event.voice, []).append(event)
    return events_by_voice


def _split_bars(tokens: Sequence[str]) -> list[list[str]]:
    bars: list[list[str]] = []
    current_bar: list[str] | None = None
    for token in tokens:
        if token == "BAR":
            if current_bar is not None:
                bars.append(current_bar)
            current_bar = []
            continue
        if current_bar is not None:
            current_bar.append(token)
    if current_bar is not None:
        bars.append(current_bar)
    return bars


def _pitched_voice_count(tokens: Sequence[str]) -> int:
    voices: set[int] = set()
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if not token.startswith("VOICE_"):
            idx += 1
            continue
        try:
            event, next_idx = parse_voice_event(tokens, idx)
        except ValueError:
            idx += 1
            continue
        if not event.is_rest:
            voices.add(event.voice)
        idx = next_idx
    return len(voices)


def _numeric_metric(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _ids_for_tokens(
    tokens: Sequence[str],
    *,
    vocab_path: str | Path | None,
    vocab: dict[str, int] | None,
) -> list[int] | None:
    resolved_vocab = vocab if vocab is not None else _load_vocab(vocab_path)
    if resolved_vocab is None:
        return None

    ids: list[int] = []
    for token in tokens:
        token_id = resolved_vocab.get(token)
        if not isinstance(token_id, int):
            return None
        ids.append(token_id)
    return ids


def _load_vocab(vocab_path: str | Path | None) -> dict[str, int] | None:
    if vocab_path is None:
        return None
    path = Path(vocab_path)
    if not path.exists():
        return None
    vocab = json.loads(path.read_text(encoding="utf-8"))
    return vocab if isinstance(vocab, dict) else None
