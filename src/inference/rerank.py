from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from src.inference.generate_v1 import GenerationConfig, GenerationResult
from src.music.counterpoint import evaluate_counterpoint_tokens
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
    "counterpoint_monophonic_position_rate": 20,
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
        "counterpoint_parallel_octaves": counterpoint.get("parallel_octaves"),
        "counterpoint_parallel_fifths": counterpoint.get("parallel_fifths"),
        "counterpoint_voice_crossings": counterpoint.get("voice_crossings"),
        "counterpoint_spacing_violations": counterpoint.get("spacing_violations"),
        "counterpoint_unresolved_dissonances": counterpoint.get("unresolved_dissonances"),
        "counterpoint_dissonance_on_strong_beat": counterpoint.get("dissonance_on_strong_beat"),
        "counterpoint_monophonic_position_rate": counterpoint.get("monophonic_position_rate"),
        "counterpoint_avg_active_voices": counterpoint.get("avg_active_voices"),
    }


def score_quality_metrics(metrics: dict[str, Any]) -> float:
    score = 0.0
    for metric, weight in QUALITY_SCORE_WEIGHTS.items():
        score += weight * _numeric_metric(metrics.get(metric))
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
