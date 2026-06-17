from dataclasses import replace
from pathlib import Path

from src.inference.generate_v1 import GenerationConfig, GenerationResult
from src.inference.rerank import (
    evaluate_novelty_metrics,
    evaluate_quality_metrics,
    is_high_copy_risk,
    repair_generation_harmonic_metadata,
    rerank_generations,
    score_quality_metrics,
)


def test_score_quality_metrics_applies_counterpoint_weights():
    score = score_quality_metrics(
        {
            "harm_mismatch_count": 1,
            "token_grammar_violations": 1,
            "counterpoint_parallel_octaves": 1,
            "counterpoint_parallel_fifths": 1,
            "counterpoint_voice_crossings": 1,
            "counterpoint_spacing_violations": 1,
            "counterpoint_unresolved_dissonances": 1,
            "counterpoint_dissonance_on_strong_beat": 1,
            "counterpoint_monophonic_position_rate": 0.25,
            "counterpoint_avg_active_voices": 3,
        }
    )

    assert score == 1381


def test_evaluate_quality_metrics_includes_repetition_and_voice_dropout_metrics():
    metrics = evaluate_quality_metrics(
        [
            "BAR",
            "ABS_VOICE_0_48",
            "ABS_VOICE_1_55",
            "POS_0",
            "VOICE_0",
            "DUR_24",
            "MEL_INT12_0",
            "HARM_OCT_0",
            "HARM_CLASS_0",
            "VOICE_1",
            "DUR_24",
            "MEL_INT12_0",
            "HARM_OCT_0",
            "HARM_CLASS_7",
            "POS_24",
            "VOICE_0",
            "DUR_24",
            "MEL_INT12_0",
            "HARM_OCT_0",
            "HARM_CLASS_0",
            "VOICE_1",
            "DUR_24",
            "MEL_INT12_0",
            "HARM_OCT_0",
            "HARM_CLASS_7",
            "BAR",
            "ABS_VOICE_0_48",
            "ABS_VOICE_1_55",
            "POS_0",
            "VOICE_0",
            "DUR_24",
            "MEL_INT12_0",
            "HARM_OCT_0",
            "HARM_CLASS_0",
            "VOICE_1",
            "DUR_24",
            "MEL_INT12_0",
            "HARM_OCT_0",
            "HARM_CLASS_7",
            "POS_24",
            "VOICE_0",
            "DUR_24",
            "MEL_INT12_0",
            "HARM_OCT_0",
            "HARM_CLASS_0",
            "VOICE_1",
            "DUR_24",
            "MEL_INT12_0",
            "HARM_OCT_0",
            "HARM_CLASS_7",
        ]
    )

    assert metrics["duplicate_bar_rate"] == 0.5
    assert metrics["pct_bars_2plus_voices"] == 100.0
    assert metrics["pct_bars_3plus_voices"] == 0.0
    assert metrics["repeated_pitch_rate"] == 1.0
    assert metrics["repeated_interval_rate"] == 1.0


def test_rerank_generations_returns_lowest_score_postprocessed_candidate():
    generated = [
        GenerationResult(ids=[0], tokens=["bad"], stopped_on_eos=False),
        GenerationResult(ids=[1], tokens=["good"], stopped_on_eos=False),
        GenerationResult(ids=[2], tokens=["middle"], stopped_on_eos=False),
    ]
    calls = []

    def generator(checkpoint_path, *, seed_tokens, generation_config, vocab_path=None, device="cpu"):
        calls.append(
            {
                "checkpoint_path": checkpoint_path,
                "seed_tokens": seed_tokens,
                "generation_config": generation_config,
                "vocab_path": vocab_path,
                "device": device,
            }
        )
        return generated[len(calls) - 1]

    def postprocess(generation):
        return replace(generation, tokens=[*generation.tokens, "REPAIRED"])

    def evaluate(tokens):
        crossings = {"bad": 2, "good": 0, "middle": 1}[tokens[0]]
        return {
            "counterpoint_voice_crossings": crossings,
            "counterpoint_avg_active_voices": 4,
        }

    config = GenerationConfig(max_length=32)
    result = rerank_generations(
        Path("/tmp/checkpoint.pt"),
        seed_tokens=["KEY_C"],
        generation_config=config,
        vocab_path=Path("/tmp/vocab.json"),
        device="cpu",
        generator=generator,
        quality_passes=3,
        postprocess_generation=postprocess,
        evaluate_fn=evaluate,
    )

    assert len(calls) == 3
    assert all(call["seed_tokens"] == ["KEY_C"] for call in calls)
    assert all(call["generation_config"] is config for call in calls)
    assert result.best.index == 1
    assert result.best.raw_generation.tokens == ["good"]
    assert result.best.generation.tokens == ["good", "REPAIRED"]
    assert [candidate.score for candidate in result.candidates] == [40, -20, 10]


def test_repair_generation_harmonic_metadata_remaps_ids_from_vocab():
    generated_tokens = [
        "BAR",
        "TIME_SIG_4_4",
        "KEY_C",
        "ABS_VOICE_0_60",
        "POS_0",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_7",
    ]
    repaired_tokens = [*generated_tokens[:-1], "HARM_CLASS_0"]
    vocab = {token: index for index, token in enumerate(sorted(set(repaired_tokens)))}

    repaired = repair_generation_harmonic_metadata(
        GenerationResult(ids=list(range(len(generated_tokens))), tokens=generated_tokens, stopped_on_eos=False),
        vocab=vocab,
    )

    assert repaired.tokens == repaired_tokens
    assert repaired.ids == [vocab[token] for token in repaired_tokens]


def test_novelty_metrics_detect_transposition_normalized_source_overlap():
    source = [
        "BAR",
        "ABS_VOICE_0_60",
        "POS_0",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "COPY_HASH_a",
    ]
    generated = [
        "BAR",
        "ABS_VOICE_0_65",
        "POS_0",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "COPY_HASH_a",
    ]

    metrics = evaluate_novelty_metrics(generated, [source], ngram=4)

    assert metrics["source_ngram_overlap_rate"] > 0
    assert metrics["fragment_chain_reuse"] == 1.0
    assert metrics["high_copy_risk"] == 1
    assert is_high_copy_risk(metrics)
