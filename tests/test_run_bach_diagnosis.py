from pathlib import Path

from scripts.run_bach_diagnosis import (
    _closing_bar_tokens,
    _plain_diagnosis,
    _selected_samples,
    _tokens_to_bars,
    _write_generated_continuation_tokens,
)


def test_selected_samples_picks_lowest_and_highest_matches():
    selected = _selected_samples(
        {
            "samples": [
                {"index": 1, "token_match_rate": 0.2},
                {"index": 2, "token_match_rate": 0.8},
                {"index": 3, "token_match_rate": 0.1},
            ]
        },
        limit=1,
    )

    assert [(sample["selection"], sample["index"]) for sample in selected] == [
        ("worst", 3),
        ("best", 2),
    ]


def test_plain_diagnosis_mentions_training_and_harmonic_drift():
    lines = _plain_diagnosis(
        teacher={"overall": {"top1_accuracy": 0.99}},
        sampled={
            "token_match_rate": {"avg": 0.3},
            "metrics": {"token_grammar_violations": {"avg": 0.0}},
        },
        greedy_grammar={
            "token_match_rate": {"avg": 0.2},
            "metrics": {"token_grammar_violations": {"avg": 0.0}},
        },
        greedy_no_grammar={"metrics": {"token_grammar_violations": {"avg": 4.0}}},
        harm_repair={
            "raw_harm_mismatch_count": {"avg": 10.0},
            "repaired_harm_mismatch_count": {"avg": 0.0},
        },
    )

    joined = "\n".join(lines)
    assert "Teacher forcing is excellent" in joined
    assert "HARM_* drift is real" in joined


def test_closing_bar_tokens_reuse_last_time_signature_and_key():
    assert _closing_bar_tokens(["KEY_D", "BAR", "TIME_SIG_3_4", "KEY_G"]) == [
        "BAR",
        "TIME_SIG_3_4",
        "KEY_G",
    ]


def test_write_generated_continuation_tokens_skips_prompt_bars(tmp_path: Path):
    generated = tmp_path / "generated_tokens.txt"
    generated.write_text(
        "KEY_C MEAS_4 "
        "BAR TIME_SIG_4_4 KEY_C POS_0 "
        "BAR TIME_SIG_4_4 KEY_C POS_24 "
        "BAR TIME_SIG_4_4 KEY_C POS_48 "
        "BAR TIME_SIG_4_4 KEY_C POS_72",
        encoding="utf-8",
    )

    out_path = _write_generated_continuation_tokens(
        generated,
        tmp_path,
        prompt_bars=2,
        continuation_bars=1,
    )

    assert _tokens_to_bars(out_path.read_text(encoding="utf-8").split()) == [
        ["BAR", "TIME_SIG_4_4", "KEY_C", "POS_48"]
    ]
