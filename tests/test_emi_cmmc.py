from __future__ import annotations

from src.emi.cmmc import (
    GRADUS_MAJOR_SCALE,
    choose_from_scale,
    get_function,
    gradus_evaluate,
    interval_translator,
    pattern_match,
    run_pattern_match,
)


def test_cmmc_interval_pattern_match_uses_cope_tolerances() -> None:
    assert interval_translator([60, 62, 59, 60]) == (2, -3, 1)
    assert pattern_match((2, 3), (1, 2), 1, amount_off=2)
    assert not pattern_match((2, 3), (1, 6), 1, amount_off=2)
    assert run_pattern_match((-2, -2), (-2, -2, -3, 2, -2, -2), intervals_off=1, amount_off=2) == 3


def test_cmmc_harmonic_function_uses_analysis_lexicon() -> None:
    assert get_function((0, (79, 64, 48, 48, 65, 65, 50, 50))) == "A1"
    assert get_function((0, (60,))) == "E4"


def test_gradus_candidate_filter_matches_printed_example() -> None:
    accepted = gradus_evaluate(
        (69, 71, 72, 76, 74, 72, 74, 72, 71, 69),
        (48, 52, 47),
        (57, 55, 57, 55, 53, 52, 50),
    )

    assert accepted == (52,)


def test_gradus_scale_choice_uses_diatonic_second_and_third_options() -> None:
    assert choose_from_scale(60, 1, GRADUS_MAJOR_SCALE) == 62
    assert choose_from_scale(60, 3, GRADUS_MAJOR_SCALE) == 64
    assert choose_from_scale(60, -1, GRADUS_MAJOR_SCALE) == 59
    assert choose_from_scale(60, -3, GRADUS_MAJOR_SCALE) == 57
