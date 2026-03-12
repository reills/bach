import pytest

from src.inference.controls import ComposeControls, build_control_prefix_tokens, normalize_compose_key


def test_normalize_compose_key_accepts_common_user_formats():
    assert normalize_compose_key(" c ") == "C"
    assert normalize_compose_key("f# minor") == "F#m"
    assert normalize_compose_key("bb MAJOR") == "Bb"


def test_build_control_prefix_tokens_normalizes_and_orders_controls():
    tokens = build_control_prefix_tokens(
        ComposeControls(
            key="bb minor",
            style="French overture / chorale",
            difficulty="very hard!",
            measures=8,
        )
    )

    assert tokens == [
        "KEY_Bbm",
        "STYLE_FRENCH_OVERTURE_CHORALE",
        "DIFFICULTY_VERY_HARD",
        "MEAS_8",
    ]


def test_build_control_prefix_tokens_rejects_non_positive_measures():
    with pytest.raises(ValueError, match="positive integer"):
        build_control_prefix_tokens(ComposeControls(measures=0))
