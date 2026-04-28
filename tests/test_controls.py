import pytest

from src.inference.controls import (
    ComposeControls,
    build_compose_seed_tokens,
    build_control_prefix_tokens,
    normalize_compose_key,
    normalize_texture,
)


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


def test_build_compose_seed_tokens_adds_polyphonic_anchors_for_texture():
    tokens = build_compose_seed_tokens(
        ComposeControls(key="C", measures=8, texture=4)
    )

    assert tokens == [
        "KEY_C",
        "MEAS_8",
        "BAR",
        "TIME_SIG_4_4",
        "KEY_C",
        "ABS_VOICE_0_48",
        "ABS_VOICE_1_55",
        "ABS_VOICE_2_64",
        "ABS_VOICE_3_72",
        "POS_0",
    ]


def test_build_compose_seed_tokens_uses_control_prefix_only_for_single_texture():
    assert build_compose_seed_tokens(
        ComposeControls(key="C", measures=8, texture=1)
    ) == ["KEY_C", "MEAS_8"]


def test_normalize_texture_clamps_supported_range():
    assert normalize_texture(None) == 1
    assert normalize_texture(0) == 1
    assert normalize_texture(3) == 3
    assert normalize_texture(8) == 4
