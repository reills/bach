import torch

import pytest

from src.utils.decoding.voice_state import (
    build_voice_leading_mask,
    normalize_voice_leading,
    voice_leading_enabled,
)


def _vocab(tokens):
    return {token: idx for idx, token in enumerate(tokens)}


def test_voice_leading_mask_blocks_out_of_range_pitches():
    vocab = _vocab(["MEL_INT12_-1", "MEL_INT12_0", "MEL_INT12_+1"])
    mask = build_voice_leading_mask(
        ["BAR", "ABS_VOICE_3_84", "POS_0", "VOICE_3", "DUR_24"],
        vocab,
        allowed_categories={"MEL_INT12"},
    )

    assert mask[vocab["MEL_INT12_+1"]].item() is False
    assert mask[vocab["MEL_INT12_0"]].item() is True
    assert mask[vocab["MEL_INT12_-1"]].item() is True


def test_voice_leading_mask_blocks_crossing_and_extreme_spacing():
    crossing_vocab = _vocab(["MEL_INT12_0", "MEL_INT12_+10"])
    crossing_prefix = [
        "BAR",
        "ABS_VOICE_0_48",
        "ABS_VOICE_1_55",
        "ABS_VOICE_2_64",
        "POS_0",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_2",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_1",
        "DUR_24",
    ]
    crossing_mask = build_voice_leading_mask(
        crossing_prefix,
        crossing_vocab,
        allowed_categories={"MEL_INT12"},
    )

    spacing_vocab = _vocab(["MEL_INT12_0", "MEL_INT12_+12"])
    spacing_prefix = [
        "BAR",
        "ABS_VOICE_2_60",
        "ABS_VOICE_3_72",
        "POS_0",
        "VOICE_2",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_3",
        "DUR_24",
    ]
    spacing_mask = build_voice_leading_mask(
        spacing_prefix,
        spacing_vocab,
        allowed_categories={"MEL_INT12"},
    )

    assert crossing_mask[crossing_vocab["MEL_INT12_+10"]].item() is False
    assert crossing_mask[crossing_vocab["MEL_INT12_0"]].item() is True
    assert spacing_mask[spacing_vocab["MEL_INT12_+12"]].item() is False
    assert spacing_mask[spacing_vocab["MEL_INT12_0"]].item() is True


def test_voice_leading_mask_blocks_parallel_octaves_and_fifths():
    octave_vocab = _vocab(["MEL_INT12_+1", "MEL_INT12_+2"])
    octave_prefix = [
        "BAR",
        "ABS_VOICE_0_48",
        "ABS_VOICE_1_60",
        "POS_0",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_1",
        "DUR_24",
        "MEL_INT12_0",
        "HARM_OCT_1",
        "HARM_CLASS_0",
        "POS_24",
        "VOICE_0",
        "DUR_24",
        "MEL_INT12_+2",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_1",
        "DUR_24",
    ]
    octave_mask = build_voice_leading_mask(
        octave_prefix,
        octave_vocab,
        allowed_categories={"MEL_INT12"},
    )

    fifth_vocab = _vocab(["MEL_INT12_+1", "MEL_INT12_+2"])
    fifth_prefix = [
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
        "MEL_INT12_+2",
        "HARM_OCT_0",
        "HARM_CLASS_0",
        "VOICE_1",
        "DUR_24",
    ]
    fifth_mask = build_voice_leading_mask(
        fifth_prefix,
        fifth_vocab,
        allowed_categories={"MEL_INT12"},
    )

    assert octave_mask[octave_vocab["MEL_INT12_+2"]].item() is False
    assert octave_mask[octave_vocab["MEL_INT12_+1"]].item() is True
    assert fifth_mask[fifth_vocab["MEL_INT12_+2"]].item() is False
    assert fifth_mask[fifth_vocab["MEL_INT12_+1"]].item() is True


def test_voice_leading_mask_does_not_mask_all_mel_tokens():
    vocab = _vocab(["MEL_INT12_+1", "MEL_INT12_+2"])
    mask = build_voice_leading_mask(
        ["BAR", "ABS_VOICE_3_84", "POS_0", "VOICE_3", "DUR_24"],
        vocab,
        allowed_categories={"MEL_INT12"},
    )

    mel_allowed = torch.tensor([mask[idx] for idx in vocab.values()], dtype=torch.bool)
    assert mel_allowed.any().item() is True


def test_voice_leading_mask_falls_back_when_state_is_incomplete():
    vocab = _vocab(["MEL_INT12_-24", "MEL_INT12_0", "MEL_INT12_+24"])
    mask = build_voice_leading_mask(
        ["BAR", "POS_0", "VOICE_0", "DUR_24"],
        vocab,
        allowed_categories={"MEL_INT12"},
    )

    assert mask.tolist() == [True, True, True]


def test_normalize_voice_leading_accepts_quality_mode_aliases():
    assert normalize_voice_leading(None) == "balanced"
    assert normalize_voice_leading("fast") == "off"
    assert normalize_voice_leading("off") == "off"
    assert normalize_voice_leading("balanced") == "balanced"
    assert normalize_voice_leading("best") == "best"
    assert voice_leading_enabled("off") is False
    assert voice_leading_enabled("balanced") is True
    assert voice_leading_enabled("best") is True


def test_normalize_voice_leading_rejects_unknown_mode():
    with pytest.raises(ValueError, match="voiceLeading"):
        normalize_voice_leading("maximum")
