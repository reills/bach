"""Tests for grammar FSM and build_grammar_mask."""

import sys
from pathlib import Path
from typing import List, Set

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.decoding.rules import (
    GrammarConstraints,
    allowed_next_categories,
    grammar_constraints,
    token_category,
)

try:
    import torch
    from src.utils.decoding.scg import build_grammar_mask
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# token_category tests
# ---------------------------------------------------------------------------

def test_token_category_bar():
    assert token_category("BAR") == "BAR"


def test_token_category_eos():
    assert token_category("<eos>") == "EOS"
    assert token_category("EOS") == "EOS"


def test_token_category_bos():
    assert token_category("<bos>") == "BOS"
    assert token_category("BOS") == "BOS"


def test_token_category_pad():
    assert token_category("<pad>") == "PAD"
    assert token_category("PAD") == "PAD"


def test_token_category_mel_int12():
    assert token_category("MEL_INT12_+2") == "MEL_INT12"
    assert token_category("MEL_INT12_-7") == "MEL_INT12"
    assert token_category("MEL_INT12_0") == "MEL_INT12"


def test_token_category_harm_oct():
    assert token_category("HARM_OCT_0") == "HARM_OCT"
    assert token_category("HARM_OCT_NA") == "HARM_OCT"


def test_token_category_harm_class():
    assert token_category("HARM_CLASS_4") == "HARM_CLASS"
    assert token_category("HARM_CLASS_NA") == "HARM_CLASS"


def test_token_category_rest():
    assert token_category("REST_24") == "REST"


def test_token_category_dur():
    assert token_category("DUR_24") == "DUR"
    assert token_category("DUR_12") == "DUR"


def test_token_category_dup():
    assert token_category("DUP_0") == "DUP"


def test_token_category_voice():
    assert token_category("VOICE_0") == "VOICE"
    assert token_category("VOICE_3") == "VOICE"


def test_token_category_str():
    assert token_category("STR_0") == "STR"
    assert token_category("STR_5") == "STR"


def test_token_category_fret():
    assert token_category("FRET_0") == "FRET"
    assert token_category("FRET_12") == "FRET"


def test_token_category_pos():
    assert token_category("POS_0") == "POS"
    assert token_category("POS_48") == "POS"


def test_token_category_abs_voice():
    assert token_category("ABS_VOICE_0_60") == "ABS_VOICE"


def test_token_category_abs_bass():
    assert token_category("ABS_BASS_48") == "ABS_BASS"


def test_token_category_abs_sop():
    assert token_category("ABS_SOP_72") == "ABS_SOP"


def test_token_category_key():
    assert token_category("KEY_C") == "KEY"
    assert token_category("KEY_Am") == "KEY"


def test_token_category_time_sig():
    assert token_category("TIME_SIG_4_4") == "TIME_SIG"


def test_token_category_tempo():
    assert token_category("TEMPO_120") == "TEMPO"


def test_token_category_style_control():
    assert token_category("STYLE_BAROQUE") == "CONTROL"


def test_token_category_difficulty_control():
    assert token_category("DIFFICULTY_EASY") == "CONTROL"


def test_token_category_meas_control():
    assert token_category("MEAS_4") == "CONTROL"


def test_token_category_other():
    assert token_category("UNKNOWN_TOKEN_XYZ") == "OTHER"
    assert token_category("") == "OTHER"


# ---------------------------------------------------------------------------
# grammar_constraints / allowed_next_categories
# ---------------------------------------------------------------------------

def test_initial_state_allows_bar():
    """Before any tokens, BAR must be allowed."""
    cats = allowed_next_categories([])
    assert "BAR" in cats


def test_initial_state_allows_key_time_sig():
    """Before BAR, KEY and TIME_SIG tokens are allowed."""
    cats = allowed_next_categories([])
    assert "KEY" in cats
    assert "TIME_SIG" in cats


def test_initial_state_no_voice():
    """VOICE tokens should not be allowed before a BAR and POS."""
    cats = allowed_next_categories([])
    assert "VOICE" not in cats


def test_after_bar_allows_pos():
    cats = allowed_next_categories(["BAR"])
    assert "POS" in cats


def test_after_bar_no_voice_yet():
    """After BAR but before POS, VOICE should not be allowed."""
    cats = allowed_next_categories(["BAR"])
    assert "VOICE" not in cats


def test_after_bar_pos_allows_voice():
    cats = allowed_next_categories(["BAR", "POS_0"])
    assert "VOICE" in cats


def test_after_voice_unanchored_only_rest():
    """After VOICE_0 with no anchor, only REST should be allowed."""
    cats = allowed_next_categories(["BAR", "POS_0", "VOICE_0"])
    assert "REST" in cats
    assert "DUR" not in cats


def test_after_voice_anchored_allows_dur():
    """After VOICE_0 with ABS_BASS_* anchor, DUR should be allowed."""
    cats = allowed_next_categories(["BAR", "ABS_BASS_48", "POS_0", "VOICE_0"])
    assert "DUR" in cats
    assert "REST" in cats


def test_after_voice_abs_voice_anchor():
    """ABS_VOICE_0_60 anchors voice 0, so DUR allowed after VOICE_0."""
    cats = allowed_next_categories(["BAR", "ABS_VOICE_0_60", "POS_0", "VOICE_0"])
    assert "DUR" in cats


def test_after_dur_allows_mel_int():
    cats = allowed_next_categories(["BAR", "ABS_BASS_48", "POS_0", "VOICE_0", "DUR_24"])
    assert "MEL_INT12" in cats


def test_after_dur_allows_dup():
    cats = allowed_next_categories(["BAR", "ABS_BASS_48", "POS_0", "VOICE_0", "DUR_24"])
    assert "DUP" in cats


def test_after_dup_allows_mel_int():
    cats = allowed_next_categories([
        "BAR", "ABS_BASS_48", "POS_0", "VOICE_0", "DUR_24", "DUP_0"
    ])
    assert "MEL_INT12" in cats
    assert "DUP" not in cats


def test_after_mel_int_allows_harm_oct():
    cats = allowed_next_categories([
        "BAR", "ABS_BASS_48", "POS_0", "VOICE_0", "DUR_24", "MEL_INT12_+0"
    ])
    assert "HARM_OCT" in cats
    assert "HARM_CLASS" not in cats


def test_after_harm_oct_allows_harm_class():
    cats = allowed_next_categories([
        "BAR", "ABS_BASS_48", "POS_0", "VOICE_0", "DUR_24",
        "MEL_INT12_+0", "HARM_OCT_0"
    ])
    assert "HARM_CLASS" in cats
    assert "HARM_OCT" not in cats


def test_after_harm_class_allows_voice_or_bar():
    cats = allowed_next_categories([
        "BAR", "ABS_BASS_48", "POS_0", "VOICE_0", "DUR_24",
        "MEL_INT12_+0", "HARM_OCT_0", "HARM_CLASS_0"
    ])
    assert "VOICE" in cats or "BAR" in cats or "POS" in cats


def test_after_harm_class_allows_str():
    """After a complete pitched event, STR (tab) tokens can follow."""
    cats = allowed_next_categories([
        "BAR", "ABS_BASS_48", "POS_0", "VOICE_0", "DUR_24",
        "MEL_INT12_+0", "HARM_OCT_0", "HARM_CLASS_0"
    ])
    assert "STR" in cats


def test_after_str_allows_fret():
    cats = allowed_next_categories([
        "BAR", "ABS_BASS_48", "POS_0", "VOICE_0", "DUR_24",
        "MEL_INT12_+0", "HARM_OCT_0", "HARM_CLASS_0", "STR_2"
    ])
    assert "FRET" in cats
    assert "VOICE" not in cats


def test_after_fret_back_to_free():
    cats = allowed_next_categories([
        "BAR", "ABS_BASS_48", "POS_0", "VOICE_0", "DUR_24",
        "MEL_INT12_+0", "HARM_OCT_0", "HARM_CLASS_0", "STR_2", "FRET_3"
    ])
    assert "VOICE" in cats or "BAR" in cats or "POS" in cats


def test_rest_event_returns_to_free():
    """After VOICE + REST, should be back to free (VOICE/POS/BAR allowed)."""
    cats = allowed_next_categories(["BAR", "POS_0", "VOICE_0", "REST_24"])
    assert "VOICE" in cats or "BAR" in cats or "POS" in cats


def test_eos_allowed_at_free_state():
    constraints = grammar_constraints(["BAR", "POS_0", "VOICE_0", "REST_24"])
    assert constraints.allow_eos is True


def test_eos_not_allowed_mid_event():
    """EOS should not be allowed in the middle of a pitched event."""
    constraints = grammar_constraints([
        "BAR", "ABS_BASS_48", "POS_0", "VOICE_0", "DUR_24"
    ])
    assert constraints.allow_eos is False


def test_abs_sop_anchors_voice_3():
    """ABS_SOP_* should anchor voice 3."""
    cats = allowed_next_categories(["BAR", "ABS_SOP_72", "POS_0", "VOICE_3"])
    assert "DUR" in cats


def test_multiple_voices_anchored():
    cats = allowed_next_categories([
        "BAR", "ABS_BASS_48", "ABS_SOP_72", "POS_0", "VOICE_1"
    ])
    # VOICE_1 not anchored -> only REST
    assert "REST" in cats
    assert "DUR" not in cats


def test_multiple_voices_all_anchored():
    cats = allowed_next_categories([
        "BAR",
        "ABS_VOICE_0_48", "ABS_VOICE_1_55", "ABS_VOICE_2_60", "ABS_VOICE_3_67",
        "POS_0", "VOICE_1"
    ])
    assert "DUR" in cats


def test_grammar_constraints_returns_dataclass():
    result = grammar_constraints(["BAR"])
    assert isinstance(result, GrammarConstraints)
    assert hasattr(result, "allowed_categories")
    assert hasattr(result, "allow_eos")


def test_grammar_constraints_frozen():
    result = grammar_constraints(["BAR"])
    with pytest.raises((AttributeError, TypeError)):
        result.allow_eos = True


def test_bar_resets_position_state():
    """After a BAR, position state should reset so VOICE is not allowed before POS."""
    cats = allowed_next_categories([
        "BAR", "POS_0", "VOICE_0", "REST_24",
        "BAR",
    ])
    assert "VOICE" not in cats


def test_key_token_in_pre_bar_context():
    cats = allowed_next_categories(["KEY_C"])
    assert "BAR" in cats


def test_control_token_allowed_pre_bar():
    cats = allowed_next_categories(["MEAS_4"])
    assert "BAR" in cats


def test_long_sequence_remains_valid():
    """A realistic multi-bar prefix should not crash the FSM."""
    prefix = [
        "KEY_C", "MEAS_4",
        "BAR",
        "ABS_BASS_48", "ABS_SOP_72",
        "POS_0",
        "VOICE_0", "DUR_24", "MEL_INT12_+0", "HARM_OCT_0", "HARM_CLASS_0",
        "VOICE_3", "DUR_24", "MEL_INT12_+7", "HARM_OCT_0", "HARM_CLASS_7",
        "POS_24",
        "VOICE_0", "REST_12",
        "BAR",
        "POS_0",
        "VOICE_0", "DUR_48", "MEL_INT12_+2", "HARM_OCT_0", "HARM_CLASS_2",
    ]
    result = grammar_constraints(prefix)
    assert isinstance(result, GrammarConstraints)
    assert len(result.allowed_categories) > 0


# ---------------------------------------------------------------------------
# build_grammar_mask tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_build_grammar_mask_shape():
    vocab = {
        "BAR": 0, "<pad>": 1, "KEY_C": 2, "POS_0": 3,
        "VOICE_0": 4, "DUR_24": 5, "MEL_INT12_+0": 6,
        "HARM_OCT_0": 7, "HARM_CLASS_0": 8, "REST_24": 9,
        "<eos>": 10,
    }
    mask = build_grammar_mask([], vocab, allowed_categories={"BAR", "KEY", "TIME_SIG"})
    assert mask.shape == (max(vocab.values()) + 1,)
    assert mask.dtype == torch.bool


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_build_grammar_mask_bar_allowed():
    vocab = {"BAR": 0, "POS_0": 1, "VOICE_0": 2, "<eos>": 3}
    mask = build_grammar_mask([], vocab, allowed_categories={"BAR"}, allow_eos=False)
    assert mask[0].item() is True   # BAR
    assert mask[1].item() is False  # POS_0
    assert mask[2].item() is False  # VOICE_0


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_build_grammar_mask_eos_allowed():
    vocab = {"BAR": 0, "<eos>": 1, "POS_0": 2}
    mask = build_grammar_mask([], vocab, allowed_categories={"BAR"}, allow_eos=True)
    assert mask[1].item() is True   # <eos>


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_build_grammar_mask_eos_blocked():
    vocab = {"BAR": 0, "<eos>": 1, "DUR_24": 2}
    mask = build_grammar_mask([], vocab, allowed_categories={"DUR"}, allow_eos=False)
    assert mask[1].item() is False  # <eos> blocked
    assert mask[2].item() is True   # DUR_24 allowed


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_build_grammar_mask_mel_int_allowed():
    vocab = {"MEL_INT12_+0": 0, "MEL_INT12_+2": 1, "HARM_OCT_0": 2, "BAR": 3}
    mask = build_grammar_mask([], vocab, allowed_categories={"MEL_INT12"}, allow_eos=False)
    assert mask[0].item() is True
    assert mask[1].item() is True
    assert mask[2].item() is False  # HARM_OCT not in allowed
    assert mask[3].item() is False  # BAR not in allowed


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_build_grammar_mask_integration_after_bar():
    """Integration test: mask after BAR should include POS but not VOICE."""
    vocab = {
        "BAR": 0, "POS_0": 1, "VOICE_0": 2, "KEY_C": 3, "TIME_SIG_4_4": 4,
        "ABS_BASS_48": 5, "<eos>": 6,
    }
    prefix = ["BAR"]
    constraints = grammar_constraints(prefix)
    mask = build_grammar_mask(
        prefix, vocab,
        allowed_categories=set(constraints.allowed_categories),
        allow_eos=constraints.allow_eos,
    )
    assert mask[1].item() is True   # POS_0
    assert mask[2].item() is False  # VOICE_0


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
def test_build_grammar_mask_integration_after_dur():
    """After DUR, MEL_INT12 and DUP should be unmasked."""
    vocab = {
        "BAR": 0, "POS_0": 1, "VOICE_0": 2, "ABS_BASS_48": 3,
        "DUR_24": 4, "MEL_INT12_+0": 5, "MEL_INT12_+2": 6,
        "DUP_0": 7, "HARM_OCT_0": 8, "HARM_CLASS_0": 9, "REST_24": 10,
    }
    prefix = ["BAR", "ABS_BASS_48", "POS_0", "VOICE_0", "DUR_24"]
    constraints = grammar_constraints(prefix)
    mask = build_grammar_mask(
        prefix, vocab,
        allowed_categories=set(constraints.allowed_categories),
        allow_eos=constraints.allow_eos,
    )
    assert mask[5].item() is True   # MEL_INT12_+0
    assert mask[6].item() is True   # MEL_INT12_+2
    assert mask[7].item() is True   # DUP_0
    assert mask[8].item() is False  # HARM_OCT_0 not yet
    assert mask[0].item() is False  # BAR not yet
