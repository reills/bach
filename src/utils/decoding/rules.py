from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import torch


# ---------------------------------------------------------------------------
# Token category helpers
# ---------------------------------------------------------------------------

_EXACT_CATEGORY: Dict[str, str] = {
    "BAR": "BAR",
    "<eos>": "EOS",
    "EOS": "EOS",
    "<bos>": "BOS",
    "BOS": "BOS",
    "<pad>": "PAD",
    "PAD": "PAD",
}

_PREFIX_CATEGORY: List[Tuple[str, str]] = [
    ("MEL_INT12_", "MEL_INT12"),
    ("HARM_OCT_", "HARM_OCT"),
    ("HARM_CLASS_", "HARM_CLASS"),
    ("REST_", "REST"),
    ("DUR_", "DUR"),
    ("DUP_", "DUP"),
    ("VOICE_", "VOICE"),
    ("STR_", "STR"),
    ("FRET_", "FRET"),
    ("POS_", "POS"),
    ("ABS_VOICE_", "ABS_VOICE"),
    ("ABS_BASS_", "ABS_BASS"),
    ("ABS_SOP_", "ABS_SOP"),
    ("ABS_LOW_", "ABS_LOW"),
    ("ABS_HIGH_", "ABS_HIGH"),
    ("REF_VOICE_", "REF_VOICE"),
    ("TIME_SIG_", "TIME_SIG"),
    ("KEY_", "KEY"),
    ("STYLE_", "CONTROL"),
    ("DIFFICULTY_", "CONTROL"),
    ("MEAS_", "CONTROL"),
    ("TEMPO_", "TEMPO"),
]


def token_category(token: str) -> str:
    """Return the category string for a token, e.g. 'MEL_INT12', 'VOICE', 'BAR'."""
    exact_category = _EXACT_CATEGORY.get(token)
    if exact_category is not None:
        return exact_category
    for prefix, category in _PREFIX_CATEGORY:
        if token.startswith(prefix):
            return category
    return "OTHER"


# ---------------------------------------------------------------------------
# Grammar FSM
# ---------------------------------------------------------------------------

_PRE_BAR_ALLOWED: FrozenSet[str] = frozenset({"BAR", "TIME_SIG", "KEY", "TEMPO", "CONTROL"})
_IN_BAR_NO_POS_ALLOWED: FrozenSet[str] = frozenset(
    {
        "BAR",
        "TIME_SIG",
        "KEY",
        "TEMPO",
        "POS",
        "ABS_VOICE",
        "ABS_BASS",
        "ABS_SOP",
        "ABS_LOW",
        "ABS_HIGH",
        "REF_VOICE",
    }
)
_IN_POS_ALLOWED: FrozenSet[str] = frozenset(
    {
        "BAR",
        "POS",
        "VOICE",
        "ABS_VOICE",
        "ABS_BASS",
        "ABS_SOP",
        "ABS_LOW",
        "ABS_HIGH",
        "REF_VOICE",
    }
)


@dataclass(frozen=True)
class GrammarConstraints:
    allowed_categories: FrozenSet[str]
    allow_eos: bool


@dataclass
class _GrammarState:
    event_state: str = "FREE"
    in_bar: bool = False
    current_pos: bool = False
    anchored_voices: Set[int] = field(default_factory=set)
    pending_voice: Optional[int] = None


def _parse_voice_index(token: str) -> Optional[int]:
    if not token.startswith("VOICE_"):
        return None
    try:
        return int(token.split("_", 1)[1])
    except ValueError:
        return None


def _parse_abs_voice_index(token: str) -> Optional[int]:
    parts = token.split("_")
    if len(parts) != 4 or parts[0] != "ABS" or parts[1] != "VOICE":
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def _free_allowed_categories(state: _GrammarState) -> FrozenSet[str]:
    if not state.in_bar:
        return _PRE_BAR_ALLOWED
    if not state.current_pos:
        return _IN_BAR_NO_POS_ALLOWED
    return _IN_POS_ALLOWED


def _after_voice_allowed_categories(state: _GrammarState) -> FrozenSet[str]:
    if state.pending_voice is None or state.pending_voice not in state.anchored_voices:
        return frozenset({"REST"})
    return frozenset({"DUR", "REST"})


def _constraints_for_state(state: _GrammarState) -> GrammarConstraints:
    if state.event_state == "AFTER_VOICE":
        allowed = _after_voice_allowed_categories(state)
    elif state.event_state == "AFTER_DUR":
        allowed = frozenset({"DUP", "MEL_INT12"})
    elif state.event_state == "AFTER_DUP":
        allowed = frozenset({"MEL_INT12"})
    elif state.event_state == "AFTER_MEL":
        allowed = frozenset({"HARM_OCT"})
    elif state.event_state == "AFTER_HARM_OCT":
        allowed = frozenset({"HARM_CLASS"})
    elif state.event_state == "AFTER_HARM_CLASS":
        allowed = _free_allowed_categories(state) | frozenset({"STR"})
    elif state.event_state == "AFTER_STR":
        allowed = frozenset({"FRET"})
    else:
        allowed = _free_allowed_categories(state)
    return GrammarConstraints(allowed_categories=allowed, allow_eos=(state.event_state == "FREE"))


def _advance_free_state(state: _GrammarState, token: str, category: str) -> None:
    if category == "BAR":
        state.in_bar = True
        state.current_pos = False
        state.pending_voice = None
        return
    if category == "POS":
        if state.in_bar:
            state.current_pos = True
        state.pending_voice = None
        return
    if category == "ABS_VOICE":
        voice = _parse_abs_voice_index(token)
        if voice is not None:
            state.anchored_voices.add(voice)
        return
    if category == "ABS_BASS":
        state.anchored_voices.add(0)
        return
    if category == "ABS_SOP":
        state.anchored_voices.add(3)
        return
    if category == "VOICE":
        if state.in_bar and state.current_pos:
            state.event_state = "AFTER_VOICE"
            state.pending_voice = _parse_voice_index(token)
        return


def _fsm_next_state(state: _GrammarState, token: str) -> None:
    category = token_category(token)

    if state.event_state == "FREE":
        _advance_free_state(state, token, category)
        return

    if state.event_state == "AFTER_VOICE":
        if category == "REST":
            state.event_state = "FREE"
            state.pending_voice = None
            return
        if category == "DUR" and state.pending_voice is not None and state.pending_voice in state.anchored_voices:
            state.event_state = "AFTER_DUR"
            return
    elif state.event_state == "AFTER_DUR":
        if category == "DUP":
            state.event_state = "AFTER_DUP"
            return
        if category == "MEL_INT12":
            state.event_state = "AFTER_MEL"
            return
    elif state.event_state == "AFTER_DUP":
        if category == "MEL_INT12":
            state.event_state = "AFTER_MEL"
            return
    elif state.event_state == "AFTER_MEL":
        if category == "HARM_OCT":
            state.event_state = "AFTER_HARM_OCT"
            return
    elif state.event_state == "AFTER_HARM_OCT":
        if category == "HARM_CLASS":
            state.event_state = "AFTER_HARM_CLASS"
            return
    elif state.event_state == "AFTER_HARM_CLASS":
        if category == "STR":
            state.event_state = "AFTER_STR"
            return
    elif state.event_state == "AFTER_STR":
        if category == "FRET":
            state.event_state = "FREE"
            state.pending_voice = None
            return

    state.event_state = "FREE"
    state.pending_voice = None
    _advance_free_state(state, token, category)


def grammar_constraints(prefix_tokens: List[str]) -> GrammarConstraints:
    state = _GrammarState()
    for token in prefix_tokens:
        _fsm_next_state(state, token)
    return _constraints_for_state(state)


def allowed_next_categories(prefix_tokens: List[str]) -> Set[str]:
    """Return the set of allowed next token categories given a prefix of decoded tokens."""
    return set(grammar_constraints(prefix_tokens).allowed_categories)


class MusicRules:
    def __init__(self, vocab: Dict[str, int]) -> None:
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

        # Pre-cache some categories
        self.mel_tokens = {k: v for k, v in vocab.items() if k.startswith("MEL_INT12_")}
        self.harm_class_tokens = {k: v for k, v in vocab.items() if k.startswith("HARM_CLASS_")}

        # Consonant classes: 0 (Unison), 3 (m3), 4 (M3), 7 (P5), 8 (m6), 9 (M6)
        self.consonant_classes = {"0", "3", "4", "7", "8", "9"}

    def apply_rules(self, logits: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Rescores logits based on musical rules.
        alpha: strictness (0.0 = no rules, 1.0 = strict)
        """
        if alpha <= 0:
            return logits

        # We'll use a modified logit set
        new_logits = logits.clone()

        for token, idx in self.vocab.items():
            penalty = 0.0

            # 1. Melodic Leap Penalty (|MEL_INT12| > 7)
            if token.startswith("MEL_INT12_"):
                try:
                    val = int(token.split("_")[-1].replace("+", ""))
                    if abs(val) > 7:
                        penalty += 2.0 * alpha
                except ValueError:
                    pass

            # 2. Consonance Bias (HARM_CLASS)
            if token.startswith("HARM_CLASS_"):
                cls = token.split("_")[-1]
                if cls != "NA" and cls not in self.consonant_classes:
                    penalty += 1.0 * alpha

            if penalty > 0:
                new_logits[:, idx] -= penalty

        return new_logits
