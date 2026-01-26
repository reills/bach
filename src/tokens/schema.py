from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class EventSpec:
    """
    Defines the tokenization schema version and parameters.
    """
    version: str = "remi_tab_v1"
    tpq: int = 24
    # Dictionary of token types to their vocabulary lists (or descriptions)
    tokens: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class DescriptorSpec:
    """
    Defines the available bar-level descriptors.
    """
    fields: List[str] = field(default_factory=lambda: [
        "TIME_SIG",
        "KEY",
        "CHORD_FN", # Simplified to just 'CHORD' usually (Root+Quality)
        "DENSITY",  # Onsets per bar
        "CADENCE",  # Optional: PAC, IAC, Half
        "DIFFICULTY" # Optional: derived from intervals/speed
    ])

@dataclass
class BarPlan:
    """
    Container for the high-level descriptors of a single bar.
    These are used as conditions/labels for the model.
    """
    bar_index: int
    time_sig: str          # e.g., "4/4"
    key: str               # e.g., "Cm"
    density_bucket: str    # e.g., "DENSITY_LOW"
    pitch_range: Optional[int] = None  # semitone span (max - min MIDI pitch)
    polyphony_max: Optional[int] = None  # max simultaneous onsets at any POS
    chord_token: Optional[str] = None # e.g., "C:min"
    cadence_token: Optional[str] = None
    difficulty_token: Optional[str] = None

    def to_token_string(self) -> str:
        """
        Returns a string of descriptor tokens to prepend to the bar.
        Example: "TIME_SIG_4_4 KEY_Cm DENSITY_LOW"
        """
        toks = [
            f"TIME_SIG_{self.time_sig.replace('/', '_')}",
            f"KEY_{self.key}",
            self.density_bucket
        ]
        if self.chord_token:
            toks.append(self.chord_token)
        if self.cadence_token:
            toks.append(self.cadence_token)
        # Note: DIFFICULTY is often a piece-level tag, but can be bar-level.
        return " ".join(toks)
