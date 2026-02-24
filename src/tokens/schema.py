from dataclasses import dataclass, field
from typing import Dict, List, Optional

SCHEMA_VERSION = "remi_tab_v1"
STAGE1_TPQ = 24


def _default_event_tokens() -> Dict[str, List[str]]:
    return {
        "structural": [
            "BAR",
            "TIME_SIG_*",
            "KEY_*",
            "TEMPO_*",
            "POS_*",
        ],
        "voice_events": [
            "VOICE_{0..K-1}",
            "DUR_{ticks}",
            "REST_{ticks}",
        ],
        "anchors": [
            "ABS_VOICE_{v}_{MIDI}",
            "ABS_LOW_{MIDI}",
            "ABS_HIGH_{MIDI}",
        ],
        "intervals": [
            "MEL_INT12_{-24..+24}",
            "HARM_OCT_{-2..4|NA}",
            "HARM_CLASS_{0..11|NA}",
        ],
        "doubling": ["DUP_{1..N}"],
    }


@dataclass
class EventSpec:
    """
    Stage-1 event schema. This schema is versioned and fixed to TPQ=24.
    """

    version: str = SCHEMA_VERSION
    tpq: int = STAGE1_TPQ
    tokens: Dict[str, List[str]] = field(default_factory=_default_event_tokens)

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported event schema version: {self.version!r}; expected {SCHEMA_VERSION!r}"
            )
        if self.tpq != STAGE1_TPQ:
            raise ValueError(f"Stage-1 requires tpq={STAGE1_TPQ}, got {self.tpq}")


@dataclass
class DescriptorSpec:
    """
    Declares supported bar-level descriptor fields for Stage-1 data rows.
    """

    fields: List[str] = field(
        default_factory=lambda: [
            "TIME_SIG",
            "KEY",
            "CHORD_FN",
            "DENSITY",
            "CADENCE",
            "DIFFICULTY",
        ]
    )


@dataclass
class BarPlan:
    """
    Container for high-level descriptors of a single bar.
    """

    bar_index: int
    time_sig: str
    key: str
    density_bucket: str
    pitch_range: Optional[int] = None
    polyphony_max: Optional[int] = None
    chord_token: Optional[str] = None
    cadence_token: Optional[str] = None
    difficulty_token: Optional[str] = None

    def to_token_string(self) -> str:
        toks = [
            f"TIME_SIG_{self.time_sig.replace('/', '_')}",
            f"KEY_{self.key}",
            self.density_bucket,
        ]
        if self.chord_token:
            toks.append(self.chord_token)
        if self.cadence_token:
            toks.append(self.cadence_token)
        return " ".join(toks)
