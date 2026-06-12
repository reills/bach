from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from src.tabber import DEFAULT_MAX_FRET, STANDARD_GUITAR_TUNING

OctaveShiftPolicy = Literal["none", "below_range", "outside_range"]
GuitarDifficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class GuitarArrangementSettings:
    difficulty: GuitarDifficulty = "medium"
    tuning: tuple[int, ...] = STANDARD_GUITAR_TUNING
    max_fret: int = DEFAULT_MAX_FRET
    octave_shift_policy: OctaveShiftPolicy = "outside_range"
    allow_drop_notes: bool = True
    preserve_melody: bool = True
    preserve_bass: bool = True
    max_hand_span_frets: int | None = None
    max_notes_per_onset: int | None = None
    preferred_position: int | None = None
    target_instrument: str = "classical_guitar"
    midi_program: int = 24

    def __post_init__(self) -> None:
        normalized_tuning = _normalize_tuning(self.tuning)
        object.__setattr__(self, "tuning", normalized_tuning)
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"unsupported guitar difficulty: {self.difficulty}")
        _validate_int("max_fret", self.max_fret)
        if self.max_fret < 0:
            raise ValueError("max_fret must be non-negative")
        if self.max_hand_span_frets is not None:
            _validate_int("max_hand_span_frets", self.max_hand_span_frets)
            if self.max_hand_span_frets < 0:
                raise ValueError("max_hand_span_frets must be non-negative")
        if self.max_notes_per_onset is not None:
            _validate_int("max_notes_per_onset", self.max_notes_per_onset)
            if self.max_notes_per_onset <= 0:
                raise ValueError("max_notes_per_onset must be positive")
        if self.preferred_position is not None:
            _validate_int("preferred_position", self.preferred_position)
            if self.preferred_position < 0:
                raise ValueError("preferred_position must be non-negative")
        if self.octave_shift_policy not in ("none", "below_range", "outside_range"):
            raise ValueError(f"unsupported octave_shift_policy: {self.octave_shift_policy}")
        if not self.target_instrument:
            raise ValueError("target_instrument must be non-empty")
        _validate_int("midi_program", self.midi_program)
        if not 0 <= self.midi_program <= 127:
            raise ValueError("midi_program must be between 0 and 127")

    @property
    def lowest_pitch(self) -> int:
        return min(self.tuning)

    @property
    def highest_pitch(self) -> int:
        return max(open_pitch + self.max_fret for open_pitch in self.tuning)

    @property
    def resolved_max_hand_span_frets(self) -> int:
        if self.max_hand_span_frets is not None:
            return self.max_hand_span_frets
        return {
            "easy": 4,
            "medium": 5,
            "hard": 7,
        }[self.difficulty]

    @property
    def resolved_max_notes_per_onset(self) -> int:
        if self.max_notes_per_onset is not None:
            return min(self.max_notes_per_onset, len(self.tuning))
        return {
            "easy": 3,
            "medium": 4,
            "hard": 5,
        }[self.difficulty]

    @classmethod
    def for_legacy_compose(cls, *, max_fret: int = DEFAULT_MAX_FRET) -> "GuitarArrangementSettings":
        return cls(
            difficulty="hard",
            max_fret=max_fret,
            octave_shift_policy="below_range",
            allow_drop_notes=False,
            max_hand_span_frets=12,
            max_notes_per_onset=len(STANDARD_GUITAR_TUNING),
        )


def _normalize_tuning(tuning: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(tuning)
    if not normalized:
        raise ValueError("tuning must define at least one string")
    for pitch in normalized:
        _validate_int("tuning pitch", pitch)
        if not 0 <= pitch <= 127:
            raise ValueError("tuning pitches must be valid MIDI values")
    return normalized


def _validate_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
