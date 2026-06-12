from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class DroppedNoteDiagnostic:
    source_event_id: str
    start_tick: int
    voice_id: int
    pitch_midi: int
    reason: str


@dataclass(frozen=True)
class OctaveShiftDiagnostic:
    source_event_id: str
    start_tick: int
    voice_id: int
    original_pitch_midi: int
    arranged_pitch_midi: int
    semitone_shift: int


@dataclass(frozen=True)
class ImpossibleChordDiagnostic:
    onset_tick: int
    source_event_ids: list[str]
    reason: str


@dataclass(frozen=True)
class RangeChangeDiagnostic:
    source_event_id: str
    start_tick: int
    original_pitch_midi: int
    arranged_pitch_midi: int
    reason: str


@dataclass(frozen=True)
class HandPositionCompromiseDiagnostic:
    onset_tick: int
    source_event_ids: list[str]
    min_fret: int
    max_fret: int
    span_frets: int
    reason: str


@dataclass(frozen=True)
class GuitarConversionDiagnostics:
    dropped_notes: list[DroppedNoteDiagnostic] = field(default_factory=list)
    octave_shifted_notes: list[OctaveShiftDiagnostic] = field(default_factory=list)
    impossible_chords: list[ImpossibleChordDiagnostic] = field(default_factory=list)
    range_changes: list[RangeChangeDiagnostic] = field(default_factory=list)
    hand_position_compromises: list[HandPositionCompromiseDiagnostic] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "droppedNotes": [asdict(item) for item in self.dropped_notes],
            "octaveShiftedNotes": [asdict(item) for item in self.octave_shifted_notes],
            "impossibleChords": [asdict(item) for item in self.impossible_chords],
            "rangeChanges": [asdict(item) for item in self.range_changes],
            "handPositionCompromises": [asdict(item) for item in self.hand_position_compromises],
            "warnings": list(self.warnings),
        }
