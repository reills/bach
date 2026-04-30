from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.emi.fragments import Fragment, extract_fragments
from src.instrumental_v3.representation import FIELD_NAMES as V3_FIELD_NAMES, STATE_HOLD, STATE_NOTE, SliceEvent
from src.instrumental_v4.representation import V4_FIELD_NAMES, V4_FEATURE_SPECS, V4Piece

PHRASE_ROLE_NAMES = [
    "UNKNOWN",
    "OPENING",
    "SUBJECT_ENTRY",
    "ANSWER_ENTRY",
    "COUNTERSUBJECT",
    "EPISODE",
    "SEQUENCE",
    "CADENTIAL_PREP",
    "CADENCE",
    "CLOSING",
]

CONTOUR_BUCKET_NAMES = [
    "UNKNOWN",
    "STATIC",
    "ASCENDING_STEPWISE",
    "DESCENDING_STEPWISE",
    "ASCENDING_LEAPY",
    "DESCENDING_LEAPY",
    "ARCH",
    "INVERTED_ARCH",
    "ZIGZAG",
    "REPEATED_NOTE",
    "MIXED",
]

RHYTHM_BUCKET_NAMES = [
    "UNKNOWN",
    "EVEN_16THS",
    "EVEN_8THS",
    "EVEN_QUARTERS",
    "LONG_SHORT",
    "SHORT_LONG",
    "DOTTED",
    "SYNCOPATED",
    "SUSPENSION",
    "MIXED",
]

PHRASE_ROLE_TO_ID = {name: idx for idx, name in enumerate(PHRASE_ROLE_NAMES)}
CONTOUR_BUCKET_TO_ID = {name: idx for idx, name in enumerate(CONTOUR_BUCKET_NAMES)}
RHYTHM_BUCKET_TO_ID = {name: idx for idx, name in enumerate(RHYTHM_BUCKET_NAMES)}

V5_EMI_FIELD_NAMES = [
    "phrase_role",
    "fragment_contour_bucket",
    "fragment_rhythm_bucket",
]

V5_FIELD_NAMES = list(V4_FIELD_NAMES) + V5_EMI_FIELD_NAMES
V5_FEATURE_SPECS = {
    **V4_FEATURE_SPECS,
    "phrase_role": len(PHRASE_ROLE_NAMES),
    "fragment_contour_bucket": len(CONTOUR_BUCKET_NAMES),
    "fragment_rhythm_bucket": len(RHYTHM_BUCKET_NAMES),
}


@dataclass(frozen=True)
class V5Piece:
    piece_id: str
    source_path: str
    tpq: int
    grid_ticks: int
    time_signature: str
    key: str | None
    key_pc: int
    mode: int
    bar_len_ticks: int
    steps_per_bar: int
    rows: list[list[int]]
    fragments: list[Fragment]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fragments"] = [fragment.to_dict() for fragment in self.fragments]
        return data


def build_v5_piece(
    piece: V4Piece,
    *,
    length_slices: int = 8,
    hop_slices: int = 4,
    min_notes: int = 2,
) -> V5Piece:
    v3_piece = _v4_to_slice_piece(piece)
    fragments = extract_fragments(
        v3_piece,
        length_slices=length_slices,
        hop_slices=hop_slices,
        min_notes=min_notes,
    )
    rows = _annotate_rows(piece.rows, fragments)
    return V5Piece(
        piece_id=piece.piece_id,
        source_path=piece.source_path,
        tpq=piece.tpq,
        grid_ticks=piece.grid_ticks,
        time_signature=piece.time_signature,
        key=piece.key,
        key_pc=piece.key_pc,
        mode=piece.mode,
        bar_len_ticks=piece.bar_len_ticks,
        steps_per_bar=piece.steps_per_bar,
        rows=rows,
        fragments=fragments,
    )


def classify_contour_bucket(melodic_intervals: list[int]) -> str:
    if not melodic_intervals:
        return "UNKNOWN"
    if all(interval == 0 for interval in melodic_intervals):
        return "REPEATED_NOTE"

    nonzero = [interval for interval in melodic_intervals if interval != 0]
    if not nonzero:
        return "STATIC"
    signs = [1 if interval > 0 else -1 for interval in nonzero]
    abs_intervals = [abs(interval) for interval in nonzero]
    stepwise = all(interval <= 2 for interval in abs_intervals)

    if all(sign > 0 for sign in signs):
        return "ASCENDING_STEPWISE" if stepwise else "ASCENDING_LEAPY"
    if all(sign < 0 for sign in signs):
        return "DESCENDING_STEPWISE" if stepwise else "DESCENDING_LEAPY"

    changes = sum(1 for idx in range(1, len(signs)) if signs[idx] != signs[idx - 1])
    if changes == 1 and signs[0] > 0 and signs[-1] < 0:
        return "ARCH"
    if changes == 1 and signs[0] < 0 and signs[-1] > 0:
        return "INVERTED_ARCH"
    if changes >= 2:
        return "ZIGZAG"
    return "MIXED"


def classify_rhythm_bucket(rhythm_steps: list[int], state_pattern: list[int] | None = None) -> str:
    if not rhythm_steps:
        return "UNKNOWN"
    if len(rhythm_steps) >= 2 and rhythm_steps[0] >= 4 and (state_pattern or [])[1:2] == [STATE_HOLD]:
        return "SUSPENSION"
    if all(step == rhythm_steps[0] for step in rhythm_steps):
        if rhythm_steps[0] == 1:
            return "EVEN_16THS"
        if rhythm_steps[0] == 2:
            return "EVEN_8THS"
        if rhythm_steps[0] == 4:
            return "EVEN_QUARTERS"
        return "MIXED"
    if _has_dotted_pair(rhythm_steps):
        return "DOTTED"
    if _looks_syncopated(rhythm_steps):
        return "SYNCOPATED"
    if len(rhythm_steps) >= 2 and rhythm_steps[0] > rhythm_steps[1]:
        return "LONG_SHORT"
    if len(rhythm_steps) >= 2 and rhythm_steps[0] < rhythm_steps[1]:
        return "SHORT_LONG"
    return "MIXED"


def phrase_role_id(role: str | None) -> int:
    if role == "CADENTIAL_PREPARATION":
        role = "CADENTIAL_PREP"
    return PHRASE_ROLE_TO_ID.get(role or "UNKNOWN", PHRASE_ROLE_TO_ID["UNKNOWN"])


def contour_bucket_id(name: str | None) -> int:
    return CONTOUR_BUCKET_TO_ID.get(name or "UNKNOWN", CONTOUR_BUCKET_TO_ID["UNKNOWN"])


def rhythm_bucket_id(name: str | None) -> int:
    return RHYTHM_BUCKET_TO_ID.get(name or "UNKNOWN", RHYTHM_BUCKET_TO_ID["UNKNOWN"])


def _annotate_rows(rows: list[list[int]], fragments: list[Fragment]) -> list[list[int]]:
    by_row: list[list[Fragment]] = [[] for _ in rows]
    for fragment in fragments:
        end = min(len(rows), fragment.start_slice + fragment.length_slices)
        for idx in range(fragment.start_slice, end):
            by_row[idx].append(fragment)

    annotated: list[list[int]] = []
    for idx, row in enumerate(rows):
        fragment = _select_fragment_for_row(row, by_row[idx])
        if fragment is None:
            extras = [
                PHRASE_ROLE_TO_ID["UNKNOWN"],
                CONTOUR_BUCKET_TO_ID["UNKNOWN"],
                RHYTHM_BUCKET_TO_ID["UNKNOWN"],
            ]
        else:
            extras = [
                phrase_role_id(fragment.phrase_role),
                contour_bucket_id(classify_contour_bucket(fragment.melodic_intervals)),
                rhythm_bucket_id(classify_rhythm_bucket(fragment.rhythm_steps, fragment.state_pattern)),
            ]
        annotated.append(row[:] + extras)
    return annotated


def _select_fragment_for_row(row: list[int], candidates: list[Fragment]) -> Fragment | None:
    if not candidates:
        return None
    active_note_voices = [
        voice for voice in (0, 1) if row[V4_FIELD_NAMES.index(f"v{voice}_state")] == STATE_NOTE
    ]
    for voice in reversed(active_note_voices):
        for fragment in candidates:
            if fragment.voice == voice:
                return fragment
    return sorted(candidates, key=lambda fragment: (fragment.start_slice, fragment.voice))[0]


def _v4_to_slice_piece(piece: V4Piece) -> _SlicePiece:
    return _SlicePiece(
        piece_id=piece.piece_id,
        source_path=piece.source_path,
        grid_ticks=piece.grid_ticks,
        steps_per_bar=piece.steps_per_bar,
        key=piece.key,
        key_pc=piece.key_pc,
        mode=piece.mode,
        slices=[SliceEvent(row[: len(V3_FIELD_NAMES)]) for row in piece.rows],
    )


def _has_dotted_pair(steps: list[int]) -> bool:
    return any((a, b) in {(3, 1), (1, 3), (6, 2), (2, 6)} for a, b in zip(steps, steps[1:]))


def _looks_syncopated(steps: list[int]) -> bool:
    return len(steps) >= 3 and any(step == 3 for step in steps) and not _has_dotted_pair(steps)


@dataclass(frozen=True)
class _SlicePiece:
    piece_id: str
    source_path: str
    grid_ticks: int
    steps_per_bar: int
    key: str | None
    key_pc: int
    mode: int
    slices: list[SliceEvent]
