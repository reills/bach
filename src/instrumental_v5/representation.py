from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.emi.buckets import (
    CADENCE_TYPE_NAMES,
    CONTOUR_BUCKET_NAMES,
    HARMONIC_FUNCTION_NAMES,
    RHYTHM_BUCKET_NAMES,
    SPEAC_LABEL_NAMES,
    classify_contour_bucket,
    classify_rhythm_bucket,
)
from src.emi.fragments import Fragment, extract_fragments
from src.emi.planner import (
    PHRASE_ROLE_NAMES,
    build_phrase_plan,
    cadence_type_id,
    harmonic_function_id,
    phrase_role_id,
    plan_step_for_row,
    speac_label_id,
)
from src.instrumental_v3.representation import FIELD_NAMES as V3_FIELD_NAMES, STATE_NOTE, SliceEvent
from src.instrumental_v4.representation import V4_FIELD_NAMES, V4_FEATURE_SPECS, V4Piece

PHRASE_ROLE_TO_ID = {name: idx for idx, name in enumerate(PHRASE_ROLE_NAMES)}
SPEAC_LABEL_TO_ID = {name: idx for idx, name in enumerate(SPEAC_LABEL_NAMES)}
CADENCE_TYPE_TO_ID = {name: idx for idx, name in enumerate(CADENCE_TYPE_NAMES)}
HARMONIC_FUNCTION_TO_ID = {name: idx for idx, name in enumerate(HARMONIC_FUNCTION_NAMES)}
CONTOUR_BUCKET_TO_ID = {name: idx for idx, name in enumerate(CONTOUR_BUCKET_NAMES)}
RHYTHM_BUCKET_TO_ID = {name: idx for idx, name in enumerate(RHYTHM_BUCKET_NAMES)}

V5_EMI_FIELD_NAMES = [
    "phrase_role",
    "speac_label",
    "cadence_target",
    "harmonic_function",
    "local_key_pc",
    "retrieved_contour_bucket",
    "retrieved_rhythm_bucket",
]

V5_FIELD_NAMES = list(V4_FIELD_NAMES) + V5_EMI_FIELD_NAMES
V5_FEATURE_SPECS = {
    **V4_FEATURE_SPECS,
    "phrase_role": len(PHRASE_ROLE_NAMES),
    "speac_label": len(SPEAC_LABEL_NAMES),
    "cadence_target": len(CADENCE_TYPE_NAMES),
    "harmonic_function": len(HARMONIC_FUNCTION_NAMES),
    "local_key_pc": 13,
    "retrieved_contour_bucket": len(CONTOUR_BUCKET_NAMES),
    "retrieved_rhythm_bucket": len(RHYTHM_BUCKET_NAMES),
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
    measures = max(1, (len(piece.rows) + piece.steps_per_bar - 1) // piece.steps_per_bar)
    plan = build_phrase_plan(
        measures=measures,
        key=piece.key,
        key_pc=piece.key_pc,
        mode=piece.mode,
        texture=2,
    )
    rows = _annotate_rows(piece.rows, fragments, steps_per_bar=piece.steps_per_bar, plan=plan)
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


def contour_bucket_id(name: str | None) -> int:
    return CONTOUR_BUCKET_TO_ID.get(name or "UNKNOWN", CONTOUR_BUCKET_TO_ID["UNKNOWN"])


def rhythm_bucket_id(name: str | None) -> int:
    return RHYTHM_BUCKET_TO_ID.get(name or "UNKNOWN", RHYTHM_BUCKET_TO_ID["UNKNOWN"])


def _annotate_rows(
    rows: list[list[int]],
    fragments: list[Fragment],
    *,
    steps_per_bar: int,
    plan,
) -> list[list[int]]:
    by_row: list[list[Fragment]] = [[] for _ in rows]
    for fragment in fragments:
        end = min(len(rows), fragment.start_slice + fragment.length_slices)
        for idx in range(fragment.start_slice, end):
            by_row[idx].append(fragment)

    annotated: list[list[int]] = []
    for idx, row in enumerate(rows):
        plan_step = plan_step_for_row(idx, steps_per_bar=steps_per_bar, plan=plan)
        fragment = _select_fragment_for_row(row, by_row[idx])
        if fragment is None:
            contour_id = CONTOUR_BUCKET_TO_ID["UNKNOWN"]
            rhythm_id = RHYTHM_BUCKET_TO_ID["UNKNOWN"]
        else:
            contour_id = contour_bucket_id(
                fragment.contour_bucket or classify_contour_bucket(fragment.melodic_intervals)
            )
            rhythm_id = rhythm_bucket_id(
                fragment.rhythm_bucket or classify_rhythm_bucket(fragment.rhythm_steps, fragment.state_pattern)
            )
        extras = [
            phrase_role_id(plan_step.phrase_role),
            speac_label_id(plan_step.speac_label),
            cadence_type_id(plan_step.cadence_target),
            harmonic_function_id(plan_step.harmonic_function),
            max(0, min(12, int(plan_step.local_key_pc))),
            contour_id,
            rhythm_id,
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
