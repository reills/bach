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
from src.emi.cmmc import CMMC_FUNCTION_NAMES, cmmc_function_id
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
from src.instrumental_v3.representation import STATE_HOLD
from src.instrumental_v4.representation import V4_FIELD_NAMES, V4_FEATURE_SPECS, V4Piece

PHRASE_ROLE_TO_ID = {name: idx for idx, name in enumerate(PHRASE_ROLE_NAMES)}
SPEAC_LABEL_TO_ID = {name: idx for idx, name in enumerate(SPEAC_LABEL_NAMES)}
CMMC_FUNCTION_TO_ID = {name: idx for idx, name in enumerate(CMMC_FUNCTION_NAMES)}
CADENCE_TYPE_TO_ID = {name: idx for idx, name in enumerate(CADENCE_TYPE_NAMES)}
HARMONIC_FUNCTION_TO_ID = {name: idx for idx, name in enumerate(HARMONIC_FUNCTION_NAMES)}
CONTOUR_BUCKET_TO_ID = {name: idx for idx, name in enumerate(CONTOUR_BUCKET_NAMES)}
RHYTHM_BUCKET_TO_ID = {name: idx for idx, name in enumerate(RHYTHM_BUCKET_NAMES)}

CP_MAX_MOTION = 24
CP_MAX_SPACING = 19
CP_MOTION_TYPE_NAMES = ["UNKNOWN", "STATIC", "OBLIQUE", "CONTRARY", "SIMILAR", "PARALLEL"]
CP_MOTION_TYPE_TO_ID = {name: idx for idx, name in enumerate(CP_MOTION_TYPE_NAMES)}

V5_COUNTERPOINT_FIELD_NAMES = [
    "cp_v0_motion",
    "cp_v1_motion",
    "cp_motion_type",
    "cp_prev_interval_class",
    "cp_curr_interval_class",
    "cp_parallel_perfect",
    "cp_direct_perfect",
    "cp_voice_crossing",
    "cp_spacing_violation",
]

V5_EMI_FIELD_NAMES = [
    "phrase_role",
    "speac_label",
    "cmmc_function",
    "cadence_target",
    "harmonic_function",
    "local_key_pc",
    "retrieved_contour_bucket",
    "retrieved_rhythm_bucket",
]

V5_FIELD_NAMES = list(V4_FIELD_NAMES) + V5_COUNTERPOINT_FIELD_NAMES + V5_EMI_FIELD_NAMES
V5_FEATURE_SPECS = {
    **V4_FEATURE_SPECS,
    "cp_v0_motion": (CP_MAX_MOTION * 2) + 2,
    "cp_v1_motion": (CP_MAX_MOTION * 2) + 2,
    "cp_motion_type": len(CP_MOTION_TYPE_NAMES),
    "cp_prev_interval_class": 13,
    "cp_curr_interval_class": 13,
    "cp_parallel_perfect": 2,
    "cp_direct_perfect": 2,
    "cp_voice_crossing": 2,
    "cp_spacing_violation": 2,
    "phrase_role": len(PHRASE_ROLE_NAMES),
    "speac_label": len(SPEAC_LABEL_NAMES),
    "cmmc_function": len(CMMC_FUNCTION_NAMES),
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
    rows = _annotate_rows(
        apply_counterpoint_features(piece.rows),
        fragments,
        steps_per_bar=piece.steps_per_bar,
        plan=plan,
    )
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


def apply_counterpoint_features(rows: list[list[int]]) -> list[list[int]]:
    """Append transition-level voice-leading targets derived from adjacent sonorities."""

    out: list[list[int]] = []
    previous_row: list[int] | None = None
    for row in rows:
        features = counterpoint_features_for_transition(previous_row, row)
        out.append(row[:] + features)
        previous_row = row
    return out


def counterpoint_features_for_transition(previous_row: list[int] | None, current_row: list[int]) -> list[int]:
    previous_active = None if previous_row is None else (_active_pitch(previous_row, 0), _active_pitch(previous_row, 1))
    current_active = (_active_pitch(current_row, 0), _active_pitch(current_row, 1))
    previous_interval_class = 0 if previous_active is None else _interval_class_id(previous_active)
    return _counterpoint_features(previous_active, current_active, previous_interval_class)


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
            phrase_role = plan_step.phrase_role
            speac_label = plan_step.speac_label
            cmmc_function = plan_step.cmmc_function
            cadence_target = plan_step.cadence_target
            harmonic_function = plan_step.harmonic_function
            local_key_pc = plan_step.local_key_pc
            contour_id = CONTOUR_BUCKET_TO_ID["UNKNOWN"]
            rhythm_id = RHYTHM_BUCKET_TO_ID["UNKNOWN"]
        else:
            phrase_role = fragment.phrase_role
            speac_label = fragment.speac_label
            cmmc_function = fragment.cmmc_function
            cadence_target = fragment.cadence_type
            harmonic_function = fragment.harmonic_function
            local_key_pc = fragment.local_key_pc
            contour_id = contour_bucket_id(
                fragment.contour_bucket or classify_contour_bucket(fragment.melodic_intervals)
            )
            rhythm_id = rhythm_bucket_id(
                fragment.rhythm_bucket or classify_rhythm_bucket(fragment.rhythm_steps, fragment.state_pattern)
            )
        extras = [
            phrase_role_id(phrase_role),
            speac_label_id(speac_label),
            cmmc_function_id(cmmc_function),
            cadence_type_id(cadence_target),
            harmonic_function_id(harmonic_function),
            max(0, min(12, int(local_key_pc))),
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


def _counterpoint_features(
    previous_active: tuple[int | None, int | None] | None,
    current_active: tuple[int | None, int | None],
    previous_interval_class: int,
) -> list[int]:
    current_interval_class = _interval_class_id(current_active)
    crossing = int(
        current_active[0] is not None and current_active[1] is not None and current_active[0] > current_active[1]
    )
    spacing_violation = int(
        current_active[0] is not None
        and current_active[1] is not None
        and current_active[1] - current_active[0] > CP_MAX_SPACING
    )
    if (
        previous_active is None
        or previous_active[0] is None
        or previous_active[1] is None
        or current_active[0] is None
        or current_active[1] is None
    ):
        return [
            0,
            0,
            CP_MOTION_TYPE_TO_ID["UNKNOWN"],
            previous_interval_class,
            current_interval_class,
            0,
            0,
            crossing,
            spacing_violation,
        ]

    motion0 = current_active[0] - previous_active[0]
    motion1 = current_active[1] - previous_active[1]
    motion_type = _motion_type(motion0, motion1)
    same_direction = _same_nonzero_direction(motion0, motion1)
    curr_pc = current_interval_class - 1 if current_interval_class > 0 else None
    prev_pc = previous_interval_class - 1 if previous_interval_class > 0 else None
    parallel_perfect = int(
        same_direction and prev_pc in {0, 7} and curr_pc == prev_pc
    )
    direct_perfect = int(
        same_direction
        and curr_pc in {0, 7}
        and curr_pc != prev_pc
        and (abs(motion0) > 2 or abs(motion1) > 2)
    )
    return [
        _encode_motion(motion0),
        _encode_motion(motion1),
        CP_MOTION_TYPE_TO_ID[motion_type],
        previous_interval_class,
        current_interval_class,
        parallel_perfect,
        direct_perfect,
        crossing,
        spacing_violation,
    ]


def _active_pitch(row: list[int], voice: int) -> int | None:
    state = row[V4_FIELD_NAMES.index(f"v{voice}_state")]
    pitch = row[V4_FIELD_NAMES.index(f"v{voice}_pitch")]
    return int(pitch) if state in {STATE_NOTE, STATE_HOLD} and pitch > 0 else None


def _interval_class_id(active: tuple[int | None, int | None]) -> int:
    if active[0] is None or active[1] is None:
        return 0
    return abs(active[1] - active[0]) % 12 + 1


def _encode_motion(delta: int) -> int:
    clipped = max(-CP_MAX_MOTION, min(CP_MAX_MOTION, int(delta)))
    return clipped + CP_MAX_MOTION + 1


def _motion_type(motion0: int, motion1: int) -> str:
    if motion0 == 0 and motion1 == 0:
        return "STATIC"
    if motion0 == 0 or motion1 == 0:
        return "OBLIQUE"
    if motion0 * motion1 < 0:
        return "CONTRARY"
    if motion0 == motion1:
        return "PARALLEL"
    if _same_nonzero_direction(motion0, motion1):
        return "SIMILAR"
    return "UNKNOWN"


def _same_nonzero_direction(left: int, right: int) -> bool:
    return (left > 0 and right > 0) or (left < 0 and right < 0)


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
