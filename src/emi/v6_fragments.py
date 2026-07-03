from __future__ import annotations

from collections import Counter
from typing import Iterable

from src.emi.buckets import classify_contour_bucket, classify_rhythm_bucket
from src.emi.cmmc import cmmc_function_for_role
from src.emi.fragments import Fragment, hash_signature, summarize_fragments
from src.emi.planner import (
    cadence_type_for_role,
    harmonic_function_for_role,
    phrase_role_to_speac,
)
from src.instrumental_v6.representation import (
    GLOBAL_FIELD_NAMES,
    ROLE_TO_ID,
    STATE_HOLD,
    STATE_NOTE,
    VOICE_FIELD_NAMES,
    InstrumentalV6Piece,
)

ROLE_ID_TO_NAME = {value: key for key, value in ROLE_TO_ID.items()}


def extract_v6_fragments(
    piece: InstrumentalV6Piece,
    *,
    length_slices: int,
    hop_slices: int | None = None,
    min_notes: int = 2,
) -> list[Fragment]:
    if length_slices <= 0:
        raise ValueError("length_slices must be positive")
    if min_notes <= 0:
        raise ValueError("min_notes must be positive")
    if not piece.voice_rows:
        return []
    hop = hop_slices if hop_slices is not None else max(1, length_slices // 2)
    if hop <= 0:
        raise ValueError("hop_slices must be positive")

    fragments: list[Fragment] = []
    for voice in range(piece.voice_count):
        for start in range(0, max(0, len(piece.voice_rows) - length_slices + 1), hop):
            notes = _note_events(piece, voice, start, length_slices)
            if len(notes) < min_notes:
                continue
            end = start + length_slices
            state_pattern = [
                int(row[voice][_voice_col("state")])
                for row in piece.voice_rows[start:end]
            ]
            melodic_intervals = [
                notes[index][2] - notes[index - 1][2]
                for index in range(1, len(notes))
            ]
            rhythm_steps = [duration for _, duration, _, _ in notes]
            verticals = _vertical_intervals(piece, voice, start, end)
            phrase_role = _window_role(piece.global_rows[start:end])
            local_key_pc = _window_local_key(piece.global_rows[start:end], piece.key_pc)
            cmmc_function = cmmc_function_for_role(phrase_role)
            cadence_type = cadence_type_for_role(phrase_role)
            harmonic_function = harmonic_function_for_role(phrase_role)
            contour_bucket = classify_contour_bucket(melodic_intervals)
            rhythm_bucket = classify_rhythm_bucket(rhythm_steps, state_pattern)
            contour_hash = hash_signature({"mel": melodic_intervals, "rhythm": rhythm_steps})
            copy_hash = hash_signature(
                {
                    "mel": melodic_intervals,
                    "rhythm": rhythm_steps,
                    "degrees": [notes[0][3], notes[-1][3]],
                    "verticals": verticals,
                    "states": state_pattern,
                }
            )
            transposition_hash = hash_signature(
                {
                    "mel": melodic_intervals,
                    "rhythm": rhythm_steps,
                    "degree_span": notes[-1][3] - notes[0][3],
                    "states": state_pattern,
                    "cadence": cadence_type,
                    "cmmc": cmmc_function,
                }
            )
            fingerprint = hash_signature(
                {
                    "piece": piece.piece_id,
                    "voice": voice,
                    "start": start,
                    "mel": melodic_intervals,
                    "rhythm": rhythm_steps,
                    "states": state_pattern,
                    "role": phrase_role,
                    "cmmc": cmmc_function,
                }
            )
            fragments.append(
                Fragment(
                    id=f"{piece.piece_id}_v{voice}_s{start}_l{length_slices}",
                    piece_id=piece.piece_id,
                    source_path=piece.source_path,
                    voice=voice,
                    start_slice=start,
                    length_slices=length_slices,
                    start_bar=piece.global_rows[start][_global_col("bar")],
                    start_pos=piece.global_rows[start][_global_col("pos")],
                    beats=length_slices * piece.grid_ticks / 24.0,
                    phrase_role=phrase_role,
                    key=piece.key,
                    key_pc=piece.key_pc,
                    mode=piece.mode,
                    start_pitch=notes[0][2],
                    end_pitch=notes[-1][2],
                    start_degree=notes[0][3],
                    end_degree=notes[-1][3],
                    melodic_intervals=melodic_intervals,
                    rhythm_steps=rhythm_steps,
                    vertical_intervals=verticals,
                    state_pattern=state_pattern,
                    contour_hash=contour_hash,
                    fingerprint=fingerprint,
                    speac_label=phrase_role_to_speac(phrase_role),
                    cmmc_function=cmmc_function,
                    cadence_type=cadence_type,
                    contour_bucket=contour_bucket,
                    rhythm_bucket=rhythm_bucket,
                    local_key_pc=local_key_pc,
                    harmonic_function=harmonic_function,
                    entry_degree=notes[0][3],
                    exit_degree=notes[-1][3],
                    min_pitch=min(note[2] for note in notes),
                    max_pitch=max(note[2] for note in notes),
                    copy_hash=copy_hash,
                    transposition_hash=transposition_hash,
                )
            )
    return fragments


def summarize_v6_fragments(fragments: Iterable[Fragment]) -> dict[str, object]:
    return summarize_fragments(fragments)


def _note_events(
    piece: InstrumentalV6Piece,
    voice: int,
    start: int,
    length: int,
) -> list[tuple[int, int, int, int]]:
    events: list[tuple[int, int, int, int]] = []
    end = start + length
    for row_index in range(start, end):
        voice_row = piece.voice_rows[row_index][voice]
        if voice_row[_voice_col("state")] != STATE_NOTE:
            continue
        pitch = int(voice_row[_voice_col("pitch")])
        if pitch <= 0:
            continue
        duration = max(1, int(voice_row[_voice_col("dur")]))
        degree = int(voice_row[_voice_col("degree")])
        events.append((row_index - start, duration, pitch, degree))
    return events


def _vertical_intervals(
    piece: InstrumentalV6Piece,
    voice: int,
    start: int,
    end: int,
) -> list[int]:
    intervals: list[int] = []
    for row in piece.voice_rows[start:end]:
        pitch = _active_pitch(row[voice])
        if pitch is None:
            continue
        for other_voice in range(piece.voice_count):
            if other_voice == voice:
                continue
            other_pitch = _active_pitch(row[other_voice])
            if other_pitch is not None:
                intervals.append(abs(other_pitch - pitch))
    return intervals


def _window_role(global_rows: list[list[int]]) -> str:
    if not global_rows:
        return "UNKNOWN"
    role_col = _global_col("section_role")
    counts = Counter(ROLE_ID_TO_NAME.get(row[role_col], "UNKNOWN") for row in global_rows)
    role, _ = counts.most_common(1)[0]
    return role


def _window_local_key(global_rows: list[list[int]], fallback: int) -> int:
    if not global_rows:
        return fallback
    local_key_col = _global_col("local_key_pc")
    counts = Counter(int(row[local_key_col]) for row in global_rows)
    local_key, _ = counts.most_common(1)[0]
    return max(0, min(12, local_key))


def _active_pitch(voice_row: list[int]) -> int | None:
    state = voice_row[_voice_col("state")]
    pitch = voice_row[_voice_col("pitch")]
    return pitch if state in {STATE_NOTE, STATE_HOLD} and pitch > 0 else None


def _global_col(name: str) -> int:
    return GLOBAL_FIELD_NAMES.index(name)


def _voice_col(name: str) -> int:
    return VOICE_FIELD_NAMES.index(name)
