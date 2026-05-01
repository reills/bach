from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Protocol

from src.emi.buckets import classify_contour_bucket, classify_rhythm_bucket
from src.emi.planner import cadence_type_for_role, harmonic_function_for_role, phrase_role_to_speac
from src.instrumental_v3.representation import (
    FIELD_NAMES,
    STATE_NOTE,
    SliceEvent,
)


PHRASE_ROLES = [
    "OPENING",
    "SUBJECT_ENTRY",
    "ANSWER_ENTRY",
    "EPISODE",
    "SEQUENCE",
    "CADENTIAL_PREPARATION",
    "CADENCE",
    "CLOSING",
]


class SlicePiece(Protocol):
    piece_id: str
    source_path: str
    grid_ticks: int
    steps_per_bar: int
    key: str | None
    key_pc: int
    mode: int
    slices: list[SliceEvent]


@dataclass(frozen=True)
class Fragment:
    id: str
    piece_id: str
    source_path: str
    voice: int
    start_slice: int
    length_slices: int
    start_bar: int
    start_pos: int
    beats: float
    phrase_role: str
    key: str | None
    key_pc: int
    mode: int
    start_pitch: int | None
    end_pitch: int | None
    start_degree: int
    end_degree: int
    melodic_intervals: list[int]
    rhythm_steps: list[int]
    vertical_intervals: list[int]
    state_pattern: list[int]
    contour_hash: str
    fingerprint: str
    speac_label: str = "UNKNOWN"
    cadence_type: str = "UNKNOWN"
    contour_bucket: str = "UNKNOWN"
    rhythm_bucket: str = "UNKNOWN"
    local_key_pc: int = 12
    harmonic_function: str = "UNKNOWN"
    entry_degree: int = 0
    exit_degree: int = 0
    min_pitch: int | None = None
    max_pitch: int | None = None
    copy_hash: str = ""
    transposition_hash: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Fragment":
        return cls(
            id=str(data["id"]),
            piece_id=str(data["piece_id"]),
            source_path=str(data["source_path"]),
            voice=int(data["voice"]),
            start_slice=int(data["start_slice"]),
            length_slices=int(data["length_slices"]),
            start_bar=int(data["start_bar"]),
            start_pos=int(data["start_pos"]),
            beats=float(data["beats"]),
            phrase_role=str(data["phrase_role"]),
            key=data.get("key") if data.get("key") is None else str(data.get("key")),
            key_pc=int(data["key_pc"]),
            mode=int(data["mode"]),
            start_pitch=_optional_int(data.get("start_pitch")),
            end_pitch=_optional_int(data.get("end_pitch")),
            start_degree=int(data["start_degree"]),
            end_degree=int(data["end_degree"]),
            melodic_intervals=[int(v) for v in data["melodic_intervals"]],  # type: ignore[index]
            rhythm_steps=[int(v) for v in data["rhythm_steps"]],  # type: ignore[index]
            vertical_intervals=[int(v) for v in data["vertical_intervals"]],  # type: ignore[index]
            state_pattern=[int(v) for v in data["state_pattern"]],  # type: ignore[index]
            contour_hash=str(data["contour_hash"]),
            fingerprint=str(data["fingerprint"]),
            speac_label=str(data.get("speac_label", "UNKNOWN")),
            cadence_type=str(data.get("cadence_type", "UNKNOWN")),
            contour_bucket=str(data.get("contour_bucket", "UNKNOWN")),
            rhythm_bucket=str(data.get("rhythm_bucket", "UNKNOWN")),
            local_key_pc=int(data.get("local_key_pc", data.get("key_pc", 12))),
            harmonic_function=str(data.get("harmonic_function", "UNKNOWN")),
            entry_degree=int(data.get("entry_degree", data.get("start_degree", 0))),
            exit_degree=int(data.get("exit_degree", data.get("end_degree", 0))),
            min_pitch=_optional_int(data.get("min_pitch")),
            max_pitch=_optional_int(data.get("max_pitch")),
            copy_hash=str(data.get("copy_hash", "")),
            transposition_hash=str(data.get("transposition_hash", "")),
        )


@dataclass(frozen=True)
class FragmentQuery:
    voice: int | None = None
    phrase_role: str | None = None
    key_pc: int | None = None
    mode: int | None = None
    start_degree: int | None = None
    start_pitch: int | None = None
    previous_end_pitch: int | None = None
    previous_end_degree: int | None = None
    target_beats: float | None = None
    avoid_piece_id: str | None = None
    contour_hash: str | None = None
    speac_label: str | None = None
    cadence_type: str | None = None
    contour_bucket: str | None = None
    rhythm_bucket: str | None = None
    local_key_pc: int | None = None
    harmonic_function: str | None = None
    avoid_copy_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FragmentMatch:
    fragment: Fragment
    score: float
    reasons: dict[str, float]


def extract_fragments(
    piece: SlicePiece,
    *,
    length_slices: int,
    hop_slices: int | None = None,
    min_notes: int = 2,
) -> list[Fragment]:
    """Mine short EMI-style interval/rhythm cells from an instrumental v3 piece.

    The extractor intentionally stores compact signatures rather than long copied
    passages. Retrieval can then condition a model on contour/rhythm/role without
    forcing direct cut-and-paste continuation.
    """

    if length_slices <= 0:
        raise ValueError("length_slices must be positive")
    if min_notes <= 0:
        raise ValueError("min_notes must be positive")
    rows = [slice_.values for slice_ in piece.slices]
    if not rows:
        return []
    hop = hop_slices if hop_slices is not None else max(1, length_slices // 2)
    if hop <= 0:
        raise ValueError("hop_slices must be positive")

    fragments: list[Fragment] = []
    for voice in (0, 1):
        for start in range(0, max(0, len(rows) - length_slices + 1), hop):
            window = rows[start : start + length_slices]
            notes = _note_events(window, voice)
            if len(notes) < min_notes:
                continue
            state_pattern = [_voice_state(row, voice) for row in window]
            melodic_intervals = [notes[idx][2] - notes[idx - 1][2] for idx in range(1, len(notes))]
            rhythm_steps = [dur for _, dur, _, _ in notes]
            verticals = [_decode_vertical(row) for row in window if _decode_vertical(row) > 0]
            start_bar = window[0][_col("bar")]
            start_pos = window[0][_col("pos")]
            phrase_role = infer_phrase_role(rows, start, length_slices, voice, piece.steps_per_bar)
            speac_label = phrase_role_to_speac(phrase_role)
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
                    start_bar=start_bar,
                    start_pos=start_pos,
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
                    speac_label=speac_label,
                    cadence_type=cadence_type,
                    contour_bucket=contour_bucket,
                    rhythm_bucket=rhythm_bucket,
                    local_key_pc=piece.key_pc,
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


def infer_phrase_role(rows: list[list[int]], start: int, length_slices: int, voice: int, steps_per_bar: int) -> str:
    if not rows:
        return "EPISODE"
    start_bar = rows[start][_col("bar")]
    end = min(len(rows), start + length_slices)
    window = rows[start:end]
    phrase_positions = [row[_col("phrase_pos")] for row in window]
    cadence_rate = sum(row[_col("cadence_zone")] for row in window) / max(1, len(window))

    if start_bar == 0 and _first_note_index(rows, voice) in range(start, end):
        return "SUBJECT_ENTRY" if voice == 1 else "ANSWER_ENTRY"
    if start_bar <= 1:
        return "OPENING"
    if cadence_rate >= 0.75:
        final_degrees = [_voice_degree(row, voice) for row in window if _voice_state(row, voice) == STATE_NOTE]
        if final_degrees and final_degrees[-1] in {1, 5}:
            return "CADENCE"
        return "CADENTIAL_PREPARATION"
    if start_bar >= max(0, rows[-1][_col("bar")] - 1):
        return "CLOSING"
    if _looks_sequential(rows, start, length_slices, voice, steps_per_bar):
        return "SEQUENCE"
    if phrase_positions and phrase_positions[0] in {2, 3, 4, 5}:
        return "EPISODE"
    return "EPISODE"


def rank_fragments(query: FragmentQuery, fragments: Iterable[Fragment], *, limit: int = 16) -> list[FragmentMatch]:
    matches = [score_fragment(query, fragment) for fragment in fragments]
    matches.sort(key=lambda match: (-match.score, match.fragment.id))
    return matches[:limit]


def score_fragment(query: FragmentQuery, fragment: Fragment) -> FragmentMatch:
    reasons: dict[str, float] = {}

    def add(name: str, value: float) -> None:
        reasons[name] = value

    add("base", 1.0)
    if query.voice is not None:
        add("voice", 1.0 if fragment.voice == query.voice else -0.75)
    if query.phrase_role is not None:
        add("phrase_role", 1.5 if fragment.phrase_role == query.phrase_role else _role_transition_score(query.phrase_role, fragment.phrase_role))
    if query.key_pc is not None:
        add("key_pc", 0.75 if fragment.key_pc == query.key_pc else -0.25)
    if query.local_key_pc is not None:
        add("local_key_pc", 0.6 if fragment.local_key_pc == query.local_key_pc else -0.2)
    if query.mode is not None:
        add("mode", 0.5 if fragment.mode == query.mode else -0.25)
    if query.speac_label is not None:
        add("speac", 0.6 if fragment.speac_label == query.speac_label else -0.2)
    if query.cadence_type is not None:
        add("cadence", 0.7 if fragment.cadence_type == query.cadence_type else -0.2)
    if query.harmonic_function is not None:
        add("harmonic_function", 0.5 if fragment.harmonic_function == query.harmonic_function else -0.15)
    if query.start_degree is not None:
        add("start_degree", _distance_score(query.start_degree, fragment.start_degree, exact=0.8, near=0.25, penalty=-0.4))
    if query.previous_end_degree is not None:
        add("degree_continuity", _distance_score(query.previous_end_degree, fragment.start_degree, exact=0.75, near=0.2, penalty=-0.35))
    if query.previous_end_pitch is not None and fragment.start_pitch is not None:
        leap = abs(fragment.start_pitch - query.previous_end_pitch)
        add("pitch_continuity", 0.8 if leap <= 2 else 0.35 if leap <= 7 else -0.6 if leap > 12 else -0.15)
    if query.start_pitch is not None and fragment.start_pitch is not None:
        add("register", 0.4 if abs(fragment.start_pitch - query.start_pitch) <= 7 else -0.25)
    if query.target_beats is not None:
        add("beats", max(-0.5, 0.5 - abs(fragment.beats - query.target_beats) * 0.25))
    if query.contour_hash is not None:
        add("contour", 0.8 if fragment.contour_hash == query.contour_hash else 0.0)
    if query.contour_bucket is not None:
        add("contour_bucket", 0.45 if fragment.contour_bucket == query.contour_bucket else 0.0)
    if query.rhythm_bucket is not None:
        add("rhythm_bucket", 0.45 if fragment.rhythm_bucket == query.rhythm_bucket else 0.0)
    if query.avoid_piece_id is not None and fragment.piece_id == query.avoid_piece_id:
        add("novelty_piece", -1.5)
    if fragment.copy_hash and fragment.copy_hash in query.avoid_copy_hashes:
        add("novelty_copy_hash", -3.0)

    score = sum(reasons.values())
    return FragmentMatch(fragment=fragment, score=score, reasons=reasons)


def summarize_fragments(fragments: Iterable[Fragment]) -> dict[str, object]:
    items = list(fragments)
    role_counts = Counter(fragment.phrase_role for fragment in items)
    piece_counts = Counter(fragment.piece_id for fragment in items)
    contour_counts = Counter(fragment.contour_hash for fragment in items)
    cadence_counts = Counter(fragment.cadence_type for fragment in items)
    harmonic_counts = Counter(fragment.harmonic_function for fragment in items)
    copy_counts = Counter(fragment.copy_hash for fragment in items if fragment.copy_hash)
    return {
        "fragment_count": len(items),
        "piece_count": len(piece_counts),
        "role_counts": dict(sorted(role_counts.items())),
        "cadence_counts": dict(sorted(cadence_counts.items())),
        "harmonic_function_counts": dict(sorted(harmonic_counts.items())),
        "top_pieces": dict(piece_counts.most_common(10)),
        "unique_contours": len(contour_counts),
        "reused_contours": sum(1 for count in contour_counts.values() if count > 1),
        "unique_copy_hashes": len(copy_counts),
        "reused_copy_hashes": sum(1 for count in copy_counts.values() if count > 1),
    }


def fragment_to_jsonl(fragment: Fragment) -> str:
    return json.dumps(fragment.to_dict(), sort_keys=True)


def fragment_from_jsonl(line: str) -> Fragment:
    return Fragment.from_dict(json.loads(line))


def hash_signature(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _note_events(rows: list[list[int]], voice: int) -> list[tuple[int, int, int, int]]:
    events: list[tuple[int, int, int, int]] = []
    for idx, row in enumerate(rows):
        if _voice_state(row, voice) != STATE_NOTE:
            continue
        pitch = _voice_pitch(row, voice)
        if pitch <= 0:
            continue
        dur = max(1, _voice_duration(row, voice))
        degree = _voice_degree(row, voice)
        events.append((idx, dur, pitch, degree))
    return events


def _looks_sequential(rows: list[list[int]], start: int, length_slices: int, voice: int, steps_per_bar: int) -> bool:
    if length_slices < 4 or steps_per_bar <= 0:
        return False
    prev_start = start - length_slices
    if prev_start < 0:
        return False
    current = _interval_rhythm_signature(rows[start : start + length_slices], voice)
    previous = _interval_rhythm_signature(rows[prev_start:start], voice)
    if not current or not previous:
        return False
    current_intervals, current_rhythm = current
    previous_intervals, previous_rhythm = previous
    return current_rhythm == previous_rhythm and current_intervals == previous_intervals


def _interval_rhythm_signature(rows: list[list[int]], voice: int) -> tuple[list[int], list[int]] | None:
    notes = _note_events(rows, voice)
    if len(notes) < 2:
        return None
    intervals = [notes[idx][2] - notes[idx - 1][2] for idx in range(1, len(notes))]
    rhythm = [dur for _, dur, _, _ in notes]
    return intervals, rhythm


def _first_note_index(rows: list[list[int]], voice: int) -> int | None:
    for idx, row in enumerate(rows):
        if _voice_state(row, voice) == STATE_NOTE and _voice_pitch(row, voice) > 0:
            return idx
    return None


def _role_transition_score(requested: str, actual: str) -> float:
    compatible = {
        "SUBJECT_ENTRY": {"ANSWER_ENTRY": 0.6, "EPISODE": 0.15},
        "ANSWER_ENTRY": {"EPISODE": 0.6, "SEQUENCE": 0.35},
        "EPISODE": {"SEQUENCE": 0.5, "CADENTIAL_PREPARATION": 0.35, "CADENCE": 0.2},
        "SEQUENCE": {"SEQUENCE": 0.4, "CADENTIAL_PREPARATION": 0.45, "EPISODE": 0.2},
        "CADENTIAL_PREPARATION": {"CADENCE": 0.8, "CLOSING": 0.2},
        "CADENCE": {"CLOSING": 0.45, "SUBJECT_ENTRY": 0.1},
    }
    return compatible.get(requested, {}).get(actual, -0.35)


def _distance_score(value: int, candidate: int, *, exact: float, near: float, penalty: float) -> float:
    if value == candidate:
        return exact
    if value > 0 and candidate > 0 and abs(value - candidate) <= 1:
        return near
    return penalty


def _decode_vertical(row: list[int]) -> int:
    encoded = row[_col("vertical_interval")]
    return max(0, encoded - 1)


def _voice_state(row: list[int], voice: int) -> int:
    return row[_col(f"v{voice}_state")]


def _voice_pitch(row: list[int], voice: int) -> int:
    return row[_col(f"v{voice}_pitch")]


def _voice_duration(row: list[int], voice: int) -> int:
    return row[_col(f"v{voice}_dur")]


def _voice_degree(row: list[int], voice: int) -> int:
    return row[_col(f"v{voice}_degree")]


def _col(name: str) -> int:
    return FIELD_NAMES.index(name)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
