from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from src.instrumental_v6.representation import (
    DEVELOPMENT_TO_ID,
    GLOBAL_FIELD_NAMES,
    ROLE_TO_ID,
    STATE_NOTE,
    VOICE_FIELD_NAMES,
)

_SECTION_NAMES = ("opening", "middle", "closing")
_ROLE_ID_TO_NAME = {value: key for key, value in ROLE_TO_ID.items()}
_DEVELOPMENT_ID_TO_NAME = {value: key for key, value in DEVELOPMENT_TO_ID.items()}


def evaluate_global_coherence(
    global_rows: Sequence[Sequence[int]],
    voice_rows: Sequence[Sequence[Sequence[int]]],
    *,
    voice_count: int,
    steps_per_bar: int,
    subject: Sequence[int] | None = None,
    phrase_bars: int = 8,
) -> dict[str, object]:
    """Measure long-range musical organization for a v6 row sequence.

    This is deliberately symbolic and transposition-invariant: it checks whether
    the generated span keeps a recognizable subject contour alive, contains
    planned cadential regions, uses contrasting development roles, and has a
    coherent local-key arc. It complements local counterpoint metrics rather than
    replacing them.
    """

    if not global_rows or not voice_rows:
        return _empty_report()
    if steps_per_bar <= 0:
        raise ValueError("steps_per_bar must be positive")
    active_voice_count = max(1, min(voice_count, len(voice_rows[0])))
    resolved_subject = list(subject or _infer_subject(voice_rows, active_voice_count))
    subject_hits = _subject_hits(
        voice_rows,
        resolved_subject,
        voice_count=active_voice_count,
    )
    bars = max(1, max(int(row[_global_col("bar")]) for row in global_rows) + 1)
    role_by_bar = _dominant_bar_values(global_rows, "section_role", _ROLE_ID_TO_NAME)
    development_by_bar = _dominant_bar_values(global_rows, "development", _DEVELOPMENT_ID_TO_NAME)
    role_counts = Counter(role_by_bar.values())
    development_counts = Counter(development_by_bar.values())
    cadence_bars = sorted(
        {
            bar
            for bar, role in role_by_bar.items()
            if role in {"CADENCE", "CADENTIAL_PREP"}
        }
        | {
            int(row[_global_col("bar")])
            for row in global_rows
            if int(row[_global_col("cadence_zone")]) > 0
        }
    )
    local_keys = [
        int(row[_global_col("local_key_pc")])
        for row in global_rows[::steps_per_bar]
        if int(row[_global_col("local_key_pc")]) < 12
    ]
    local_key_changes = sum(
        left != right for left, right in zip(local_keys, local_keys[1:])
    )
    subject_coverage_score = sum(
        1 for section in _SECTION_NAMES if subject_hits["section_hits"][section] > 0
    ) / len(_SECTION_NAMES)
    cadence_count_score = _cadence_count_score(
        cadence_bars,
        bars=bars,
        phrase_bars=phrase_bars,
    )
    cadence_spacing_score = _cadence_spacing_score(
        cadence_bars,
        bars=bars,
        phrase_bars=phrase_bars,
    )
    development_score = _development_score(role_counts, development_counts)
    key_arc_score = _key_arc_score(local_key_changes, bars=bars)
    phrase_balance_score = _entropy_score(role_counts)
    coherence_score = round(
        100.0
        * (
            0.34 * subject_coverage_score
            + 0.22 * cadence_count_score
            + 0.14 * cadence_spacing_score
            + 0.14 * development_score
            + 0.08 * key_arc_score
            + 0.08 * phrase_balance_score
        ),
        4,
    )
    return {
        "bars": bars,
        "subject": resolved_subject,
        **subject_hits,
        "subject_coverage_score": round(subject_coverage_score, 4),
        "cadence_bars": cadence_bars,
        "cadence_count_score": round(cadence_count_score, 4),
        "cadence_spacing_score": round(cadence_spacing_score, 4),
        "role_counts": dict(sorted(role_counts.items())),
        "development_counts": dict(sorted(development_counts.items())),
        "development_score": round(development_score, 4),
        "local_key_sequence": local_keys,
        "local_key_change_count": local_key_changes,
        "key_arc_score": round(key_arc_score, 4),
        "phrase_balance_score": round(phrase_balance_score, 4),
        "coherence_score": coherence_score,
    }


def _empty_report() -> dict[str, object]:
    return {
        "bars": 0,
        "subject": [],
        "subject_head_hits": 0,
        "subject_inversion_hits": 0,
        "voice_subject_head_hits": [],
        "section_hits": dict.fromkeys(_SECTION_NAMES, 0),
        "subject_coverage_score": 0.0,
        "cadence_bars": [],
        "cadence_count_score": 0.0,
        "cadence_spacing_score": 0.0,
        "role_counts": {},
        "development_counts": {},
        "development_score": 0.0,
        "local_key_sequence": [],
        "local_key_change_count": 0,
        "key_arc_score": 0.0,
        "phrase_balance_score": 0.0,
        "coherence_score": 0.0,
    }


def _subject_hits(
    voice_rows: Sequence[Sequence[Sequence[int]]],
    subject: list[int],
    *,
    voice_count: int,
) -> dict[str, object]:
    section_hits = dict.fromkeys(_SECTION_NAMES, 0)
    voice_hits = [0] * voice_count
    exact_hits = 0
    inversion_hits = 0
    if not subject:
        return {
            "subject_head_hits": 0,
            "subject_inversion_hits": 0,
            "voice_subject_head_hits": voice_hits,
            "section_hits": section_hits,
        }
    head_length = min(4, max(2, len(subject)))
    head = subject[:head_length]
    inverted = [-interval for interval in head]
    total_rows = max(1, len(voice_rows))
    for voice in range(voice_count):
        attacks = _attacks(voice_rows, voice)
        intervals = [right[1] - left[1] for left, right in zip(attacks, attacks[1:])]
        for index in range(0, len(intervals) - head_length + 1):
            window = intervals[index : index + head_length]
            hit_type = "exact" if window == head else "inversion" if window == inverted else ""
            if not hit_type:
                continue
            row_index = attacks[index][0]
            section = _SECTION_NAMES[min(2, row_index * 3 // total_rows)]
            section_hits[section] += 1
            voice_hits[voice] += 1
            if hit_type == "exact":
                exact_hits += 1
            else:
                inversion_hits += 1
    return {
        "subject_head_hits": exact_hits,
        "subject_inversion_hits": inversion_hits,
        "voice_subject_head_hits": voice_hits,
        "section_hits": section_hits,
    }


def _infer_subject(
    voice_rows: Sequence[Sequence[Sequence[int]]],
    voice_count: int,
) -> list[int]:
    for voice in reversed(range(voice_count)):
        attacks = _attacks(voice_rows, voice)[:9]
        if len(attacks) >= 4:
            return [right[1] - left[1] for left, right in zip(attacks, attacks[1:])]
    return []


def _attacks(
    voice_rows: Sequence[Sequence[Sequence[int]]],
    voice: int,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for row_index, row in enumerate(voice_rows):
        if row[voice][_voice_col("state")] != STATE_NOTE:
            continue
        pitch = int(row[voice][_voice_col("pitch")])
        if pitch > 0:
            out.append((row_index, pitch))
    return out


def _dominant_bar_values(
    global_rows: Sequence[Sequence[int]],
    field: str,
    names: dict[int, str],
) -> dict[int, str]:
    field_col = _global_col(field)
    bar_col = _global_col("bar")
    by_bar: dict[int, Counter[str]] = {}
    for row in global_rows:
        bar = int(row[bar_col])
        name = names.get(int(row[field_col]), "UNKNOWN")
        by_bar.setdefault(bar, Counter()).update([name])
    return {
        bar: counts.most_common(1)[0][0]
        for bar, counts in by_bar.items()
    }


def _cadence_count_score(
    cadence_bars: list[int],
    *,
    bars: int,
    phrase_bars: int,
) -> float:
    target = max(1, bars // max(1, phrase_bars))
    return min(1.0, len(cadence_bars) / target)


def _cadence_spacing_score(
    cadence_bars: list[int],
    *,
    bars: int,
    phrase_bars: int,
) -> float:
    if not cadence_bars:
        return 0.0
    anchors = [-1, *cadence_bars, bars - 1]
    max_gap = max(right - left for left, right in zip(anchors, anchors[1:]))
    allowed = max(1, phrase_bars * 2)
    return max(0.0, min(1.0, 1.0 - max(0, max_gap - allowed) / allowed))


def _development_score(
    role_counts: Counter[str],
    development_counts: Counter[str],
) -> float:
    needed_roles = [
        {"SUBJECT_ENTRY", "ANSWER_ENTRY"},
        {"EPISODE"},
        {"SEQUENCE"},
        {"CADENCE"},
    ]
    role_score = sum(
        1 for choices in needed_roles if any(role_counts.get(choice, 0) for choice in choices)
    ) / len(needed_roles)
    thematic_return = any(
        development_counts.get(name, 0)
        for name in ("RECAP", "STRETTO", "INVERSION", "SUBJECT")
    )
    return min(1.0, role_score + (0.15 if thematic_return else 0.0))


def _key_arc_score(local_key_changes: int, *, bars: int) -> float:
    if bars < 12:
        return 1.0
    if local_key_changes == 0:
        return 0.45
    upper = max(2, bars // 4)
    if local_key_changes <= upper:
        return 1.0
    return max(0.25, 1.0 - (local_key_changes - upper) / max(1, upper))


def _entropy_score(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return min(1.0, entropy / math.log(len(counts)))


def _global_col(name: str) -> int:
    return GLOBAL_FIELD_NAMES.index(name)


def _voice_col(name: str) -> int:
    return VOICE_FIELD_NAMES.index(name)
