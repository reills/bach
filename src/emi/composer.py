from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Literal, Sequence

from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.emi.fragments import Fragment, FragmentQuery, fragment_from_jsonl, rank_fragments
from src.inference.controls import normalize_compose_key, normalize_texture


EMI_ENGINE_VERSION = "emi_symbolic_v1"
TPQ = 24
BAR_TICKS = TPQ * 4
STEP_TICKS = TPQ // 2
STEPS_PER_BAR = BAR_TICKS // STEP_TICKS

EngineRole = Literal[
    "SUBJECT_ENTRY",
    "ANSWER_ENTRY",
    "COUNTERSUBJECT",
    "EPISODE",
    "SEQUENCE",
    "CADENTIAL_PREPARATION",
    "CADENCE",
    "CLOSING",
]

SPEACLabel = Literal["S", "P", "E", "A", "C"]

SPEAC_SUCCESSORS: dict[SPEACLabel, set[SPEACLabel]] = {
    "S": {"P", "E", "A"},
    "P": {"S", "A", "C"},
    "E": {"S", "P", "A", "C"},
    "A": {"E", "C"},
    "C": {"S", "P", "E", "A"},
}

INTERVAL_TENSION = {
    0: 0.0,
    1: 1.0,
    2: 0.8,
    3: 0.225,
    4: 0.2,
    5: 0.55,
    6: 0.65,
    7: 0.1,
    8: 0.275,
    9: 0.25,
    10: 0.7,
    11: 0.9,
}

_KEY_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)


@dataclass(frozen=True)
class SignatureCell:
    name: str
    role: EngineRole
    intervals: tuple[int, ...]
    speac: SPEACLabel


@dataclass(frozen=True)
class EmiComposerConfig:
    key: str = "C"
    measures: int = 8
    texture: int = 2
    seed: int = 0
    tempo: int = 92
    fragment_path: Path | None = None


@dataclass(frozen=True)
class EmiComposition:
    score: CanonicalScore
    diagnostics: dict[str, object]


_BUILT_IN_CELLS: dict[EngineRole, tuple[SignatureCell, ...]] = {
    "SUBJECT_ENTRY": (
        SignatureCell("subject_head", "SUBJECT_ENTRY", (2, 2, 3, -2, -1, -2, -2), "S"),
        SignatureCell("subject_turn", "SUBJECT_ENTRY", (2, 1, 2, -1, -2, 4, -2), "S"),
    ),
    "ANSWER_ENTRY": (
        SignatureCell("answer_fifth", "ANSWER_ENTRY", (2, 2, 1, 2, -2, -2, -1), "A"),
        SignatureCell("answer_inversion", "ANSWER_ENTRY", (-2, -1, -2, 3, 2, -1, 2), "A"),
    ),
    "COUNTERSUBJECT": (
        SignatureCell("countersubject_step", "COUNTERSUBJECT", (-2, 4, -2, -1, 2, -2, 1), "E"),
        SignatureCell("countersubject_turn", "COUNTERSUBJECT", (4, -2, -2, 1, 2, -1, -2), "E"),
    ),
    "EPISODE": (
        SignatureCell("episode_sequence", "EPISODE", (2, -1, 2, -1, 2, -1, -3), "E"),
        SignatureCell("episode_contrary", "EPISODE", (-2, 1, -2, 1, -2, 3, 2), "E"),
    ),
    "SEQUENCE": (
        SignatureCell("sequence_rising", "SEQUENCE", (2, -1, 2, -1, 2, -1, 2), "P"),
        SignatureCell("sequence_falling", "SEQUENCE", (-2, 1, -2, 1, -2, 1, -2), "P"),
    ),
    "CADENTIAL_PREPARATION": (
        SignatureCell("dominant_prep", "CADENTIAL_PREPARATION", (2, 2, -1, -2, 1, -2, 2), "P"),
    ),
    "CADENCE": (
        SignatureCell("cadence_arrival", "CADENCE", (-2, -1, 2, -2, 0, 2, -2), "C"),
    ),
    "CLOSING": (
        SignatureCell("closing_tail", "CLOSING", (2, -2, -1, 2, -2, 0, 0), "C"),
    ),
}


def compose_emi(config: EmiComposerConfig) -> EmiComposition:
    """Compose a notation-first EMI-inspired contrapuntal score.

    This is intentionally an EMI-like reconstruction, not a claim of historical
    EMI compatibility. It uses protected subject/countersubject cells, SPEAC-like
    role sequencing, optional fragment retrieval, and contrapuntal post-checks.
    """

    if config.measures <= 0:
        raise ValueError("measures must be a positive integer")
    texture = normalize_texture(config.texture)
    key, key_pc, mode = _key_context(config.key)
    rng = Random(config.seed)
    fragments = _load_fragments(config.fragment_path)
    role_plan = _role_plan(config.measures)
    pitch_grid, signature_usage = _compose_pitch_grid(
        measures=config.measures,
        texture=texture,
        key_pc=key_pc,
        mode=mode,
        rng=rng,
        fragments=fragments,
        role_plan=role_plan,
    )
    _soften_parallel_perfects(pitch_grid, texture=texture, key_pc=key_pc, mode=mode)
    _enforce_vertical_order(pitch_grid, texture=texture)

    score = _grid_to_score(
        pitch_grid,
        key=key,
        measures=config.measures,
        tempo=config.tempo,
        texture=texture,
    )
    diagnostics = {
        "emiVersion": EMI_ENGINE_VERSION,
        "key": key,
        "keyPc": key_pc,
        "mode": "minor" if mode == 1 else "major",
        "texture": texture,
        "measures": config.measures,
        "rolePlan": role_plan,
        "speacLabels": [_role_to_speac(role) for role in role_plan],
        "speacTensions": _speac_tensions(pitch_grid, measures=config.measures, texture=texture),
        "signatureUsage": signature_usage,
        "fragmentPath": str(config.fragment_path) if config.fragment_path is not None else None,
        "fragmentCount": len(fragments),
        "usedFragmentIds": [
            item["fragmentId"]
            for item in signature_usage
            if item.get("fragmentId") is not None
        ],
    }
    return EmiComposition(score=score, diagnostics=diagnostics)


def _compose_pitch_grid(
    *,
    measures: int,
    texture: int,
    key_pc: int,
    mode: int,
    rng: Random,
    fragments: Sequence[Fragment],
    role_plan: Sequence[EngineRole],
) -> tuple[list[list[int | None]], list[dict[str, object]]]:
    total_steps = measures * STEPS_PER_BAR
    grid: list[list[int | None]] = [[None for _ in range(total_steps)] for _ in range(texture)]
    usage: list[dict[str, object]] = []
    previous_end: list[int | None] = [None for _ in range(texture)]

    for bar in range(measures):
        for voice in range(texture):
            entry_bar = min(voice, max(0, measures - 2))
            if bar < entry_bar:
                continue
            role = _voice_role(bar, voice, role_plan, entry_bar=entry_bar)
            if role in {"CADENCE", "CLOSING"}:
                pitches = _cadence_pitches(
                    voice=voice,
                    texture=texture,
                    key_pc=key_pc,
                    mode=mode,
                    previous_pitch=previous_end[voice],
                )
                fragment = None
                cell_name = "cadence_formula"
            else:
                cell, fragment = _select_cell(
                    role=role,
                    voice=voice,
                    key_pc=key_pc,
                    mode=mode,
                    previous_pitch=previous_end[voice],
                    fragments=fragments,
                    rng=rng,
                )
                anchor = _anchor_pitch(
                    voice=voice,
                    texture=texture,
                    bar=bar,
                    role=role,
                    key_pc=key_pc,
                    mode=mode,
                    previous_pitch=previous_end[voice],
                )
                intervals = _voice_transform(cell.intervals, voice=voice, bar=bar, role=role)
                pitches = _cell_pitches(anchor, intervals, key_pc=key_pc, mode=mode)
                cell_name = cell.name

            start = bar * STEPS_PER_BAR
            for offset, pitch in enumerate(pitches[:STEPS_PER_BAR]):
                grid[voice][start + offset] = pitch
            previous_end[voice] = pitches[min(len(pitches), STEPS_PER_BAR) - 1]
            usage.append(
                {
                    "bar": bar,
                    "voice": voice,
                    "role": role,
                    "cell": cell_name,
                    "fragmentId": fragment.id if fragment is not None else None,
                }
            )

    return grid, usage


def _grid_to_score(
    pitch_grid: Sequence[Sequence[int | None]],
    *,
    key: str,
    measures: int,
    tempo: int,
    texture: int,
) -> CanonicalScore:
    score_measures = [
        Measure(
            id=f"emi-m{i}",
            index=i,
            start_tick=i * BAR_TICKS,
            length_ticks=BAR_TICKS,
        )
        for i in range(measures)
    ]
    events: list[Event] = []
    for voice, line in enumerate(pitch_grid):
        idx = 0
        while idx < len(line):
            pitch = line[idx]
            end = idx + 1
            while end < len(line) and line[end] == pitch:
                end += 1
            events.append(
                Event(
                    id=f"emi-v{voice}-e{len(events)}",
                    start_tick=idx * STEP_TICKS,
                    dur_tick=(end - idx) * STEP_TICKS,
                    voice_id=voice,
                    pitch_midi=pitch,
                    velocity=min(96, 68 + voice * 5),
                )
            )
            idx = end
    events.sort(key=lambda event: (event.start_tick, event.voice_id, event.id))
    return CanonicalScore(
        header=ScoreHeader(
            tpq=TPQ,
            key_sig_map={0: key},
            time_sig_map={0: "4/4"},
            tempo_map={0: tempo},
        ),
        measures=score_measures,
        parts=[
            Part(
                info=PartInfo(
                    id="part-0",
                    instrument="classical_guitar",
                    tuning=[40, 45, 50, 55, 59, 64],
                    midi_program=24,
                ),
                events=events,
            )
        ],
    )


def _load_fragments(path: Path | None) -> list[Fragment]:
    if path is None or not path.exists():
        return []
    fragments: list[Fragment] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                fragments.append(fragment_from_jsonl(stripped))
    return fragments


def _select_cell(
    *,
    role: EngineRole,
    voice: int,
    key_pc: int,
    mode: int,
    previous_pitch: int | None,
    fragments: Sequence[Fragment],
    rng: Random,
) -> tuple[SignatureCell, Fragment | None]:
    if fragments:
        matches = rank_fragments(
            FragmentQuery(
                voice=voice if voice < 2 else None,
                phrase_role=role,
                key_pc=key_pc,
                mode=mode,
                previous_end_pitch=previous_pitch,
            ),
            fragments,
            limit=4,
        )
        if matches and matches[0].score > 0:
            fragment = matches[min(len(matches) - 1, rng.randrange(min(2, len(matches))))].fragment
            return _fragment_to_cell(fragment, fallback_role=role), fragment

    cells = _BUILT_IN_CELLS.get(role) or _BUILT_IN_CELLS["EPISODE"]
    return cells[rng.randrange(len(cells))], None


def _fragment_to_cell(fragment: Fragment, *, fallback_role: EngineRole) -> SignatureCell:
    intervals = tuple(max(-7, min(7, value)) for value in fragment.melodic_intervals[:7])
    if len(intervals) < 7:
        fallback = (_BUILT_IN_CELLS.get(fallback_role) or _BUILT_IN_CELLS["EPISODE"])[0].intervals
        intervals = (*intervals, *fallback)[:7]
    role = fragment.phrase_role if fragment.phrase_role in _BUILT_IN_CELLS else fallback_role
    return SignatureCell(
        name=f"fragment:{fragment.contour_hash}",
        role=role,  # type: ignore[arg-type]
        intervals=intervals,
        speac=_role_to_speac(role),  # type: ignore[arg-type]
    )


def _role_plan(measures: int) -> list[EngineRole]:
    if measures == 1:
        return ["CADENCE"]
    plan: list[EngineRole] = []
    for bar in range(measures):
        if bar == 0:
            plan.append("SUBJECT_ENTRY")
        elif bar == 1:
            plan.append("ANSWER_ENTRY")
        elif bar == measures - 1:
            plan.append("CADENCE")
        elif bar == measures - 2:
            plan.append("CADENTIAL_PREPARATION")
        elif bar % 2 == 0:
            plan.append("SEQUENCE")
        else:
            plan.append("EPISODE")
    return plan


def _voice_role(
    bar: int,
    voice: int,
    role_plan: Sequence[EngineRole],
    *,
    entry_bar: int,
) -> EngineRole:
    if bar == entry_bar:
        return "SUBJECT_ENTRY" if voice % 2 == 0 else "ANSWER_ENTRY"
    if bar == len(role_plan) - 1:
        return "CADENCE"
    if bar == len(role_plan) - 2:
        return "CADENTIAL_PREPARATION"
    if bar > entry_bar and (bar + voice) % 4 == 0:
        return "COUNTERSUBJECT"
    return role_plan[bar]


def _role_to_speac(role: EngineRole) -> SPEACLabel:
    return {
        "SUBJECT_ENTRY": "S",
        "ANSWER_ENTRY": "A",
        "COUNTERSUBJECT": "E",
        "EPISODE": "E",
        "SEQUENCE": "P",
        "CADENTIAL_PREPARATION": "P",
        "CADENCE": "C",
        "CLOSING": "C",
    }[role]


def _voice_transform(
    intervals: Sequence[int],
    *,
    voice: int,
    bar: int,
    role: EngineRole,
) -> tuple[int, ...]:
    if role in {"SUBJECT_ENTRY", "ANSWER_ENTRY"}:
        return tuple(intervals)
    direction = -1 if (voice + bar) % 2 else 1
    return tuple(direction * value for value in intervals)


def _cell_pitches(anchor: int, intervals: Sequence[int], *, key_pc: int, mode: int) -> list[int]:
    pitches = [_snap_to_scale(anchor, key_pc=key_pc, mode=mode)]
    current = anchor
    for interval in intervals:
        current += interval
        pitches.append(_snap_to_scale(current, key_pc=key_pc, mode=mode))
    while len(pitches) < STEPS_PER_BAR:
        pitches.append(pitches[-1])
    return pitches[:STEPS_PER_BAR]


def _cadence_pitches(
    *,
    voice: int,
    texture: int,
    key_pc: int,
    mode: int,
    previous_pitch: int | None,
) -> list[int]:
    pc_by_voice = {
        1: (0,),
        2: (0, 0),
        3: (0, 4 if mode == 0 else 3, 0),
        4: (0, 7, 4 if mode == 0 else 3, 0),
    }[texture]
    ranges = _voice_ranges(texture)
    target = _fit_pitch_to_range(48 + key_pc + pc_by_voice[voice], *ranges[voice])
    if previous_pitch is None:
        previous_pitch = target
    approach = [
        previous_pitch,
        _step_toward(previous_pitch, target, key_pc=key_pc, mode=mode),
        _step_toward(previous_pitch, target, key_pc=key_pc, mode=mode),
        target + (2 if voice % 2 else -2),
        target,
        target,
        target,
        target,
    ]
    return [_fit_pitch_to_range(_snap_to_scale(pitch, key_pc=key_pc, mode=mode), *ranges[voice]) for pitch in approach]


def _anchor_pitch(
    *,
    voice: int,
    texture: int,
    bar: int,
    role: EngineRole,
    key_pc: int,
    mode: int,
    previous_pitch: int | None,
) -> int:
    centers = {
        1: (60,),
        2: (50, 69),
        3: (48, 60, 72),
        4: (45, 55, 64, 74),
    }[texture]
    ranges = _voice_ranges(texture)
    if previous_pitch is not None and role not in {"SUBJECT_ENTRY", "ANSWER_ENTRY"}:
        base = previous_pitch + ((bar % 3) - 1) * 2
    else:
        base = centers[voice] + key_pc
    if role == "ANSWER_ENTRY":
        base += 7
    return _fit_pitch_to_range(_snap_to_scale(base, key_pc=key_pc, mode=mode), *ranges[voice])


def _enforce_vertical_order(pitch_grid: Sequence[list[int | None]], *, texture: int) -> None:
    ranges = _voice_ranges(texture)
    total_steps = len(pitch_grid[0]) if pitch_grid else 0
    for idx in range(total_steps):
        lower_pitch: int | None = None
        for voice in range(texture):
            pitch = pitch_grid[voice][idx]
            if pitch is None:
                continue
            low, high = ranges[voice]
            adjusted = _fit_pitch_to_range(pitch, low, high)
            if lower_pitch is not None:
                while adjusted <= lower_pitch + 2:
                    adjusted += 12
                while adjusted - lower_pitch > 19 and adjusted - 12 > lower_pitch + 2:
                    adjusted -= 12
            pitch_grid[voice][idx] = max(0, min(127, adjusted))
            lower_pitch = pitch_grid[voice][idx]


def _soften_parallel_perfects(
    pitch_grid: Sequence[list[int | None]],
    *,
    texture: int,
    key_pc: int,
    mode: int,
) -> None:
    total_steps = len(pitch_grid[0]) if pitch_grid else 0
    for idx in range(1, total_steps):
        for lower in range(texture):
            for upper in range(lower + 1, texture):
                prev_low = pitch_grid[lower][idx - 1]
                prev_high = pitch_grid[upper][idx - 1]
                curr_low = pitch_grid[lower][idx]
                curr_high = pitch_grid[upper][idx]
                if None in {prev_low, prev_high, curr_low, curr_high}:
                    continue
                low_motion = curr_low - prev_low  # type: ignore[operator]
                high_motion = curr_high - prev_high  # type: ignore[operator]
                if low_motion == 0 or high_motion == 0 or (low_motion > 0) != (high_motion > 0):
                    continue
                prev_class = abs(prev_high - prev_low) % 12  # type: ignore[operator]
                curr_class = abs(curr_high - curr_low) % 12  # type: ignore[operator]
                if prev_class == curr_class and curr_class in {0, 7}:
                    direction = -1 if high_motion > 0 else 1
                    pitch_grid[upper][idx] = _step_in_scale(
                        curr_high,  # type: ignore[arg-type]
                        direction,
                        key_pc=key_pc,
                        mode=mode,
                    )


def _speac_tensions(
    pitch_grid: Sequence[Sequence[int | None]],
    *,
    measures: int,
    texture: int,
) -> list[float]:
    tensions: list[float] = []
    for bar in range(measures):
        bar_tension = 0.0
        samples = 0
        for step in range(STEPS_PER_BAR):
            idx = bar * STEPS_PER_BAR + step
            pitches = [pitch_grid[voice][idx] for voice in range(texture) if pitch_grid[voice][idx] is not None]
            for left_idx, left in enumerate(pitches):
                for right in pitches[left_idx + 1 :]:
                    bar_tension += INTERVAL_TENSION[abs(right - left) % 12]
                    samples += 1
        tensions.append(round(bar_tension / samples, 4) if samples else 0.0)
    return tensions


def _key_context(key: str) -> tuple[str, int, int]:
    normalized = normalize_compose_key(key)
    mode = 1 if normalized.endswith("m") else 0
    tonic = normalized[:-1] if mode else normalized
    return normalized, _KEY_PC[tonic], mode


def _snap_to_scale(pitch: int, *, key_pc: int, mode: int) -> int:
    scale = _MINOR_SCALE if mode == 1 else _MAJOR_SCALE
    candidates = []
    for octave in range(-2, 3):
        base = ((pitch // 12) + octave) * 12 + key_pc
        candidates.extend(base + degree for degree in scale)
    return min(candidates, key=lambda candidate: (abs(candidate - pitch), candidate))


def _step_toward(pitch: int, target: int, *, key_pc: int, mode: int) -> int:
    if pitch == target:
        return target
    return _step_in_scale(pitch, 1 if target > pitch else -1, key_pc=key_pc, mode=mode)


def _step_in_scale(pitch: int, direction: int, *, key_pc: int, mode: int) -> int:
    direction = 1 if direction >= 0 else -1
    scale_pitch = _snap_to_scale(pitch, key_pc=key_pc, mode=mode)
    probe = scale_pitch + direction
    while _snap_to_scale(probe, key_pc=key_pc, mode=mode) == scale_pitch:
        probe += direction
    return _snap_to_scale(probe, key_pc=key_pc, mode=mode)


def _fit_pitch_to_range(pitch: int, low: int, high: int) -> int:
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    return max(low, min(high, pitch))


def _voice_ranges(texture: int) -> tuple[tuple[int, int], ...]:
    return {
        1: ((55, 76),),
        2: ((43, 60), (62, 81)),
        3: ((43, 57), (55, 70), (67, 84)),
        4: ((40, 55), (52, 66), (60, 74), (67, 84)),
    }[texture]
