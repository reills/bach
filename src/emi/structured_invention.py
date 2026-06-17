from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import mean
from typing import Sequence

from src.api.canonical import CanonicalScore, Event, Measure, Part, PartInfo, ScoreHeader
from src.inference.controls import normalize_compose_key

TPQ = 24
BAR_TICKS = TPQ * 4
STEP_TICKS = TPQ // 4  # sixteenth-note grid
STEPS_PER_BAR = BAR_TICKS // STEP_TICKS

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
_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 11)

_SUBJECT = (0, 4, 2, 1, 0, -1, 0, 1, 2, 3, 4, 2, 1, 0, -1, 0)
_COUNTERSUBJECT = (0, -1, -2, -3, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1)
_EPISODE_UP = (0, 1, 2, 1, 3, 2, 1, 0)
_EPISODE_DOWN = (0, -1, -2, -1, -3, -2, -1, 0)
_CADENCE_UPPER = (3, 2, 1, 0, 1, 2, 1, -1, 0, 0, 2, 1, 0, 0, 0, 0)
_CADENCE_LOWER = (4, 4, 3, 4, 2, 3, 4, 4, 0, 0, -3, -2, 0, 0, 0, 0)

_FORM = (
    "subject",
    "answer",
    "sequence_down",
    "episode",
    "relative_major",
    "sequence_up",
    "dominant_prep",
    "half_cadence",
    "inversion",
    "sequence_down",
    "arch_peak",
    "descent",
    "return_lower",
    "return_upper",
    "cadential_prep",
    "final_cadence",
)


@dataclass(frozen=True)
class StructuredInventionConfig:
    key: str = "D minor"
    measures: int = 16
    seed: int = 0
    tempo: int = 84
    title: str = "Structured invention"


@dataclass(frozen=True)
class StructuredInvention:
    score: CanonicalScore
    diagnostics: dict[str, object]


def compose_structured_invention(config: StructuredInventionConfig) -> StructuredInvention:
    if config.measures < 8:
        raise ValueError("structured invention needs at least 8 measures")
    key, key_pc, mode = _key_context(config.key)
    rng = Random(config.seed)
    form = _form_plan(config.measures)
    grid = _build_grid(form=form, key_pc=key_pc, mode=mode, seed=config.seed, rng=rng)
    _repair_verticals(grid, key_pc=key_pc, mode=mode)
    _soften_parallel_perfects(grid, key_pc=key_pc, mode=mode)
    score = _grid_to_score(grid, key=key, tempo=config.tempo, title=config.title)
    diagnostics = _diagnostics(grid, form=form, key=key, seed=config.seed)
    return StructuredInvention(score=score, diagnostics=diagnostics)


def _build_grid(*, form: Sequence[str], key_pc: int, mode: int, seed: int, rng: Random) -> list[list[int | None]]:
    total_steps = len(form) * STEPS_PER_BAR
    grid: list[list[int | None]] = [[None for _ in range(total_steps)] for _ in range(2)]
    tonic = _tonic_midi(key_pc)

    for bar, role in enumerate(form):
        lower: list[int]
        upper: list[int]
        if role == "subject":
            upper = _motif(tonic, 7, _SUBJECT, mode=mode, voice=1)
            lower = _support_bar(tonic, ((0, 4), (0, 4)), mode=mode, voice=0, seed=seed + bar)
        elif role == "answer":
            lower = _motif(tonic, -3, _SUBJECT, mode=mode, voice=0)
            upper = _eighthen(_motif(tonic, 9, _COUNTERSUBJECT, mode=mode, voice=1))
        elif role == "sequence_down":
            upper = _episode_bar(tonic, 10 - (bar % 3), direction=-1, mode=mode, voice=1)
            lower = _eighthen(_episode_bar(tonic, -5 - (bar % 2), direction=1, mode=mode, voice=0))
        elif role == "episode":
            upper = _eighthen(_motif(tonic, 8, _COUNTERSUBJECT, mode=mode, voice=1))
            lower = _support_bar(tonic, ((3, 0), (5, 4)), mode=mode, voice=0, seed=seed + bar)
        elif role == "relative_major":
            upper = _motif(tonic, 9, _SUBJECT, mode=mode, voice=1, truncate=8) + _episode_fragment(tonic, 10, -1, mode=mode, voice=1)
            lower = _support_bar(tonic, ((2, 6), (5, 2)), mode=mode, voice=0, seed=seed + bar)
        elif role == "sequence_up":
            upper = _episode_bar(tonic, 8 + (bar % 2), direction=1, mode=mode, voice=1)
            lower = _eighthen(_episode_bar(tonic, -7 + (bar % 3), direction=-1, mode=mode, voice=0))
        elif role == "dominant_prep":
            upper = _episode_bar(tonic, 9, direction=-1, mode=mode, voice=1)
            lower = _pedal_bar(tonic, 4, mode=mode, voice=0)
        elif role == "half_cadence":
            upper = _cadence_line(tonic, 4, mode=mode, voice=1, final_degree=4)
            lower = _cadence_line(tonic, -3, mode=mode, voice=0, final_degree=-3)
        elif role == "inversion":
            upper = _motif(tonic, 11, tuple(-x for x in _SUBJECT), mode=mode, voice=1)
            lower = _eighthen(_motif(tonic, -5, _COUNTERSUBJECT, mode=mode, voice=0))
        elif role == "arch_peak":
            upper = _raise_line(_episode_bar(tonic, 12, direction=1, mode=mode, voice=1), semitones=12, high=91)
            lower = _eighthen(_episode_bar(tonic, -1, direction=-1, mode=mode, voice=0))
        elif role == "descent":
            upper = _episode_bar(tonic, 11, direction=-1, mode=mode, voice=1)
            lower = _support_bar(tonic, ((3, 4), (1, 4)), mode=mode, voice=0, seed=seed + bar)
        elif role == "return_lower":
            lower = _motif(tonic, -7, _SUBJECT, mode=mode, voice=0)
            upper = _eighthen(_motif(tonic, 8, _COUNTERSUBJECT, mode=mode, voice=1))
        elif role == "return_upper":
            upper = _motif(tonic, 7, _SUBJECT, mode=mode, voice=1)
            lower = _support_bar(tonic, ((0, 3), (4, 0)), mode=mode, voice=0, seed=seed + bar)
        elif role == "cadential_prep":
            upper = _episode_bar(tonic, 8, direction=-1, mode=mode, voice=1)
            lower = _pedal_bar(tonic, 4, mode=mode, voice=0)
        elif role == "final_cadence":
            upper = _cadence_formula(tonic, 7, _CADENCE_UPPER, mode=mode, voice=1)
            lower = _cadence_formula(tonic, -7, _CADENCE_LOWER, mode=mode, voice=0)
        else:
            upper = _episode_bar(tonic, 8 + rng.randrange(3), direction=-1, mode=mode, voice=1)
            lower = _support_bar(tonic, ((0, 4), (3, 4)), mode=mode, voice=0, seed=seed + bar)

        _write_bar(grid, bar, 0, lower)
        _write_bar(grid, bar, 1, upper)
    return grid


def _form_plan(measures: int) -> list[str]:
    if measures == len(_FORM):
        return list(_FORM)
    if measures < len(_FORM):
        middle = list(_FORM[2:-2])
        keep = max(0, measures - 4)
        return ["subject", "answer", *middle[:keep], "cadential_prep", "final_cadence"][:measures]
    extra = measures - len(_FORM)
    expansion = ["episode", "sequence_down", "sequence_up", "descent"] * ((extra + 3) // 4)
    return list(_FORM[:12]) + expansion[:extra] + list(_FORM[12:])


def _motif(
    tonic: int,
    anchor_degree: int,
    offsets: Sequence[int],
    *,
    mode: int,
    voice: int,
    truncate: int | None = None,
) -> list[int]:
    use_offsets = tuple(offsets[: truncate or len(offsets)])
    pitches = [_degree_to_pitch(tonic, anchor_degree + degree, mode=mode) for degree in use_offsets]
    while len(pitches) < STEPS_PER_BAR:
        pitches.append(pitches[-1])
    return _fit_line_to_voice(pitches[:STEPS_PER_BAR], voice)


def _episode_fragment(tonic: int, anchor_degree: int, direction: int, *, mode: int, voice: int) -> list[int]:
    offsets = _EPISODE_UP if direction > 0 else _EPISODE_DOWN
    pitches = [_degree_to_pitch(tonic, anchor_degree + degree, mode=mode) for degree in offsets]
    return _fit_line_to_voice(pitches, voice)


def _episode_bar(tonic: int, anchor_degree: int, direction: int, *, mode: int, voice: int) -> list[int]:
    first = _episode_fragment(tonic, anchor_degree, direction, mode=mode, voice=voice)
    second = _episode_fragment(tonic, anchor_degree + direction * 2, direction, mode=mode, voice=voice)
    return (first + second)[:STEPS_PER_BAR]


def _support_bar(
    tonic: int,
    roots: tuple[tuple[int, int], tuple[int, int]],
    *,
    mode: int,
    voice: int,
    seed: int,
) -> list[int]:
    rng = Random(seed)
    out: list[int] = []
    patterns = ((0, 2, 1, 2), (0, 1, 2, 1))
    for half, pair in enumerate(roots):
        root = pair[half % len(pair)]
        chord = (root, root + 2, root + 4)
        pattern = patterns[(voice + half + rng.randrange(2)) % 2]
        for item in pattern:
            pitch = _degree_to_pitch(tonic, chord[item], mode=mode)
            out.extend([pitch, pitch])
    return _fit_line_to_voice(out[:STEPS_PER_BAR], voice)


def _pedal_bar(tonic: int, pedal_degree: int, *, mode: int, voice: int) -> list[int]:
    degrees = (pedal_degree, pedal_degree + 4, pedal_degree + 2, pedal_degree + 4) * 4
    pitches = [_degree_to_pitch(tonic, degree, mode=mode) for degree in degrees]
    return _fit_line_to_voice(pitches, voice)


def _cadence_line(tonic: int, anchor_degree: int, *, mode: int, voice: int, final_degree: int) -> list[int]:
    approach = (anchor_degree, anchor_degree - 1, anchor_degree, anchor_degree + 1, anchor_degree + 2, anchor_degree + 1, final_degree, final_degree)
    pitches = [_degree_to_pitch(tonic, degree, mode=mode) for degree in approach]
    stretched = [pitch for pitch in pitches for _ in range(2)]
    return _fit_line_to_voice(stretched[:STEPS_PER_BAR], voice)


def _cadence_formula(tonic: int, anchor_degree: int, offsets: Sequence[int], *, mode: int, voice: int) -> list[int]:
    pitches = [_degree_to_pitch(tonic, anchor_degree + degree, mode=mode) for degree in offsets]
    return _fit_line_to_voice(pitches[:STEPS_PER_BAR], voice)


def _write_bar(grid: list[list[int | None]], bar: int, voice: int, pitches: Sequence[int]) -> None:
    start = bar * STEPS_PER_BAR
    for idx, pitch in enumerate(pitches[:STEPS_PER_BAR]):
        grid[voice][start + idx] = pitch


def _eighthen(pitches: Sequence[int]) -> list[int]:
    out: list[int] = []
    for idx in range(0, len(pitches), 2):
        pitch = pitches[idx]
        out.extend([pitch, pitch])
    while len(out) < STEPS_PER_BAR:
        out.append(out[-1])
    return out[:STEPS_PER_BAR]


def _repair_verticals(grid: list[list[int | None]], *, key_pc: int, mode: int) -> None:
    for idx in range(len(grid[0])):
        low = grid[0][idx]
        high = grid[1][idx]
        if low is None or high is None:
            continue
        while high <= low + 7:
            high += 12
        while high - low > 31:
            high -= 12
        if high <= low + 4:
            high = _step_in_scale(high + 7, 1, key_pc=key_pc, mode=mode)
        grid[1][idx] = min(96, high)
        grid[0][idx] = max(36, low)


def _soften_parallel_perfects(grid: list[list[int | None]], *, key_pc: int, mode: int) -> None:
    for idx in range(1, len(grid[0])):
        prev_low, prev_high = grid[0][idx - 1], grid[1][idx - 1]
        low, high = grid[0][idx], grid[1][idx]
        if None in {prev_low, prev_high, low, high}:
            continue
        prev_ic = abs(prev_high - prev_low) % 12  # type: ignore[operator]
        curr_ic = abs(high - low) % 12  # type: ignore[operator]
        motion_low = low - prev_low  # type: ignore[operator]
        motion_high = high - prev_high  # type: ignore[operator]
        if prev_ic == curr_ic and curr_ic in {0, 7} and motion_low * motion_high > 0:
            direction = -1 if motion_high > 0 else 1
            grid[1][idx] = _step_in_scale(high, direction, key_pc=key_pc, mode=mode)  # type: ignore[arg-type]


def _grid_to_score(grid: Sequence[Sequence[int | None]], *, key: str, tempo: int, title: str) -> CanonicalScore:
    measures = [Measure(id=f"m{i}", index=i, start_tick=i * BAR_TICKS, length_ticks=BAR_TICKS) for i in range(len(grid[0]) // STEPS_PER_BAR)]
    events: list[Event] = []
    for voice, line in enumerate(grid):
        idx = 0
        while idx < len(line):
            pitch = line[idx]
            end = idx + 1
            while end < len(line) and line[end] == pitch:
                end += 1
            if pitch is not None:
                events.append(
                    Event(
                        id=f"inv-v{voice}-n{len(events)}",
                        start_tick=idx * STEP_TICKS,
                        dur_tick=(end - idx) * STEP_TICKS,
                        voice_id=voice,
                        pitch_midi=int(pitch),
                        velocity=72 if voice == 0 else 82,
                    )
                )
            idx = end
    events.sort(key=lambda event: (event.start_tick, event.voice_id, event.id))
    return CanonicalScore(
        header=ScoreHeader(tpq=TPQ, key_sig_map={0: key}, time_sig_map={0: "4/4"}, tempo_map={0: tempo}),
        measures=measures,
        parts=[Part(PartInfo(id="P1", instrument="piano", midi_program=0), events=events)],
    )


def _diagnostics(grid: Sequence[Sequence[int | None]], *, form: Sequence[str], key: str, seed: int) -> dict[str, object]:
    upper_by_bar = []
    lower_by_bar = []
    for bar in range(len(form)):
        start = bar * STEPS_PER_BAR
        end = start + STEPS_PER_BAR
        upper = [pitch for pitch in grid[1][start:end] if pitch is not None]
        lower = [pitch for pitch in grid[0][start:end] if pitch is not None]
        upper_by_bar.append(round(mean(upper), 3) if upper else 0.0)
        lower_by_bar.append(round(mean(lower), 3) if lower else 0.0)
    return {
        "engine": "structured_invention_v1",
        "key": key,
        "seed": seed,
        "form": list(form),
        "subjectBars": [idx for idx, role in enumerate(form) if role in {"subject", "answer", "return_lower", "return_upper"}],
        "registerArchUpper": upper_by_bar,
        "registerArchLower": lower_by_bar,
        "highestUpperBar": max(range(len(upper_by_bar)), key=lambda idx: upper_by_bar[idx]),
        "finalLowerPitch": grid[0][-1],
        "finalUpperPitch": grid[1][-1],
        "eventCount": _count_note_events(grid),
    }


def _key_context(key: str) -> tuple[str, int, int]:
    normalized = normalize_compose_key(key)
    mode = 1 if normalized.endswith("m") else 0
    tonic = normalized[:-1] if mode else normalized
    return normalized, _KEY_PC[tonic], mode


def _count_note_events(grid: Sequence[Sequence[int | None]]) -> int:
    count = 0
    for line in grid:
        previous = object()
        for pitch in line:
            if pitch is not None and pitch != previous:
                count += 1
            previous = pitch
    return count


def _tonic_midi(key_pc: int) -> int:
    pitch = 60 + key_pc
    if pitch > 66:
        pitch -= 12
    return pitch


def _degree_to_pitch(tonic: int, degree: int, *, mode: int) -> int:
    scale = _MINOR_SCALE if mode == 1 else _MAJOR_SCALE
    octave, index = divmod(degree, 7)
    return tonic + octave * 12 + scale[index]


def _fit_line_to_voice(pitches: Sequence[int], voice: int) -> list[int]:
    low, high, center = ((43, 64, 54), (60, 88, 74))[voice]
    shifted = list(pitches)
    while mean(shifted) < center - 5:
        shifted = [pitch + 12 for pitch in shifted]
    while mean(shifted) > center + 5:
        shifted = [pitch - 12 for pitch in shifted]
    out = []
    for pitch in shifted:
        while pitch < low:
            pitch += 12
        while pitch > high:
            pitch -= 12
        out.append(max(low, min(high, pitch)))
    return out


def _raise_line(pitches: Sequence[int], *, semitones: int, high: int) -> list[int]:
    return [pitch + semitones if pitch + semitones <= high else pitch for pitch in pitches]


def _step_in_scale(pitch: int, direction: int, *, key_pc: int, mode: int) -> int:
    direction = 1 if direction >= 0 else -1
    tonic = _tonic_midi(key_pc)
    candidates = [_degree_to_pitch(tonic, degree, mode=mode) for degree in range(-28, 36)]
    current = min(candidates, key=lambda candidate: (abs(candidate - pitch), candidate))
    ordered = sorted(set(candidates))
    pos = ordered.index(current)
    next_pos = max(0, min(len(ordered) - 1, pos + direction))
    return ordered[next_pos]
