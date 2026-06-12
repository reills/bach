from __future__ import annotations

from dataclasses import dataclass
from math import inf


@dataclass(frozen=True)
class PitchOption:
    pitch: int | None
    score: float


def voice_range(voice: int, voice_count: int) -> tuple[int, int]:
    if not 0 <= voice < voice_count:
        raise ValueError("voice is outside the active voice axis")
    if voice_count == 1:
        return 40, 84
    center = round(43 + (32 * voice / (voice_count - 1)))
    return max(28, center - 13), min(96, center + 13)


def select_counterpoint_pitches(
    options_by_voice: list[list[PitchOption]],
    previous_pitches: list[int | None],
    *,
    strong_beat: bool,
    beam_size: int = 96,
    strict: bool = True,
) -> tuple[list[int | None], float]:
    if len(options_by_voice) != len(previous_pitches):
        raise ValueError("pitch options and previous pitches must share a voice axis")
    beam: list[tuple[float, list[int | None]]] = [(0.0, [])]
    for voice, options in enumerate(options_by_voice):
        expanded: list[tuple[float, list[int | None]]] = []
        for score, assigned in beam:
            for option in options:
                transition = _transition_score(
                    assigned,
                    option.pitch,
                    previous_pitches,
                    voice=voice,
                    strong_beat=strong_beat,
                    strict=strict,
                )
                if transition == -inf:
                    continue
                expanded.append((score + option.score + transition, [*assigned, option.pitch]))
        if not expanded:
            if strict:
                beam = [(score - 100.0, [*assigned, None]) for score, assigned in beam]
                continue
            fallback = [options[0].pitch if options else None for options in options_by_voice]
            return _repair_order(fallback), -1e9
        expanded.sort(key=lambda item: item[0], reverse=True)
        beam = expanded[: max(1, beam_size)]
    return beam[0][1], beam[0][0]


def creates_parallel_perfect(
    previous_left: int | None,
    previous_right: int | None,
    current_left: int | None,
    current_right: int | None,
) -> bool:
    if None in {previous_left, previous_right, current_left, current_right}:
        return False
    assert previous_left is not None
    assert previous_right is not None
    assert current_left is not None
    assert current_right is not None
    left_motion = current_left - previous_left
    right_motion = current_right - previous_right
    return (
        left_motion != 0
        and right_motion != 0
        and left_motion * right_motion > 0
        and abs(previous_right - previous_left) % 12 in {0, 7}
        and abs(current_right - current_left) % 12
        == abs(previous_right - previous_left) % 12
    )


def _transition_score(
    assigned: list[int | None],
    pitch: int | None,
    previous: list[int | None],
    *,
    voice: int,
    strong_beat: bool,
    strict: bool,
) -> float:
    if pitch is None:
        return -0.15
    score = 0.0
    previous_pitch = previous[voice]
    if previous_pitch is not None:
        motion = pitch - previous_pitch
        leap = abs(motion)
        if leap <= 2:
            score += 0.8
        elif leap > 12:
            score -= 8.0 + (leap - 12) * 1.5
        elif leap > 7:
            score -= 2.0 + (leap - 7) * 0.8
        if motion == 0:
            score -= 0.5

    active_left = [(index, value) for index, value in enumerate(assigned) if value is not None]
    if active_left and active_left[-1][1] >= pitch:
        if strict:
            return -inf
        score -= 80.0 + (active_left[-1][1] - pitch) * 4.0

    for left, left_pitch in active_left:
        distance = pitch - left_pitch
        interval_class = abs(distance) % 12
        if strict and left == voice - 1 and distance > (24 if left == 0 else 19):
            return -inf
        if distance < 3:
            score -= 4.0
        if interval_class in {3, 4, 8, 9}:
            score += 1.6
        elif interval_class == 7:
            score += 0.8
        elif interval_class == 0:
            score -= 1.0
        elif strong_beat:
            score -= 2.2
        else:
            score -= 0.35

        if creates_parallel_perfect(
            previous[left],
            previous[voice],
            left_pitch,
            pitch,
        ):
            if strict:
                return -inf
            score -= 60.0
        previous_left = previous[left]
        if previous_left is not None and previous_pitch is not None:
            left_motion = left_pitch - previous_left
            right_motion = pitch - previous_pitch
            if left_motion * right_motion < 0:
                score += 0.7
            elif left_motion == 0 or right_motion == 0:
                score += 0.25
            elif interval_class in {0, 7} and (
                abs(left_motion) > 2 or abs(right_motion) > 2
            ):
                score -= 8.0
            else:
                score -= 0.25
    return score


def _repair_order(pitches: list[int | None]) -> list[int | None]:
    repaired = pitches[:]
    floor: int | None = None
    for index, pitch in enumerate(repaired):
        if pitch is None:
            continue
        if floor is not None and pitch <= floor:
            pitch = floor + 1
            repaired[index] = pitch
        floor = pitch
    return repaired
