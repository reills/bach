from __future__ import annotations

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

SPEAC_LABEL_NAMES = ["UNKNOWN", "S", "P", "E", "A", "C"]

CADENCE_TYPE_NAMES = [
    "UNKNOWN",
    "NONE",
    "HALF",
    "AUTHENTIC",
    "DECEPTIVE",
    "PLAGAL",
]

HARMONIC_FUNCTION_NAMES = [
    "UNKNOWN",
    "TONIC",
    "DOMINANT",
    "PREDOMINANT",
    "SEQUENTIAL",
    "CADENTIAL",
    "OTHER",
]


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
    if len(rhythm_steps) >= 2 and rhythm_steps[0] >= 4 and (state_pattern or [])[1:2] == [1]:
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


def _has_dotted_pair(steps: list[int]) -> bool:
    return any((a, b) in {(3, 1), (1, 3), (6, 2), (2, 6)} for a, b in zip(steps, steps[1:]))


def _looks_syncopated(steps: list[int]) -> bool:
    return len(steps) >= 3 and any(step == 3 for step in steps) and not _has_dotted_pair(steps)
