from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from src.instrumental_v3.representation import FIELD_NAMES, STATE_NOTE

BEAT = 1000
DEFAULT_PATTERN_SIZE = 8
DEFAULT_PATTERN_THRESHOLD = 2
DEFAULT_INTERVALS_OFF = 1
DEFAULT_AMOUNT_OFF = 2

CMMC_FUNCTION_NAMES = [
    "UNKNOWN",
    "C1",
    "P1",
    "A1",
    "A2",
    "C4",
    "P2",
    "C2",
    "S1",
    "S3",
    "E1",
    "E3",
    "C3",
    "E2",
    "E4",
    "A3",
    "P3",
    "P4",
    "S4",
    "A4",
    "S2",
]

CMMC_FUNCTION_TO_ID = {name: idx for idx, name in enumerate(CMMC_FUNCTION_NAMES)}

ANALYSIS_LEXICON: tuple[tuple[tuple[int, ...], str], ...] = (
    (
        (
            24,
            36,
            48,
            60,
            72,
            84,
            96,
            108,
            27,
            28,
            39,
            40,
            51,
            52,
            63,
            64,
            75,
            76,
            87,
            88,
            99,
            100,
            31,
            43,
            55,
            67,
            79,
            91,
            103,
        ),
        "C1",
    ),
    ((29, 41, 53, 65, 77, 89, 101, 33, 93, 105, 81, 45, 57, 69, 24, 36, 48, 60, 72, 84, 96, 108), "P1"),
    (
        (
            31,
            43,
            55,
            67,
            79,
            91,
            103,
            35,
            47,
            59,
            71,
            83,
            95,
            107,
            26,
            38,
            50,
            62,
            74,
            86,
            98,
            29,
            41,
            53,
            65,
            77,
            89,
            101,
        ),
        "A1",
    ),
    ((35, 47, 59, 71, 83, 95, 107, 26, 38, 50, 62, 74, 86, 98, 29, 41, 53, 65, 77, 89, 101), "A2"),
    ((28, 40, 52, 64, 76, 88, 100, 31, 43, 55, 67, 79, 91, 103, 35, 47, 59, 71, 83, 95, 107), "C4"),
    ((26, 38, 50, 62, 74, 86, 98, 89, 101, 77, 65, 29, 41, 53, 93, 105, 81, 69, 33, 45, 57), "P2"),
    ((33, 45, 57, 69, 81, 93, 105, 24, 36, 48, 60, 72, 84, 96, 108, 28, 40, 52, 64, 76, 88, 100), "C2"),
    ((26, 38, 50, 62, 74, 86, 98, 30, 33, 42, 54, 66, 78, 90, 102, 45, 57, 69, 81, 93, 105), "S1"),
    ((28, 40, 52, 64, 76, 88, 100, 32, 35, 44, 56, 68, 80, 92, 104, 47, 59, 71, 83, 95, 107), "S3"),
    ((33, 45, 57, 69, 81, 93, 105, 25, 37, 49, 61, 73, 85, 97, 28, 40, 52, 64, 76, 88, 100), "E1"),
    ((35, 47, 59, 71, 83, 95, 107, 27, 39, 51, 63, 75, 87, 99, 30, 42, 54, 66, 78, 90, 102), "E3"),
    (
        (
            24,
            36,
            48,
            60,
            72,
            84,
            96,
            108,
            28,
            31,
            34,
            40,
            52,
            64,
            76,
            88,
            100,
            43,
            55,
            67,
            79,
            91,
            103,
            46,
            58,
            70,
            82,
            94,
            106,
        ),
        "C3",
    ),
    (
        (
            25,
            37,
            49,
            61,
            73,
            85,
            97,
            28,
            31,
            34,
            40,
            52,
            64,
            76,
            88,
            100,
            43,
            55,
            67,
            79,
            91,
            103,
            46,
            58,
            70,
            82,
            94,
            106,
        ),
        "E2",
    ),
    (
        (
            27,
            39,
            51,
            63,
            75,
            87,
            99,
            24,
            30,
            33,
            36,
            48,
            60,
            72,
            84,
            96,
            108,
            42,
            54,
            66,
            78,
            90,
            102,
            45,
            57,
            69,
            81,
            93,
            105,
        ),
        "E4",
    ),
    (
        (
            32,
            44,
            56,
            68,
            80,
            92,
            104,
            26,
            29,
            35,
            38,
            50,
            62,
            74,
            86,
            98,
            41,
            53,
            65,
            77,
            89,
            101,
            47,
            59,
            71,
            83,
            95,
            107,
        ),
        "A3",
    ),
    ((32, 44, 56, 68, 80, 92, 104, 24, 30, 36, 48, 60, 72, 84, 96, 108, 42, 54, 66, 78, 90, 102), "P3"),
    ((25, 37, 49, 61, 73, 85, 97, 29, 32, 41, 53, 65, 77, 89, 101, 44, 56, 68, 80, 92, 104), "P4"),
    ((30, 42, 54, 66, 78, 90, 102, 25, 37, 49, 61, 73, 85, 97, 34, 46, 58, 70, 82, 94, 106), "S4"),
    ((27, 39, 51, 63, 75, 87, 99, 31, 43, 55, 67, 79, 91, 103, 34, 46, 58, 70, 82, 94, 106), "A4"),
    ((34, 46, 58, 70, 82, 94, 106, 26, 38, 50, 62, 74, 86, 98, 29, 41, 53, 65, 77, 89, 101), "S2"),
)

INTERVAL_TENSIONS: tuple[float, ...] = (
    0.0,
    1.0,
    0.8,
    0.225,
    0.2,
    0.55,
    0.65,
    0.1,
    0.275,
    0.25,
    0.7,
    0.9,
    0.0,
    1.0,
    0.8,
    0.225,
    0.2,
    0.55,
    0.65,
    0.1,
    0.275,
    0.25,
    0.7,
    0.9,
    0.0,
    1.0,
    0.8,
    0.225,
    0.2,
    0.55,
)

ROOT_STRENGTHS_AND_ROOTS: dict[int, tuple[int, int]] = {
    7: (0, 1),
    5: (5, 2),
    4: (0, 3),
    8: (8, 4),
    3: (0, 5),
    9: (9, 6),
    2: (2, 7),
    10: (0, 8),
    1: (1, 9),
    11: (0, 10),
    0: (0, 11),
    6: (6, 12),
}

METRIC_TENSION_TABLE: dict[int, dict[int, int]] = {
    4: {1: 2, 2: 2, 3: 6, 4: 2},
    2: {1: 2, 2: 2},
    3: {1: 2, 2: 2, 3: 2},
    6: {1: 2, 2: 2, 3: 2, 4: 8, 5: 4, 6: 3},
    9: {1: 2, 2: 2, 3: 2, 4: 8, 5: 4, 6: 3, 7: 14, 8: 8, 9: 4},
}

GRADUS_MAJOR_SCALE = (
    36,
    38,
    40,
    41,
    43,
    45,
    47,
    48,
    50,
    52,
    53,
    55,
    57,
    59,
    60,
    62,
    64,
    65,
    67,
    69,
    71,
    72,
    74,
    76,
    77,
    79,
    81,
    83,
    84,
    86,
    88,
    89,
    91,
    93,
    95,
    96,
)
GRADUS_ILLEGAL_VERTICALS = (
    0,
    1,
    2,
    5,
    6,
    10,
    11,
    13,
    14,
    17,
    18,
    22,
    23,
    25,
    26,
    29,
    30,
    34,
    35,
    -1,
    -2,
    -3,
    -4,
    -5,
    -6,
    -7,
    -8,
)
GRADUS_ILLEGAL_PARALLEL_MOTIONS = ((7, 7), (12, 12), (19, 19), (24, 24))
GRADUS_ILLEGAL_DOUBLE_SKIPS = (
    (3, 3),
    (3, 4),
    (3, -3),
    (3, -4),
    (-3, -3),
    (-3, -4),
    (-3, 3),
    (-3, 4),
    (4, 3),
    (4, 4),
    (4, -3),
    (4, -4),
    (-4, -3),
    (-4, -4),
    (-4, 3),
    (-4, 4),
)
GRADUS_DIRECT_FIFTHS_AND_OCTAVES = ((9, 7), (8, 7), (21, 19), (20, 19))


@dataclass(frozen=True)
class CmmcEvent:
    start: int
    pitch: int
    duration: int
    channel: int
    velocity: int = 96
    voice: int = 0
    slice_index: int = 0

    def as_lisp_tuple(self) -> tuple[int, int, int, int, int]:
        return (self.start, self.pitch, self.duration, self.channel, self.velocity)


@dataclass(frozen=True)
class CmmcPattern:
    count: int
    start: int
    intervals: tuple[int, ...]
    events: tuple[CmmcEvent, ...] = ()


@dataclass(frozen=True)
class CmmcHarmonicPoint:
    start: int
    function: str
    speac_label: str
    harmonic_function: str
    pitches: tuple[int, ...]
    tension: float


@dataclass(frozen=True)
class CmmcCadence:
    start: int
    function: str
    cadence_type: str


@dataclass(frozen=True)
class CmmcFragmentAnalysis:
    phrase_role: str
    speac_label: str
    cmmc_function: str
    cadence_type: str
    harmonic_function: str
    local_key_pc: int


@dataclass(frozen=True)
class CmmcPieceAnalysis:
    rows: tuple[tuple[int, ...], ...]
    events: tuple[CmmcEvent, ...]
    steps_per_bar: int
    grid_ticks: int
    key_pc: int
    mode: int
    harmonic_points: tuple[CmmcHarmonicPoint, ...]
    cadences: tuple[CmmcCadence, ...]
    signatures_by_voice: dict[int, tuple[CmmcPattern, ...]]

    def fragment_analysis(self, start: int, length_slices: int, voice: int) -> CmmcFragmentAnalysis:
        end = min(len(self.rows), start + length_slices)
        function = self.function_for_window(start, end)
        cadence_type = self.cadence_for_window(start, end)
        role = self.role_for_window(start, end, voice, cadence_type)
        speac = speac_label_for_cmmc_function(function)
        harmonic = harmonic_function_for_cmmc_function(function, cadence_type=cadence_type)
        return CmmcFragmentAnalysis(
            phrase_role=role,
            speac_label=speac,
            cmmc_function=function,
            cadence_type=cadence_type,
            harmonic_function=harmonic,
            local_key_pc=self.local_key_for_role(role),
        )

    def function_for_window(self, start: int, end: int) -> str:
        if not self.harmonic_points:
            return "UNKNOWN"
        start_time = self.row_time(start)
        end_time = self.row_time(max(start, end - 1)) + self.grid_ticks
        candidates = [point.function for point in self.harmonic_points if start_time <= point.start < end_time]
        if not candidates:
            previous = [point.function for point in self.harmonic_points if point.start <= start_time]
            candidates = previous[-1:] if previous else []
        if not candidates:
            return "UNKNOWN"
        return Counter(candidates).most_common(1)[0][0]

    def cadence_for_window(self, start: int, end: int) -> str:
        start_time = self.row_time(start)
        end_time = self.row_time(max(start, end - 1)) + self.grid_ticks
        for cadence in self.cadences:
            if start_time <= cadence.start < end_time:
                return cadence.cadence_type
        return "NONE"

    def role_for_window(self, start: int, end: int, voice: int, cadence_type: str) -> str:
        if not self.rows:
            return "EPISODE"
        start_bar = self.rows[start][field_index("bar")]
        last_bar = self.rows[-1][field_index("bar")]
        if cadence_type == "AUTHENTIC":
            return "CADENCE" if start_bar >= max(0, last_bar - 1) else "CADENTIAL_PREPARATION"
        if cadence_type == "HALF":
            return "CADENTIAL_PREPARATION"
        if start_bar == last_bar and start_bar > 0:
            return "CLOSING"
        if start_bar == max(0, last_bar - 1) and start_bar > 0:
            return "CADENTIAL_PREPARATION"

        first_idx = first_note_index(self.rows, voice)
        if first_idx is not None and start <= first_idx < end:
            return "SUBJECT_ENTRY" if cmmc_channel_for_voice(voice) == 1 else "ANSWER_ENTRY"
        if start_bar == 0:
            return "OPENING"

        if self.window_has_repeated_signature(start, end, voice):
            return "SEQUENCE"
        if self.window_has_prior_signature(start, end, voice):
            return "EPISODE"
        return "EPISODE"

    def window_has_repeated_signature(self, start: int, end: int, voice: int) -> bool:
        patterns = self.signatures_by_voice.get(voice, ())
        if not patterns:
            return False
        start_time = self.row_time(start)
        end_time = self.row_time(max(start, end - 1)) + self.grid_ticks
        return any(start_time <= pattern.start < end_time and pattern.count > DEFAULT_PATTERN_THRESHOLD for pattern in patterns)

    def window_has_prior_signature(self, start: int, end: int, voice: int) -> bool:
        patterns = self.signatures_by_voice.get(voice, ())
        if not patterns:
            return False
        start_time = self.row_time(start)
        end_time = self.row_time(max(start, end - 1)) + self.grid_ticks
        intervals_in_window = {
            pattern.intervals for pattern in patterns if start_time <= pattern.start < end_time
        }
        if not intervals_in_window:
            return False
        return any(pattern.start < start_time and pattern.intervals in intervals_in_window for pattern in patterns)

    def local_key_for_role(self, role: str) -> int:
        if self.key_pc >= 12:
            return 12
        if role == "ANSWER_ENTRY":
            return (self.key_pc + 7) % 12
        if role == "SEQUENCE":
            return (self.key_pc + 2) % 12
        return self.key_pc

    def row_time(self, row_index: int) -> int:
        if row_index < 0 or row_index >= len(self.rows):
            row_index = max(0, min(len(self.rows) - 1, row_index))
        row = self.rows[row_index]
        return (row[field_index("bar")] * self.steps_per_bar + row[field_index("pos")]) * self.grid_ticks


def analyze_rows(
    rows: Sequence[Sequence[int]],
    *,
    steps_per_bar: int,
    grid_ticks: int,
    key_pc: int = 12,
    mode: int = 0,
) -> CmmcPieceAnalysis:
    row_tuples = tuple(tuple(int(value) for value in row) for row in rows)
    events = tuple(events_from_rows(row_tuples, steps_per_bar=steps_per_bar, grid_ticks=grid_ticks))
    harmonic_points = tuple(analyze_harmonic_points(events))
    cadences = tuple(cadences_from_harmonic_points(harmonic_points))
    signatures_by_voice = {
        voice: tuple(
            simple_matcher(
                events,
                channel=cmmc_channel_for_voice(voice),
                pattern_size=min(DEFAULT_PATTERN_SIZE, max(2, len(_voice_note_events(row_tuples, voice)))),
                threshold=1,
            )
        )
        for voice in (0, 1)
    }
    return CmmcPieceAnalysis(
        rows=row_tuples,
        events=events,
        steps_per_bar=steps_per_bar,
        grid_ticks=grid_ticks,
        key_pc=max(0, min(12, int(key_pc))),
        mode=max(0, min(2, int(mode))),
        harmonic_points=harmonic_points,
        cadences=cadences,
        signatures_by_voice=signatures_by_voice,
    )


def events_from_rows(
    rows: Sequence[Sequence[int]],
    *,
    steps_per_bar: int,
    grid_ticks: int,
    voices: Sequence[int] = (0, 1),
) -> list[CmmcEvent]:
    events: list[CmmcEvent] = []
    for idx, row in enumerate(rows):
        start = (int(row[field_index("bar")]) * steps_per_bar + int(row[field_index("pos")])) * grid_ticks
        for voice in voices:
            if row[field_index(f"v{voice}_state")] != STATE_NOTE:
                continue
            pitch = int(row[field_index(f"v{voice}_pitch")])
            if pitch <= 0:
                continue
            duration = max(1, int(row[field_index(f"v{voice}_dur")])) * grid_ticks
            events.append(
                CmmcEvent(
                    start=start,
                    pitch=pitch,
                    duration=duration,
                    channel=cmmc_channel_for_voice(voice),
                    voice=voice,
                    slice_index=idx,
                )
            )
    return sorted(events, key=lambda event: (event.start, event.channel, event.pitch))


def interval_translator(midi_list: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(midi_list[idx + 1]) - int(midi_list[idx]) for idx in range(len(midi_list) - 1))


def pattern_match(
    pattern_1: Sequence[int],
    pattern_2: Sequence[int],
    number_wrong_possible: int = DEFAULT_INTERVALS_OFF,
    *,
    amount_off: int = DEFAULT_AMOUNT_OFF,
) -> bool:
    if len(pattern_1) != len(pattern_2):
        return False
    wrong = int(number_wrong_possible)
    for left, right in zip(pattern_1, pattern_2):
        if wrong == -1 or abs(int(left) - int(right)) > amount_off:
            return False
        if int(left) != int(right):
            wrong -= 1
    return True


def run_pattern_match(
    pattern: Sequence[int],
    patterns: Sequence[int],
    *,
    intervals_off: int = DEFAULT_INTERVALS_OFF,
    amount_off: int = DEFAULT_AMOUNT_OFF,
) -> int:
    pattern_tuple = tuple(pattern)
    if not pattern_tuple:
        return 0
    matches = 0
    for idx in range(0, len(patterns) - len(pattern_tuple) + 1):
        if pattern_match(pattern_tuple, patterns[idx : idx + len(pattern_tuple)], intervals_off, amount_off=amount_off):
            matches += 1
    return matches


def simple_matcher(
    events: Sequence[CmmcEvent],
    *,
    channel: int = 1,
    pattern_size: int = DEFAULT_PATTERN_SIZE,
    threshold: int = DEFAULT_PATTERN_THRESHOLD,
    intervals_off: int = DEFAULT_INTERVALS_OFF,
    amount_off: int = DEFAULT_AMOUNT_OFF,
) -> list[CmmcPattern]:
    notes = get_ontimes_and_pitches(get_channel(channel, events))
    return find_the_matches(
        notes,
        notes,
        pattern_size=pattern_size,
        threshold=threshold,
        intervals_off=intervals_off,
        amount_off=amount_off,
    )


def find_the_matches(
    work_1: Sequence[tuple[int, int]],
    work_2: Sequence[tuple[int, int]],
    *,
    pattern_size: int = DEFAULT_PATTERN_SIZE,
    threshold: int = DEFAULT_PATTERN_THRESHOLD,
    intervals_off: int = DEFAULT_INTERVALS_OFF,
    amount_off: int = DEFAULT_AMOUNT_OFF,
) -> list[CmmcPattern]:
    if pattern_size < 2:
        return []
    patterns: list[CmmcPattern] = []
    idx = 0
    work_2_intervals = interval_translator([pitch for _, pitch in work_2])
    while len(work_1) - idx >= pattern_size:
        pattern_notes = work_1[idx : idx + pattern_size]
        intervals = interval_translator([pitch for _, pitch in pattern_notes])
        count = run_pattern_match(intervals, work_2_intervals, intervals_off=intervals_off, amount_off=amount_off)
        if count > threshold:
            patterns.append(CmmcPattern(count=count, start=pattern_notes[0][0], intervals=intervals))
            idx += pattern_size
        else:
            idx += 1
    return patterns


def top_level_matcher(
    event_lists: Sequence[Sequence[CmmcEvent]],
    *,
    pattern_size: int = DEFAULT_PATTERN_SIZE,
    threshold: int = DEFAULT_PATTERN_THRESHOLD,
    intervals_off: int = DEFAULT_INTERVALS_OFF,
    amount_off: int = DEFAULT_AMOUNT_OFF,
    channel: int = 1,
) -> list[tuple[int, tuple[CmmcEvent, ...]]]:
    return rank_the_matches(
        challenge_the_matches(
            add_the_matches(
                match_the_database_music(
                    event_lists,
                    pattern_size=pattern_size,
                    intervals_off=intervals_off,
                    amount_off=amount_off,
                    channel=channel,
                ),
                intervals_off=intervals_off,
                amount_off=amount_off,
                channel=channel,
            ),
            threshold=threshold,
        )
    )


def match_the_database_music(
    database_music: Sequence[Sequence[CmmcEvent]],
    *,
    pattern_size: int,
    intervals_off: int,
    amount_off: int,
    channel: int,
) -> list[tuple[int, tuple[CmmcEvent, ...]]]:
    all_matches: list[tuple[int, tuple[CmmcEvent, ...]]] = []
    rotated = [tuple(work) for work in database_music]
    for _ in range(len(rotated)):
        first, rest = rotated[0], rotated[1:]
        all_matches.extend(
            match_the_databases(
                first,
                rest,
                pattern_size=pattern_size,
                intervals_off=intervals_off,
                amount_off=amount_off,
                channel=channel,
            )
        )
        rotated = rotated[1:] + rotated[:1]
    return all_matches


def match_the_databases(
    music_from_db: Sequence[CmmcEvent],
    music_from_dbs: Sequence[Sequence[CmmcEvent]],
    *,
    pattern_size: int,
    intervals_off: int,
    amount_off: int,
    channel: int,
) -> list[tuple[int, tuple[CmmcEvent, ...]]]:
    matches: list[tuple[int, tuple[CmmcEvent, ...]]] = []
    for other in music_from_dbs:
        matches.extend(
            find_event_matches(
                music_from_db,
                other,
                pattern_size=pattern_size,
                intervals_off=intervals_off,
                amount_off=amount_off,
                channel=channel,
            )
        )
    return matches


def find_event_matches(
    work_1: Sequence[CmmcEvent],
    work_2: Sequence[CmmcEvent],
    *,
    pattern_size: int,
    intervals_off: int,
    amount_off: int,
    channel: int,
) -> list[tuple[int, tuple[CmmcEvent, ...]]]:
    matches: list[tuple[int, tuple[CmmcEvent, ...]]] = []
    idx = 0
    ordered = tuple(sorted(work_1, key=lambda event: (event.start, event.channel)))
    while idx < len(ordered) and len(get_channel(channel, ordered[idx:])) >= pattern_size:
        pattern_events = group_channel_pattern(pattern_size, ordered[idx:], channel=channel)
        count = find_matches(
            pattern_events,
            work_2,
            intervals_off=intervals_off,
            amount_off=amount_off,
            channel=channel,
        )
        if count > 0:
            matches.append((count, tuple(pattern_events)))
        next_idx = next_match_note_index(ordered, idx + 1, channel=channel)
        if next_idx is None:
            break
        idx = next_idx
    return matches


def find_matches(
    pattern: Sequence[CmmcEvent],
    patterns: Sequence[CmmcEvent],
    *,
    intervals_off: int,
    amount_off: int,
    channel: int,
) -> int:
    pattern_intervals = interval_translator([event.pitch for event in get_channel(channel, pattern)])
    target_intervals = interval_translator([event.pitch for event in get_channel(channel, patterns)])
    return run_pattern_match(pattern_intervals, target_intervals, intervals_off=intervals_off, amount_off=amount_off)


def group_channel_pattern(number: int, events: Sequence[CmmcEvent], *, channel: int) -> list[CmmcEvent]:
    grouped: list[CmmcEvent] = []
    remaining = number
    for event in events:
        grouped.append(event)
        if event.channel == channel:
            remaining -= 1
            if remaining == 0:
                break
    return grouped


def add_the_matches(
    matched_list: Sequence[tuple[int, tuple[CmmcEvent, ...]]],
    *,
    intervals_off: int,
    amount_off: int,
    channel: int,
) -> list[tuple[int, tuple[CmmcEvent, ...]]]:
    pending = list(matched_list)
    combined: list[tuple[int, tuple[CmmcEvent, ...]]] = []
    while pending:
        current = pending.pop(0)
        related = meta_matcher(current, pending, intervals_off=intervals_off, amount_off=amount_off, channel=channel)
        if related:
            combined.append((current[0] + sum(match[0] for match in related), current[1]))
            pending = [match for match in pending if match not in related]
        else:
            combined.append(current)
    return combined


def meta_matcher(
    first_matched_list: tuple[int, tuple[CmmcEvent, ...]],
    second_matched_list: Sequence[tuple[int, tuple[CmmcEvent, ...]]],
    *,
    intervals_off: int,
    amount_off: int,
    channel: int,
) -> list[tuple[int, tuple[CmmcEvent, ...]]]:
    first_intervals = interval_translator([event.pitch for event in get_channel(channel, first_matched_list[1])])
    matches = []
    for candidate in second_matched_list:
        candidate_intervals = interval_translator([event.pitch for event in get_channel(channel, candidate[1])])
        if pattern_match(first_intervals, candidate_intervals, intervals_off, amount_off=amount_off):
            matches.append(candidate)
    return matches


def challenge_the_matches(
    finds: Sequence[tuple[int, tuple[CmmcEvent, ...]]],
    *,
    threshold: int = DEFAULT_PATTERN_THRESHOLD,
) -> list[tuple[int, tuple[CmmcEvent, ...]]]:
    return [find for find in finds if find[0] >= threshold]


def rank_the_matches(finds: Sequence[tuple[int, tuple[CmmcEvent, ...]]]) -> list[tuple[int, tuple[CmmcEvent, ...]]]:
    return sorted(finds, key=lambda item: (-item[0], item[1][0].start if item[1] else 0))


def get_channel(channel: int, events: Sequence[CmmcEvent]) -> list[CmmcEvent]:
    return [event for event in events if event.channel == channel]


def get_ontimes_and_pitches(events: Sequence[CmmcEvent]) -> list[tuple[int, int]]:
    return [(event.start, event.pitch) for event in sorted(events, key=lambda event: (event.start, event.pitch))]


def next_match_note_index(events: Sequence[CmmcEvent], start: int, *, channel: int) -> int | None:
    for idx in range(start, len(events)):
        if events[idx].channel == channel:
            return idx
    return None


def analyze_harmonic_points(events: Sequence[CmmcEvent]) -> list[CmmcHarmonicPoint]:
    groups: dict[int, list[CmmcEvent]] = defaultdict(list)
    for event in events:
        groups[event.start].append(event)
    points: list[CmmcHarmonicPoint] = []
    for start in sorted(groups):
        pitches = tuple(event.pitch for event in sorted(groups[start], key=lambda event: event.channel))
        function = get_function((start, pitches))
        speac = speac_label_for_cmmc_function(function)
        tension = vertical_tension(pitches)
        points.append(
            CmmcHarmonicPoint(
                start=start,
                function=function,
                speac_label=speac,
                harmonic_function=harmonic_function_for_cmmc_function(function, cadence_type="NONE"),
                pitches=pitches,
                tension=tension,
            )
        )
    return points


def get_function(chord_notes: tuple[int, Sequence[int]]) -> str:
    _, pitches = chord_notes
    if len(pitches) < 2:
        return "E4"
    return compare_them(tuple(int(pitch) for pitch in pitches), ANALYSIS_LEXICON)[1]


def compare_them(
    harmonic_notes: Sequence[int],
    harmonic_functions: Sequence[tuple[Sequence[int], str]],
) -> tuple[Sequence[int], str]:
    counts = count_harmonic_notes(harmonic_notes, harmonic_functions)
    highest = max(counts)
    return harmonic_functions[counts.index(highest)]


def count_harmonic_notes(
    harmonic_notes: Sequence[int],
    harmonic_functions: Sequence[tuple[Sequence[int], str]],
) -> list[int]:
    return [my_count(harmonic_notes, function_notes) for function_notes, _ in harmonic_functions]


def my_count(list_1: Sequence[int], list_2: Sequence[int]) -> int:
    counts = Counter(int(value) for value in list_2)
    return sum(counts[int(value)] for value in list_1)


def cadences_from_harmonic_points(
    harmonic_points: Sequence[CmmcHarmonicPoint],
    *,
    cadence_minimum: int = 9000,
) -> list[CmmcCadence]:
    function_timing_lists = [(point.start, point.function) for point in harmonic_points]
    return return_best_cadences(function_timing_lists, cadence_minimum=cadence_minimum)


def return_best_cadences(
    function_timing_lists: Sequence[tuple[int, str]],
    *,
    cadence_minimum: int = 9000,
) -> list[CmmcCadence]:
    cadences: list[CmmcCadence] = []
    distance = 0
    previous: tuple[int, str] | None = None
    minor_flag = 0
    for idx, current in enumerate(function_timing_lists):
        start, function = current
        next_function = function_timing_lists[idx + 1][1] if idx + 1 < len(function_timing_lists) else None
        emitted: tuple[int, str] | None = None
        if minor_flag > 0 and function == "C2" and distance > cadence_minimum:
            emitted = (start, "C1")
        elif minor_flag > 0 and function == "C4" and distance > cadence_minimum:
            emitted = (start, "A1")
        elif previous is not None and previous[1] == "A1" and function == "C1" and distance > cadence_minimum:
            emitted = current
        elif distance > cadence_minimum and function == "A1" and next_function != "C1":
            emitted = current
        elif function == "C1" and distance > cadence_minimum:
            emitted = current
        elif function == "A1" and not next4(function_timing_lists[idx + 1 :]) and distance > cadence_minimum:
            emitted = current

        if emitted is not None:
            cadences.append(CmmcCadence(start=emitted[0], function=emitted[1], cadence_type=cadence_type_for_cmmc_function(emitted[1])))
            distance = 0
        else:
            distance += BEAT
        previous = current
        minor_flag = set_minor_flag(current, minor_flag)
    return cadences


def set_minor_flag(item: tuple[int, str], flag: int) -> int:
    return 4 if item[1] in {"S3", "A3", "P4"} else max(0, flag - 1)


def next4(lists: Sequence[tuple[int, str]]) -> bool:
    return any(function == "C1" for _, function in lists[:4])


def run_the_speac_weightings(
    events: Sequence[CmmcEvent],
    begin_beat: int,
    total_beats: int,
    meter: int,
) -> list[float]:
    beat_groups = beat_event_groups(events)
    vertical = [vertical_tension(tuple(event.pitch for event in group)) for group in beat_groups]
    metric = map_metric_tensions(begin_beat, total_beats, meter)
    duration = compute_duration_tensions(beat_groups, vertical)
    approach = get_root_motion_weightings(beat_groups)
    n = min(total_beats, len(vertical), len(metric), len(duration), len(approach))
    return [round(vertical[idx] + metric[idx] + duration[idx] + approach[idx], 4) for idx in range(n)]


def map_metric_tensions(start_beat: int, total_beats: int, meter: int) -> list[float]:
    values: list[float] = []
    beat = int(start_beat)
    table = METRIC_TENSION_TABLE[meter]
    for _ in range(total_beats):
        values.append((beat * 0.1) / table[beat])
        beat = 1 if beat == meter else beat + 1
    return values


def beat_event_groups(events: Sequence[CmmcEvent]) -> list[list[CmmcEvent]]:
    grouped: dict[int, list[CmmcEvent]] = defaultdict(list)
    for event in events:
        grouped[event.start].append(event)
    return [sorted(grouped[start], key=lambda event: event.channel) for start in sorted(grouped)]


def vertical_tension(pitches: Sequence[int]) -> float:
    arranged = remove_octaves(sorted(int(pitch) for pitch in pitches))
    if len(arranged) < 2:
        return 0.0
    bass = arranged[0]
    intervals = [pitch - bass for pitch in arranged[1:]]
    return round(sum(interval_tension(interval) for interval in intervals), 2)


def interval_tension(interval: int) -> float:
    interval = abs(int(interval))
    if interval < len(INTERVAL_TENSIONS):
        return INTERVAL_TENSIONS[interval]
    reduced = interval % 12
    octave_penalty = (interval // 12) * 0.02
    return INTERVAL_TENSIONS[reduced] + octave_penalty


def remove_octaves(notes: Sequence[int]) -> list[int]:
    output: list[int] = []
    seen_pcs: set[int] = set()
    for note in notes:
        pc = int(note) % 12
        if pc not in seen_pcs:
            output.append(int(note))
            seen_pcs.add(pc)
    return output


def compute_duration_tensions(beat_groups: Sequence[Sequence[CmmcEvent]], vertical_tensions: Sequence[float]) -> list[float]:
    starts = [group[0].start for group in beat_groups if group]
    durations = get_durations(starts)
    values: list[float] = []
    for duration, vertical in zip(durations, vertical_tensions):
        values.append(round((duration / 4000.0) * 0.1 + vertical * 0.1, 2))
    return values


def get_durations(ontimes: Sequence[int]) -> list[int]:
    if not ontimes:
        return []
    if len(ontimes) == 1:
        return [0]
    durations = [int(ontimes[idx + 1]) - int(ontimes[idx]) for idx in range(len(ontimes) - 1)]
    durations.append(durations[-1])
    return durations


def get_root_motion_weightings(beat_groups: Sequence[Sequence[CmmcEvent]]) -> list[float]:
    roots = get_chord_roots(beat_groups)
    if not roots:
        return []
    return [0.0] + [interval_tension(abs(left - right) % 12) for left, right in zip(roots, roots[1:])]


def get_chord_roots(beat_groups: Sequence[Sequence[CmmcEvent]]) -> list[int]:
    roots: list[int] = []
    for group in beat_groups:
        pitches = sorted({event.pitch for event in group})
        if not pitches:
            continue
        intervals = derive(pitches)
        strongest = find_strongest_root_interval(intervals)
        interval_pair = find_interval_in_chord(strongest, pitches)
        roots.append(find_upper_lower(strongest, interval_pair))
    return roots


def derive(pitches: Sequence[int]) -> list[int]:
    return sorted(set(derive_all_intervals(pitches)))


def derive_all_intervals(pitches: Sequence[int]) -> list[int]:
    if len(pitches) <= 1:
        return [0]
    output: list[int] = []
    for idx in range(len(pitches)):
        first = pitches[idx]
        for pitch in pitches[idx:]:
            output.append((int(pitch) - int(first)) % 12)
    return output


def find_strongest_root_interval(intervals: Sequence[int]) -> int:
    ranked = [(ROOT_STRENGTHS_AND_ROOTS[int(interval)][1], int(interval)) for interval in intervals]
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def find_interval_in_chord(interval: int, chord: Sequence[int]) -> tuple[int, int]:
    if len(chord) == 1:
        return (int(chord[0]), int(chord[0]))
    for left in chord:
        for right in chord:
            if (int(right) - int(left)) % 12 == interval:
                return (int(right), int(left))
    return (int(chord[0]), int(chord[0]))


def find_upper_lower(root: int, interval: tuple[int, int]) -> int:
    root_placement, _ = ROOT_STRENGTHS_AND_ROOTS[int(root)]
    return interval[0] if root_placement == 0 else interval[1]


def cmmc_function_for_role(role: str) -> str:
    normalized = "CADENTIAL_PREP" if role == "CADENTIAL_PREPARATION" else role
    return {
        "OPENING": "S1",
        "SUBJECT_ENTRY": "S1",
        "ANSWER_ENTRY": "A1",
        "COUNTERSUBJECT": "E1",
        "EPISODE": "E4",
        "SEQUENCE": "P2",
        "CADENTIAL_PREP": "P1",
        "CADENCE": "C1",
        "CLOSING": "C1",
    }.get(normalized, "UNKNOWN")


def speac_label_for_cmmc_function(function: str) -> str:
    if not function or function == "UNKNOWN":
        return "UNKNOWN"
    return function[0].upper()


def cadence_type_for_cmmc_function(function: str) -> str:
    if function in {"C1", "C2", "C3", "C4"}:
        return "AUTHENTIC"
    if function in {"A1", "A2", "A3", "A4"}:
        return "HALF"
    return "NONE"


def harmonic_function_for_cmmc_function(function: str, *, cadence_type: str = "NONE") -> str:
    if function == "UNKNOWN":
        return "UNKNOWN"
    prefix = function[0].upper()
    if cadence_type != "NONE" and prefix in {"A", "C"}:
        return "CADENTIAL" if prefix == "C" else "DOMINANT"
    if prefix == "C":
        return "TONIC"
    if prefix == "A":
        return "DOMINANT"
    if prefix == "P":
        return "PREDOMINANT"
    if prefix == "S":
        return "SEQUENTIAL"
    return "OTHER"


def cmmc_function_id(function: str | None) -> int:
    return CMMC_FUNCTION_TO_ID.get((function or "UNKNOWN").upper(), 0)


def gradus_evaluate(
    cantus_firmus: Sequence[int],
    choices: Sequence[int],
    last_notes: Sequence[int],
    *,
    rules: Iterable[tuple[int | None, tuple[int | None, ...], tuple[int | None, ...]]] = (),
    temporary_rules: Iterable[tuple[int | None, tuple[int | None, ...], tuple[int | None, ...]]] = (),
) -> tuple[int, ...]:
    accepted: list[int] = []
    for choice in choices:
        extended = tuple(last_notes) + (int(choice),)
        if create_rule(cantus_firmus, extended) in set(rules) | set(temporary_rules):
            continue
        if test_for_vertical_dissonance(cantus_firmus[len(last_notes)], choice):
            continue
        if test_for_parallel_octaves_and_fifths(cantus_firmus[: len(last_notes) + 1], choice, last_notes):
            continue
        if test_for_leaps(extended):
            continue
        if test_for_simultaneous_leaps(cantus_firmus[: len(last_notes) + 1], choice, last_notes):
            continue
        if test_for_direct_fifths(cantus_firmus[: len(last_notes) + 1], choice, last_notes):
            continue
        if test_for_consecutive_motions(cantus_firmus[: len(last_notes) + 1], choice, last_notes):
            continue
        accepted.append(int(choice))
    return tuple(accepted)


def create_choices(scale: Sequence[int], last_choice: int) -> tuple[int, int, int, int]:
    return (
        choose_from_scale(last_choice, 1, scale),
        choose_from_scale(last_choice, 3, scale),
        choose_from_scale(last_choice, -1, scale),
        choose_from_scale(last_choice, -3, scale),
    )


def choose_from_scale(current_note: int, interval_class: int, scale: Sequence[int]) -> int:
    scale_list = list(scale)
    diatonic_interval = abs(get_diatonic_interval(interval_class))
    if interval_class > 0:
        idx = scale_list.index(current_note)
        return scale_list[idx + diatonic_interval]
    reversed_scale = list(reversed(scale_list))
    idx = reversed_scale.index(current_note)
    return reversed_scale[idx + diatonic_interval]


def get_diatonic_interval(interval_class: int) -> int:
    return {1: 1, 2: 1, 3: 2, 4: 2, -1: -1, -2: -1, -3: -2, -4: -2}.get(interval_class, 1)


def create_rule(cantus_firmus: Sequence[int], new_notes: Sequence[int]) -> tuple[int | None, tuple[int | None, ...], tuple[int | None, ...]]:
    the_list = tuple(new_notes[-4:])
    cf_start = len(new_notes) - len(the_list)
    cf_notes = tuple(cantus_firmus[cf_start : cf_start + len(the_list)])
    return create_interval_rule((cf_notes, the_list))


def create_interval_rule(rule: tuple[Sequence[int], Sequence[int]]) -> tuple[int | None, tuple[int | None, ...], tuple[int | None, ...]]:
    vertical = find_scale_intervals((rule[0][0], rule[1][0]), GRADUS_MAJOR_SCALE)
    return (
        vertical[0] if vertical else None,
        tuple(find_scale_intervals(rule[0], GRADUS_MAJOR_SCALE)),
        tuple(find_scale_intervals(rule[1], GRADUS_MAJOR_SCALE)),
    )


def find_scale_intervals(notes: Sequence[int], scale: Sequence[int]) -> list[int | None]:
    output: list[int | None] = []
    scale_list = list(scale)
    for first, second in zip(notes, notes[1:]):
        if second is None:
            output.append(None)
            continue
        first_idx = scale_list.index(int(first))
        second_idx = scale_list.index(int(second))
        output.append(second_idx - first_idx)
    return output


def test_for_vertical_dissonance(cantus_firmus_note: int, choice: int) -> bool:
    return int(cantus_firmus_note) - int(choice) in GRADUS_ILLEGAL_VERTICALS


def test_for_parallel_octaves_and_fifths(cantus_firmus: Sequence[int], choice: int, last_notes: Sequence[int]) -> bool:
    if len(cantus_firmus) < 2 or len(last_notes) < 1:
        return False
    pair = (abs(int(cantus_firmus[-2]) - int(last_notes[-1])), abs(int(cantus_firmus[-1]) - int(choice)))
    return pair in GRADUS_ILLEGAL_PARALLEL_MOTIONS


def test_for_leaps(extended_last_notes: Sequence[int]) -> bool:
    if len(extended_last_notes) < 3:
        return False
    last_motion = int(extended_last_notes[-2]) - int(extended_last_notes[-1])
    previous_motion = int(extended_last_notes[-3]) - int(extended_last_notes[-2])
    if (last_motion, previous_motion) in GRADUS_ILLEGAL_DOUBLE_SKIPS:
        return True
    if abs(previous_motion) > 2 and not opposite_sign((last_motion, previous_motion)):
        return True
    return False


def test_for_simultaneous_leaps(cantus_firmus: Sequence[int], choice: int, last_notes: Sequence[int]) -> bool:
    if len(cantus_firmus) < 2 or len(last_notes) < 1:
        return False
    return skipp(cantus_firmus[-2:]) and skipp(tuple(last_notes[-1:]) + (int(choice),))


def test_for_direct_fifths(cantus_firmus: Sequence[int], choice: int, last_notes: Sequence[int]) -> bool:
    if len(cantus_firmus) < 2 or len(last_notes) < 1:
        return False
    return tuple(get_verticals(cantus_firmus[-2:], tuple(last_notes[-1:]) + (int(choice),))) in GRADUS_DIRECT_FIFTHS_AND_OCTAVES


def test_for_consecutive_motions(cantus_firmus: Sequence[int], choice: int, last_notes: Sequence[int]) -> bool:
    if len(cantus_firmus) <= 3 or len(last_notes) <= 2:
        return False
    last_four_cf = tuple(cantus_firmus[-4:])
    last_four_newline = tuple(last_notes[-3:]) + (int(choice),)
    return not any(
        opposite_sign((get_intervals(last_four_cf[idx : idx + 2])[0], get_intervals(last_four_newline[idx : idx + 2])[0]))
        for idx in range(3)
    )


def skipp(notes: Sequence[int]) -> bool:
    return abs(int(notes[1]) - int(notes[0])) > 2


def get_verticals(cantus_firmus: Sequence[int], new_line: Sequence[int]) -> list[int]:
    return [int(cf) - int(cp) for cf, cp in zip(cantus_firmus, new_line)]


def get_intervals(notes: Sequence[int]) -> list[int]:
    return [int(notes[idx + 1]) - int(notes[idx]) for idx in range(len(notes) - 1)]


def opposite_sign(numbers: Sequence[int]) -> bool:
    return (int(numbers[0]) < 0 < int(numbers[1])) or (int(numbers[0]) > 0 > int(numbers[1]))


def first_note_index(rows: Sequence[Sequence[int]], voice: int) -> int | None:
    for idx, row in enumerate(rows):
        if row[field_index(f"v{voice}_state")] == STATE_NOTE and row[field_index(f"v{voice}_pitch")] > 0:
            return idx
    return None


def cmmc_channel_for_voice(voice: int) -> int:
    return 1 if voice == 1 else 2


def field_index(name: str) -> int:
    return FIELD_NAMES.index(name)


def _voice_note_events(rows: Sequence[Sequence[int]], voice: int) -> list[int]:
    return [
        int(row[field_index(f"v{voice}_pitch")])
        for row in rows
        if row[field_index(f"v{voice}_state")] == STATE_NOTE and int(row[field_index(f"v{voice}_pitch")]) > 0
    ]
