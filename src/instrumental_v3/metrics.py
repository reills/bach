from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict

from src.instrumental_v3.representation import FIELD_NAMES, STATE_HOLD, STATE_NOTE, STATE_REST, SliceEvent


@dataclass(frozen=True)
class CounterpointReport:
    slice_count: int
    v0_note_rate: float
    v1_note_rate: float
    v0_stuck_rate: float
    v1_stuck_rate: float
    v0_max_same_pitch_run: int
    v1_max_same_pitch_run: int
    v0_same_pitch_run_rate: float
    v1_same_pitch_run_rate: float
    repeated_sonority_rate: float
    voice_crossing_rate: float
    parallel_fifth_octave_rate: float
    empty_slice_rate: float
    invalid_pitch_state_rate: float
    melodic_interval_distribution: dict[str, int]
    vertical_interval_distribution: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_slices(slices: list[SliceEvent] | list[list[int]]) -> CounterpointReport:
    rows = [s.values if isinstance(s, SliceEvent) else s for s in slices]
    if not rows:
        return CounterpointReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {})

    def col(name: str) -> int:
        return FIELD_NAMES.index(name)

    empty = 0
    invalid = 0
    crossing = 0
    repeated = 0
    parallel = 0
    prev_sonority: tuple[int, int] | None = None
    prev_interval_pc: int | None = None
    prev_low_high: tuple[int, int] | None = None
    mel_counter: Counter[str] = Counter()
    vert_counter: Counter[str] = Counter()
    note_counts = [0, 0]
    hold_runs = [0, 0]
    max_hold_runs = [0, 0]
    same_pitch_runs = [0, 0]
    max_same_pitch_runs = [0, 0]
    same_pitch_excess = [0, 0]
    previous_active_pitch: list[int | None] = [None, None]

    for row in rows:
        states = [row[col("v0_state")], row[col("v1_state")]]
        pitches = [row[col("v0_pitch")], row[col("v1_pitch")]]
        if all(state == STATE_REST for state in states):
            empty += 1
        for voice in range(2):
            if states[voice] == STATE_NOTE:
                note_counts[voice] += 1
                hold_runs[voice] = 0
                encoded = row[col(f"v{voice}_mel")]
                if encoded > 0:
                    mel_counter[str(encoded - 25)] += 1
            elif states[voice] == STATE_HOLD:
                hold_runs[voice] += 1
                max_hold_runs[voice] = max(max_hold_runs[voice], hold_runs[voice])
            else:
                hold_runs[voice] = 0

            active_pitch = pitches[voice] if states[voice] in {STATE_NOTE, STATE_HOLD} and pitches[voice] > 0 else None
            if active_pitch is not None and active_pitch == previous_active_pitch[voice]:
                same_pitch_runs[voice] += 1
            elif active_pitch is not None:
                same_pitch_runs[voice] = 1
            else:
                same_pitch_runs[voice] = 0
            previous_active_pitch[voice] = active_pitch
            max_same_pitch_runs[voice] = max(max_same_pitch_runs[voice], same_pitch_runs[voice])
            if same_pitch_runs[voice] > 8:
                same_pitch_excess[voice] += 1

            if states[voice] in {STATE_NOTE, STATE_HOLD} and pitches[voice] <= 0:
                invalid += 1
            if states[voice] == STATE_REST and pitches[voice] != 0:
                invalid += 1

        if pitches[0] > 0 and pitches[1] > 0:
            if pitches[0] > pitches[1]:
                crossing += 1
            interval = abs(pitches[1] - pitches[0])
            vert_counter[str(interval)] += 1
            sonority = (pitches[0], pitches[1])
            if prev_sonority == sonority:
                repeated += 1
            interval_pc = interval % 12
            if prev_interval_pc in {0, 7} and interval_pc == prev_interval_pc and prev_low_high is not None:
                if pitches[0] != prev_low_high[0] and pitches[1] != prev_low_high[1]:
                    direction0 = pitches[0] - prev_low_high[0]
                    direction1 = pitches[1] - prev_low_high[1]
                    if direction0 * direction1 > 0:
                        parallel += 1
            prev_sonority = sonority
            prev_interval_pc = interval_pc
            prev_low_high = (pitches[0], pitches[1])

    n = len(rows)
    return CounterpointReport(
        slice_count=n,
        v0_note_rate=note_counts[0] / n,
        v1_note_rate=note_counts[1] / n,
        v0_stuck_rate=max_hold_runs[0] / n,
        v1_stuck_rate=max_hold_runs[1] / n,
        v0_max_same_pitch_run=max_same_pitch_runs[0],
        v1_max_same_pitch_run=max_same_pitch_runs[1],
        v0_same_pitch_run_rate=same_pitch_excess[0] / n,
        v1_same_pitch_run_rate=same_pitch_excess[1] / n,
        repeated_sonority_rate=repeated / max(1, n - 1),
        voice_crossing_rate=crossing / n,
        parallel_fifth_octave_rate=parallel / max(1, n - 1),
        empty_slice_rate=empty / n,
        invalid_pitch_state_rate=invalid / max(1, n * 2),
        melodic_interval_distribution=dict(mel_counter),
        vertical_interval_distribution=dict(vert_counter),
    )


def source_overlap_report(
    generated: list[SliceEvent] | list[list[int]],
    sources: list[list[SliceEvent] | list[list[int]]],
    *,
    ngram: int = 16,
) -> dict[str, float | int]:
    generated_rows = [_musical_signature(s.values if isinstance(s, SliceEvent) else s) for s in generated]
    source_rows_by_piece = [
        [_musical_signature(s.values if isinstance(s, SliceEvent) else s) for s in source]
        for source in sources
    ]
    if len(generated_rows) < ngram:
        return {
            "ngram": ngram,
            "generated_ngrams": 0,
            "source_ngram_overlap_rate": 0.0,
            "max_contiguous_source_match": 0,
        }

    source_ngrams = set()
    for source_rows in source_rows_by_piece:
        for idx in range(0, max(0, len(source_rows) - ngram + 1)):
            source_ngrams.add(tuple(source_rows[idx : idx + ngram]))

    total = 0
    hits = 0
    for idx in range(0, len(generated_rows) - ngram + 1):
        total += 1
        if tuple(generated_rows[idx : idx + ngram]) in source_ngrams:
            hits += 1

    return {
        "ngram": ngram,
        "generated_ngrams": total,
        "source_ngram_overlap_rate": hits / max(1, total),
        "max_contiguous_source_match": _max_contiguous_match(generated_rows, source_rows_by_piece),
    }


def _musical_signature(row: list[int]) -> tuple[int, ...]:
    def col(name: str) -> int:
        return FIELD_NAMES.index(name)

    # Ignore bar/position/key identity. Keep the actual musical event content.
    return (
        row[col("v0_state")],
        row[col("v0_pitch")],
        row[col("v0_mel")],
        row[col("v0_dur")],
        row[col("v1_state")],
        row[col("v1_pitch")],
        row[col("v1_mel")],
        row[col("v1_dur")],
        row[col("vertical_interval")],
    )


def _max_contiguous_match(
    generated_rows: list[tuple[int, ...]],
    source_rows_by_piece: list[list[tuple[int, ...]]],
) -> int:
    best = 0
    for source_rows in source_rows_by_piece:
        current = [0] * (len(source_rows) + 1)
        for generated_row in generated_rows:
            previous = 0
            for source_idx, source_row in enumerate(source_rows, start=1):
                saved = current[source_idx]
                if generated_row == source_row:
                    current[source_idx] = previous + 1
                    best = max(best, current[source_idx])
                else:
                    current[source_idx] = 0
                previous = saved
    return best
