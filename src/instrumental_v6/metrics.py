from __future__ import annotations

from src.instrumental_v6.representation import (
    GLOBAL_FIELD_NAMES,
    PAIR_FIELD_NAMES,
    STATE_HOLD,
    STATE_NOTE,
    STATE_REST,
    VOICE_FIELD_NAMES,
)


def evaluate_piece_rows(
    global_rows: list[list[int]],
    voice_rows: list[list[list[int]]],
    pair_rows: list[list[list[list[int]]]],
    *,
    voice_count: int,
) -> dict[str, object]:
    if not global_rows:
        return {"slice_count": 0}
    note_counts = [0] * voice_count
    active_counts = [0] * voice_count
    hold_runs = [0] * voice_count
    max_hold_runs = [0] * voice_count
    invalid = empty = crossing = repeated = parallel = strong_dissonance = strong_pairs = 0
    note_attacks = tonal_outliers = strong_attacks = strong_tonal_outliers = 0
    previous_sonority: tuple[int | None, ...] | None = None
    for row_index, global_row in enumerate(global_rows):
        states = [voice_rows[row_index][voice][_voice_col("state")] for voice in range(voice_count)]
        pitches = [
            voice_rows[row_index][voice][_voice_col("pitch")]
            if states[voice] in {STATE_NOTE, STATE_HOLD}
            else 0
            for voice in range(voice_count)
        ]
        active = tuple(pitch if pitch > 0 else None for pitch in pitches)
        if all(state == STATE_REST for state in states):
            empty += 1
        for voice in range(voice_count):
            if states[voice] in {STATE_NOTE, STATE_HOLD}:
                active_counts[voice] += 1
            if states[voice] == STATE_NOTE:
                note_counts[voice] += 1
                note_attacks += 1
                pitch = pitches[voice]
                strong = global_row[_global_col("pos")] % 4 == 0
                if strong:
                    strong_attacks += 1
                if pitch > 0 and not _fits_tonal_context(pitch, global_row):
                    tonal_outliers += 1
                    if strong:
                        strong_tonal_outliers += 1
                hold_runs[voice] = 0
            elif states[voice] == STATE_HOLD:
                hold_runs[voice] += 1
                max_hold_runs[voice] = max(max_hold_runs[voice], hold_runs[voice])
            else:
                hold_runs[voice] = 0
            raw_pitch = voice_rows[row_index][voice][_voice_col("pitch")]
            if states[voice] in {STATE_NOTE, STATE_HOLD} and raw_pitch <= 0:
                invalid += 1
            if states[voice] == STATE_REST and raw_pitch != 0:
                invalid += 1
        ordered = [pitch for pitch in active if pitch is not None]
        if any(left >= right for left, right in zip(ordered, ordered[1:])):
            crossing += 1
        if previous_sonority == active:
            repeated += 1
        previous_sonority = active
        strong = global_row[_global_col("pos")] % 4 == 0
        for left in range(voice_count):
            for right in range(left + 1, voice_count):
                pair = pair_rows[row_index][left][right]
                parallel += pair[_pair_col("parallel_perfect")]
                if strong and active[left] is not None and active[right] is not None:
                    strong_pairs += 1
                    if abs(active[right] - active[left]) % 12 not in {0, 3, 4, 7, 8, 9}:
                        strong_dissonance += 1
    count = len(global_rows)
    pair_count = voice_count * (voice_count - 1) // 2
    repeated_note_rates: list[float] = []
    short_loop_rates: list[float] = []
    max_repeated_note_attacks: list[int] = []
    max_static_pitch_rows: list[int] = []
    for voice in range(voice_count):
        attack_pitches = [
            row[voice][_voice_col("pitch")]
            for row in voice_rows
            if row[voice][_voice_col("state")] == STATE_NOTE
            and row[voice][_voice_col("pitch")] > 0
        ]
        active_pitches = [
            (
                row[voice][_voice_col("pitch")]
                if row[voice][_voice_col("state")] in {STATE_NOTE, STATE_HOLD}
                else None
            )
            for row in voice_rows
        ]
        repeated_note_rates.append(_repeated_note_attack_rate(attack_pitches))
        short_loop_rates.append(_short_loop_rate(attack_pitches))
        max_repeated_note_attacks.append(_max_run(attack_pitches))
        max_static_pitch_rows.append(_max_run(active_pitches, ignore=None))
    return {
        "slice_count": count,
        "voice_count": voice_count,
        "voice_note_rates": [value / count for value in note_counts],
        "voice_active_rates": [value / count for value in active_counts],
        "voice_stuck_rates": [value / count for value in max_hold_runs],
        "voice_repeated_note_attack_rates": repeated_note_rates,
        "voice_short_loop_rates": short_loop_rates,
        "voice_max_repeated_note_attacks": max_repeated_note_attacks,
        "voice_max_static_pitch_rows": max_static_pitch_rows,
        "voice_crossing_rate": crossing / count,
        "parallel_fifth_octave_rate": parallel / max(1, (count - 1) * pair_count),
        "strong_beat_dissonance_rate": strong_dissonance / max(1, strong_pairs),
        "tonal_outlier_rate": tonal_outliers / max(1, note_attacks),
        "strong_beat_tonal_outlier_rate": strong_tonal_outliers / max(1, strong_attacks),
        "repeated_sonority_rate": repeated / max(1, count - 1),
        "empty_slice_rate": empty / count,
        "invalid_pitch_state_rate": invalid / max(1, count * voice_count),
        **_cadence_metrics(global_rows, voice_rows, voice_count=voice_count),
    }


def source_overlap_report(
    generated: list[list[list[int]]],
    sources: list[list[list[list[int]]]],
    *,
    voice_count: int,
    ngram: int = 16,
) -> dict[str, float | int]:
    generated_signatures = [_signature(row, voice_count) for row in generated]
    source_signatures = [
        [_signature(row, voice_count) for row in source]
        for source in sources
    ]
    source_ngrams = {
        tuple(rows[index : index + ngram])
        for rows in source_signatures
        for index in range(max(0, len(rows) - ngram + 1))
    }
    total = max(0, len(generated_signatures) - ngram + 1)
    hits = sum(
        tuple(generated_signatures[index : index + ngram]) in source_ngrams
        for index in range(total)
    )
    return {
        "ngram": ngram,
        "generated_ngrams": total,
        "source_ngram_overlap_rate": hits / max(1, total),
        "max_contiguous_source_match": _max_match(generated_signatures, source_signatures),
    }


def _signature(row: list[list[int]], voice_count: int) -> tuple[int, ...]:
    values: list[int] = []
    for voice in range(voice_count):
        values.extend(row[voice][:_voice_col("tie")])
    return tuple(values)


def _max_match(
    generated: list[tuple[int, ...]],
    sources: list[list[tuple[int, ...]]],
) -> int:
    best = 0
    for source in sources:
        current = [0] * (len(source) + 1)
        for generated_row in generated:
            previous = 0
            for index, source_row in enumerate(source, start=1):
                saved = current[index]
                current[index] = previous + 1 if generated_row == source_row else 0
                best = max(best, current[index])
                previous = saved
    return best


def _voice_col(name: str) -> int:
    return VOICE_FIELD_NAMES.index(name)


def _pair_col(name: str) -> int:
    return PAIR_FIELD_NAMES.index(name)


def _global_col(name: str) -> int:
    return GLOBAL_FIELD_NAMES.index(name)


def _fits_tonal_context(pitch: int, global_row: list[int]) -> bool:
    mode = global_row[_global_col("mode")]
    scale = {0, 2, 3, 5, 7, 8, 10, 11} if mode == 1 else {0, 2, 4, 5, 7, 9, 11}
    pitch_class = pitch % 12
    key_pc = global_row[_global_col("key_pc")]
    local_key_pc = global_row[_global_col("local_key_pc")]
    keys = [value for value in {key_pc, local_key_pc} if value < 12]
    return any((pitch_class - tonic) % 12 in scale for tonic in keys)


def _cadence_metrics(
    global_rows: list[list[int]],
    voice_rows: list[list[list[int]]],
    *,
    voice_count: int,
) -> dict[str, object]:
    final = _active_sonority(voice_rows[-1], voice_count=voice_count)
    final_start = len(voice_rows) - 1
    while (
        final_start > 0
        and _active_sonority(voice_rows[final_start - 1], voice_count=voice_count) == final
    ):
        final_start -= 1
    penultimate = (
        _active_sonority(voice_rows[final_start - 1], voice_count=voice_count)
        if final_start > 0
        else tuple([None] * voice_count)
    )
    key_pc = global_rows[-1][_global_col("key_pc")]
    mode = global_rows[-1][_global_col("mode")]
    if key_pc >= 12:
        return {
            "final_tonic_bass": False,
            "final_tonic_sonority": False,
            "penultimate_dominant_sonority": False,
            "authentic_cadence_proxy": False,
            "final_sonority_stability_rows": len(voice_rows) - final_start,
            "final_sonority_pitch_classes": [],
            "penultimate_sonority_pitch_classes": [],
        }

    third = 3 if mode == 1 else 4
    tonic_classes = {(key_pc + value) % 12 for value in (0, third, 7)}
    dominant_classes = {(key_pc + value) % 12 for value in (7, 11, 2)}
    final_classes = {pitch % 12 for pitch in final if pitch is not None}
    penultimate_classes = {pitch % 12 for pitch in penultimate if pitch is not None}
    final_tonic_bass = final[0] is not None and final[0] % 12 == key_pc
    final_tonic = (
        all(pitch is not None for pitch in final)
        and final_tonic_bass
        and bool(final_classes)
        and final_classes <= tonic_classes
    )
    dominant_root = (key_pc + 7) % 12
    dominant = (
        all(pitch is not None for pitch in penultimate)
        and penultimate[0] is not None
        and penultimate[0] % 12 == dominant_root
        and bool(penultimate_classes)
        and penultimate_classes <= dominant_classes
    )
    return {
        "final_tonic_bass": final_tonic_bass,
        "final_tonic_sonority": final_tonic,
        "penultimate_dominant_sonority": dominant,
        "authentic_cadence_proxy": final_tonic and dominant,
        "final_sonority_stability_rows": len(voice_rows) - final_start,
        "final_sonority_pitch_classes": sorted(final_classes),
        "penultimate_sonority_pitch_classes": sorted(penultimate_classes),
    }


def _active_sonority(
    row: list[list[int]],
    *,
    voice_count: int,
) -> tuple[int | None, ...]:
    return tuple(
        voice[_voice_col("pitch")]
        if voice[_voice_col("state")] in {STATE_NOTE, STATE_HOLD}
        and voice[_voice_col("pitch")] > 0
        else None
        for voice in row[:voice_count]
    )


def _repeated_note_attack_rate(pitches: list[int]) -> float:
    if len(pitches) < 2:
        return 0.0
    return sum(left == right for left, right in zip(pitches, pitches[1:])) / (len(pitches) - 1)


def _short_loop_rate(pitches: list[int]) -> float:
    if len(pitches) < 4:
        return 0.0
    looped = 0
    eligible = 0
    for end in range(3, len(pitches)):
        eligible += 1
        if any(
            end + 1 >= period * 2
            and pitches[end - period + 1 : end + 1]
            == pitches[end - (period * 2) + 1 : end - period + 1]
            for period in range(1, min(4, (end + 1) // 2) + 1)
        ):
            looped += 1
    return looped / max(1, eligible)


def _max_run(values: list[int | None], *, ignore: int | None = -1) -> int:
    best = run = 0
    previous: int | None | object = object()
    for value in values:
        if value == ignore:
            previous = object()
            run = 0
            continue
        if value == previous:
            run += 1
        else:
            previous = value
            run = 1
        best = max(best, run)
    return best
