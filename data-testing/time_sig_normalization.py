"""
Time signature normalization function for eventizer.py

Add this to src/tokens/eventizer.py around line 300 (before eventize_musicxml)
Then call it from eventize_musicxml around line 451 (right after rebar check)
"""

from collections import Counter
from fractions import Fraction
from typing import Dict, List, Tuple
import music21


def _equivalent_meters(sig1: Tuple[int, int], sig2: Tuple[int, int]) -> bool:
    """Check if two time signatures have the same bar duration."""
    dur1 = Fraction(sig1[0] * 4, sig1[1])  # quarter notes
    dur2 = Fraction(sig2[0] * 4, sig2[1])
    return dur1 == dur2 and sig1 != sig2


def _normalize_time_signatures(
    score: music21.stream.Score,
) -> Tuple[music21.stream.Score, Dict[str, int]]:
    """
    Normalize time signature metadata to remove engraving artifacts.

    Specifically:
    1. Detect "prevailing meter" (most frequent time signature)
    2. Rewrite isolated equivalent-meter blips to prevailing meter
       (e.g., single measure of 2/2 in a 4/4 piece where both = 4 quarters)
    3. Return normalized score + stats dict

    Args:
        score: music21.stream.Score to normalize

    Returns:
        (normalized_score, stats_dict) where stats_dict contains:
            - 'time_sig_rewrites': count of measures rewritten
            - 'time_sig_equivalent_changes': count of equivalent meter changes detected
    """
    stats: Dict[str, int] = {
        "time_sig_rewrites": 0,
        "time_sig_equivalent_changes": 0,
    }

    parts = list(score.parts)
    if not parts:
        return score, stats

    # We'll analyze the first part as reference (assume all parts are synchronized)
    # In most Bach scores, all parts share the same time signatures
    part = parts[0]
    measures = list(part.getElementsByClass(music21.stream.Measure))

    if len(measures) < 3:  # Need at least 3 measures to detect isolated blips
        return score, stats

    # Pass 1: Collect time signatures per measure
    measure_time_sigs: List[Tuple[int, Tuple[int, int]]] = []  # (measure_idx, (beats, beat_type))

    for i, measure in enumerate(measures):
        ts_list = measure.getElementsByClass(music21.meter.TimeSignature)
        if ts_list:
            ts = ts_list[0]
            measure_time_sigs.append((i, (ts.numerator, ts.denominator)))

    if len(measure_time_sigs) < 2:
        return score, stats  # No changes to detect

    # Build full time signature sequence (propagate forward)
    current_sig = measure_time_sigs[0][1]
    all_sigs: List[Tuple[int, int]] = []
    ts_idx = 0

    for i in range(len(measures)):
        if ts_idx < len(measure_time_sigs) and measure_time_sigs[ts_idx][0] == i:
            current_sig = measure_time_sigs[ts_idx][1]
            ts_idx += 1
        all_sigs.append(current_sig)

    # Pass 2: Find prevailing meter
    sig_counts = Counter(all_sigs)
    prevailing = sig_counts.most_common(1)[0][0]

    # Pass 3: Detect equivalent meter changes
    for i in range(len(measure_time_sigs) - 1):
        curr_idx, curr_sig = measure_time_sigs[i]
        next_idx, next_sig = measure_time_sigs[i + 1]

        if _equivalent_meters(curr_sig, next_sig):
            stats["time_sig_equivalent_changes"] += 1

    # Pass 4: Detect and rewrite isolated blips
    # An isolated blip is a single-measure time sig that differs from neighbors
    # and is equivalent to the prevailing meter
    for i in range(len(all_sigs)):
        curr_sig = all_sigs[i]

        # Get previous and next signatures (with boundary handling)
        prev_sig = all_sigs[i - 1] if i > 0 else curr_sig
        next_sig = all_sigs[i + 1] if i < len(all_sigs) - 1 else curr_sig

        # Check if this is an isolated blip
        is_blip = (
            curr_sig != prev_sig
            and curr_sig != next_sig
            and prev_sig == next_sig
            and _equivalent_meters(curr_sig, prev_sig)
        )

        if is_blip:
            # Rewrite this measure's time signature across ALL parts
            for part in parts:
                part_measures = list(part.getElementsByClass(music21.stream.Measure))
                if i < len(part_measures):
                    measure = part_measures[i]
                    # Remove old time signature
                    old_ts_list = measure.getElementsByClass(music21.meter.TimeSignature)
                    for old_ts in old_ts_list:
                        measure.remove(old_ts)
                    # Insert new time signature
                    new_ts = music21.meter.TimeSignature(f"{prevailing[0]}/{prevailing[1]}")
                    measure.insert(0, new_ts)

            stats["time_sig_rewrites"] += 1

    return score, stats


# Example usage in eventize_musicxml (around line 451):
"""
    if _needs_rebar(warnings_list):
        try:
            score = score.makeMeasures(inPlace=False)
            score = score.makeNotation(inPlace=False)
        except Exception:
            pass

    # ADD THIS:
    score, ts_stats = _normalize_time_signatures(score)
    # Optionally: log ts_stats to metadata or aggregate stats

    if voice_mode not in {"auto", "parts", "pitch", "events"}:
        raise ValueError(f"unsupported voice_mode: {voice_mode}")
    parts = list(score.parts)
    ...
"""
