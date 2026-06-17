from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

MEL_INT12_MIN = -24
MEL_INT12_MAX = 24
HARM_OCT_MIN = -2
HARM_OCT_MAX = 4


@dataclass
class IntervalRepairStats:
    mel_clamped: int = 0
    harm_oct_clamped: int = 0


def format_signed_interval(value: int) -> str:
    if value > 0:
        return f"+{value}"
    if value < 0:
        return str(value)
    return "0"


def compute_reference_pitch(active_pitches: Iterable[int]) -> Optional[int]:
    ref_pitch: Optional[int] = None
    for pitch in active_pitches:
        if ref_pitch is None or pitch < ref_pitch:
            ref_pitch = pitch
    return ref_pitch


def compute_melodic_interval(
    pitch: int,
    base_pitch: int,
    *,
    qa_mode: bool,
    stats: Optional[IntervalRepairStats] = None,
) -> int:
    mel_int = pitch - base_pitch
    if MEL_INT12_MIN <= mel_int <= MEL_INT12_MAX:
        return mel_int

    if qa_mode:
        raise ValueError(
            f"MEL_INT12 out of range: {mel_int} (allowed: {MEL_INT12_MIN}..{MEL_INT12_MAX})"
        )

    clamped = max(MEL_INT12_MIN, min(MEL_INT12_MAX, mel_int))
    if stats is not None:
        stats.mel_clamped += 1
    return clamped


def compute_harmonic_interval(
    pitch: int,
    ref_pitch: Optional[int],
    *,
    qa_mode: bool,
    stats: Optional[IntervalRepairStats] = None,
) -> Tuple[Optional[int], Optional[int]]:
    if ref_pitch is None:
        return None, None

    diff = pitch - ref_pitch
    octv, klass = divmod(diff, 12)
    if HARM_OCT_MIN <= octv <= HARM_OCT_MAX:
        return octv, klass

    if qa_mode:
        raise ValueError(
            f"HARM_OCT out of range: {octv} (allowed: {HARM_OCT_MIN}..{HARM_OCT_MAX})"
        )

    clamped_oct = max(HARM_OCT_MIN, min(HARM_OCT_MAX, octv))
    if stats is not None:
        stats.harm_oct_clamped += 1
    return clamped_oct, klass


def melodic_token_for_pitch(
    pitch: int,
    base_pitch: int,
    *,
    qa_mode: bool,
    stats: Optional[IntervalRepairStats] = None,
) -> str:
    mel_int = compute_melodic_interval(
        pitch, base_pitch, qa_mode=qa_mode, stats=stats
    )
    return f"MEL_INT12_{format_signed_interval(mel_int)}"


def harmonic_tokens_for_pitch(
    pitch: int,
    ref_pitch: Optional[int],
    *,
    qa_mode: bool,
    stats: Optional[IntervalRepairStats] = None,
) -> Tuple[str, str]:
    octv, klass = compute_harmonic_interval(
        pitch, ref_pitch, qa_mode=qa_mode, stats=stats
    )
    if octv is None or klass is None:
        return "HARM_OCT_NA", "HARM_CLASS_NA"
    return f"HARM_OCT_{octv}", f"HARM_CLASS_{klass}"
