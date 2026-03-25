from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.tokens.intervals import compute_reference_pitch, harmonic_tokens_for_pitch
from src.tokens.validator import _infer_num_voices, _parse_time_sig_token, validate_harm_tokens


@dataclass
class HarmRepairResult:
    tokens: List[str]
    repaired_event_count: int
    mismatch_count_before: int
    mismatch_count_after: int
    skipped_event_count: int
    errors_before: List[str] = field(default_factory=list)
    errors_after: List[str] = field(default_factory=list)


def _count_mismatch_errors(errors: List[str]) -> int:
    return sum(1 for error in errors if error.startswith("Mismatch at "))


def repair_harm_tokens(tokens: List[str], tpq: int = 24) -> HarmRepairResult:
    """
    Deterministic post-process pass that rewrites only the HARM_OCT_* /
    HARM_CLASS_* pair for each pitched voice event so they match the repo's
    reference-pitch logic.

    Does NOT change MEL_INT12_*, insert anchors, fix malformed grammar, or
    alter bar/position layout.  Events that cannot be reconstructed (no pitch
    anchor, out-of-range harmonic interval, truncated payload) are left
    unchanged and counted as skipped.
    """
    errors_before = validate_harm_tokens(tokens, tpq=tpq)

    result_tokens = list(tokens)
    repaired_event_count = 0
    skipped_event_count = 0

    bar_len_ticks = tpq * 4
    num_voices = _infer_num_voices(tokens)
    prev_pitch: List[Optional[int]] = [None] * num_voices
    active_until: List[int] = [0] * num_voices

    current_bar_idx = -1
    bar_start = 0

    idx = 0
    n = len(tokens)

    while idx < n:
        tok = tokens[idx]

        if tok == "BAR":
            if current_bar_idx >= 0:
                bar_start += bar_len_ticks
            current_bar_idx += 1
            idx += 1
            continue

        if tok.startswith("TIME_SIG_"):
            num_sig, denom_sig = _parse_time_sig_token(tok)
            bar_len_ticks = int(round((num_sig * (4.0 / denom_sig)) * tpq))
            idx += 1
            continue

        if tok.startswith("KEY_") or tok.startswith("TEMPO_"):
            idx += 1
            continue

        if tok.startswith("ABS_BASS_"):
            prev_pitch[0] = int(tok.split("_")[-1])
            idx += 1
            continue

        if tok.startswith("ABS_SOP_"):
            if len(prev_pitch) <= 3:
                prev_pitch.extend([None] * (4 - len(prev_pitch)))
                active_until.extend([0] * (4 - len(active_until)))
            prev_pitch[3] = int(tok.split("_")[-1])
            idx += 1
            continue

        if tok.startswith("ABS_VOICE_"):
            parts = tok.split("_")
            if len(parts) == 4:
                v = int(parts[2])
                pitch = int(parts[3])
                if v >= len(prev_pitch):
                    extra = v + 1 - len(prev_pitch)
                    prev_pitch.extend([None] * extra)
                    active_until.extend([0] * extra)
                prev_pitch[v] = pitch
            idx += 1
            continue

        if (
            tok.startswith("ABS_LOW_")
            or tok.startswith("ABS_HIGH_")
            or tok.startswith("REF_VOICE_")
        ):
            idx += 1
            continue

        if tok.startswith("POS_"):
            pos_tick = int(tok.split("_")[1])
            abs_t = bar_start + pos_tick

            # ── Lookahead pass: reconstruct onsets at this position ──────────
            # Mirrors validate_harm_tokens exactly so ref_pitch is identical.
            onsets_at_t: dict[int, int] = {}
            temp_prev = prev_pitch.copy()
            temp_idx = idx + 1
            while temp_idx < n:
                tt = tokens[temp_idx]
                if tt == "BAR" or tt.startswith("POS_"):
                    break

                if tt.startswith("ABS_BASS_"):
                    temp_prev[0] = int(tt.split("_")[-1])
                    temp_idx += 1
                    continue

                if tt.startswith("ABS_SOP_"):
                    if len(temp_prev) <= 3:
                        temp_prev.extend([None] * (4 - len(temp_prev)))
                    temp_prev[3] = int(tt.split("_")[-1])
                    temp_idx += 1
                    continue

                if tt.startswith("ABS_VOICE_"):
                    parts = tt.split("_")
                    if len(parts) == 4:
                        v = int(parts[2])
                        pitch = int(parts[3])
                        if v >= len(temp_prev):
                            temp_prev.extend([None] * (v + 1 - len(temp_prev)))
                        temp_prev[v] = pitch
                    temp_idx += 1
                    continue

                if (
                    tt.startswith("ABS_LOW_")
                    or tt.startswith("ABS_HIGH_")
                    or tt.startswith("REF_VOICE_")
                ):
                    temp_idx += 1
                    continue

                if tt.startswith("VOICE_"):
                    v_idx = int(tt.split("_", 1)[1])
                    if temp_idx + 1 < n and tokens[temp_idx + 1].startswith("REST_"):
                        temp_idx += 2
                        continue

                    if temp_idx + 2 >= n:
                        break

                    dur_tok = tokens[temp_idx + 1]
                    next_idx = temp_idx + 2
                    if next_idx < n and tokens[next_idx].startswith("DUP_"):
                        next_idx += 1
                    if next_idx + 2 >= n:
                        break

                    mel_tok = tokens[next_idx]
                    if not dur_tok.startswith("DUR_") or not mel_tok.startswith("MEL_INT12_"):
                        break

                    mel_val = int(mel_tok.split("_")[-1])
                    if v_idx >= len(temp_prev):
                        temp_prev.extend([None] * (v_idx + 1 - len(temp_prev)))
                    base = temp_prev[v_idx] if temp_prev[v_idx] is not None else 60
                    pitch_val = base + mel_val
                    onsets_at_t[v_idx] = pitch_val
                    temp_prev[v_idx] = pitch_val
                    temp_idx = next_idx + 3
                    continue

                temp_idx += 1

            active_pitches: list[int] = []
            for v in range(len(prev_pitch)):
                if active_until[v] > abs_t and prev_pitch[v] is not None:
                    active_pitches.append(prev_pitch[v])
            active_pitches.extend(onsets_at_t.values())
            ref_pitch = compute_reference_pitch(active_pitches)

            # ── Second pass: rewrite HARM_* tokens ───────────────────────────
            idx += 1
            while idx < n:
                tt = tokens[idx]
                if tt == "BAR" or tt.startswith("POS_"):
                    break

                if tt.startswith("ABS_BASS_"):
                    prev_pitch[0] = int(tt.split("_")[-1])
                    idx += 1
                    continue

                if tt.startswith("ABS_SOP_"):
                    if len(prev_pitch) <= 3:
                        prev_pitch.extend([None] * (4 - len(prev_pitch)))
                        active_until.extend([0] * (4 - len(active_until)))
                    prev_pitch[3] = int(tt.split("_")[-1])
                    idx += 1
                    continue

                if tt.startswith("ABS_VOICE_"):
                    parts = tt.split("_")
                    if len(parts) == 4:
                        v = int(parts[2])
                        pitch = int(parts[3])
                        if v >= len(prev_pitch):
                            extra = v + 1 - len(prev_pitch)
                            prev_pitch.extend([None] * extra)
                            active_until.extend([0] * extra)
                        prev_pitch[v] = pitch
                    idx += 1
                    continue

                if (
                    tt.startswith("ABS_LOW_")
                    or tt.startswith("ABS_HIGH_")
                    or tt.startswith("REF_VOICE_")
                ):
                    idx += 1
                    continue

                if tt.startswith("VOICE_"):
                    v_idx = int(tt.split("_", 1)[1])
                    if v_idx >= len(prev_pitch):
                        extra = v_idx + 1 - len(prev_pitch)
                        prev_pitch.extend([None] * extra)
                        active_until.extend([0] * extra)

                    # Rest event – leave unchanged, no state update needed
                    if idx + 1 < n and tokens[idx + 1].startswith("REST_"):
                        idx += 2
                        continue

                    # Truncated: not enough tokens for a full pitched event
                    if idx + 2 >= n:
                        skipped_event_count += 1
                        break

                    dur_tok = tokens[idx + 1]
                    next_idx = idx + 2
                    if next_idx < n and tokens[next_idx].startswith("DUP_"):
                        next_idx += 1
                    if next_idx + 2 >= n:
                        skipped_event_count += 1
                        idx = next_idx
                        continue

                    mel_tok = tokens[next_idx]
                    harm_oct_idx = next_idx + 1
                    harm_cls_idx = next_idx + 2
                    harm_oct_tok = tokens[harm_oct_idx]
                    harm_cls_tok = tokens[harm_cls_idx]

                    if not dur_tok.startswith("DUR_") or not mel_tok.startswith(
                        "MEL_INT12_"
                    ):
                        skipped_event_count += 1
                        idx = next_idx + 3
                        continue

                    if not harm_oct_tok.startswith("HARM_OCT_") or not harm_cls_tok.startswith(
                        "HARM_CLASS_"
                    ):
                        skipped_event_count += 1
                        idx = next_idx + 3
                        continue

                    dur = int(dur_tok.split("_")[-1])
                    mel_val = int(mel_tok.split("_")[-1])

                    if prev_pitch[v_idx] is None:
                        # No anchor: update state with default (mirrors validator),
                        # but skip the harmonic rewrite.
                        prev_pitch[v_idx] = 60 + mel_val
                        active_until[v_idx] = abs_t + dur
                        skipped_event_count += 1
                        idx = next_idx + 3
                        continue

                    pitch_val = prev_pitch[v_idx] + mel_val
                    prev_pitch[v_idx] = pitch_val
                    active_until[v_idx] = abs_t + dur

                    try:
                        exp_oct, exp_cls = harmonic_tokens_for_pitch(
                            pitch_val, ref_pitch, qa_mode=True
                        )
                    except ValueError:
                        skipped_event_count += 1
                        idx = next_idx + 3
                        continue

                    if (
                        result_tokens[harm_oct_idx] != exp_oct
                        or result_tokens[harm_cls_idx] != exp_cls
                    ):
                        result_tokens[harm_oct_idx] = exp_oct
                        result_tokens[harm_cls_idx] = exp_cls
                        repaired_event_count += 1

                    idx = next_idx + 3
                    continue

                idx += 1
            continue

        idx += 1

    errors_after = validate_harm_tokens(result_tokens, tpq=tpq)

    return HarmRepairResult(
        tokens=result_tokens,
        repaired_event_count=repaired_event_count,
        mismatch_count_before=_count_mismatch_errors(errors_before),
        mismatch_count_after=_count_mismatch_errors(errors_after),
        skipped_event_count=skipped_event_count,
        errors_before=errors_before,
        errors_after=errors_after,
    )
