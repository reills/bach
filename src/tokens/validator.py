from typing import List, Optional, Tuple


def _infer_num_voices(tokens: List[str]) -> int:
    max_idx = -1
    for tok in tokens:
        if tok.startswith("VOICE_"):
            try:
                v = int(tok.split("_", 1)[1])
            except ValueError:
                continue
            max_idx = max(max_idx, v)
        elif tok.startswith("ABS_VOICE_"):
            parts = tok.split("_")
            if len(parts) == 4:
                try:
                    v = int(parts[2])
                except ValueError:
                    continue
                max_idx = max(max_idx, v)
        elif tok.startswith("ABS_BASS_"):
            max_idx = max(max_idx, 0)
        elif tok.startswith("ABS_SOP_"):
            max_idx = max(max_idx, 3)
    return max_idx + 1 if max_idx >= 0 else 1


def _parse_time_sig_token(token: str) -> Tuple[int, int]:
    parts = token.split("_")
    if len(parts) != 4:
        raise ValueError(f"bad time signature token: {token}")
    return int(parts[2]), int(parts[3])


def validate_harm_tokens(tokens: List[str], tpq: int = 24) -> List[str]:
    """
    Validates that HARM_OCT and HARM_CLASS tokens in the stream match
    the expected values calculated from the pitch and reference pitch.

    Returns a list of error messages (empty if valid).
    """
    errors = []
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
            num, denom = _parse_time_sig_token(tok)
            bar_len_ticks = int(round((num * (4.0 / denom)) * tpq))
            idx += 1
            continue

        if tok.startswith("KEY_"):
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
            if len(parts) != 4:
                errors.append(f"bad ABS_VOICE token: {tok}")
                idx += 1
                continue
            v = int(parts[2])
            pitch = int(parts[3])
            if v >= len(prev_pitch):
                extra = v + 1 - len(prev_pitch)
                prev_pitch.extend([None] * extra)
                active_until.extend([0] * extra)
            prev_pitch[v] = pitch
            idx += 1
            continue

        if tok.startswith("ABS_LOW_") or tok.startswith("ABS_HIGH_"):
            idx += 1
            continue

        if tok.startswith("REF_VOICE_"):
            idx += 1
            continue

        if tok.startswith("POS_"):
            pos_tick = int(tok.split("_")[1])
            abs_t = bar_start + pos_tick

            onsets_at_t = {}
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

                if tt.startswith("ABS_LOW_") or tt.startswith("ABS_HIGH_"):
                    temp_idx += 1
                    continue

                if tt.startswith("REF_VOICE_"):
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
                    if temp_prev[v_idx] is None:
                        errors.append(
                            f"Error at token {temp_idx} ({tt}): Voice {v_idx} has no previous pitch anchor."
                        )
                        base = 60
                    else:
                        base = temp_prev[v_idx]
                    pitch = base + mel_val
                    onsets_at_t[v_idx] = pitch
                    temp_prev[v_idx] = pitch
                    temp_idx = next_idx + 3
                    continue

                temp_idx += 1

            active_pitches = []
            for v in range(len(prev_pitch)):
                if active_until[v] > abs_t and prev_pitch[v] is not None:
                    active_pitches.append(prev_pitch[v])
            active_pitches.extend(onsets_at_t.values())
            ref_pitch = min(active_pitches) if active_pitches else None

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

                if tt.startswith("ABS_LOW_") or tt.startswith("ABS_HIGH_"):
                    idx += 1
                    continue

                if tt.startswith("REF_VOICE_"):
                    idx += 1
                    continue

                if tt.startswith("VOICE_"):
                    v_idx = int(tt.split("_", 1)[1])
                    if v_idx >= len(prev_pitch):
                        extra = v_idx + 1 - len(prev_pitch)
                        prev_pitch.extend([None] * extra)
                        active_until.extend([0] * extra)

                    if idx + 1 < n and tokens[idx + 1].startswith("REST_"):
                        idx += 2
                        continue

                    if idx + 2 >= n:
                        errors.append(f"truncated VOICE event at index {idx}")
                        break

                    dur_tok = tokens[idx + 1]
                    next_idx = idx + 2
                    if next_idx < n and tokens[next_idx].startswith("DUP_"):
                        next_idx += 1
                    if next_idx + 2 >= n:
                        errors.append(f"truncated VOICE event at index {idx}")
                        break

                    mel_tok = tokens[next_idx]
                    harm_oct_tok = tokens[next_idx + 1]
                    harm_cls_tok = tokens[next_idx + 2]

                    if not dur_tok.startswith("DUR_") or not mel_tok.startswith("MEL_INT12_"):
                        errors.append(f"bad VOICE event starting at {tt}")
                        idx = next_idx + 3
                        continue

                    dur = int(dur_tok.split("_")[-1])
                    mel_val = int(mel_tok.split("_")[-1])

                    if prev_pitch[v_idx] is None:
                        errors.append(
                            f"Error at token {idx} ({tt}): Voice {v_idx} has no previous pitch anchor."
                        )
                        base = 60
                    else:
                        base = prev_pitch[v_idx]
                    pitch = base + mel_val
                    prev_pitch[v_idx] = pitch
                    active_until[v_idx] = abs_t + dur

                    if ref_pitch is None:
                        if harm_oct_tok != "HARM_OCT_NA" or harm_cls_tok != "HARM_CLASS_NA":
                            errors.append(
                                f"Mismatch at {idx} (Voice {v_idx}, POS {pos_tick}): "
                                f"Expected NA/NA (no ref), got {harm_oct_tok}/{harm_cls_tok}"
                            )
                    else:
                        diff = pitch - ref_pitch
                        expected_oct, expected_cls = divmod(diff, 12)

                        got_oct_str = harm_oct_tok.replace("HARM_OCT_", "")
                        got_cls_str = harm_cls_tok.replace("HARM_CLASS_", "")
                        if got_oct_str == "NA":
                            errors.append(
                                f"Mismatch at {idx} (Voice {v_idx}, POS {pos_tick}): "
                                f"Expected {expected_oct}/{expected_cls}, got NA"
                            )
                        else:
                            got_oct = int(got_oct_str)
                            got_cls = int(got_cls_str)
                            if got_oct != expected_oct or got_cls != expected_cls:
                                errors.append(
                                    f"Mismatch at {idx} (Voice {v_idx}, POS {pos_tick}): "
                                    f"Pitch {pitch} (Ref {ref_pitch}). Expected HARM "
                                    f"{expected_oct}/{expected_cls}, got {got_oct}/{got_cls}"
                                )

                    idx = next_idx + 3
                    continue

                idx += 1
            continue

        idx += 1

    return errors
