from typing import Any, Dict, List, Optional, Tuple

from src.tokens.schema import BarPlan


def _extract_time_sig(tokens: List[str], fallback: Optional[str]) -> str:
    for tok in tokens:
        if tok.startswith("TIME_SIG_"):
            parts = tok.split("_")
            if len(parts) != 4:
                raise ValueError(f"bad TIME_SIG token: {tok}")
            return f"{parts[2]}/{parts[3]}"
    if fallback is not None:
        return fallback
    raise ValueError("missing TIME_SIG token and no fallback provided")


def _extract_key(tokens: List[str], fallback: Optional[str]) -> str:
    for tok in tokens:
        if tok.startswith("KEY_"):
            return tok[len("KEY_") :]
    if fallback is not None:
        return fallback
    raise ValueError("missing KEY token and no fallback provided")


def _classify_density(onset_count: int) -> str:
    if onset_count < 4:
        return "DENSITY_LOW"
    if onset_count <= 8:
        return "DENSITY_MED"
    return "DENSITY_HIGH"


def _compute_pitch_range(
    onset_pitches: List[int], active_pitches: List[int]
) -> Optional[int]:
    combined = onset_pitches + active_pitches
    if not combined:
        return None
    return max(combined) - min(combined)


def _bar_len_ticks(time_sig: str, tpq: int) -> int:
    numerator, denominator = time_sig.split("/", 1)
    bar_ql = int(numerator) * (4.0 / int(denominator))
    return int(round(bar_ql * tpq))


def _init_running_state(
    running_state: Optional[Dict[str, Any]],
    num_voices: int,
) -> Dict[str, Any]:
    if running_state is None:
        running_state = {}
    prev_pitch = running_state.get("prev_pitch")
    active_until = running_state.get("active_until")
    if prev_pitch is None:
        prev_pitch = {}
    if active_until is None:
        active_until = {}
    for v in range(num_voices):
        prev_pitch.setdefault(v, None)
        active_until.setdefault(v, 0)
    state = {
        "prev_pitch": dict(prev_pitch),
        "active_until": dict(active_until),
        "num_voices": num_voices,
        "time_sig": running_state.get("time_sig"),
        "key": running_state.get("key"),
        "bar_len_ticks": running_state.get("bar_len_ticks"),
    }
    return state


def _shift_active_until(state: Dict[str, Any], bar_len_ticks: int) -> None:
    active_until = state["active_until"]
    for v in range(state["num_voices"]):
        active_until[v] = max(0, active_until[v] - bar_len_ticks)


def _active_pitches_at_bar_start(state: Dict[str, Any]) -> List[int]:
    active_pitches = []
    for v in range(state["num_voices"]):
        if state["active_until"][v] > 0 and state["prev_pitch"][v] is not None:
            active_pitches.append(state["prev_pitch"][v])
    return active_pitches


def _infer_num_voices(
    tokens: List[str], running_state: Optional[Dict[str, Any]]
) -> int:
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
    if running_state and running_state.get("num_voices") is not None:
        max_idx = max(max_idx, running_state["num_voices"] - 1)
    if max_idx >= 0:
        return max_idx + 1
    return running_state.get("num_voices", 1) if running_state else 1


def _reconstruct_pitches_and_polyphony(
    tokens: List[str],
    running_state: Dict[str, Any],
    tpq: int,
) -> Tuple[List[int], Dict[str, Any], int]:
    prev_pitch = dict(running_state["prev_pitch"])
    active_until = dict(running_state["active_until"])
    num_voices = running_state["num_voices"]
    onset_pitches: List[int] = []
    polyphony_max = 0

    current_pos: Optional[int] = None
    current_pos_onsets = 0

    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]

        if tok == "BAR":
            idx += 1
            continue

        if tok.startswith("TIME_SIG_") or tok.startswith("KEY_"):
            idx += 1
            continue

        if tok.startswith("ABS_BASS_"):
            prev_pitch[0] = int(tok.split("_")[-1])
            num_voices = max(num_voices, 1)
            active_until.setdefault(0, 0)
            idx += 1
            continue

        if tok.startswith("ABS_SOP_"):
            prev_pitch[3] = int(tok.split("_")[-1])
            num_voices = max(num_voices, 4)
            active_until.setdefault(3, 0)
            idx += 1
            continue

        if tok.startswith("ABS_VOICE_"):
            parts = tok.split("_")
            if len(parts) != 4:
                raise ValueError(f"bad ABS_VOICE token: {tok}")
            v = int(parts[2])
            prev_pitch[v] = int(parts[3])
            num_voices = max(num_voices, v + 1)
            active_until.setdefault(v, 0)
            idx += 1
            continue

        if tok.startswith("REF_VOICE_"):
            idx += 1
            continue

        if tok.startswith("ABS_LOW_") or tok.startswith("ABS_HIGH_"):
            idx += 1
            continue

        if tok.startswith("POS_"):
            current_pos = int(tok.split("_")[1])
            current_pos_onsets = 0
            idx += 1
            continue

        if tok.startswith("VOICE_"):
            if current_pos is None:
                raise ValueError(f"VOICE token before POS: {tok}")
            v = int(tok.split("_")[1])
            if v >= num_voices:
                num_voices = v + 1
                prev_pitch.setdefault(v, None)
                active_until.setdefault(v, 0)

            if idx + 1 < len(tokens) and tokens[idx + 1].startswith("REST_"):
                idx += 2
                continue

            if idx + 2 >= len(tokens):
                raise ValueError(f"truncated VOICE event at index {idx}")

            dur_tok = tokens[idx + 1]
            next_idx = idx + 2
            if next_idx < len(tokens) and tokens[next_idx].startswith("DUP_"):
                next_idx += 1
            if next_idx >= len(tokens):
                raise ValueError(f"truncated VOICE event at index {idx}")
            mel_tok = tokens[next_idx]
            if next_idx + 2 >= len(tokens):
                raise ValueError(f"truncated VOICE event at index {idx}")

            if not dur_tok.startswith("DUR_") or not mel_tok.startswith("MEL_INT12_"):
                raise ValueError(f"bad VOICE event starting at {tok}")

            if prev_pitch[v] is None:
                raise ValueError(f"missing anchor before VOICE_{v} at index {idx}")

            duration_ticks = int(dur_tok.split("_")[-1])
            mel_val = int(mel_tok.split("_")[-1])

            pitch = prev_pitch[v] + mel_val
            prev_pitch[v] = pitch
            active_until[v] = current_pos + duration_ticks

            onset_pitches.append(pitch)
            current_pos_onsets += 1
            polyphony_max = max(polyphony_max, current_pos_onsets)

            idx = next_idx + 3
            continue

        idx += 1

    updated_state = dict(running_state)
    updated_state["prev_pitch"] = prev_pitch
    updated_state["active_until"] = active_until
    updated_state["num_voices"] = num_voices
    return onset_pitches, updated_state, polyphony_max


def compute_bar_plan(
    tokens: List[str],
    bar_index: int,
    running_state: Optional[Dict[str, Any]] = None,
    tpq: int = 24,
    num_voices: Optional[int] = None,
) -> Tuple[BarPlan, Dict[str, Any]]:
    if not tokens or tokens[0] != "BAR":
        raise ValueError("bar token list must start with BAR")

    inferred_voices = _infer_num_voices(tokens, running_state)
    if num_voices is None:
        num_voices = inferred_voices
    else:
        num_voices = max(num_voices, inferred_voices)
    if running_state and running_state.get("num_voices"):
        num_voices = max(num_voices, running_state["num_voices"])
    if num_voices < 1:
        num_voices = 1

    state = _init_running_state(running_state, num_voices)

    prev_bar_len = state.get("bar_len_ticks")
    if prev_bar_len is None and state.get("time_sig"):
        prev_bar_len = _bar_len_ticks(state["time_sig"], tpq)
    if bar_index > 0 and prev_bar_len:
        _shift_active_until(state, prev_bar_len)

    active_pitches = _active_pitches_at_bar_start(state)

    time_sig = _extract_time_sig(tokens, state.get("time_sig"))
    key = _extract_key(tokens, state.get("key"))
    bar_len = _bar_len_ticks(time_sig, tpq)

    state["time_sig"] = time_sig
    state["key"] = key
    state["bar_len_ticks"] = bar_len

    onset_pitches, state, polyphony_max = _reconstruct_pitches_and_polyphony(
        tokens, state, tpq
    )

    density_bucket = _classify_density(len(onset_pitches))
    pitch_range = _compute_pitch_range(onset_pitches, active_pitches)

    plan = BarPlan(
        bar_index=bar_index,
        time_sig=time_sig,
        key=key,
        density_bucket=density_bucket,
        pitch_range=pitch_range,
        polyphony_max=polyphony_max,
    )

    return plan, state
