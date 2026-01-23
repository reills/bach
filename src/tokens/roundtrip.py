from typing import List, Optional, Tuple

import music21


def _parse_time_sig_token(token: str) -> Tuple[int, int]:
    parts = token.split("_")
    if len(parts) != 4:
        raise ValueError(f"bad time signature token: {token}")
    return int(parts[2]), int(parts[3])


def _parse_key_token(token: str) -> music21.key.Key:
    key_str = token[len("KEY_") :]
    mode = "major"
    if key_str.endswith("m"):
        mode = "minor"
        key_str = key_str[:-1]
    tonic = key_str.replace("b", "-")
    return music21.key.Key(tonic, mode)


def _parse_abs_voice(token: str) -> Tuple[int, int]:
    parts = token.split("_")
    if len(parts) != 4:
        raise ValueError(f"bad ABS_VOICE token: {token}")
    return int(parts[2]), int(parts[3])


def _parse_pitch_from_token(token: str) -> int:
    return int(token.split("_", 1)[1])


def _parse_signed_int(token: str) -> int:
    return int(token.split("_")[-1])


def _parse_last_int(token: str) -> int:
    return int(token.split("_")[-1])


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


def _tokenize_stream(text: str) -> List[str]:
    raw = text.replace("\n", ",")
    tokens = [tok.strip() for tok in raw.split(",")]
    return [tok for tok in tokens if tok]


def tokens_to_score(tokens: List[str], tpq: int = 24) -> music21.stream.Score:
    score = music21.stream.Score()
    num_voices = _infer_num_voices(tokens)
    parts = []
    for v in range(num_voices):
        part = music21.stream.Part(id=f"Voice{v}")
        parts.append(part)
        score.append(part)

    current_time = 0
    bar_start = 0
    bar_len_ticks = tpq * 4
    time_sig: Optional[Tuple[int, int]] = None
    key_token: Optional[str] = None
    prev_pitch: List[Optional[int]] = [None] * num_voices

    i = 0
    first_bar = True
    while i < len(tokens):
        tok = tokens[i]

        if tok == "BAR":
            if not first_bar:
                bar_start += bar_len_ticks
            first_bar = False
            i += 1
            continue

        if tok.startswith("TIME_SIG_"):
            num, denom = _parse_time_sig_token(tok)
            time_sig = (num, denom)
            bar_len_ticks = int(round((num * (4.0 / denom)) * tpq))
            for part in parts:
                part.insert(bar_start / tpq, music21.meter.TimeSignature(f"{num}/{denom}"))
            i += 1
            continue

        if tok.startswith("KEY_"):
            key_token = tok
            key_obj = _parse_key_token(tok)
            for part in parts:
                part.insert(bar_start / tpq, music21.key.Key(key_obj.tonic, key_obj.mode))
            i += 1
            continue

        if tok.startswith("POS_"):
            pos_tick = _parse_pitch_from_token(tok)
            current_time = bar_start + pos_tick
            i += 1
            continue

        if tok.startswith("ABS_BASS_"):
            prev_pitch[0] = _parse_last_int(tok)
            i += 1
            continue

        if tok.startswith("ABS_SOP_"):
            if len(prev_pitch) <= 3:
                prev_pitch.extend([None] * (4 - len(prev_pitch)))
                for v in range(len(parts), 4):
                    part = music21.stream.Part(id=f"Voice{v}")
                    parts.append(part)
                    score.append(part)
            prev_pitch[3] = _parse_last_int(tok)
            i += 1
            continue

        if tok.startswith("ABS_VOICE_"):
            v, pitch = _parse_abs_voice(tok)
            if v >= len(prev_pitch):
                for idx in range(len(prev_pitch), v + 1):
                    prev_pitch.append(None)
                    part = music21.stream.Part(id=f"Voice{idx}")
                    parts.append(part)
                    score.append(part)
            prev_pitch[v] = pitch
            i += 1
            continue

        if tok.startswith("ABS_LOW_") or tok.startswith("ABS_HIGH_"):
            i += 1
            continue

        if tok.startswith("REF_VOICE_"):
            i += 1
            continue

        if tok.startswith("VOICE_"):
            voice = int(tok.split("_", 1)[1])
            if voice >= len(prev_pitch):
                for idx in range(len(prev_pitch), voice + 1):
                    prev_pitch.append(None)
                    part = music21.stream.Part(id=f"Voice{idx}")
                    parts.append(part)
                    score.append(part)
            next_tok = tokens[i + 1]
            if next_tok.startswith("REST_"):
                i += 2
                continue

            dur_tok = tokens[i + 1]
            next_idx = i + 2
            if next_idx < len(tokens) and tokens[next_idx].startswith("DUP_"):
                next_idx += 1
            mel_tok = tokens[next_idx]

            duration_ticks = _parse_pitch_from_token(dur_tok)
            mel_int = _parse_signed_int(mel_tok)

            if prev_pitch[voice] is None:
                raise ValueError(f"missing anchor for voice {voice} at token index {i}")

            pitch = prev_pitch[voice] + mel_int
            note = music21.note.Note(pitch)
            note.duration = music21.duration.Duration(duration_ticks / tpq)
            parts[voice].insert(current_time / tpq, note)
            prev_pitch[voice] = pitch

            i = next_idx + 3
            continue

        i += 1

    return score


def tokens_to_midi(tokens: List[str], midi_path: str, tpq: int = 24) -> None:
    score = tokens_to_score(tokens, tpq=tpq)
    score.write("midi", fp=midi_path)


def load_tokens_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return _tokenize_stream(f.read())
