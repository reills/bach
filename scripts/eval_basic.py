"""
eval_basic.py — lightweight evaluation of generated token streams or exported pieces.

Metrics
-------
* bar_count           : number of BAR tokens in the stream
* token_validity      : fraction of tokens that are known (from a vocab file, if provided)
* interval_range_ok   : True if all MEL_INT12 values are in [-24, 24]
* mel_int_range       : (min, max) of observed MEL_INT12 values
* voice_event_count   : total voiced (non-rest) events
* rest_event_count    : total rest events
* tab_present         : True when at least one STR_/FRET_ token is found
* tab_span_mean       : mean fret span per bar (max_fret – min_fret), only if tab_present
* tab_fret_max        : highest fret token value seen (playability proxy)
* tab_open_string_pct : fraction of fret events that are FRET_0 (open string)

Musical quality metrics (optional):
* off_key_rate         : fraction of reconstructed pitches outside the declared key.
                         Requires --key or a KEY_* token in the stream. Uses keyed
                         pitched onsets reconstructed from anchors + MEL_INT12_*.
* harm_mismatch_count  : number of HARM_OCT/HARM_CLASS tokens that do not match the
                         value expected by the interval logic (requires src package).
* duplicate_bar_rate   : fraction of bars whose token sequence exactly repeats an
                         earlier bar (crude repetition proxy).
* cadence_proxy_rate   : fraction of keyed bars whose final pitched onset lands on
                         scale degree 1 or 5. Labeled as proxy only.
* token_grammar_violations : count of malformed VOICE_* event parses.
* counterpoint_*       : voice-leading metrics from src.music.counterpoint.

Polyphony metrics:
* avg_voices_per_bar      : average distinct VOICE indices per bar
* avg_notes_per_onset     : average note events per POS (onset position)
* pct_bars_2plus_voices   : percent of bars with 2+ distinct voices
* pct_bars_3plus_voices   : percent of bars with 3+ distinct voices

Usage
-----
    # score a raw token file (one token per line or space-separated)
    python scripts/eval_basic.py --token-file generated.txt

    # score with key-awareness
    python scripts/eval_basic.py --token-file generated.txt --key C

    # score a parquet events file (must have a "tokens" column)
    python scripts/eval_basic.py --parquet events.parquet [--vocab vocab.json]

    # output JSON
    python scripts/eval_basic.py --token-file generated.txt --output-json results.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Token parsing helpers
# ---------------------------------------------------------------------------

_MEL_INT_RE = re.compile(r"^MEL_INT12_([+-]?\d+)$")
_DUR_RE = re.compile(r"^DUR_(\d+)$")
_REST_RE = re.compile(r"^REST_(\d+)$")
_STR_RE = re.compile(r"^STR_(\d+)$")
_FRET_RE = re.compile(r"^FRET_(\d+)$")
_ABS_VOICE_RE = re.compile(r"^ABS_VOICE_(\d+)_(\d+)$")


# ---------------------------------------------------------------------------
# Soft imports from src (optional — degrade gracefully)
# ---------------------------------------------------------------------------

try:
    _PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from src.tokens.tokenizer import parse_voice_event as _parse_voice_event
    from src.tokens.validator import validate_harm_tokens as _validate_harm_tokens
    from src.music.counterpoint import evaluate_counterpoint_tokens as _evaluate_counterpoint_tokens
    _HAS_SRC_HELPERS = True
except Exception:
    _HAS_SRC_HELPERS = False
    _parse_voice_event = None  # type: ignore[assignment]
    _validate_harm_tokens = None  # type: ignore[assignment]
    _evaluate_counterpoint_tokens = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Key / pitch-class helpers
# ---------------------------------------------------------------------------

_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
_TONIC_TO_PC: Dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def _key_pitch_classes(key: str) -> Optional[Set[int]]:
    """Return pitch-class set for a key string like 'C', 'Am', 'Bb', 'F#m'."""
    if not key:
        return None
    minor = key.endswith("m")
    tonic_str = key[:-1] if minor else key
    pc = _TONIC_TO_PC.get(tonic_str)
    if pc is None:
        return None
    offsets = _MINOR_SCALE if minor else _MAJOR_SCALE
    return {(pc + o) % 12 for o in offsets}


def _parse_time_sig_token(token: str) -> Tuple[int, int]:
    parts = token.split("_")
    if len(parts) != 4:
        raise ValueError(f"bad time signature token: {token}")
    return int(parts[2]), int(parts[3])


def _split_bars(tokens: Sequence[str]) -> List[List[str]]:
    bars: List[List[str]] = []
    current_bar: Optional[List[str]] = None
    for tok in tokens:
        if tok == "BAR":
            if current_bar is not None:
                bars.append(current_bar)
            current_bar = []
            continue
        if current_bar is not None:
            current_bar.append(tok)
    if current_bar is not None:
        bars.append(current_bar)
    return bars


def _count_duplicate_bars(tokens: Sequence[str]) -> int:
    seen = set()
    duplicate_count = 0
    for bar in _split_bars(tokens):
        key = tuple(bar)
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)
    return duplicate_count


def _count_grammar_violations(tokens: Sequence[str]) -> int:
    if not _HAS_SRC_HELPERS:
        return 0

    violations = 0
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok.startswith("VOICE_"):
            try:
                _, next_idx = _parse_voice_event(tokens, idx)
            except ValueError:
                violations += 1
                idx += 1
                continue
            idx = next_idx
            continue
        idx += 1
    return violations


def _infer_onset_records(
    tokens: Sequence[str],
    *,
    key_override: Optional[str] = None,
) -> List[Dict[str, Optional[int]]]:
    records: List[Dict[str, Optional[int]]] = []
    prev_pitch: Dict[int, int] = {}
    bar_len_ticks = 24 * 4
    bar_start = 0
    current_pos_tick = 0
    current_bar_idx = -1
    saw_bar = False
    current_key = key_override
    idx = 0

    while idx < len(tokens):
        tok = tokens[idx]

        if tok == "BAR":
            if saw_bar:
                bar_start += bar_len_ticks
            else:
                saw_bar = True
            current_bar_idx += 1
            idx += 1
            continue

        if tok.startswith("TIME_SIG_"):
            try:
                num, denom = _parse_time_sig_token(tok)
                bar_len_ticks = int(round((num * (4.0 / denom)) * 24))
            except ValueError:
                pass
            idx += 1
            continue

        if tok.startswith("KEY_"):
            if key_override is None:
                current_key = tok[4:]
            idx += 1
            continue

        if tok.startswith("POS_"):
            try:
                current_pos_tick = int(tok.split("_", 1)[1])
            except ValueError:
                pass
            idx += 1
            continue

        if tok.startswith("ABS_BASS_"):
            prev_pitch[0] = int(tok.split("_")[-1])
            idx += 1
            continue

        if tok.startswith("ABS_SOP_"):
            prev_pitch[3] = int(tok.split("_")[-1])
            idx += 1
            continue

        abs_voice_match = _ABS_VOICE_RE.match(tok)
        if abs_voice_match:
            prev_pitch[int(abs_voice_match.group(1))] = int(abs_voice_match.group(2))
            idx += 1
            continue

        if tok.startswith("VOICE_") and _HAS_SRC_HELPERS:
            try:
                event, next_idx = _parse_voice_event(tokens, idx)
            except ValueError:
                idx += 1
                continue

            if not event.is_rest:
                base_pitch = prev_pitch.get(event.voice)
                if base_pitch is not None:
                    pitch = base_pitch + event.mel_int
                    prev_pitch[event.voice] = pitch
                    records.append(
                        {
                            "voice": event.voice,
                            "abs_tick": bar_start + current_pos_tick,
                            "pitch": pitch,
                            "bar_index": current_bar_idx,
                            "key": current_key,
                        }
                    )
            idx = next_idx
            continue

        idx += 1

    return records


def _infer_onset_pitches(tokens: Sequence[str]) -> List[Tuple[int, int, int]]:
    records = _infer_onset_records(tokens)
    return [
        (int(record["voice"]), int(record["abs_tick"]), int(record["pitch"]))
        for record in records
        if record["voice"] is not None
        and record["abs_tick"] is not None
        and record["pitch"] is not None
    ]


def _count_off_key_onsets(
    tokens: Sequence[str],
    key_override: Optional[str] = None,
) -> Tuple[Optional[int], Optional[int]]:
    onset_records = _infer_onset_records(tokens, key_override=key_override)
    off_key_count = 0
    pitched_onset_count = 0

    for record in onset_records:
        key_name = key_override or record["key"]
        if not isinstance(key_name, str):
            continue
        key_pcs = _key_pitch_classes(key_name)
        if key_pcs is None:
            continue
        pitch = record["pitch"]
        if pitch is None:
            continue
        pitched_onset_count += 1
        if pitch % 12 not in key_pcs:
            off_key_count += 1

    if pitched_onset_count == 0:
        return None, None
    return off_key_count, pitched_onset_count


def _cadence_proxy_stats(
    tokens: Sequence[str],
    key_override: Optional[str] = None,
) -> Tuple[int, int]:
    onset_records = _infer_onset_records(tokens, key_override=key_override)
    last_onset_by_bar: Dict[int, Dict[str, Optional[int]]] = {}
    for record in onset_records:
        bar_index = record["bar_index"]
        if isinstance(bar_index, int):
            last_onset_by_bar[bar_index] = record

    hits = 0
    eligible_bars = 0
    for bar_index in sorted(last_onset_by_bar):
        record = last_onset_by_bar[bar_index]
        key_name = key_override or record["key"]
        if not isinstance(key_name, str):
            continue
        tonic_name = key_name[:-1] if key_name.endswith("m") else key_name
        tonic_pc = _TONIC_TO_PC.get(tonic_name)
        pitch = record["pitch"]
        if tonic_pc is None or pitch is None:
            continue
        eligible_bars += 1
        degree = (pitch % 12 - tonic_pc) % 12
        if degree in (0, 7):
            hits += 1

    return hits, eligible_bars


_VOICE_RE = re.compile(r"^VOICE_(\d+)$")
_POS_RE = re.compile(r"^POS_(\d+)$")


def _polyphony_stats(tokens: Sequence[str]) -> Dict:
    """Compute per-bar voice counts and per-onset note counts."""
    bars = _split_bars(tokens)
    voices_per_bar: List[int] = []
    notes_per_onset: List[int] = []

    for bar_tokens in bars:
        voices_in_bar: Set[int] = set()
        notes_at_current_pos = 0
        saw_pos = False
        for tok in bar_tokens:
            vm = _VOICE_RE.match(tok)
            if vm:
                voices_in_bar.add(int(vm.group(1)))
                notes_at_current_pos += 1
            elif _POS_RE.match(tok):
                if saw_pos and notes_at_current_pos > 0:
                    notes_per_onset.append(notes_at_current_pos)
                notes_at_current_pos = 0
                saw_pos = True
        # flush last position
        if saw_pos and notes_at_current_pos > 0:
            notes_per_onset.append(notes_at_current_pos)
        voices_per_bar.append(len(voices_in_bar))

    total_bars = len(voices_per_bar)
    if total_bars == 0:
        return {
            "avg_voices_per_bar": None,
            "avg_notes_per_onset": None,
            "pct_bars_2plus_voices": None,
            "pct_bars_3plus_voices": None,
        }

    avg_voices = sum(voices_per_bar) / total_bars
    bars_2plus = sum(1 for v in voices_per_bar if v >= 2)
    bars_3plus = sum(1 for v in voices_per_bar if v >= 3)
    avg_notes = (sum(notes_per_onset) / len(notes_per_onset)) if notes_per_onset else 0.0

    return {
        "avg_voices_per_bar": round(avg_voices, 3),
        "avg_notes_per_onset": round(avg_notes, 3),
        "pct_bars_2plus_voices": round(100.0 * bars_2plus / total_bars, 2),
        "pct_bars_3plus_voices": round(100.0 * bars_3plus / total_bars, 2),
    }


def _load_tokens_from_text(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8")
    tokens: List[str] = []
    for line in raw.splitlines():
        for tok in line.split():
            tok = tok.strip(",")
            if tok:
                tokens.append(tok)
    return tokens


def _load_tokens_from_parquet(path: Path) -> List[str]:
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required for --parquet; install it with: pip install pandas")
    df = pd.read_parquet(path)
    if "tokens" not in df.columns:
        sys.exit(f"parquet file has no 'tokens' column; found: {list(df.columns)}")
    tokens: List[str] = []
    for cell in df["tokens"]:
        if isinstance(cell, str):
            for tok in cell.split():
                tok = tok.strip(",")
                if tok:
                    tokens.append(tok)
        elif hasattr(cell, "__iter__"):
            for tok in cell:
                if tok:
                    tokens.append(str(tok))
    return tokens


def _load_vocab(path: Path) -> Optional[set]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return set(data.keys())
    if isinstance(data, list):
        return set(data)
    return None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    tokens: List[str],
    vocab: Optional[set] = None,
    key: Optional[str] = None,
) -> Dict:
    harm_mismatch_count: Optional[int] = None
    if _HAS_SRC_HELPERS:
        try:
            harm_mismatch_count = len(_validate_harm_tokens(tokens))
        except Exception:
            harm_mismatch_count = None

    grammar_violations = _count_grammar_violations(tokens)
    duplicate_bar_count = _count_duplicate_bars(tokens)
    onset_pitches = _infer_onset_pitches(tokens)
    off_key_count, pitched_onset_count = _count_off_key_onsets(tokens, key_override=key)
    cadence_proxy_hits, cadence_proxy_eligible_bars = _cadence_proxy_stats(
        tokens, key_override=key
    )

    bar_count = 0
    mel_ints: List[int] = []
    dur_count = 0
    rest_count = 0
    frets_by_bar: List[List[int]] = []
    current_bar_frets: List[int] = []
    fret_values: List[int] = []
    unknown_count = 0
    total = len(tokens)

    for tok in tokens:
        if vocab is not None and tok not in vocab:
            unknown_count += 1

        if tok == "BAR":
            if current_bar_frets:
                frets_by_bar.append(current_bar_frets)
            bar_count += 1
            current_bar_frets = []
            continue

        mel_m = _MEL_INT_RE.match(tok)
        if mel_m:
            mel_ints.append(int(mel_m.group(1)))

        if _DUR_RE.match(tok):
            dur_count += 1

        if _REST_RE.match(tok):
            rest_count += 1

        fm = _FRET_RE.match(tok)
        if fm:
            fv = int(fm.group(1))
            current_bar_frets.append(fv)
            fret_values.append(fv)

    if current_bar_frets:
        frets_by_bar.append(current_bar_frets)

    # --- derived metrics ---

    # interval range
    interval_range_ok = True
    mel_int_range: Optional[Tuple[int, int]] = None
    if mel_ints:
        mn, mx = min(mel_ints), max(mel_ints)
        mel_int_range = (mn, mx)
        interval_range_ok = (-24 <= mn) and (mx <= 24)

    # tab metrics
    tab_present = len(fret_values) > 0
    tab_span_mean: Optional[float] = None
    tab_fret_max: Optional[int] = None
    tab_open_string_pct: Optional[float] = None
    if tab_present:
        spans = [max(bar) - min(bar) for bar in frets_by_bar if bar]
        tab_span_mean = float(sum(spans) / len(spans)) if spans else 0.0
        tab_fret_max = max(fret_values)
        open_count = sum(1 for f in fret_values if f == 0)
        tab_open_string_pct = round(open_count / len(fret_values), 4)

    # token validity
    token_validity: Optional[float] = None
    if vocab is not None and total > 0:
        token_validity = round((total - unknown_count) / total, 4)

    duplicate_bar_rate: Optional[float] = None
    if bar_count > 0:
        duplicate_bar_rate = round(duplicate_bar_count / bar_count, 4)

    off_key_rate: Optional[float] = None
    if pitched_onset_count is not None and pitched_onset_count > 0 and off_key_count is not None:
        off_key_rate = round(off_key_count / pitched_onset_count, 4)

    cadence_proxy_rate: Optional[float] = None
    if cadence_proxy_eligible_bars > 0:
        cadence_proxy_rate = round(
            cadence_proxy_hits / cadence_proxy_eligible_bars, 4
        )

    polyphony = _polyphony_stats(tokens)
    counterpoint: Dict[str, object] = {}
    if _HAS_SRC_HELPERS and _evaluate_counterpoint_tokens is not None:
        try:
            counterpoint = {
                f"counterpoint_{key}": value
                for key, value in _evaluate_counterpoint_tokens(tokens).to_dict().items()
            }
        except Exception:
            counterpoint = {}

    result: Dict = {
        "token_count": total,
        "bar_count": bar_count,
        "token_validity": token_validity,
        "interval_range_ok": interval_range_ok,
        "mel_int_range": list(mel_int_range) if mel_int_range else None,
        "voice_event_count": dur_count,
        "rest_event_count": rest_count,
        "tab_present": tab_present,
        "onset_pitch_count": len(onset_pitches),
        "off_key_count": off_key_count,
        "pitched_onset_count": pitched_onset_count,
        "off_key_rate": off_key_rate,
        "harm_mismatch_count": harm_mismatch_count,
        "duplicate_bar_count": duplicate_bar_count,
        "duplicate_bar_rate": duplicate_bar_rate,
        "cadence_proxy_hits": cadence_proxy_hits,
        "cadence_proxy_eligible_bars": cadence_proxy_eligible_bars,
        "cadence_proxy_rate": cadence_proxy_rate,
        "token_grammar_violations": grammar_violations,
        **polyphony,
        **counterpoint,
    }
    if tab_present:
        result["tab_span_mean"] = round(tab_span_mean, 3) if tab_span_mean is not None else None
        result["tab_fret_max"] = tab_fret_max
        result["tab_open_string_pct"] = tab_open_string_pct

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(metrics: Dict) -> None:
    w = 28
    print("=" * 50)
    print("eval_basic — token stream metrics")
    print("=" * 50)
    for k, v in metrics.items():
        label = k.replace("_", " ").title()
        if v is None:
            display = "n/a"
        elif isinstance(v, bool):
            display = str(v)
        elif isinstance(v, float):
            display = f"{v:.4f}"
        else:
            display = str(v)
        print(f"  {label:<{w}}: {display}")
    print("=" * 50)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a generated token stream with pragmatic music metrics."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--token-file", type=Path, metavar="FILE",
                     help="plain-text file of tokens (space/newline separated)")
    src.add_argument("--parquet", type=Path, metavar="FILE",
                     help="parquet events file with a 'tokens' column")
    parser.add_argument("--vocab", type=Path, metavar="FILE",
                        help="vocab JSON (dict or list) for validity scoring")
    parser.add_argument("--key", type=str, metavar="KEY",
                        help="declared key for off_key_rate (e.g. C, Am, Bb, F#m); "
                             "if omitted the first KEY_* token in the stream is used")
    parser.add_argument("--output-json", type=Path, metavar="FILE",
                        help="write metrics as JSON to this file")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress printed report")

    args = parser.parse_args(argv)

    if args.token_file:
        tokens = _load_tokens_from_text(args.token_file)
    else:
        tokens = _load_tokens_from_parquet(args.parquet)

    vocab = _load_vocab(args.vocab) if args.vocab else None

    metrics = evaluate(tokens, vocab=vocab, key=getattr(args, "key", None))

    if not args.quiet:
        _print_report(metrics)

    if args.output_json:
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"Metrics written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
