"""
eval_basic.py — lightweight evaluation of generated token streams or exported pieces.

Metrics
-------
* bar_count          : number of BAR tokens in the stream
* token_validity     : fraction of tokens that are known (from a vocab file, if provided)
* interval_range_ok  : True if all MEL_INT12 values are in [-24, 24]
* mel_int_range      : (min, max) of observed MEL_INT12 values
* voice_event_count  : total voiced (non-rest) events
* rest_event_count   : total rest events
* tab_present        : True when at least one STR_/FRET_ token is found
* tab_span_mean      : mean fret span per bar (max_fret – min_fret), only if tab_present
* tab_fret_max       : highest fret token value seen (playability proxy)
* tab_open_string_pct: fraction of fret events that are FRET_0 (open string)

Usage
-----
    # score a raw token file (one token per line or space-separated)
    python scripts/eval_basic.py --token-file generated.txt

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
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Token parsing helpers
# ---------------------------------------------------------------------------

_MEL_INT_RE = re.compile(r"^MEL_INT12_([+-]?\d+)$")
_DUR_RE = re.compile(r"^DUR_(\d+)$")
_REST_RE = re.compile(r"^REST_(\d+)$")
_STR_RE = re.compile(r"^STR_(\d+)$")
_FRET_RE = re.compile(r"^FRET_(\d+)$")


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

def evaluate(tokens: List[str], vocab: Optional[set] = None) -> Dict:
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
        if tok == "BAR":
            bar_count += 1
            if current_bar_frets:
                frets_by_bar.append(current_bar_frets)
            current_bar_frets = []

        m = _MEL_INT_RE.match(tok)
        if m:
            mel_ints.append(int(m.group(1)))

        if _DUR_RE.match(tok):
            dur_count += 1

        if _REST_RE.match(tok):
            rest_count += 1

        fm = _FRET_RE.match(tok)
        if fm:
            fv = int(fm.group(1))
            current_bar_frets.append(fv)
            fret_values.append(fv)

        if vocab is not None and tok not in vocab:
            unknown_count += 1

    # flush last bar
    if current_bar_frets:
        frets_by_bar.append(current_bar_frets)

    # interval range sanity
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

    result: Dict = {
        "token_count": total,
        "bar_count": bar_count,
        "token_validity": token_validity,
        "interval_range_ok": interval_range_ok,
        "mel_int_range": list(mel_int_range) if mel_int_range else None,
        "voice_event_count": dur_count,
        "rest_event_count": rest_count,
        "tab_present": tab_present,
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

    metrics = evaluate(tokens, vocab=vocab)

    if not args.quiet:
        _print_report(metrics)

    if args.output_json:
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"Metrics written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
