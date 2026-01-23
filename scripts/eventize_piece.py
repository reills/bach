#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tokens.eventizer import eventize_musicxml  # noqa: E402
from src.tokens.roundtrip import tokens_to_midi  # noqa: E402


def _write_tokens(tokens, path, pretty=True):
    if not pretty:
        text = ", ".join(tokens)
        path.write_text(text, encoding="utf-8")
        return

    lines = []
    current = []
    for tok in tokens:
        if tok == "BAR" and current:
            lines.append(", ".join(current))
            current = [tok]
        else:
            current.append(tok)
    if current:
        lines.append(", ".join(current))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Eventize MusicXML into raw token strings (with optional MIDI round-trip)."
    )
    parser.add_argument("--input", required=True, help="Path to MusicXML file")
    parser.add_argument("--output", required=True, help="Path to token text output")
    parser.add_argument("--midi-out", help="Optional path for round-trip MIDI output")
    parser.add_argument("--tpq", type=int, default=24, help="Ticks per quarter")
    parser.add_argument(
        "--reentry",
        type=int,
        default=48,
        help="Ticks of silence before emitting ABS_VOICE on re-entry",
    )
    parser.add_argument(
        "--mel-range",
        type=int,
        default=24,
        help="Range for large-leap detection (semitones)",
    )
    parser.add_argument(
        "--anchor-large-leaps",
        action="store_true",
        help="Emit ABS_VOICE when leaps exceed --mel-range",
    )
    parser.add_argument(
        "--key",
        help="Override key token (e.g., Em, C, Bb). Default: infer if possible.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Write tokens grouped by BAR (default).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write tokens as a single line.",
    )

    args = parser.parse_args()
    pretty = args.pretty and not args.compact

    tokens, meta = eventize_musicxml(
        args.input,
        tpq=args.tpq,
        reentry_ticks=args.reentry,
        mel_range=args.mel_range,
        anchor_large_leaps=args.anchor_large_leaps,
        key_override=args.key,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tokens(tokens, output_path, pretty=pretty)

    print(f"wrote tokens: {output_path}")
    print(f"time signature token: {meta.time_sig_token}")
    if meta.key_token:
        print(f"key token: {meta.key_token}")
    print(f"voice mapping: {meta.mapping_note}")

    if args.midi_out:
        midi_path = Path(args.midi_out)
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        tokens_to_midi(tokens, str(midi_path), tpq=args.tpq)
        print(f"wrote round-trip midi: {midi_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
