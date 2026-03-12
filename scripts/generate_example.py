"""generate_example.py — end-to-end example generation script.

Exercises the compose pipeline (tokens → canonical score → tab → MusicXML / MIDI /
ASCII tab) and writes all outputs to an output directory.  Intended for manual QA
and demo preparation, not production batch use.

Usage
-----
# Quick demo using the built-in synthetic token stream (no model required):
    python scripts/generate_example.py --out-dir out/examples

# Generate with a trained checkpoint:
    python scripts/generate_example.py \\
        --checkpoint out/notelm_v1/notelm_step1000.pt \\
        --vocab      data/processed/vocab.json \\
        --key C --style baroque --measures 8 \\
        --out-dir    out/examples

Output files
------------
    example.musicxml   MusicXML score with fingering technical tags
    example.mid        MIDI file
    example_tab.txt    ASCII guitar tab
    tokens.txt         raw token stream (space-separated); pass to eval_basic.py
    metrics.json       eval_basic metrics (bar count, interval range, tab stats, …)
"""

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.canonical import tokens_to_canonical_score
from src.api.render import canonical_score_to_midi, canonical_score_to_musicxml
from src.tabber import DEFAULT_MAX_FRET, render_ascii_tab, tab_events


# ---------------------------------------------------------------------------
# Built-in synthetic token stream: C major scale over two 4/4 bars.
# Exercises every stage of the pipeline without requiring a trained model.
# TPQ=24 → quarter note = 24 ticks; POS values are offsets within the bar.
# VOICE_3 is used intentionally (not 0) to exercise voice remapping.
# HARM_OCT/HARM_CLASS are set to 0/0: single-voice stream, one note per onset.
# ---------------------------------------------------------------------------

_BUILTIN_TOKENS = [
    # Bar 1 — C D E F
    "BAR", "TIME_SIG_4_4", "KEY_C",
    "POS_0",  "ABS_VOICE_3_60",
    "VOICE_3", "DUR_24", "MEL_INT12_0",  "HARM_OCT_0", "HARM_CLASS_0",  # C4
    "POS_24",
    "VOICE_3", "DUR_24", "MEL_INT12_+2", "HARM_OCT_0", "HARM_CLASS_0",  # D4
    "POS_48",
    "VOICE_3", "DUR_24", "MEL_INT12_+2", "HARM_OCT_0", "HARM_CLASS_0",  # E4
    "POS_72",
    "VOICE_3", "DUR_24", "MEL_INT12_+1", "HARM_OCT_0", "HARM_CLASS_0",  # F4
    # Bar 2 — G A B C5
    "BAR", "TIME_SIG_4_4", "KEY_C",
    "POS_0",
    "VOICE_3", "DUR_24", "MEL_INT12_+2", "HARM_OCT_0", "HARM_CLASS_0",  # G4
    "POS_24",
    "VOICE_3", "DUR_24", "MEL_INT12_+2", "HARM_OCT_0", "HARM_CLASS_0",  # A4
    "POS_48",
    "VOICE_3", "DUR_24", "MEL_INT12_+2", "HARM_OCT_0", "HARM_CLASS_0",  # B4
    "POS_72",
    "VOICE_3", "DUR_24", "MEL_INT12_+1", "HARM_OCT_0", "HARM_CLASS_0",  # C5 quarter note
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_eval_basic():
    """Import eval_basic from scripts/ without requiring it to be a package."""
    spec = importlib.util.spec_from_file_location(
        "eval_basic", ROOT / "scripts" / "eval_basic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _generate_with_checkpoint(args) -> list:
    """Run the full compose pipeline using a trained NoteLM checkpoint."""
    from src.api.compose_service import compose_baseline
    from src.inference.controls import ComposeControls, build_control_prefix_tokens
    from src.inference.generate_v1 import GenerationConfig

    controls = ComposeControls(
        key=args.key,
        style=args.style,
        difficulty=args.difficulty,
        measures=args.measures,
    )
    seed_tokens = build_control_prefix_tokens(controls)
    if not seed_tokens:
        seed_tokens = ["KEY_C"]

    config = GenerationConfig(
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    result = compose_baseline(
        Path(args.checkpoint),
        seed_tokens=seed_tokens,
        generation_config=config,
        vocab_path=Path(args.vocab) if args.vocab else None,
    )
    return result.generation.tokens


def _run_pipeline(tokens: list, out_dir: Path) -> dict:
    """Convert tokens through the full pipeline and write all output files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    score = tokens_to_canonical_score(tokens)
    part = score.parts[0]

    tabbed_part = replace(
        part,
        events=tab_events(part.events, tuning=part.info.tuning, max_fret=DEFAULT_MAX_FRET),
    )
    tabbed_score = replace(score, parts=[tabbed_part])

    musicxml = canonical_score_to_musicxml(tabbed_score)
    midi_bytes = canonical_score_to_midi(tabbed_score)
    ascii_tab = render_ascii_tab(tabbed_part.events, tuning=part.info.tuning)

    xml_path = out_dir / "example.musicxml"
    mid_path = out_dir / "example.mid"
    tab_path = out_dir / "example_tab.txt"
    tok_path = out_dir / "tokens.txt"

    xml_path.write_text(musicxml, encoding="utf-8")
    mid_path.write_bytes(midi_bytes)
    tab_path.write_text(ascii_tab, encoding="utf-8")
    tok_path.write_text(" ".join(tokens), encoding="utf-8")

    return {
        "musicxml": xml_path,
        "midi": mid_path,
        "ascii_tab": tab_path,
        "tokens": tok_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate MusicXML/MIDI/ASCII-tab example outputs for QA and demo."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out/examples"),
        metavar="DIR",
        help="directory to write output files (default: out/examples)",
    )
    parser.add_argument(
        "--checkpoint", metavar="PATH",
        help="NoteLM checkpoint .pt file; omit to use the built-in synthetic token stream",
    )
    parser.add_argument(
        "--vocab", metavar="PATH",
        help="vocab JSON for the checkpoint (recommended when using --checkpoint)",
    )
    parser.add_argument("--key",        default=None, metavar="KEY",
                        help="key signature, e.g. C or Am (used with --checkpoint)")
    parser.add_argument("--style",      default=None, metavar="STYLE",
                        help="style token, e.g. baroque (used with --checkpoint)")
    parser.add_argument("--difficulty", default=None, metavar="LEVEL",
                        help="difficulty token, e.g. easy (used with --checkpoint)")
    parser.add_argument("--measures",   type=int, default=None, metavar="N",
                        help="number of measures to generate (used with --checkpoint)")
    parser.add_argument("--max-length", type=int, default=512, metavar="N",
                        help="max generation tokens (default: 512, used with --checkpoint)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p",       type=float, default=0.9)
    parser.add_argument("--no-eval",  action="store_true",
                        help="skip eval_basic metrics")
    parser.add_argument("--quiet",    action="store_true",
                        help="suppress progress output")

    args = parser.parse_args(argv)

    if not args.quiet:
        print("bach-gen example generator")
        print("=" * 40)

    # 1. Obtain token stream
    if args.checkpoint:
        if not args.quiet:
            print(f"Generating with checkpoint: {args.checkpoint}")
        tokens = _generate_with_checkpoint(args)
    else:
        if not args.quiet:
            print("No --checkpoint given — using built-in synthetic stream (C major scale, 2 bars)")
        tokens = list(_BUILTIN_TOKENS)

    if not args.quiet:
        print(f"Token stream length: {len(tokens)} tokens")

    # 2. Run compose pipeline and write files
    paths = _run_pipeline(tokens, args.out_dir)

    if not args.quiet:
        print(f"\nOutputs written to: {args.out_dir.resolve()}")
        for label, path in paths.items():
            print(f"  {label:<12}: {path}")

    # 3. Optional eval_basic metrics
    if not args.no_eval:
        eval_mod = _load_eval_basic()
        metrics = eval_mod.evaluate(tokens)
        metrics_path = args.out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        paths["metrics"] = metrics_path

        if not args.quiet:
            print(f"  {'metrics':<12}: {metrics_path}")
            print("\nMetrics:")
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
                print(f"  {label:<28}: {display}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
