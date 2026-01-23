import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Ensure project root is in path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.tokens.eventizer import eventize_musicxml
from src.tokens.roundtrip import tokens_to_midi
from src.dataio.descriptors import compute_bar_plan
from src.tokens.schema import BarPlan


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
    return max_idx + 1 if max_idx >= 0 else 0


def _maybe_roundtrip(
    tokens: List[str],
    tpq: int,
    out_dir: Path,
    label: str,
) -> None:
    midi_path = out_dir / f"{label}.roundtrip.mid"
    tokens_to_midi(tokens, str(midi_path), tpq=tpq)


def process_file(
    path: Path,
    args: argparse.Namespace,
    roundtrip_state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process a single MusicXML file into a list of bar records.
    """
    try:
        # 1. Eventize
        tokens, meta = eventize_musicxml(
            str(path),
            tpq=args.tpq,
            reentry_ticks=args.reentry,
            mel_range=args.mel_range,
            anchor_large_leaps=args.anchor_large_leaps,
            voice_mode=args.voice_mode,
            max_voices=args.max_voices,
        )
    except Exception as e:
        print(f"Skipping {path.name}: {e}")
        return [], []

    if roundtrip_state is not None and roundtrip_state["remaining"] > 0:
        try:
            _maybe_roundtrip(
                tokens,
                args.tpq,
                roundtrip_state["out_dir"],
                path.stem,
            )
        except Exception as e:
            print(f"Roundtrip failed for {path.name}: {e}")
        else:
            roundtrip_state["remaining"] -= 1

    # 2. Split into bars and compute plans
    num_voices = _infer_num_voices(tokens)
    bars_data = []
    plans_data = []
    current_bar_tokens = []
    bar_index = 0
    running_state = None

    # The token stream starts with BAR, so the first accumulation happens immediately
    # We iterate and split by "BAR" token
    
    # We need to handle the stream carefully. 
    # eventize_musicxml returns a flat list. 
    # Structure: BAR, TIME_SIG..., POS..., ... BAR ...
    
    # Helper to yield groups of tokens per bar
    # Note: The first token is BAR.
    
    iterator = iter(tokens)
    try:
        first = next(iterator)
        if first != "BAR":
            print(f"Skipping {path.name}: Stream does not start with BAR")
            return []
    except StopIteration:
         return []

    current_bar_tokens = ["BAR"]
    
    for tok in iterator:
        if tok == "BAR":
            # Finish previous bar
            try:
                plan, running_state = compute_bar_plan(
                    current_bar_tokens, 
                    bar_index, 
                    running_state, 
                    tpq=args.tpq,
                    num_voices=num_voices,
                )
                
                # Record
                plan_json = json.dumps(plan.__dict__)
                bars_data.append({
                    "piece_id": path.stem,
                    "bar_index": bar_index,
                    "tokens": " ".join(current_bar_tokens), # Space-separated string
                    "plan_json": plan_json,  # Serialized plan
                    "bar_len_ticks": running_state["bar_len_ticks"]
                })
                plans_data.append({
                    "piece_id": path.stem,
                    "bar_index": bar_index,
                    "plan_json": plan_json,
                })
                
                bar_index += 1
                current_bar_tokens = ["BAR"]
            except Exception as e:
                print(f"Error in {path.name} bar {bar_index}: {e}")
                # We might want to abort this file or skip this bar?
                # Usually better to abort file to keep continuity
                return [], []
        else:
            current_bar_tokens.append(tok)
            
    # Process the last bar
    if current_bar_tokens:
        try:
            plan, running_state = compute_bar_plan(
                current_bar_tokens, 
                bar_index, 
                running_state, 
                tpq=args.tpq,
                num_voices=num_voices,
            )
            plan_json = json.dumps(plan.__dict__)
            bars_data.append({
                "piece_id": path.stem,
                "bar_index": bar_index,
                "tokens": " ".join(current_bar_tokens),
                "plan_json": plan_json,
                "bar_len_ticks": running_state["bar_len_ticks"]
            })
            plans_data.append({
                "piece_id": path.stem,
                "bar_index": bar_index,
                "plan_json": plan_json,
            })
        except Exception as e:
            print(f"Error in {path.name} last bar: {e}")
            return [], []

    return bars_data, plans_data

def main():
    parser = argparse.ArgumentParser(description="Build dataset from MusicXML files.")
    parser.add_argument(
        "--input",
        default="data/tobis_xml",
        help="Input directory (recursive search)",
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Output directory for parquet/json",
    )
    
    # Tokenizer params
    parser.add_argument("--tpq", type=int, default=24)
    parser.add_argument("--reentry", type=int, default=48)
    parser.add_argument("--mel-range", type=int, default=24)
    parser.add_argument("--anchor-large-leaps", action="store_true")
    parser.add_argument(
        "--voice-mode",
        default="auto",
        choices=["auto", "parts", "pitch", "events"],
        help="Voice mapping strategy (auto=collapse+continuity, parts/pitch=part-based).",
    )
    parser.add_argument(
        "--max-voices",
        type=int,
        default=8,
        help="Max VOICE tracks to emit (default: 8).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of files processed (0 = no limit).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle files before applying --limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for --shuffle.",
    )
    parser.add_argument(
        "--validate-roundtrip",
        type=int,
        default=0,
        help="Roundtrip this many files to MIDI for a smoke check.",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    all_plans = []
    
    # Find all xml/mxl/musicxml files
    extensions = {".xml", ".mxl", ".musicxml"}
    files = [p for p in input_dir.rglob("*") if p.suffix.lower() in extensions]
    files = sorted(files)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(files)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    print(f"Found {len(files)} files in {input_dir}")
    
    roundtrip_state = None
    tmp_dir = None
    if args.validate_roundtrip and args.validate_roundtrip > 0:
        tmp_dir = tempfile.TemporaryDirectory()
        roundtrip_state = {
            "remaining": args.validate_roundtrip,
            "out_dir": Path(tmp_dir.name),
            "tmp_dir": tmp_dir,
        }

    for p in files:
        records, plans = process_file(p, args, roundtrip_state=roundtrip_state)
        all_records.extend(records)
        all_plans.extend(plans)
        if len(records) > 0:
            print(f"Processed {p.name}: {len(records)} bars")
            
    if not all_records:
        print("No data processed.")
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return
        
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Save to Parquet
    out_file = output_dir / "events.parquet"
    # PyArrow or fastparquet engine required
    try:
        df.to_parquet(out_file, index=False)
        print(f"Saved dataset to {out_file} ({len(df)} bars)")
    except ImportError:
        print("Error: pandas requires 'pyarrow' or 'fastparquet' to save parquet.")
        print("Saving as CSV instead...")
        df.to_csv(output_dir / "events.csv", index=False)
        print(f"Saved dataset to {output_dir / 'events.csv'}")

    if all_plans:
        plans_df = pd.DataFrame(all_plans)
        plans_file = output_dir / "barplans.parquet"
        try:
            plans_df.to_parquet(plans_file, index=False)
            print(f"Saved barplans to {plans_file} ({len(plans_df)} bars)")
        except ImportError:
            plans_df.to_csv(output_dir / "barplans.csv", index=False)
            print(f"Saved barplans to {output_dir / 'barplans.csv'}")

    # Compute and save simplified stats
    stats = {
        "total_bars": len(df),
        "total_pieces": df["piece_id"].nunique(),
        "avg_bar_len_ticks": float(df["bar_len_ticks"].mean()) if "bar_len_ticks" in df else 0,
        "total_files": len(files),
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved stats.json")

    if tmp_dir is not None:
        tmp_dir.cleanup()

if __name__ == "__main__":
    main()
