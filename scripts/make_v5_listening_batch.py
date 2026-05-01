#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_instrumental_v5 import _generate_rows, _template_piece
from src.api.render.midi import canonical_score_to_midi
from src.api.render.musicxml import canonical_score_to_musicxml
from src.instrumental_v3.metrics import evaluate_slices
from src.instrumental_v3.representation import (
    FEATURE_SPECS as V3_FEATURE_SPECS,
    FIELD_NAMES as V3_FIELD_NAMES,
    InstrumentalV3Piece,
    piece_to_canonical_score,
    slice_rows_to_piece,
)
from src.instrumental_v4.model import CompoundConfig
from src.instrumental_v5.model import build_generator
from src.instrumental_v5.representation import V5_FIELD_NAMES


RATING_COLUMNS = [
    "sample_id",
    "listened",
    "bach_like_1_5",
    "musical_coherence_1_5",
    "counterpoint_1_5",
    "interesting_1_5",
    "best_moment",
    "main_failure",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reviewable v5 listening batch with MIDI/MusicXML and rating files.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--piece-index", type=int, default=0)
    parser.add_argument("--prompt-rows", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.init()
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, max_context = _load_model(Path(args.checkpoint), device=device)
    events = pd.read_parquet(Path(args.data_dir) / "events.parquet")
    grouped = [group.sort_values("row_index").copy() for _, group in events.groupby("piece_id", sort=False)]
    if not grouped:
        raise SystemExit("events.parquet has no pieces")
    template_df = grouped[min(args.piece_index, len(grouped) - 1)]
    template = _template_piece(template_df)
    prompt = template_df[V5_FIELD_NAMES].to_numpy(dtype="int64").tolist()[: args.prompt_rows]
    if len(prompt) < 2:
        raise SystemExit("prompt must contain at least two rows")
    prompt_paths, prompt_metrics = _write_prompt(out_dir / "prompt", prompt_rows=prompt, template=template)

    manifest_samples = []
    for sample_idx in range(1, args.samples + 1):
        sample_id = f"sample_{sample_idx:03d}"
        sample_seed = args.seed + sample_idx - 1
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(sample_seed)
        sample_dir = out_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        rows = _generate_rows(
            model,
            prompt_rows=[row[:] for row in prompt],
            template=template,
            max_new_rows=args.max_new_tokens,
            device=device,
            max_context=max_context,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        generated_only_rows = _reset_positions(rows[len(prompt) :], template=template)
        paths, metrics = _write_sample(
            sample_dir,
            sample_id=sample_id,
            rows=generated_only_rows,
            template=template,
            checkpoint=str(args.checkpoint),
            prompt_rows=len(prompt),
            generated_rows=args.max_new_tokens,
            seed=sample_seed,
        )
        manifest_samples.append(
            {
                "sample_id": sample_id,
                "seed": sample_seed,
                "paths": {key: str(path.relative_to(out_dir)) for key, path in paths.items()},
                "metrics": metrics,
            }
        )

    manifest = {
        "batch_id": out_dir.name,
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "prompt": {
            "rows": len(prompt),
            "paths": {key: str(path.relative_to(out_dir)) for key, path in prompt_paths.items()},
            "metrics": prompt_metrics,
        },
        "samples": manifest_samples,
        "generation": {
            "samples": args.samples,
            "piece_index": args.piece_index,
            "prompt_rows": args.prompt_rows,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
    }
    _write_manifest(out_dir / "manifest.json", manifest)
    _write_ratings_template(out_dir / "ratings_template.csv", [sample["sample_id"] for sample in manifest_samples])
    _write_notes(out_dir / "notes.md", args=args, sample_ids=[sample["sample_id"] for sample in manifest_samples])
    print(f"Listening batch created at:\n{out_dir}/")
    print("The prompt is exported once in prompt/. Each sample contains only generated continuation audio.")
    print("Listen to the two samples and fill in ratings_template.csv or notes.md before the next modeling change.")


def _load_model(checkpoint_path: Path, *, device: torch.device) -> tuple[torch.nn.Module, int]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if list(ckpt.get("field_names", [])) != V5_FIELD_NAMES:
        raise SystemExit("checkpoint field_names do not match v5 representation")
    config = CompoundConfig(**ckpt["config"])
    model = build_generator(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config.max_seq_len


def _write_sample(
    sample_dir: Path,
    *,
    sample_id: str,
    rows: list[list[int]],
    template: InstrumentalV3Piece,
    checkpoint: str,
    prompt_rows: int,
    generated_rows: int,
    seed: int,
) -> tuple[dict[str, Path], dict[str, float | int]]:
    v3_rows = [row[: len(V3_FIELD_NAMES)] for row in rows]
    piece = slice_rows_to_piece(v3_rows, template=template, piece_id=sample_id, source_path=checkpoint)
    score = piece_to_canonical_score(piece)
    report = evaluate_slices(piece.slices)
    metrics = _sanity_metrics(report.to_dict(), template=template)

    musicxml_path = sample_dir / f"{sample_id}.musicxml"
    midi_path = sample_dir / f"{sample_id}.mid"
    rows_path = sample_dir / f"{sample_id}.rows.json"
    metrics_path = sample_dir / f"{sample_id}.metrics.json"
    musicxml_path.write_text(canonical_score_to_musicxml(score), encoding="utf-8")
    midi_path.write_bytes(canonical_score_to_midi(score))
    rows_path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "checkpoint": checkpoint,
                "seed": seed,
                "prompt_rows": prompt_rows,
                "generated_rows": generated_rows,
                "field_names": V5_FIELD_NAMES,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return (
        {
            "musicxml": musicxml_path,
            "midi": midi_path,
            "rows": rows_path,
            "metrics": metrics_path,
        },
        metrics,
    )


def _write_prompt(
    prompt_dir: Path,
    *,
    prompt_rows: list[list[int]],
    template: InstrumentalV3Piece,
) -> tuple[dict[str, Path], dict[str, float | int]]:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    return _write_sample(
        prompt_dir,
        sample_id="prompt",
        rows=_reset_positions(prompt_rows, template=template),
        template=template,
        checkpoint="source_prompt",
        prompt_rows=0,
        generated_rows=0,
        seed=0,
    )


def _reset_positions(rows: list[list[int]], *, template: InstrumentalV3Piece) -> list[list[int]]:
    out = []
    for idx, row in enumerate(rows):
        new_row = row[:]
        bar = min(V3_FEATURE_SPECS["bar"] - 1, idx // template.steps_per_bar)
        pos = min(V3_FEATURE_SPECS["pos"] - 1, idx % template.steps_per_bar)
        phrase_pos = bar % V3_FEATURE_SPECS["phrase_pos"]
        new_row[V5_FIELD_NAMES.index("bar")] = bar
        new_row[V5_FIELD_NAMES.index("pos")] = pos
        new_row[V5_FIELD_NAMES.index("phrase_pos")] = phrase_pos
        new_row[V5_FIELD_NAMES.index("cadence_zone")] = 1 if phrase_pos in {6, 7} else 0
        out.append(new_row)
    return out


def _sanity_metrics(report: dict[str, object], *, template: InstrumentalV3Piece) -> dict[str, float | int]:
    v0_note_rate = float(report["v0_note_rate"])
    v1_note_rate = float(report["v1_note_rate"])
    return {
        "invalid_pitch_state_rate": float(report["invalid_pitch_state_rate"]),
        "voice_crossing_rate": float(report["voice_crossing_rate"]),
        "stuck_voice_rate": max(float(report["v0_stuck_rate"]), float(report["v1_stuck_rate"])),
        "repeated_sonority_rate": float(report["repeated_sonority_rate"]),
        "num_bars_estimate": int((int(report["slice_count"]) + template.steps_per_bar - 1) // template.steps_per_bar),
        "num_voices_active": int(v0_note_rate > 0.01) + int(v1_note_rate > 0.01),
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _write_ratings_template(path: Path, sample_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RATING_COLUMNS)
        writer.writeheader()
        for sample_id in sample_ids:
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "listened": "",
                    "bach_like_1_5": "",
                    "musical_coherence_1_5": "",
                    "counterpoint_1_5": "",
                    "interesting_1_5": "",
                    "best_moment": "",
                    "main_failure": "",
                    "notes": "",
                }
            )


def _write_notes(path: Path, *, args: argparse.Namespace, sample_ids: list[str]) -> None:
    lines = [
        f"# Listening batch: {Path(args.out_dir).name}",
        "",
        "Checkpoint:",
        str(args.checkpoint),
        "",
        "Generation settings:",
        f"temperature = {args.temperature}",
        f"top_p = {args.top_p}",
        f"seed = {args.seed}",
        f"samples = {args.samples}",
        f"max_new_tokens = {args.max_new_tokens}",
        "",
        "## What to listen for",
        "",
        "- Does it sound like intentional Baroque counterpoint?",
        "- Are there two active voices?",
        "- Is there an opening subject-like idea?",
        "- Does the second voice imitate or answer?",
        "- Does it become random?",
        "- Does one voice get stuck?",
        "- Does it cadence?",
        "- Would I want to keep this sample?",
        "",
        "The prompt is available in `prompt/` for reference. The sample files are continuation-only.",
        "",
        "## Ratings",
        "",
    ]
    for sample_id in sample_ids:
        lines.extend(
            [
                f"### {sample_id}",
                "Bach-like:",
                "Coherence:",
                "Counterpoint:",
                "Interesting:",
                "Best moment:",
                "Main failure:",
                "Notes:",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
