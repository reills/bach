#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.render.midi import canonical_score_to_midi
from src.api.render.musicxml import canonical_score_to_musicxml
from src.emi.structured_invention import StructuredInventionConfig, compose_structured_invention

RATING_COLUMNS = [
    "sample_id",
    "listened",
    "bach_like_1_5",
    "cohesive_form_1_5",
    "subject_answer_1_5",
    "counterpoint_1_5",
    "best_moment",
    "main_failure",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structured two-voice Bach-like invention listening samples.")
    parser.add_argument("--out-dir", default="out/listening_batches/structured_invention_eval")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--key", default="D minor")
    parser.add_argument("--measures", type=int, default=16)
    parser.add_argument("--tempo", type=int, default=84)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    for idx in range(args.samples):
        sample_id = f"sample_{idx + 1:03d}"
        sample_dir = out_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        seed = args.seed + idx
        composition = compose_structured_invention(
            StructuredInventionConfig(
                key=args.key,
                measures=args.measures,
                tempo=args.tempo,
                seed=seed,
                title=sample_id,
            )
        )
        paths = _write_sample(sample_dir, sample_id=sample_id, composition=composition)
        samples.append(
            {
                "sample_id": sample_id,
                "seed": seed,
                "paths": {name: str(path.relative_to(out_dir)) for name, path in paths.items()},
                "diagnostics": composition.diagnostics,
            }
        )
        print(f"[{idx + 1}/{args.samples}] wrote {sample_id}", flush=True)

    manifest = {
        "batch_id": out_dir.name,
        "engine": "structured_invention_v1",
        "generation": {
            "key": args.key,
            "measures": args.measures,
            "tempo": args.tempo,
            "samples": args.samples,
            "seed": args.seed,
        },
        "samples": samples,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_ratings(out_dir / "ratings_template.csv", [sample["sample_id"] for sample in samples])
    _write_notes(out_dir / "notes.md", args=args, sample_ids=[sample["sample_id"] for sample in samples])
    print(f"Structured invention listening batch created at:\n{out_dir}/")


def _write_sample(sample_dir: Path, *, sample_id: str, composition) -> dict[str, Path]:
    musicxml_path = sample_dir / f"{sample_id}.musicxml"
    midi_path = sample_dir / f"{sample_id}.mid"
    diagnostics_path = sample_dir / f"{sample_id}.diagnostics.json"
    musicxml_path.write_text(canonical_score_to_musicxml(composition.score), encoding="utf-8")
    midi_path.write_bytes(canonical_score_to_midi(composition.score))
    diagnostics_path.write_text(json.dumps(composition.diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    return {"musicxml": musicxml_path, "midi": midi_path, "diagnostics": diagnostics_path}


def _write_ratings(path: Path, sample_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RATING_COLUMNS)
        writer.writeheader()
        for sample_id in sample_ids:
            writer.writerow({column: sample_id if column == "sample_id" else "" for column in RATING_COLUMNS})


def _write_notes(path: Path, *, args: argparse.Namespace, sample_ids: list[str]) -> None:
    lines = [
        f"# Structured invention listening batch: {Path(args.out_dir).name}",
        "",
        "This is a rule-structured baseline, not the V5 neural model.",
        "It is intended to test subject/answer/form coherence before pushing more neural training.",
        "",
        f"key = {args.key}",
        f"measures = {args.measures}",
        f"tempo = {args.tempo}",
        f"seed = {args.seed}",
        "",
        "## Listen for",
        "",
        "- Is there a recognizable subject and answer?",
        "- Does the middle develop sequentially instead of wandering?",
        "- Is there a register/form arch?",
        "- Does the ending cadence feel intentional?",
        "- Is it too mechanical or too busy?",
        "",
    ]
    for sample_id in sample_ids:
        lines.extend([f"### {sample_id}", "Bach-like:", "Cohesive form:", "Subject/answer:", "Counterpoint:", "Main failure:", "Notes:", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
