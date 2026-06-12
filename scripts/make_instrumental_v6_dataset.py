#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instrumental_v6.data import save_dataset
from src.instrumental_v6.metrics import evaluate_piece_rows
from src.instrumental_v6.representation import (
    DEFAULT_MAX_VOICES,
    InstrumentalV6Piece,
    parse_musicxml_movements,
)
from src.instrumental_v6.tokenize import build_tokenized_split, save_tokenized_split


@dataclass(frozen=True)
class SourceSpec:
    form: str
    voices: int | None
    path: Path


DEFAULT_CLEANED_ROOT = ROOT / "data/musicxml_cleaned/bach_counterpoint_v1/files"
DEFAULT_APPROVED_LIST = ROOT / "data/musicxml_cleaned/bach_counterpoint_v1/approved_files.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build factorized 2-6 voice instrumental_v6 dataset.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="FORM:VOICES:PATH, where VOICES is 2-6 or auto. May be repeated.",
    )
    parser.add_argument("--cleaned-root", default=str(DEFAULT_CLEANED_ROOT))
    parser.add_argument("--approved-list", default=str(DEFAULT_APPROVED_LIST))
    parser.add_argument("--output-dir", default="data/instrumental_v6/mixed_bach_v1")
    parser.add_argument("--max-voices", type=int, default=DEFAULT_MAX_VOICES)
    parser.add_argument("--max-bars", type=int, default=32)
    parser.add_argument("--limit-per-source", type=int, default=0)
    parser.add_argument(
        "--limit-movements-per-work",
        type=int,
        default=0,
        help="Keep at most this many accepted movements from each MusicXML file.",
    )
    parser.add_argument("--min-slices", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument(
        "--overfit-all",
        action="store_true",
        help="Use every accepted piece in both splits for the required tiny-overfit gate.",
    )
    parser.add_argument("--seed", type=int, default=2604)
    parser.add_argument("--no-normalize-key", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = (
        [_parse_source(value) for value in args.source]
        if args.source
        else _approved_source_specs(Path(args.cleaned_root), Path(args.approved_list))
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pieces: list[InstrumentalV6Piece] = []
    skipped: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    source_counts: dict[str, int] = {}
    for spec in specs:
        paths = _source_paths(spec.path)
        if args.limit_per_source > 0:
            paths = paths[: args.limit_per_source]
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                skipped.append({"path": str(path), "reason": "duplicate_sha256"})
                continue
            seen_hashes.add(digest)
            try:
                parsed = parse_musicxml_movements(
                    path,
                    form=spec.form,
                    target_voices=spec.voices,
                    max_voices=args.max_voices,
                    max_bars=args.max_bars,
                    normalize_key=not args.no_normalize_key,
                    max_movements=args.limit_movements_per_work,
                )
            except Exception as exc:
                skipped.append({"path": str(path), "reason": "parse_error", "error": str(exc)})
                continue
            accepted = 0
            for piece in parsed:
                if args.limit_movements_per_work > 0 and accepted >= args.limit_movements_per_work:
                    skipped.append(
                        {
                            "path": str(path),
                            "piece_id": piece.piece_id,
                            "reason": "movement_limit",
                        }
                    )
                    continue
                if len(piece.global_rows) < args.min_slices:
                    skipped.append(
                        {
                            "path": str(path),
                            "piece_id": piece.piece_id,
                            "reason": "too_short",
                            "slices": len(piece.global_rows),
                        }
                    )
                    continue
                report = evaluate_piece_rows(
                    piece.global_rows,
                    piece.voice_rows,
                    piece.pair_rows,
                    voice_count=piece.voice_count,
                )
                if float(report["invalid_pitch_state_rate"]) > 0.0:
                    skipped.append(
                        {
                            "path": str(path),
                            "piece_id": piece.piece_id,
                            "reason": "invalid_pitch_state",
                        }
                    )
                    continue
                pieces.append(piece)
                accepted += 1
                print(
                    f"ok {piece.piece_id}: form={piece.form} voices={piece.voice_count} "
                    f"meter={piece.time_signature} slices={len(piece.global_rows)}"
                )
            source_counts[spec.form] = source_counts.get(spec.form, 0) + accepted
    if not pieces:
        raise SystemExit("no v6 pieces were accepted")

    if args.overfit_all:
        train_ids = val_ids = {piece.piece_id for piece in pieces}
    else:
        split_by_work = _split_by_work(pieces, val_split=args.val_split, seed=args.seed)
        train_ids = {piece.piece_id for piece in pieces if split_by_work[_work_id(piece)] == "train"}
        val_ids = {piece.piece_id for piece in pieces if split_by_work[_work_id(piece)] == "val"}
        if not val_ids:
            val_ids = {sorted(train_ids)[-1]}
            train_ids -= val_ids

    dataset_path = output_dir / "pieces.json"
    metadata = {
        "max_voices": args.max_voices,
        "piece_count": len(pieces),
        "voice_count_distribution": _count(piece.voice_count for piece in pieces),
        "form_distribution": _count(piece.form for piece in pieces),
        "source_counts": source_counts,
        "train_piece_ids": sorted(train_ids),
        "val_piece_ids": sorted(val_ids),
        "skipped": skipped,
        "normalize_key": not args.no_normalize_key,
        "max_bars": args.max_bars,
        "limit_movements_per_work": args.limit_movements_per_work,
        "overfit_all": args.overfit_all,
        "seed": args.seed,
    }
    save_dataset(dataset_path, pieces, metadata=metadata)

    train = build_tokenized_split(
        pieces,
        piece_ids=train_ids,
        seq_len=args.seq_len,
        stride=args.stride,
    )
    val = build_tokenized_split(
        pieces,
        piece_ids=val_ids,
        seq_len=args.seq_len,
        stride=args.stride,
    )
    tokenized_dir = output_dir / "tokenized"
    save_tokenized_split(tokenized_dir / "train.pt", train, max_voices=args.max_voices)
    save_tokenized_split(tokenized_dir / "val.pt", val, max_voices=args.max_voices)
    summary = {
        **metadata,
        "dataset_path": str(dataset_path),
        "train_windows": int(train.global_values.shape[0]),
        "val_windows": int(val.global_values.shape[0]),
        "seq_len": args.seq_len,
        "stride": args.stride,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parse_source(value: str) -> SourceSpec:
    form, voices, path = value.split(":", 2)
    voice_value = None if voices.strip().lower() == "auto" else int(voices)
    return SourceSpec(form.strip(), voice_value, Path(path))


def _approved_source_specs(cleaned_root: Path, approved_list: Path) -> list[SourceSpec]:
    if not approved_list.exists():
        raise SystemExit(f"approved MusicXML list not found: {approved_list}")
    specs: list[SourceSpec] = []
    for line in approved_list.read_text(encoding="utf-8").splitlines():
        relative = line.strip()
        if not relative:
            continue
        path = cleaned_root / relative
        if path.exists():
            specs.append(
                SourceSpec(
                    _form_for_path(relative),
                    _voice_prior_for_path(relative),
                    path,
                )
            )
    if not specs:
        raise SystemExit("approved MusicXML list did not resolve to any files")
    return specs


def _form_for_path(path: str) -> str:
    normalized = path.lower()
    if "inventions" in normalized:
        return "invention"
    if "sinfonias" in normalized or "trio sonatas" in normalized:
        return "sinfonia"
    if any(name in normalized for name in ("suite", "partita", "variations")):
        return "partita"
    if "well-tempered" in normalized or "preludes and fugues" in normalized:
        return "wtc"
    if "canon" in normalized:
        return "canon"
    if "fugue" in normalized or "art of fugue" in normalized or "musical offering" in normalized:
        return "fugue"
    if "prelude" in normalized:
        return "prelude"
    if "toccata" in normalized or "fantasia" in normalized:
        return "toccata"
    return "keyboard"


def _voice_prior_for_path(path: str) -> int | None:
    normalized = path.lower()
    if "inventions" in normalized or "four duets" in normalized:
        return 2
    if "sinfonias" in normalized or "trio sonatas" in normalized:
        return 3
    if "musical offering/bwv_1079_01/" in normalized:
        return 3
    if "musical offering/bwv_1079_02/" in normalized:
        return 6
    if "musical offering/bwv_1079_03_" in normalized:
        return 3
    return None


def _source_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted({*path.glob("**/*.xml"), *path.glob("**/*.musicxml")})


def _split_by_work(
    pieces: list[InstrumentalV6Piece],
    *,
    val_split: float,
    seed: int,
) -> dict[str, str]:
    works = sorted({_work_id(piece) for piece in pieces})
    shuffled = works[:]
    random.Random(seed).shuffle(shuffled)
    val_count = min(len(works) - 1, max(1, round(len(works) * val_split))) if len(works) > 1 else 0
    val = set(shuffled[:val_count])
    return {work: ("val" if work in val else "train") for work in works}


def _work_id(piece: InstrumentalV6Piece) -> str:
    return str(Path(piece.source_path).resolve())


def _count(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:  # type: ignore[union-attr]
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
