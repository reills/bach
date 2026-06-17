#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataio.musicxml_cleaner import (
    clean_musicxml_file,
    load_meter_overrides,
    sha256_file,
)


DEFAULT_SOURCES = [
    "instrumental-works/keyboard-works",
    "instrumental-works/Art of fugue",
    "instrumental-works/Canons",
    "instrumental-works/Musical offering",
    "instrumental-works/organ-works/BWV 525-530 Trio Sonatas",
    "instrumental-works/organ-works/BWV 531-552 Preludes and Fugues",
    "instrumental-works/organ-works/BWV 553-560 Eight Short Preludes and Fugues",
    "instrumental-works/organ-works/BWV 561-563 Fantasias and Fugues",
    "instrumental-works/organ-works/BWV 564-566 Toccatas and Fugues",
    "instrumental-works/organ-works/BWV 574-581 Fugues",
    "instrumental-works/organ-works/BWV 582 Passacaglia and Fugue",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a provenance-tracked cleaned MusicXML mirror for training."
    )
    parser.add_argument("--input-root", default="data/tobis_xml")
    parser.add_argument("--output-dir", default="data/musicxml_cleaned/bach_counterpoint_v1")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="File or directory relative to input-root. May be repeated.",
    )
    parser.add_argument("--overrides", default="configs/musicxml_meter_overrides.json")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-auto-run", type=int, default=2)
    parser.add_argument("--dominant-support", type=float, default=0.7)
    parser.add_argument("--no-auto-repair", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir)
    files_dir = output_dir / "files"
    paths = _source_paths(input_root, args.source or DEFAULT_SOURCES)
    if args.max_files > 0:
        paths = paths[: args.max_files]
    overrides = load_meter_overrides(args.overrides)
    if not args.dry_run:
        files_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, object]] = []
    duplicates: list[dict[str, str]] = []
    seen_hashes: dict[str, str] = {}
    for index, source_path in enumerate(paths, start=1):
        relative = source_path.relative_to(input_root).as_posix()
        digest = sha256_file(source_path)
        if digest in seen_hashes:
            duplicates.append(
                {
                    "path": relative,
                    "duplicate_of": seen_hashes[digest],
                    "sha256": digest,
                }
            )
            continue
        seen_hashes[digest] = relative
        destination = None if args.dry_run else files_dir / relative
        report = clean_musicxml_file(
            source_path,
            destination,
            relative_path=relative,
            overrides=overrides,
            auto_repair=not args.no_auto_repair,
            max_auto_run=args.max_auto_run,
            dominant_support=args.dominant_support,
        )
        reports.append(report.to_dict())
        print(
            f"[{index}/{len(paths)}] {report.status} {relative} "
            f"movements={report.movement_count} changes={len(report.changes)} "
            f"issues={len(report.issues)}",
            flush=True,
        )

    status_counts = Counter(str(report["status"]) for report in reports)
    approved = [
        str(Path(str(report["output_path"])).relative_to(files_dir))
        for report in reports
        if report["training_approved"] and report["output_path"] is not None
    ]
    summary = {
        "input_root": str(input_root),
        "files_dir": str(files_dir),
        "source_count": len(paths),
        "unique_count": len(reports),
        "duplicate_count": len(duplicates),
        "status_counts": dict(sorted(status_counts.items())),
        "training_approved_count": len(approved),
        "review_required_count": status_counts.get("review_required", 0),
        "auto_repair": not args.no_auto_repair,
        "max_auto_run": args.max_auto_run,
        "dominant_support": args.dominant_support,
        "overrides_path": args.overrides,
    }
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "manifest.jsonl", reports)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "duplicates.json").write_text(
            json.dumps(duplicates, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_dir / "approved_files.txt").write_text(
            "".join(f"{path}\n" for path in approved),
            encoding="utf-8",
        )
        review = [report for report in reports if not bool(report["training_approved"])]
        (output_dir / "review_queue.json").write_text(
            json.dumps(review, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        override_source = Path(args.overrides)
        if override_source.exists():
            shutil.copy2(override_source, output_dir / "meter_overrides.snapshot.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _source_paths(input_root: Path, sources: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for value in sources:
        source = Path(value)
        if not source.is_absolute():
            source = input_root / source
        if source.is_file():
            paths.add(source.resolve())
            continue
        paths.update(path.resolve() for path in source.glob("**/*.xml"))
        paths.update(path.resolve() for path in source.glob("**/*.musicxml"))
    return sorted(paths)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
