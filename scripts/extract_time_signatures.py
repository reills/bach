#!/usr/bin/env python3
"""Extract time signatures and measure numbers from MusicXML files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from music21 import converter, meter, stream

SUPPORTED_SUFFIXES = {".xml", ".musicxml", ".mxl"}


def iter_musicxml_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path
        return
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield file_path


def extract_rows(file_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        score = converter.parse(str(file_path))
    except Exception as exc:  # pragma: no cover - depends on input data
        return [
            {
                "file": str(file_path),
                "part_index": "",
                "part_name": "",
                "measure": "",
                "time_signature": "",
                "status": f"PARSE_ERROR: {type(exc).__name__}: {exc}",
            }
        ]

    for part_index, part in enumerate(score.parts, start=1):
        part_name = part.partName or part.id or ""
        seen: set[tuple[object, str]] = set()
        for ts in part.recurse().getElementsByClass(meter.TimeSignature):
            measure_obj = ts.getContextByClass(stream.Measure)
            measure_number = measure_obj.number if measure_obj is not None else ""
            signature = ts.ratioString
            key = (measure_number, signature)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "file": str(file_path),
                    "part_index": part_index,
                    "part_name": part_name,
                    "measure": measure_number,
                    "time_signature": signature,
                    "status": "OK",
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract time signatures and measure numbers from MusicXML."
    )
    parser.add_argument(
        "--xml-dir",
        required=True,
        help="Directory (or single file) containing .xml/.musicxml/.mxl files.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path. Defaults to <xml-dir>/time_signatures_by_measure.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml_dir = Path(args.xml_dir)
    if not xml_dir.exists():
        print(f"Input path does not exist: {xml_dir}")
        return 1

    out_csv = Path(args.out_csv) if args.out_csv else xml_dir / "time_signatures_by_measure.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    files = list(iter_musicxml_files(xml_dir))
    if not files:
        print(f"No MusicXML files found under: {xml_dir}")
        return 1

    rows: list[dict[str, object]] = []
    for file_path in files:
        rows.extend(extract_rows(file_path))

    fieldnames = ["file", "part_index", "part_name", "measure", "time_signature", "status"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(files)} files")
    print(f"Wrote {out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
