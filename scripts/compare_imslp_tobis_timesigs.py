#!/usr/bin/env python3
"""Compare IMSLP OMR vs Tobis MusicXML time signatures by BWV."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable

from music21 import converter, meter, stream

SUPPORTED_SUFFIXES = {".xml", ".musicxml", ".mxl"}
IMSLP_FROM_NAME_RE = re.compile(r"(IMSLP\d+)", re.IGNORECASE)
IMSLP_PDF_RE = re.compile(r"BWV(\d{4})__IMSLP(\d+)", re.IGNORECASE)
BWV_IN_PATH_RE = re.compile(r"BWV_(\d{4})", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare IMSLP OMR and Tobis MusicXML time signatures by BWV."
    )
    parser.add_argument(
        "--imslp-csv",
        default="data/imslp_musicxml/time_signatures_by_measure.csv",
        help="CSV from extract_time_signatures.py for IMSLP output.",
    )
    parser.add_argument(
        "--imslp-pdf-dir",
        default="data/imslp_pdfs",
        help="Directory containing BWVxxxx__IMSLPyyyyyy-...pdf files.",
    )
    parser.add_argument(
        "--tobis-dir",
        default="data/tobis_xml",
        help="Root directory of Tobis MusicXML files.",
    )
    parser.add_argument(
        "--out-csv",
        default="data/processed/imslp_vs_tobis_timesig_diff.csv",
        help="Output comparison CSV.",
    )
    parser.add_argument(
        "--bwv",
        action="append",
        default=[],
        help="Optional BWV filter, ex: BWV_0772 (can repeat).",
    )
    return parser.parse_args()


def normalize_bwv(text: str) -> str:
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        return ""
    return f"BWV_{int(digits):04d}"


def normalize_file_field(path_text: str) -> str:
    return path_text.replace("\\", "/")


def load_imslp_mapping(pdf_dir: Path) -> tuple[dict[str, set[str]], set[str]]:
    imslp_to_bwvs: DefaultDict[str, set[str]] = defaultdict(set)
    target_bwvs: set[str] = set()

    for pdf_path in pdf_dir.glob("*.pdf"):
        match = IMSLP_PDF_RE.search(pdf_path.name)
        if not match:
            continue
        bwv = f"BWV_{match.group(1)}"
        imslp_id = f"IMSLP{match.group(2)}"
        imslp_to_bwvs[imslp_id].add(bwv)
        target_bwvs.add(bwv)

    return dict(imslp_to_bwvs), target_bwvs


def load_imslp_from_csv(
    csv_path: Path,
    imslp_to_bwvs: dict[str, set[str]],
    allowed_bwvs: set[str],
) -> tuple[
    dict[str, set[tuple[str, str]]],
    dict[str, set[str]],
    dict[str, set[str]],
]:
    pairs_by_bwv: DefaultDict[str, set[tuple[str, str]]] = defaultdict(set)
    signatures_by_bwv: DefaultDict[str, set[str]] = defaultdict(set)
    files_by_bwv: DefaultDict[str, set[str]] = defaultdict(set)

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status", "").strip() != "OK":
                continue
            file_field = row.get("file", "").strip()
            if not file_field:
                continue
            base_name = Path(normalize_file_field(file_field)).name
            imslp_match = IMSLP_FROM_NAME_RE.search(base_name)
            if not imslp_match:
                continue
            imslp_id = imslp_match.group(1).upper()
            bwvs = imslp_to_bwvs.get(imslp_id, set())
            if not bwvs:
                continue

            measure = str(row.get("measure", "")).strip()
            signature = str(row.get("time_signature", "")).strip()
            if not signature:
                continue

            for bwv in bwvs:
                if bwv not in allowed_bwvs:
                    continue
                pairs_by_bwv[bwv].add((measure, signature))
                signatures_by_bwv[bwv].add(signature)
                files_by_bwv[bwv].add(base_name)

    return dict(pairs_by_bwv), dict(signatures_by_bwv), dict(files_by_bwv)


def iter_tobis_files(tobis_dir: Path, allowed_bwvs: set[str]) -> Iterable[tuple[str, Path]]:
    for file_path in sorted(tobis_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        match = BWV_IN_PATH_RE.search(file_path.as_posix())
        if not match:
            continue
        bwv = f"BWV_{match.group(1)}"
        if bwv in allowed_bwvs:
            yield bwv, file_path


def extract_pairs(path: Path) -> tuple[set[tuple[str, str]], set[str], str]:
    try:
        score = converter.parse(str(path))
    except Exception as exc:
        return set(), set(), f"{type(exc).__name__}: {exc}"

    pairs: set[tuple[str, str]] = set()
    signatures: set[str] = set()
    for part in score.parts:
        seen: set[tuple[str, str]] = set()
        for ts in part.recurse().getElementsByClass(meter.TimeSignature):
            measure_obj = ts.getContextByClass(stream.Measure)
            measure = str(measure_obj.number) if measure_obj is not None else ""
            signature = ts.ratioString
            key = (measure, signature)
            if key in seen:
                continue
            seen.add(key)
            pairs.add(key)
            signatures.add(signature)

    return pairs, signatures, ""


def load_tobis_data(
    tobis_dir: Path,
    allowed_bwvs: set[str],
) -> tuple[
    dict[str, set[tuple[str, str]]],
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, list[str]],
]:
    pairs_by_bwv: DefaultDict[str, set[tuple[str, str]]] = defaultdict(set)
    signatures_by_bwv: DefaultDict[str, set[str]] = defaultdict(set)
    files_by_bwv: DefaultDict[str, set[str]] = defaultdict(set)
    errors_by_bwv: DefaultDict[str, list[str]] = defaultdict(list)

    for bwv, file_path in iter_tobis_files(tobis_dir, allowed_bwvs):
        files_by_bwv[bwv].add(file_path.as_posix())
        pairs, signatures, err = extract_pairs(file_path)
        if err:
            errors_by_bwv[bwv].append(f"{file_path.as_posix()}: {err}")
            continue
        pairs_by_bwv[bwv].update(pairs)
        signatures_by_bwv[bwv].update(signatures)

    return (
        dict(pairs_by_bwv),
        dict(signatures_by_bwv),
        dict(files_by_bwv),
        dict(errors_by_bwv),
    )


def measure_sort_key(text: str) -> tuple[int, str]:
    try:
        return (0, f"{int(text):09d}")
    except Exception:
        return (1, text)


def format_pair_list(values: set[tuple[str, str]]) -> str:
    sorted_pairs = sorted(values, key=lambda x: (measure_sort_key(x[0]), x[1]))
    return "; ".join(f"{m if m else '?'}:{s}" for m, s in sorted_pairs)


def format_string_set(values: set[str]) -> str:
    return "; ".join(sorted(values))


def main() -> int:
    args = parse_args()

    imslp_csv = Path(args.imslp_csv)
    imslp_pdf_dir = Path(args.imslp_pdf_dir)
    tobis_dir = Path(args.tobis_dir)
    out_csv = Path(args.out_csv)

    if not imslp_csv.exists():
        print(f"Missing IMSLP CSV: {imslp_csv}")
        return 1
    if not imslp_pdf_dir.exists():
        print(f"Missing IMSLP PDF dir: {imslp_pdf_dir}")
        return 1
    if not tobis_dir.exists():
        print(f"Missing Tobis dir: {tobis_dir}")
        return 1

    imslp_to_bwvs, mapped_bwvs = load_imslp_mapping(imslp_pdf_dir)
    if not imslp_to_bwvs:
        print(f"No BWV->IMSLP mapping found under: {imslp_pdf_dir}")
        return 1

    selected_bwvs = {normalize_bwv(b) for b in args.bwv if normalize_bwv(b)}
    if selected_bwvs:
        allowed_bwvs = mapped_bwvs.intersection(selected_bwvs)
    else:
        allowed_bwvs = set(mapped_bwvs)

    imslp_pairs, imslp_signatures, imslp_files = load_imslp_from_csv(
        imslp_csv,
        imslp_to_bwvs,
        allowed_bwvs,
    )
    tobis_pairs, tobis_signatures, tobis_files, tobis_errors = load_tobis_data(
        tobis_dir,
        allowed_bwvs,
    )

    rows: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()
    all_bwvs = sorted(allowed_bwvs)

    for bwv in all_bwvs:
        ip = imslp_pairs.get(bwv, set())
        tp = tobis_pairs.get(bwv, set())
        isigs = imslp_signatures.get(bwv, set())
        tsigs = tobis_signatures.get(bwv, set())

        imslp_only_signatures = isigs - tsigs
        tobis_only_signatures = tsigs - isigs
        imslp_only_pairs = ip - tp
        tobis_only_pairs = tp - ip

        if not ip and not tp:
            status = "NO_DATA"
        elif not ip:
            status = "MISSING_IMSLP"
        elif not tp:
            status = "MISSING_TOBIS"
        elif ip == tp:
            status = "MATCH"
        elif isigs == tsigs:
            status = "MEASURE_DIFF"
        else:
            status = "SIGNATURE_DIFF"

        status_counts[status] += 1

        imslp_ids = sorted(
            {
                imslp_id
                for imslp_id, bwvs in imslp_to_bwvs.items()
                if bwv in bwvs
            }
        )

        rows.append(
            {
                "bwv": bwv,
                "status": status,
                "imslp_ids": "; ".join(imslp_ids),
                "imslp_files": "; ".join(sorted(imslp_files.get(bwv, set()))),
                "tobis_files": "; ".join(sorted(tobis_files.get(bwv, set()))),
                "imslp_signatures": format_string_set(isigs),
                "tobis_signatures": format_string_set(tsigs),
                "imslp_only_signatures": format_string_set(imslp_only_signatures),
                "tobis_only_signatures": format_string_set(tobis_only_signatures),
                "imslp_measure_signature_count": str(len(ip)),
                "tobis_measure_signature_count": str(len(tp)),
                "imslp_only_measure_signatures": format_pair_list(imslp_only_pairs),
                "tobis_only_measure_signatures": format_pair_list(tobis_only_pairs),
                "tobis_parse_errors": " | ".join(tobis_errors.get(bwv, [])),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "bwv",
            "status",
            "imslp_ids",
            "imslp_files",
            "tobis_files",
            "imslp_signatures",
            "tobis_signatures",
            "imslp_only_signatures",
            "tobis_only_signatures",
            "imslp_measure_signature_count",
            "tobis_measure_signature_count",
            "imslp_only_measure_signatures",
            "tobis_only_measure_signatures",
            "tobis_parse_errors",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Compared {len(rows)} BWV entries")
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
