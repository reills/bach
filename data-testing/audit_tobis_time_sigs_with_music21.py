#!/usr/bin/env python3
"""
Audit Tobis MusicXML files for noisy/spurious time-signature metadata.

What this script checks:
1) Equivalent meter flips in explicit <time> events (e.g. 4/4 -> 8/8).
2) Isolated one-measure equivalent-meter blips (A -> B -> A).
3) Optional comparison to music21 corpus "golden" Bach scores by BWV.
4) XML-level time signature usage across all parts.
5) Tuplet density (3:2 triplets vs total notes) and derived compound-feel tag.

Outputs:
- CSV with one row per file
- JSON summary

Example:
  python data-testing/audit_tobis_time_sigs_with_music21.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from music21 import converter, corpus, meter, stream  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "music21 is required. Install with: pip install music21\n"
        f"Import error: {exc}"
    )


Sig = Tuple[int, int]
SeqItem = Tuple[int, Optional[Sig]]  # (measure_number, active_time_signature)
EventItem = Tuple[int, int, Sig]  # (measure_index, measure_number, explicit_time_signature)


def sig_to_str(sig: Optional[Sig]) -> str:
    if sig is None:
        return "NA"
    return f"{sig[0]}/{sig[1]}"


def _ratio_counter_to_str(counter: Counter) -> str:
    if not counter:
        return ""
    return "|".join(f"{key}={count}" for key, count in counter.most_common())


def _parse_ratio_counts(raw: str) -> Counter:
    counter: Counter = Counter()
    if not raw:
        return counter
    for chunk in raw.split("|"):
        if not chunk or "=" not in chunk:
            continue
        key, val = chunk.split("=", 1)
        try:
            counter[key] += int(val)
        except ValueError:
            continue
    return counter


def parse_xml_metrics(xml_path: Path) -> Dict[str, object]:
    result: Dict[str, object] = {
        "xml_parse_ok": False,
        "xml_parse_error": "",
        "xml_parts": 0,
        "xml_time_sig_events": 0,
        "xml_unique_sigs": "NA",
        "xml_total_notes": 0,
        "xml_time_mod_notes": 0,
        "xml_time_mod_ratio": 0.0,
        "xml_triplet_3_2_notes": 0,
        "xml_triplet_3_2_ratio": 0.0,
        "xml_triplet_3_2_of_time_mod": 0.0,
        "xml_tuplet_ratio_counts": "",
    }

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as exc:
        result["xml_parse_error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["xml_parse_ok"] = True
    result["xml_parts"] = len(root.findall(".//part"))

    time_sig_counts: Counter = Counter()
    for time_elem in root.findall(".//time"):
        beats = time_elem.find("beats")
        beat_type = time_elem.find("beat-type")
        if beats is None or beat_type is None:
            continue
        try:
            sig = (int(beats.text), int(beat_type.text))
        except (TypeError, ValueError):
            continue
        time_sig_counts[sig] += 1

    if time_sig_counts:
        unique_sigs = sorted(time_sig_counts)
        result["xml_unique_sigs"] = "|".join(sig_to_str(sig) for sig in unique_sigs)
        result["xml_time_sig_events"] = sum(time_sig_counts.values())

    total_notes = 0
    time_mod_notes = 0
    triplet_3_2_notes = 0
    ratio_counts: Counter = Counter()

    for note in root.findall(".//note"):
        if note.find("grace") is not None:
            continue
        if note.find("pitch") is None:
            continue

        total_notes += 1
        time_mod = note.find("time-modification")
        if time_mod is None:
            continue
        time_mod_notes += 1

        actual = time_mod.find("actual-notes")
        normal = time_mod.find("normal-notes")
        if actual is None or normal is None:
            continue
        try:
            actual_val = int(actual.text)
            normal_val = int(normal.text)
        except (TypeError, ValueError):
            continue

        ratio_key = f"{actual_val}:{normal_val}"
        ratio_counts[ratio_key] += 1
        if actual_val == 3 and normal_val == 2:
            triplet_3_2_notes += 1

    result["xml_total_notes"] = total_notes
    result["xml_time_mod_notes"] = time_mod_notes
    result["xml_triplet_3_2_notes"] = triplet_3_2_notes
    if total_notes:
        result["xml_time_mod_ratio"] = round(time_mod_notes / total_notes, 6)
        result["xml_triplet_3_2_ratio"] = round(triplet_3_2_notes / total_notes, 6)
    if time_mod_notes:
        result["xml_triplet_3_2_of_time_mod"] = round(triplet_3_2_notes / time_mod_notes, 6)
    result["xml_tuplet_ratio_counts"] = _ratio_counter_to_str(ratio_counts)
    return result


def equivalent_meters(a: Optional[Sig], b: Optional[Sig]) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return False
    dur_a = Fraction(a[0] * 4, a[1])
    dur_b = Fraction(b[0] * 4, b[1])
    return dur_a == dur_b


def extract_bwv(path: Path) -> Optional[int]:
    # Prefer the filename stem (e.g. BWV_0001_1.xml -> 1), which is usually
    # the real work id. Folder names like "BWV 001-020" are just range buckets.
    stem_match = re.search(r"BWV[_\s-]*(\d{1,4})(?!\d)", path.stem, flags=re.IGNORECASE)
    if stem_match:
        return int(stem_match.group(1))

    # Fallback: search full path and take the last BWV-like occurrence.
    matches = re.findall(r"BWV[_\s-]*(\d{1,4})(?!\d)", str(path), flags=re.IGNORECASE)
    if matches:
        return int(matches[-1])
    return None


def is_chorale_path(path: Path) -> bool:
    return "chorale" in str(path).lower()


def find_xml_files(dirs: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for base in dirs:
        if not base.exists():
            continue
        if base.is_file():
            if base.suffix.lower() == ".xml":
                files.append(base)
            continue
        files.extend(sorted(base.rglob("*.xml")))
    return files


def pick_reference_part(score_obj: stream.Score) -> Optional[stream.Part]:
    parts = list(score_obj.parts)
    if not parts:
        return None
    # Use the part with most measures to reduce missing-measure edge cases.
    return max(parts, key=lambda p: len(list(p.getElementsByClass(stream.Measure))))


def collect_active_sequence(part: stream.Part) -> List[SeqItem]:
    measures = list(part.getElementsByClass(stream.Measure))
    active: List[Tuple[int, Optional[Sig]]] = []
    current: Optional[Sig] = None

    for idx, m in enumerate(measures):
        num = m.number if m.number is not None else idx + 1
        ts = m.timeSignature
        if ts is not None:
            current = (int(ts.numerator), int(ts.denominator))
        active.append((int(num), current))
    return active


def collect_explicit_events(part: stream.Part) -> List[EventItem]:
    measures = list(part.getElementsByClass(stream.Measure))
    events: List[EventItem] = []
    for idx, m in enumerate(measures):
        num = m.number if m.number is not None else idx + 1
        for ts in m.getElementsByClass(meter.TimeSignature):
            sig: Sig = (int(ts.numerator), int(ts.denominator))
            events.append((idx, int(num), sig))
            break
    return events


def detect_equivalent_event_changes(events: Sequence[EventItem]) -> List[Tuple[int, Sig, Sig]]:
    changes: List[Tuple[int, Sig, Sig]] = []
    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]
        if equivalent_meters(prev[2], curr[2]):
            changes.append((curr[1], prev[2], curr[2]))
    return changes


def detect_isolated_equivalent_blips(active_seq: Sequence[SeqItem]) -> List[Tuple[int, Sig, Sig]]:
    blips: List[Tuple[int, Sig, Sig]] = []
    for i in range(1, len(active_seq) - 1):
        prev_sig = active_seq[i - 1][1]
        curr_sig = active_seq[i][1]
        next_sig = active_seq[i + 1][1]
        if (
            curr_sig is not None
            and prev_sig is not None
            and next_sig is not None
            and curr_sig != prev_sig
            and curr_sig != next_sig
            and prev_sig == next_sig
            and equivalent_meters(curr_sig, prev_sig)
        ):
            blips.append((active_seq[i][0], prev_sig, curr_sig))
    return blips


def dominant_sig(active_seq: Sequence[SeqItem]) -> Optional[Sig]:
    counter = Counter(sig for _, sig in active_seq if sig is not None)
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def unique_sigs(active_seq: Sequence[SeqItem]) -> List[Sig]:
    sigs = sorted({sig for _, sig in active_seq if sig is not None})
    return list(sigs)


def compare_sequences(
    source_seq: Sequence[SeqItem], trusted_seq: Sequence[SeqItem]
) -> Tuple[int, int]:
    mismatches = 0
    equiv_mismatches = 0
    n = min(len(source_seq), len(trusted_seq))
    for i in range(n):
        s = source_seq[i][1]
        t = trusted_seq[i][1]
        if s is None or t is None:
            continue
        if s != t:
            mismatches += 1
            if equivalent_meters(s, t):
                equiv_mismatches += 1
    return mismatches, equiv_mismatches


def get_trusted_data(
    bwv: int, trusted_cache: Dict[int, Dict[str, object]]
) -> Dict[str, object]:
    if bwv in trusted_cache:
        return trusted_cache[bwv]

    try:
        score = corpus.parse(f"bach/bwv{bwv}")
        part = pick_reference_part(score)
        if part is None:
            result = {
                "available": False,
                "reason": "no_parts",
            }
        else:
            seq = collect_active_sequence(part)
            result = {
                "available": True,
                "active_seq": seq,
                "first_sig": seq[0][1] if seq else None,
                "dominant_sig": dominant_sig(seq),
            }
    except Exception as exc:
        result = {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    trusted_cache[bwv] = result
    return result


def should_compare_to_trusted(path: Path, scope: str) -> bool:
    if scope == "none":
        return False
    if scope == "all":
        return True
    return is_chorale_path(path)


def analyze_file(
    xml_path: Path,
    trusted_scope: str,
    trusted_cache: Dict[int, Dict[str, object]],
    compound_triplet_threshold: float,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "path": str(xml_path),
        "bwv": extract_bwv(xml_path),
        "parse_ok": False,
        "parse_error": "",
        "measures": 0,
        "unique_sigs": "",
        "dominant_sig": "NA",
        "explicit_time_events": 0,
        "equivalent_event_changes": 0,
        "isolated_equivalent_blips": 0,
        "flag_suspicious": False,
        "trusted_checked": False,
        "trusted_available": False,
        "trusted_first_sig": "NA",
        "source_first_sig": "NA",
        "trusted_first_mismatch": False,
        "trusted_first_equiv_mismatch": False,
        "trusted_measure_mismatches": 0,
        "trusted_measure_equiv_mismatches": 0,
        "compound_feel": False,
    }

    xml_metrics = parse_xml_metrics(xml_path)
    row.update(xml_metrics)
    if row.get("xml_parse_ok"):
        triplet_ratio = float(row.get("xml_triplet_3_2_ratio", 0.0))
        row["compound_feel"] = triplet_ratio >= compound_triplet_threshold

    try:
        score = converter.parse(str(xml_path))
    except Exception as exc:
        row["parse_error"] = f"{type(exc).__name__}: {exc}"
        return row

    part = pick_reference_part(score)
    if part is None:
        row["parse_error"] = "No parts found in score"
        return row

    active_seq = collect_active_sequence(part)
    events = collect_explicit_events(part)
    equiv_changes = detect_equivalent_event_changes(events)
    isolated_blips = detect_isolated_equivalent_blips(active_seq)
    dom = dominant_sig(active_seq)
    uniq = unique_sigs(active_seq)

    row["parse_ok"] = True
    row["measures"] = len(active_seq)
    row["unique_sigs"] = "|".join(sig_to_str(s) for s in uniq)
    row["dominant_sig"] = sig_to_str(dom)
    row["explicit_time_events"] = len(events)
    row["equivalent_event_changes"] = len(equiv_changes)
    row["isolated_equivalent_blips"] = len(isolated_blips)
    row["flag_suspicious"] = bool(equiv_changes or isolated_blips)
    row["source_first_sig"] = sig_to_str(active_seq[0][1]) if active_seq else "NA"

    bwv = row["bwv"]
    if isinstance(bwv, int) and should_compare_to_trusted(xml_path, trusted_scope):
        row["trusted_checked"] = True
        trusted = get_trusted_data(bwv, trusted_cache)
        if bool(trusted.get("available")):
            row["trusted_available"] = True
            trusted_first = trusted.get("first_sig")
            row["trusted_first_sig"] = sig_to_str(trusted_first if isinstance(trusted_first, tuple) else None)

            source_first = active_seq[0][1] if active_seq else None
            if isinstance(trusted_first, tuple) and source_first is not None and source_first != trusted_first:
                row["trusted_first_mismatch"] = True
                row["trusted_first_equiv_mismatch"] = equivalent_meters(source_first, trusted_first)

            trusted_seq = trusted.get("active_seq")
            if isinstance(trusted_seq, list):
                mm, emm = compare_sequences(active_seq, trusted_seq)
                row["trusted_measure_mismatches"] = mm
                row["trusted_measure_equiv_mismatches"] = emm

    return row


def pct(n: int, d: int) -> str:
    if d == 0:
        return "0.0%"
    return f"{(100.0 * n / d):.1f}%"


def write_csv(rows: Sequence[Dict[str, object]], out_csv: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Dict[str, object]], compound_triplet_threshold: float) -> Dict[str, object]:
    total = len(rows)
    parse_ok = [r for r in rows if r["parse_ok"]]
    parse_errors = total - len(parse_ok)

    xml_ok = [r for r in rows if r.get("xml_parse_ok")]
    xml_parse_errors = total - len(xml_ok)
    total_notes = sum(int(r.get("xml_total_notes", 0)) for r in xml_ok)
    total_time_mod = sum(int(r.get("xml_time_mod_notes", 0)) for r in xml_ok)
    total_triplet_3_2 = sum(int(r.get("xml_triplet_3_2_notes", 0)) for r in xml_ok)
    files_with_triplets = [r for r in xml_ok if int(r.get("xml_triplet_3_2_notes", 0)) > 0]
    files_with_compound = [r for r in xml_ok if r.get("compound_feel")]

    tuplet_ratio_counter: Counter = Counter()
    for r in xml_ok:
        raw = str(r.get("xml_tuplet_ratio_counts", ""))
        tuplet_ratio_counter.update(_parse_ratio_counts(raw))

    suspicious = [r for r in parse_ok if r["flag_suspicious"]]
    with_equiv_changes = [r for r in parse_ok if int(r["equivalent_event_changes"]) > 0]
    with_blips = [r for r in parse_ok if int(r["isolated_equivalent_blips"]) > 0]
    with_multi_unique = [r for r in parse_ok if len(str(r["unique_sigs"]).split("|")) > 1]

    trusted_checked = [r for r in parse_ok if r["trusted_checked"]]
    trusted_available = [r for r in trusted_checked if r["trusted_available"]]
    trusted_first_mismatch = [r for r in trusted_available if r["trusted_first_mismatch"]]
    trusted_first_equiv = [r for r in trusted_available if r["trusted_first_equiv_mismatch"]]
    trusted_measure_mismatch = [r for r in trusted_available if int(r["trusted_measure_mismatches"]) > 0]
    trusted_measure_equiv = [
        r for r in trusted_available if int(r["trusted_measure_equiv_mismatches"]) > 0
    ]

    dominant_counter = Counter(str(r["dominant_sig"]) for r in parse_ok if r["dominant_sig"] != "NA")

    summary: Dict[str, object] = {
        "total_files": total,
        "parse_ok": len(parse_ok),
        "parse_errors": parse_errors,
        "xml_parse_ok": len(xml_ok),
        "xml_parse_errors": xml_parse_errors,
        "xml_total_notes": total_notes,
        "xml_time_mod_notes": total_time_mod,
        "xml_triplet_3_2_notes": total_triplet_3_2,
        "xml_triplet_3_2_ratio_overall": round(total_triplet_3_2 / total_notes, 6)
        if total_notes
        else 0.0,
        "xml_time_mod_ratio_overall": round(total_time_mod / total_notes, 6)
        if total_notes
        else 0.0,
        "files_with_triplet_3_2": len(files_with_triplets),
        "files_with_compound_feel": len(files_with_compound),
        "compound_triplet_threshold": compound_triplet_threshold,
        "xml_tuplet_ratio_distribution": dict(tuplet_ratio_counter.most_common()),
        "flag_suspicious": len(suspicious),
        "flag_suspicious_pct": pct(len(suspicious), len(parse_ok)),
        "with_equivalent_event_changes": len(with_equiv_changes),
        "with_isolated_equivalent_blips": len(with_blips),
        "with_multiple_unique_signatures": len(with_multi_unique),
        "trusted_checked": len(trusted_checked),
        "trusted_available": len(trusted_available),
        "trusted_first_mismatch": len(trusted_first_mismatch),
        "trusted_first_equivalent_mismatch": len(trusted_first_equiv),
        "trusted_measure_mismatch_any": len(trusted_measure_mismatch),
        "trusted_measure_equivalent_mismatch_any": len(trusted_measure_equiv),
        "dominant_signature_distribution": dict(dominant_counter.most_common()),
    }
    return summary


def print_console_summary(summary: Dict[str, object], out_csv: Path, out_json: Path) -> None:
    print("=== Time Signature Audit Summary ===")
    print(f"Total files: {summary['total_files']}")
    print(f"Parsed OK: {summary['parse_ok']}")
    print(f"Parse errors: {summary['parse_errors']}")
    print(f"XML parsed OK: {summary['xml_parse_ok']}")
    print(f"XML parse errors: {summary['xml_parse_errors']}")
    print(
        "Triplet 3:2 notes (overall): "
        f"{summary['xml_triplet_3_2_notes']} / {summary['xml_total_notes']} "
        f"({summary['xml_triplet_3_2_ratio_overall']})"
    )
    print(f"Compound-feel triplet threshold: {summary['compound_triplet_threshold']}")
    print(f"Files with any 3:2 triplets: {summary['files_with_triplet_3_2']}")
    print(f"Files tagged compound feel: {summary['files_with_compound_feel']}")
    print(
        "Flagged suspicious (equivalent flip and/or isolated equivalent blip): "
        f"{summary['flag_suspicious']} ({summary['flag_suspicious_pct']})"
    )
    print(f"Files with equivalent event changes: {summary['with_equivalent_event_changes']}")
    print(f"Files with isolated equivalent blips: {summary['with_isolated_equivalent_blips']}")
    print(f"Files with multiple unique signatures: {summary['with_multiple_unique_signatures']}")
    print(f"Trusted checked: {summary['trusted_checked']}")
    print(f"Trusted available: {summary['trusted_available']}")
    print(f"Trusted first-measure mismatches: {summary['trusted_first_mismatch']}")
    print(
        "Trusted first-measure equivalent mismatches: "
        f"{summary['trusted_first_equivalent_mismatch']}"
    )
    print(f"Trusted any-measure mismatches: {summary['trusted_measure_mismatch_any']}")
    print(
        "Trusted any-measure equivalent mismatches: "
        f"{summary['trusted_measure_equivalent_mismatch_any']}"
    )
    print(f"CSV report: {out_csv}")
    print(f"JSON summary: {out_json}")


def default_dirs(project_root: Path) -> List[Path]:
    return [
        project_root / "data/tobis_xml",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Tobis MusicXML for spurious/equivalent time-signature noise."
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=None,
        help="Directories or .xml files to scan. Defaults to Cantatas + chorales.",
    )
    parser.add_argument(
        "--trusted-scope",
        choices=["none", "chorales", "all"],
        default="chorales",
        help="Where to attempt music21 corpus BWV comparison.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick testing (first N files after sorting).",
    )
    parser.add_argument(
        "--out-prefix",
        default="data-testing/time_sig_audit",
        help="Output prefix. Produces <prefix>_files.csv and <prefix>_summary.json.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N files.",
    )
    parser.add_argument(
        "--compound-triplet-threshold",
        type=float,
        default=0.25,
        help=(
            "Triplet 3:2 ratio (triplet notes / total notes) to tag compound_feel."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    scan_dirs = [Path(d) for d in args.dirs] if args.dirs else default_dirs(project_root)
    scan_dirs = [d if d.is_absolute() else (project_root / d) for d in scan_dirs]

    files = find_xml_files(scan_dirs)
    if args.limit is not None:
        files = files[: max(0, args.limit)]

    print(f"Scanning {len(files)} XML files...")
    trusted_cache: Dict[int, Dict[str, object]] = {}
    rows: List[Dict[str, object]] = []

    for i, path in enumerate(files, start=1):
        rows.append(
            analyze_file(
                path,
                args.trusted_scope,
                trusted_cache,
                args.compound_triplet_threshold,
            )
        )
        if args.progress_every > 0 and i % args.progress_every == 0:
            print(f"  processed {i}/{len(files)}")

    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = project_root / out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_csv = out_prefix.with_name(out_prefix.name + "_files.csv")
    out_json = out_prefix.with_name(out_prefix.name + "_summary.json")

    write_csv(rows, out_csv)
    summary = summarize(rows, args.compound_triplet_threshold)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_console_summary(summary, out_csv, out_json)


if __name__ == "__main__":
    main()
