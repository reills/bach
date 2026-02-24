#!/usr/bin/env python3
"""
Build an IMSLP download queue for time-signature suspicious Tobis files.

Inputs:
  - data-testing/time_sig_audit_files.csv
  - data/complete-works.html (IMSLP "List of works" page HTML)

Output:
  - data-testing/imslp_suspicious_queue.csv
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

IMSLP_BASE = "https://imslp.org"


def _find_work_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    for table in soup.find_all("table", class_="wikitable"):
        header = table.find("tr")
        if not header:
            continue
        ths = [th.get_text(" ", strip=True) for th in header.find_all("th")]
        if "BWV" in ths and "Title" in ths:
            return table
    return None


def _parse_bwv_from_href(href: str) -> Optional[int]:
    m = re.search(r"BWV[_\s-]*(\d{1,4})", href, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def _parse_range_from_href(href: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"BWV[_\s-]*(\d{1,4})\s*-\s*(\d{1,4})", href, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _normalize_link(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return f"{IMSLP_BASE}{href}"


def build_bwv_index(html_path: Path) -> Dict[int, List[Tuple[str, str]]]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    table = _find_work_table(soup)
    if table is None:
        raise RuntimeError("Could not find BWV work table in HTML.")

    header = table.find("tr")
    ths = [th.get_text(" ", strip=True) for th in header.find_all("th")]
    idx_bwv = ths.index("BWV")
    idx_title = ths.index("Title")

    bwv_to_links: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) <= max(idx_bwv, idx_title):
            continue

        link_tag = tds[idx_title].find("a", href=True)
        if not link_tag:
            continue

        href = link_tag["href"]
        title_text = link_tag.get_text(strip=True)
        rng = _parse_range_from_href(href)
        if rng is not None:
            lo, hi = rng
            for bwv in range(lo, hi + 1):
                bwv_to_links[bwv].append((href, title_text))
            continue

        bwv = _parse_bwv_from_href(href)
        if bwv is None:
            continue
        bwv_to_links[bwv].append((href, title_text))

    return bwv_to_links


def select_best_link(bwv: int, candidates: Iterable[Tuple[str, str]]) -> Tuple[Optional[str], str]:
    """
    Return the first candidate link for this BWV. Returns (link, match_type).
    match_type in {"exact", "range", "none"}.
    """
    for href, _title in candidates:
        if _parse_range_from_href(href) is not None:
            return _normalize_link(href), "range"
        return _normalize_link(href), "exact"
    return None, "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build IMSLP queue from suspicious time-sig audit.")
    parser.add_argument(
        "--audit-csv",
        default="data-testing/time_sig_audit_files.csv",
        help="Path to audit CSV",
    )
    parser.add_argument(
        "--works-html",
        default="data/complete-works.html",
        help="Local IMSLP List-of-works HTML",
    )
    parser.add_argument(
        "--out-csv",
        default="data-testing/imslp_suspicious_queue.csv",
        help="Output queue CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_csv = Path(args.audit_csv)
    works_html = Path(args.works_html)
    out_csv = Path(args.out_csv)

    if not audit_csv.exists():
        raise SystemExit(f"Missing audit CSV: {audit_csv}")
    if not works_html.exists():
        raise SystemExit(f"Missing works HTML: {works_html}")

    bwv_index = build_bwv_index(works_html)

    df = pd.read_csv(audit_csv)
    if "flag_suspicious" not in df.columns or "bwv" not in df.columns:
        raise SystemExit("Audit CSV missing required columns: 'flag_suspicious' and/or 'bwv'")

    suspicious = df[df["flag_suspicious"] == True].copy()
    rows = []
    for bwv_val, group in suspicious.groupby("bwv"):
        if pd.isna(bwv_val):
            continue
        try:
            bwv = int(bwv_val)
        except (TypeError, ValueError):
            continue

        candidates = bwv_index.get(bwv, [])
        best_link, match_type = select_best_link(bwv, candidates)
        alt_links = [
            _normalize_link(href) for href, _ in candidates if _normalize_link(href) != best_link
        ]
        paths = " | ".join(sorted(set(group["path"].dropna().astype(str).tolist())))

        rows.append(
            {
                "bwv": bwv,
                "paths": paths,
                "imslp_work_url": best_link or "",
                "imslp_match_type": match_type,
                "imslp_alternates": " | ".join(alt_links),
                "source_paths": len(group),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
