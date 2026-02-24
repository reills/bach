#!/usr/bin/env python3
"""
Download the first PDF from each IMSLP work page in the queue.

Queue CSV columns:
  - bwv
  - imslp_work_url

Output: PDFs saved into --out-dir (default: data/imslp_pdfs)
"""

from __future__ import annotations

import argparse
import gzip
import http.cookiejar as cookiejar
import math
import re
import time
from html import unescape
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

import pandas as pd
from bs4 import BeautifulSoup

IMSLP_BASE = "https://imslp.org"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
_OPENER = None
_COOKIE_JAR: Optional[cookiejar.CookieJar] = None


def _open(req: Request):
    if _OPENER is not None:
        return _OPENER.open(req)
    return urlopen(req)


def _read_response(resp) -> bytes:
    data = resp.read()
    encoding = (resp.headers.get("Content-Encoding") or "").lower()
    if "gzip" in encoding:
        try:
            data = gzip.decompress(data)
        except OSError:
            pass
    return data


def _fetch_text(url: str) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with _open(req) as resp:  # nosec - user provided URL list
        return _read_response(resp).decode("utf-8", errors="ignore")


def _fetch_url_state(url: str):
    req = Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
        },
    )
    with _open(req) as resp:  # nosec - user provided URL list
        data = _read_response(resp)
        final_url = resp.geturl()
        ctype = (resp.headers.get("Content-Type") or "").lower()

    is_pdf = (
        data.startswith(b"%PDF-")
        or "application/pdf" in ctype
        or final_url.lower().endswith(".pdf")
    )
    return final_url, data.decode("utf-8", errors="ignore"), is_pdf


def _fetch_html(url: str) -> BeautifulSoup:
    return BeautifulSoup(_fetch_text(url), "html.parser")


def _normalize_url(href: str, base: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return urljoin(base, href)


def _split_pipe_urls(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []

    text = str(value).strip()
    if not text:
        return []
    if text.lower() in {"nan", "none", "null"}:
        return []

    urls = []
    for part in text.split("|"):
        part = part.strip()
        if not part:
            continue
        if part.lower() in {"nan", "none", "null"}:
            continue
        urls.append(part)
    return urls


def _is_range_work_url(url: str) -> bool:
    return bool(re.search(r"BWV[_ ]\d{1,4}-\d{1,4}", unquote(url), re.I))


def _extract_bwv_specific_work_urls(work_soup: BeautifulSoup, bwv: int, base_url: str) -> list[str]:
    candidates = []
    seen = set()
    base_parsed = urlparse(base_url)
    base_path = unquote(base_parsed.path or "").lower()
    bwv_pat = re.compile(rf"bwv_0*{bwv}(?:[^0-9]|$)", re.I)

    for a in work_soup.find_all("a", href=True):
        url = _normalize_url(a["href"], base_url)
        parsed = urlparse(url)
        path = unquote(parsed.path or "").lower()
        if parsed.query or parsed.fragment:
            continue
        if not path.startswith("/wiki/"):
            continue
        if "/wiki/special:" in path or "/wiki/file:" in path:
            continue
        if path == base_path:
            continue
        if not bwv_pat.search(path):
            continue
        if url in seen:
            continue
        seen.add(url)
        candidates.append(url)
    return candidates


def _iter_candidate_downloads(work_soup: BeautifulSoup, work_url: str):
    tab_ids = [
        "tabScore1",  # Full Scores
        "tabScore2",  # Parts
        "tabScore3",  # Vocal Scores
        "tabArrTrans",  # Arrangements and Transcriptions
        "tabScore5",  # Books
        "tabScore6",  # Other
    ]
    seen = set()

    def _maybe_add(href: str) -> Optional[str]:
        if not href:
            return None
        url = _normalize_url(href, work_url)
        if url in seen:
            return None
        seen.add(url)
        return url

    def _yield_from_container(container: BeautifulSoup):
        for a in container.find_all("a", href=True):
            href = a["href"]
            if "Special:ImagefromIndex" in href:
                url = _maybe_add(href)
                if url:
                    yield url
        for a in container.find_all("a", href=True):
            href = a["href"]
            if ".pdf" in href.lower():
                url = _maybe_add(href)
                if url:
                    yield url

    for tab_id in tab_ids:
        tab = work_soup.find(id=tab_id)
        if not tab:
            continue
        yield from _yield_from_container(tab)

    # Fallback: any remaining ImagefromIndex or PDF links on the page.
    yield from _yield_from_container(work_soup)


def _extract_publisher_info_text(we_block: BeautifulSoup) -> str:
    table = we_block.find("table", class_=re.compile(r"\bwe_edition_info\b"))
    if not table:
        return ""

    for tr in table.find_all("tr"):
        th = tr.find("th")
        if not th:
            continue
        th_text = th.get_text(" ", strip=True).lower()
        if "publisher" not in th_text:
            continue
        td = tr.find("td")
        if td:
            return td.get_text(" ", strip=True)
    return ""


def _extract_edition_table_text(we_block: BeautifulSoup) -> str:
    table = we_block.find("table", class_=re.compile(r"\bwe_edition_info\b"))
    if not table:
        return ""
    return table.get_text(" ", strip=True)


def _rank_complete_score_candidates(work_soup: BeautifulSoup, work_url: str) -> list[str]:
    """
    Return ranked "Complete Score" download links from Full Scores.
    Ranking favors:
      1) typeset complete scores
      2) high-quality urtext scans (e.g., NBA/Baerenreiter, 600dpi)
      3) other non-manuscript complete scores

    This mirrors the manual IMSLP flow:
      Full Scores -> first "Complete Score" -> Special:ImagefromIndex...
    """
    tab = work_soup.find(id="tabScore1")
    if not tab:
        return []

    we_blocks = tab.find_all("div", class_="we", recursive=False)
    candidates = []
    order = 0

    for we_block in we_blocks:
        publisher_info = _extract_publisher_info_text(we_block)
        edition_text = _extract_edition_table_text(we_block)
        meta_text = f"{publisher_info} {edition_text}".lower()
        has_manuscript = "manuscript" in meta_text

        for a in we_block.find_all("a", href=True):
            href = a.get("href") or ""
            if "Special:ImagefromIndex" not in href:
                continue
            text = a.get_text(" ", strip=True).lower()
            if "complete score" in text:
                file_block = a.find_parent("div", id=re.compile(r"^IMSLP"))
                file_text = file_block.get_text(" ", strip=True).lower() if file_block else ""
                is_typeset = "typeset" in file_text
                is_scanned = "scanned by" in file_text
                scanned_unknown = "scanned by unknown" in file_text
                has_urtext = (
                    "urtext" in meta_text
                    or "neue bach-ausgabe" in meta_text
                    or "bärenreiter" in meta_text
                    or "ba " in meta_text
                )
                has_high_dpi = bool(re.search(r"\b(?:600|1200)\s*dpi\b", meta_text))

                # Lower is better.
                if is_typeset and not is_scanned:
                    quality_rank = 0
                elif is_typeset:
                    quality_rank = 1
                elif has_urtext and has_high_dpi:
                    quality_rank = 2
                elif has_urtext:
                    quality_rank = 3
                elif has_high_dpi:
                    quality_rank = 4
                elif is_scanned and not scanned_unknown:
                    quality_rank = 5
                else:
                    quality_rank = 6

                candidates.append(
                    {
                        "url": _normalize_url(href, work_url),
                        "has_manuscript": has_manuscript,
                        "quality_rank": quality_rank,
                        "order": order,
                    }
                )
                order += 1

    non_manuscript = [c for c in candidates if not c["has_manuscript"]]
    if not non_manuscript:
        return []

    ranked = sorted(non_manuscript, key=lambda c: (c["quality_rank"], c["order"]))
    return [c["url"] for c in ranked]


def _extract_timed_download_url(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    wait_span = soup.find(id="sm_dl_wait")
    if not wait_span:
        return None
    data_id = (wait_span.get("data-id") or "").strip()
    if not data_id:
        return None
    url = unescape(data_id).strip()
    if url.startswith("//"):
        url = f"https:{url}"
    return _normalize_url(url, base_url)


def _extract_gate_wait_seconds(html: str, soup: BeautifulSoup) -> Optional[float]:
    # Some pages place the countdown directly in the #sm_dl_wait text.
    wait_span = soup.find(id="sm_dl_wait")
    if wait_span:
        text = wait_span.get_text(" ", strip=True)
        m = re.search(r"continue in\s+(\d+)\s+seconds?", text, re.I)
        if m:
            return float(m.group(1))

    # Otherwise the wait value is usually in the localized JS bundle.
    m = re.search(r'"js-a4"\s*:\s*"(\d+)"', html)
    if m:
        return float(m.group(1))
    return None


def _resolve_continue_download(
    imagefromindex_url: str, timeout_s: float = 35.0, poll_s: float = 1.0
) -> Optional[str]:
    base_url = imagefromindex_url
    fetched_url, html, is_pdf = _fetch_url_state(base_url)
    if is_pdf:
        return fetched_url
    deadline = time.time() + max(0.0, timeout_s)
    timed_url_fallback: Optional[str] = None

    while True:
        soup = BeautifulSoup(html, "html.parser")

        # Accept disclaimer if present.
        accept = soup.find("a", href=re.compile(r"Special:IMSLPDisclaimerAccept", re.I))
        if accept and accept.get("href"):
            base_url = _normalize_url(accept["href"], base_url)
            fetched_url, html, is_pdf = _fetch_url_state(base_url)
            if is_pdf:
                return fetched_url
            continue

        # Look for a direct "continue to your download" link.
        for a in soup.find_all("a", href=True):
            text = a.get_text(" ", strip=True).lower()
            if "continue" in text and "download" in text:
                return _normalize_url(a["href"], base_url)

        # Timed gate pages often store the mirror URL in #sm_dl_wait[data-id].
        # Keep polling until continue-link appears or timeout expires.
        timed_url = _extract_timed_download_url(soup, base_url)
        if timed_url:
            timed_url_fallback = timed_url
            now = time.time()
            if now < deadline:
                gate_wait = _extract_gate_wait_seconds(html, soup)
                if gate_wait is not None and gate_wait > 0:
                    time.sleep(min(gate_wait + 0.5, max(0.0, deadline - now)))
                else:
                    time.sleep(min(max(0.2, poll_s), max(0.0, deadline - now)))
                if time.time() < deadline:
                    fetched_url, html, is_pdf = _fetch_url_state(base_url)
                    if is_pdf:
                        return fetched_url
                    continue
            return timed_url_fallback

        # Meta refresh redirects sometimes contain the final URL.
        meta = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
        if meta and meta.get("content"):
            m = re.search(r"url=([^;]+)", meta["content"], re.I)
            if m:
                return _normalize_url(m.group(1).strip(), base_url)

        # Regex for any direct PDF URL in the page HTML.
        m = re.search(r"https?://[^\"'<>\\s]+\\.pdf", html, re.I)
        if m:
            return m.group(0)

        # Fallback: first .pdf link
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".pdf" in href.lower():
                return _normalize_url(href, base_url)

        # Last fallback: handler URL (still often needs another pass).
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "IMSLPImageHandler" in href:
                return _normalize_url(href, base_url)

        if time.time() >= deadline:
            break
        time.sleep(min(max(0.2, poll_s), max(0.0, deadline - time.time())))
        fetched_url, html, is_pdf = _fetch_url_state(base_url)
        if is_pdf:
            return fetched_url

    return timed_url_fallback


def _download_file(url: str, out_path: Path, referer: Optional[str] = None) -> None:
    headers = {
        "User-Agent": UA,
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    }
    if referer:
        headers["Referer"] = referer
    req = Request(url, headers=headers)
    with _open(req) as resp:  # nosec - user provided URL list
        data = _read_response(resp)
        ctype = (resp.headers.get("Content-Type") or "").lower()

    if data.startswith(b"<!DOCTYPE html") or data.startswith(b"<html") or b"IMSLP - Bot Check" in data:
        raise ValueError("IMSLP bot-check page returned (export browser cookies and use --cookies)")
    if not data.startswith(b"%PDF-"):
        raise ValueError(f"non-PDF response (content-type: {ctype or 'unknown'})")

    out_path.write_bytes(data)


def _safe_filename(bwv: int, url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        name = f"bwv_{bwv:04d}.pdf"
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return f"BWV{bwv:04d}__{name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download IMSLP PDFs from queue.")
    parser.add_argument(
        "--queue-csv",
        default="data-testing/imslp_suspicious_queue.csv",
        help="Queue CSV with imslp_work_url per BWV",
    )
    parser.add_argument(
        "--out-dir",
        default="data/imslp_pdfs",
        help="Output directory for PDFs",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Sleep seconds between downloads",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max downloads (for testing)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Path to Netscape/Mozilla cookies.txt exported from your browser",
    )
    parser.add_argument(
        "--cookie-jar-out",
        default=None,
        help="Optional path to save cookies after the run (Netscape format).",
    )
    parser.add_argument(
        "--gate-timeout",
        type=float,
        default=35.0,
        help="Max seconds to wait for the IMSLP timed download gate",
    )
    parser.add_argument(
        "--gate-poll",
        type=float,
        default=1.0,
        help="Polling interval (seconds) while waiting for timed download gate",
    )
    return parser.parse_args()


def _init_opener(cookies_path: Optional[str]) -> None:
    global _OPENER, _COOKIE_JAR
    jar = cookiejar.MozillaCookieJar()
    if cookies_path:
        jar.load(cookies_path, ignore_discard=True, ignore_expires=True)
    _COOKIE_JAR = jar
    _OPENER = build_opener(HTTPCookieProcessor(jar))


def main() -> None:
    args = parse_args()
    global _OPENER
    _init_opener(args.cookies)
    # Preflight to let IMSLP set session cookies (if any).
    try:
        _fetch_text(f"{IMSLP_BASE}/")
    except Exception:
        pass

    queue_path = Path(args.queue_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(queue_path)
    if "bwv" not in df.columns or "imslp_work_url" not in df.columns:
        raise SystemExit("Queue CSV missing required columns: 'bwv' and/or 'imslp_work_url'")

    count = 0
    for _, row in df.iterrows():
        bwv_val = row.get("bwv")
        work_url = str(row.get("imslp_work_url") or "").strip()
        if not work_url:
            print(f"BWV {bwv_val}: missing IMSLP URL, skipping")
            continue

        try:
            bwv = int(bwv_val)
        except (TypeError, ValueError):
            print(f"BWV {bwv_val}: invalid BWV value, skipping")
            continue

        print(f"BWV {bwv}: {work_url}")
        if not args.overwrite:
            existing_any = sorted(out_dir.glob(f"BWV{bwv:04d}__*.pdf"))
            if existing_any:
                print(f"  exists: {existing_any[0]} (BWV already downloaded), skipping")
                count += 1
                if args.max is not None and count >= args.max:
                    break
                time.sleep(max(0.0, args.sleep))
                continue

        downloaded = False
        stop_for_bot_check = False

        primary_soup = _fetch_html(work_url)
        work_url_candidates = []
        seen_work_urls = set()

        def _add_work_url(url: str) -> None:
            if not url:
                return
            norm = _normalize_url(url, work_url)
            if norm in seen_work_urls:
                return
            seen_work_urls.add(norm)
            work_url_candidates.append(norm)

        match_type = str(row.get("imslp_match_type") or "").strip().lower()
        alternates = _split_pipe_urls(row.get("imslp_alternates"))

        if match_type == "range" or _is_range_work_url(work_url):
            for specific_url in _extract_bwv_specific_work_urls(primary_soup, bwv, work_url):
                _add_work_url(specific_url)

        for alt in alternates:
            _add_work_url(alt)

        _add_work_url(work_url)

        for candidate_work_url in work_url_candidates:
            if candidate_work_url == work_url:
                candidate_soup = primary_soup
            else:
                try:
                    candidate_soup = _fetch_html(candidate_work_url)
                except Exception as exc:
                    print(f"  candidate page failed: {candidate_work_url} ({exc})")
                    continue

            candidates = _rank_complete_score_candidates(candidate_soup, candidate_work_url)
            if not candidates:
                continue
            for download_url in candidates:
                referer = candidate_work_url
                if "Special:ImagefromIndex" in download_url:
                    final_url = _resolve_continue_download(
                        download_url, timeout_s=args.gate_timeout, poll_s=args.gate_poll
                    )
                    if not final_url:
                        continue
                    referer = download_url
                else:
                    final_url = download_url

                filename = _safe_filename(bwv, final_url)
                out_path = out_dir / filename
                if out_path.exists() and not args.overwrite:
                    print(f"  exists: {out_path}, skipping")
                    downloaded = True
                    break

                try:
                    _download_file(final_url, out_path, referer=referer)
                    print(f"  saved: {out_path}")
                    downloaded = True
                    break
                except ValueError as exc:
                    print(f"  download failed: {exc}")
                    if "bot-check" in str(exc).lower():
                        stop_for_bot_check = True
                        break
                    continue

            if downloaded or stop_for_bot_check:
                break

        if not downloaded:
            print("  no downloadable PDF found, skipping")
            continue

        count += 1
        if args.max is not None and count >= args.max:
            break
        time.sleep(max(0.0, args.sleep))

    if args.cookie_jar_out and _COOKIE_JAR is not None:
        try:
            _COOKIE_JAR.save(args.cookie_jar_out, ignore_discard=True, ignore_expires=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
