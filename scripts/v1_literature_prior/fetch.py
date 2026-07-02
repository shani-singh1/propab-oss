"""Fetch papers from arXiv and Semantic Scholar (no Propab service imports)."""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

_ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}
_ARXIV_MIN_INTERVAL_SEC = 3.5
_last_arxiv_at = 0.0


def _throttle_arxiv() -> None:
    global _last_arxiv_at
    now = time.monotonic()
    wait = _ARXIV_MIN_INTERVAL_SEC - (now - _last_arxiv_at)
    if wait > 0:
        time.sleep(wait)
    _last_arxiv_at = time.monotonic()


def _arxiv_id_from_entry(entry: ET.Element) -> str:
    raw = entry.find("a:id", _ARXIV_NS)
    if raw is None or not raw.text:
        return ""
    m = re.search(r"arxiv\.org/abs/([^/\s]+)", raw.text)
    return m.group(1) if m else ""


def fetch_arxiv(query: str, *, max_results: int = 15) -> list[dict[str, Any]]:
    _throttle_arxiv()
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    })
    url = f"http://export.arxiv.org/api/query?{params}"
    with urllib.request.urlopen(url, timeout=90) as resp:
        root = ET.fromstring(resp.read())
    papers: list[dict[str, Any]] = []
    for entry in root.findall("a:entry", _ARXIV_NS):
        arxiv_id = _arxiv_id_from_entry(entry)
        title_el = entry.find("a:title", _ARXIV_NS)
        summary_el = entry.find("a:summary", _ARXIV_NS)
        published_el = entry.find("a:published", _ARXIV_NS)
        authors = [
            (a.find("a:name", _ARXIV_NS).text or "").strip()
            for a in entry.findall("a:author", _ARXIV_NS)
            if a.find("a:name", _ARXIV_NS) is not None
        ]
        if not arxiv_id:
            continue
        papers.append({
            "id": arxiv_id,
            "source": "arxiv",
            "title": (title_el.text or "").strip().replace("\n", " ") if title_el is not None else "",
            "abstract": (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else "",
            "year": int((published_el.text or "1970")[:4]) if published_el is not None else None,
            "authors": authors[:6],
            "citation_count": None,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return papers


def fetch_semantic_scholar(
    query: str,
    *,
    limit: int = 20,
    retries: int = 4,
) -> list[dict[str, Any]]:
    if httpx is None:
        return []
    fields = "title,abstract,year,citationCount,externalIds,authors,url"
    params = {"query": query, "limit": limit, "fields": fields}
    delay = 8.0
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params=params,
                    headers={"User-Agent": "propab-v1-literature-prior/0.1"},
                )
            if resp.status_code == 429:
                time.sleep(delay)
                delay *= 1.8
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception:
            if attempt == retries - 1:
                return []
            time.sleep(delay)
            delay *= 1.5
    else:
        return []

    papers: list[dict[str, Any]] = []
    for row in data.get("data") or []:
        if not isinstance(row, dict):
            continue
        ext = row.get("externalIds") or {}
        arxiv_id = ext.get("ArXiv") or ext.get("arXiv")
        paper_id = arxiv_id or row.get("paperId") or ""
        if not paper_id:
            continue
        authors = [
            (a.get("name") or "").strip()
            for a in (row.get("authors") or [])
            if isinstance(a, dict)
        ]
        papers.append({
            "id": str(arxiv_id or paper_id),
            "source": "semantic_scholar",
            "title": (row.get("title") or "").strip(),
            "abstract": (row.get("abstract") or "").strip(),
            "year": row.get("year"),
            "authors": authors[:6],
            "citation_count": row.get("citationCount"),
            "url": row.get("url") or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""),
            "semantic_scholar_id": row.get("paperId"),
        })
    return papers


def merge_and_rank_papers(
    batches: list[list[dict[str, Any]]],
    *,
    max_papers: int = 30,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for batch in batches:
        for paper in batch:
            pid = str(paper.get("id") or "").strip()
            norm = pid.lower().replace("arxiv:", "")
            if not norm or norm in seen:
                continue
            seen.add(norm)
            merged.append(paper)

    def _rank_key(p: dict[str, Any]) -> tuple[int, int, str]:
        cites = p.get("citation_count")
        cite_score = int(cites) if isinstance(cites, int) else 0
        year = p.get("year")
        year_score = int(year) if isinstance(year, int) else 0
        return (cite_score, year_score, str(p.get("title") or ""))

    merged.sort(key=_rank_key, reverse=True)
    return merged[:max_papers]


def fetch_domain_papers(
    domain_cfg: dict[str, object],
    *,
    max_papers: int = 30,
    per_query: int = 12,
) -> list[dict[str, Any]]:
    batches: list[list[dict[str, Any]]] = []
    for q in domain_cfg.get("arxiv_queries") or []:
        if isinstance(q, str):
            batches.append(fetch_arxiv(q, max_results=per_query))
    for q in domain_cfg.get("semantic_scholar_queries") or []:
        if isinstance(q, str):
            batches.append(fetch_semantic_scholar(q, limit=per_query))
            time.sleep(2.0)
    return merge_and_rank_papers(batches, max_papers=max_papers)
