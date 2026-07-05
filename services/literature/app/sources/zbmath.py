"""
zbMATH source — MSC-code search over the math-specific review catalog.

zbMATH's real value is its 1-paragraph expert review of each paper's main
result: better structured than an abstract, and covers older/non-English
papers Semantic Scholar indexes poorly. We use the public zbMATH Open API.
"""
from __future__ import annotations

from typing import Any

import httpx

from services.literature.app.models import FullTextDocument, RawDocument, TabMatch
from services.literature.app.sources.base import BaseSource, RateLimiter

_BASE = "https://api.zbmath.org/v1"


class ZbmathSource(BaseSource):
    name = "zbmath"
    supported_domains: tuple[str, ...] = ("math_combinatorics", "graph_invariants")

    def __init__(self, *, min_interval_sec: float = 1.0, http_timeout: float = 30.0,
                 user_agent: str = "propab-literature/0.1") -> None:
        super().__init__(http_timeout=http_timeout, user_agent=user_agent)
        self._rate_limiter = RateLimiter(min_interval_sec)

    async def search(self, query: str, profile: dict[str, Any]) -> list[RawDocument]:
        msc_codes = (profile.get("classification_codes", {}) or {}).get("zbmath", [])
        client = await self._get_client()
        await self._rate_limiter.wait()
        params: dict[str, Any] = {"search_string": query, "results_per_page": 50}
        if msc_codes:
            params["search_string"] = f"{query} msc:{','.join(msc_codes)}"
        try:
            resp = await client.get(f"{_BASE}/document/_search", params=params)
        except httpx.HTTPError:
            return []
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except Exception:
            return []
        out = []
        for item in data.get("result", []) or []:
            out.append(
                RawDocument(
                    source="zbmath",
                    external_id=str(item.get("zbmath_id") or item.get("id") or ""),
                    title=(item.get("title", {}) or {}).get("title", "") if isinstance(item.get("title"), dict) else str(item.get("title", "")),
                    authors=", ".join(a.get("name", "") for a in item.get("contributors", {}).get("authors", []) or []) if isinstance(item.get("contributors"), dict) else "",
                    year=item.get("year") or 0,
                    doi=item.get("doi", "") or "",
                    url=item.get("source", {}).get("series", "") if isinstance(item.get("source"), dict) else "",
                    abstract=item.get("review_text", "") or item.get("summary", "") or "",
                    extra={"msc": item.get("msc", [])},
                )
            )
        return out

    async def fetch_full_text(self, doc: RawDocument) -> FullTextDocument:
        return FullTextDocument(
            source="zbmath",
            external_id=doc.external_id,
            title=doc.title,
            authors=doc.authors,
            year=doc.year,
            doi=doc.doi,
            url=doc.url,
            body_text=doc.abstract,
            extraction_method="abstract_only",
            extraction_quality=0.5,  # zbMATH reviews are dense summaries, not full text
        )

    async def check_tabulated(self, values: dict[str, Any]) -> list[TabMatch]:
        return []

    async def health(self) -> bool:
        try:
            client = await self._get_client()
            resp = await client.get(f"{_BASE}/document/_search", params={"search_string": "test"}, timeout=10.0)
            return resp.status_code in (200, 400)
        except httpx.HTTPError:
            return False
