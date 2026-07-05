"""Crossref source — DOI resolution and full-text link discovery.

Used to resolve a bare DOI (from a bibliography annotation or a domain
profile's ``canonical_surveys``) into metadata and, where publishers expose
it, a full-text link. Crossref itself never has full text; ``fetch_full_text``
returns metadata plus the resolved link so the pipeline can hand off to a
format-specific fetch (e.g. arXiv) when the link points there.
"""
from __future__ import annotations

from typing import Any

import httpx

from services.literature.app.models import FullTextDocument, RawDocument, TabMatch
from services.literature.app.sources.base import BaseSource, RateLimiter

_BASE = "https://api.crossref.org"


class CrossrefSource(BaseSource):
    name = "crossref"
    supported_domains: tuple[str, ...] = ()

    def __init__(self, *, min_interval_sec: float = 1.0, http_timeout: float = 30.0,
                 user_agent: str = "propab-literature/0.1") -> None:
        super().__init__(http_timeout=http_timeout, user_agent=user_agent)
        self._rate_limiter = RateLimiter(min_interval_sec)

    async def search(self, query: str, profile: dict[str, Any]) -> list[RawDocument]:
        client = await self._get_client()
        await self._rate_limiter.wait()
        try:
            resp = await client.get(f"{_BASE}/works", params={"query.bibliographic": query, "rows": 30})
        except httpx.HTTPError:
            return []
        if resp.status_code != 200:
            return []
        items = resp.json().get("message", {}).get("items", [])
        return [_to_raw_doc(item) for item in items]

    async def resolve_doi(self, doi: str) -> RawDocument | None:
        client = await self._get_client()
        await self._rate_limiter.wait()
        try:
            resp = await client.get(f"{_BASE}/works/{doi}")
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        return _to_raw_doc(resp.json().get("message", {}))

    async def fetch_full_text(self, doc: RawDocument) -> FullTextDocument:
        links = doc.extra.get("links", [])
        return FullTextDocument(
            source="crossref",
            external_id=doc.external_id,
            title=doc.title,
            authors=doc.authors,
            year=doc.year,
            doi=doc.doi,
            url=doc.url,
            body_text=doc.abstract,
            extraction_method="abstract_only",
            extraction_quality=0.1 if not links else 0.2,
        )

    async def check_tabulated(self, values: dict[str, Any]) -> list[TabMatch]:
        return []

    async def health(self) -> bool:
        try:
            client = await self._get_client()
            resp = await client.get(f"{_BASE}/works", params={"query": "test", "rows": 1}, timeout=10.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False


def _to_raw_doc(item: dict[str, Any]) -> RawDocument:
    authors = ", ".join(
        f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", []) or []
    )
    doi = item.get("DOI", "") or ""
    title_list = item.get("title") or [""]
    year = 0
    date_parts = (item.get("issued", {}) or {}).get("date-parts", [[0]])
    if date_parts and date_parts[0]:
        year = date_parts[0][0] or 0
    links = [l.get("URL", "") for l in item.get("link", []) or []]
    return RawDocument(
        source="crossref",
        external_id=doi,
        title=title_list[0] if title_list else "",
        authors=authors,
        year=year,
        doi=doi,
        url=item.get("URL", "") or (f"https://doi.org/{doi}" if doi else ""),
        abstract=item.get("abstract", "") or "",
        extra={"links": links},
    )
