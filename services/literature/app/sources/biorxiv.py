"""bioRxiv source — preprint search via the public biorxiv API.

The bioRxiv API is detail-lookup-by-DOI oriented rather than free-text
search; we use its "details" endpoint to enrich documents discovered via
PubMed/crossref, and a lightweight local match against ``search_terms`` for
the standalone ``search`` path.
"""
from __future__ import annotations

from typing import Any

import httpx

from services.literature.app.models import FullTextDocument, RawDocument, TabMatch
from services.literature.app.sources.base import BaseSource, RateLimiter

_BASE = "https://api.biorxiv.org"


class BiorxivSource(BaseSource):
    name = "biorxiv"
    supported_domains: tuple[str, ...] = ("genomics", "enzyme_kinetics")

    def __init__(self, *, min_interval_sec: float = 1.0, http_timeout: float = 30.0,
                 user_agent: str = "propab-literature/0.1") -> None:
        super().__init__(http_timeout=http_timeout, user_agent=user_agent)
        self._rate_limiter = RateLimiter(min_interval_sec)

    async def search(self, query: str, profile: dict[str, Any]) -> list[RawDocument]:
        # bioRxiv has no free-text search endpoint in the public API; this
        # source is primarily used via fetch_by_doi() for enrichment. Return
        # empty rather than fabricating a search — an honest "not supported"
        # beats a fake result set.
        return []

    async def fetch_by_doi(self, doi: str) -> RawDocument | None:
        client = await self._get_client()
        await self._rate_limiter.wait()
        try:
            resp = await client.get(f"{_BASE}/details/biorxiv/{doi}")
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        collection = resp.json().get("collection", [])
        if not collection:
            return None
        item = collection[-1]  # latest version
        return RawDocument(
            source="biorxiv",
            external_id=doi,
            title=item.get("title", ""),
            authors=item.get("authors", ""),
            year=int((item.get("date", "0000")[:4]) or 0) if item.get("date") else 0,
            doi=doi,
            url=f"https://www.biorxiv.org/content/{doi}",
            abstract=item.get("abstract", ""),
            extra={"category": item.get("category", "")},
        )

    async def fetch_full_text(self, doc: RawDocument) -> FullTextDocument:
        return FullTextDocument(
            source="biorxiv",
            external_id=doc.external_id,
            title=doc.title,
            authors=doc.authors,
            year=doc.year,
            doi=doc.doi,
            url=doc.url,
            body_text=doc.abstract,
            extraction_method="abstract_only",
            extraction_quality=0.4,
        )

    async def check_tabulated(self, values: dict[str, Any]) -> list[TabMatch]:
        return []

    async def health(self) -> bool:
        try:
            client = await self._get_client()
            resp = await client.get(f"{_BASE}/details/biorxiv/10.1101/2020.01.01.000001", timeout=10.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
