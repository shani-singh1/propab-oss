"""
Semantic Scholar source — citation graph traversal.

Given ``seed_papers`` from the domain profile, finds every paper that cites
each seed (paginated), clusters citing papers by recency ("frontier") and by
citation count ("established"), and surfaces the sentence in each citing
paper's abstract that references the seed's result. This is also the engine
behind two-level citation-depth crawling (agent3.md Step 6): ``references()``
walks the other direction (what a seed cites).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from services.literature.app.models import FullTextDocument, RawDocument, TabMatch
from services.literature.app.sources.base import BaseSource, RateLimiter

_BASE = "https://api.semanticscholar.org/graph/v1"
_PAPER_FIELDS = "title,abstract,year,externalIds,authors,citationCount"


class SemanticScholarSource(BaseSource):
    name = "semantic_scholar"
    supported_domains: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        api_key: str = "",
        min_interval_sec: float = 1.0,
        max_citations_per_seed: int = 1000,
        http_timeout: float = 30.0,
        user_agent: str = "propab-literature/0.1",
    ) -> None:
        super().__init__(http_timeout=http_timeout, user_agent=user_agent)
        self._api_key = api_key
        self._rate_limiter = RateLimiter(min_interval_sec)
        self._max_citations = max_citations_per_seed

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key} if self._api_key else {}

    async def search(self, query: str, profile: dict[str, Any]) -> list[RawDocument]:
        client = await self._get_client()
        resp = None
        # The unauthenticated public tier (no api_key) is heavily rate-limited
        # and returns 429 under any real-world concurrent usage; a couple of
        # backed-off retries turn "always empty without a key" into "usually
        # works, just slower" rather than silently returning nothing.
        for attempt in range(3):
            await self._rate_limiter.wait()
            resp = await client.get(
                f"{_BASE}/paper/search",
                params={"query": query, "fields": _PAPER_FIELDS, "limit": 100},
                headers=self._headers(),
            )
            if resp.status_code != 429:
                break
            if attempt < 2:
                await asyncio.sleep(2.0 * (2**attempt))
        if resp is None or resp.status_code != 200:
            return []
        data = resp.json()
        return [_to_raw_doc(p) for p in data.get("data", []) if p]

    async def citations_of(self, paper_id: str) -> list[RawDocument]:
        """Papers that cite ``paper_id`` (a DOI, arXiv id prefixed 'arXiv:',
        or S2 paper id), paginated up to ``max_citations_per_seed``."""
        client = await self._get_client()
        out: list[RawDocument] = []
        offset = 0
        page_size = 100
        while offset < self._max_citations:
            await self._rate_limiter.wait()
            resp = await client.get(
                f"{_BASE}/paper/{paper_id}/citations",
                params={"fields": _PAPER_FIELDS, "limit": page_size, "offset": offset},
                headers=self._headers(),
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            batch = data.get("data", [])
            if not batch:
                break
            for item in batch:
                paper = item.get("citingPaper")
                if paper:
                    doc = _to_raw_doc(paper)
                    doc.extra["contexts"] = item.get("contexts", [])
                    doc.extra["intents"] = item.get("intents", [])
                    out.append(doc)
            offset += len(batch)
            if len(batch) < page_size:
                break
        return out

    async def references_of(self, paper_id: str) -> list[RawDocument]:
        """Papers ``paper_id`` cites — used for two-level citation-depth crawl."""
        await self._rate_limiter.wait()
        client = await self._get_client()
        resp = await client.get(
            f"{_BASE}/paper/{paper_id}/references",
            params={"fields": _PAPER_FIELDS, "limit": 200},
            headers=self._headers(),
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        out = []
        for item in data.get("data", []):
            paper = item.get("citedPaper")
            if paper:
                out.append(_to_raw_doc(paper))
        return out

    def cluster(self, citing_docs: list[RawDocument]) -> dict[str, list[RawDocument]]:
        """Frontier (last 3 years) vs. established (>=20 citations) clustering,
        used by the retriever/gap_mapper to find where open questions live."""
        current_year = time.gmtime().tm_year
        frontier = [d for d in citing_docs if d.year and current_year - d.year <= 3]
        established = [d for d in citing_docs if (d.extra.get("citationCount") or 0) >= 20]
        other = [d for d in citing_docs if d not in frontier and d not in established]
        return {"frontier": frontier, "established": established, "other": other}

    async def fetch_full_text(self, doc: RawDocument) -> FullTextDocument:
        # Semantic Scholar itself only reliably has abstracts; delegate to
        # arXiv/crossref for full text when an external id is available. Here
        # we return the richest thing we have: abstract + citation contexts.
        contexts = "\n".join(doc.extra.get("contexts", []) or [])
        body = f"{doc.abstract}\n\nCiting contexts:\n{contexts}" if contexts else doc.abstract
        return FullTextDocument(
            source="semantic_scholar",
            external_id=doc.external_id,
            title=doc.title,
            authors=doc.authors,
            year=doc.year,
            doi=doc.doi,
            url=doc.url,
            body_text=body,
            extraction_method="abstract_only",
            extraction_quality=0.0,
        )

    async def check_tabulated(self, values: dict[str, Any]) -> list[TabMatch]:
        return []

    async def health(self) -> bool:
        try:
            client = await self._get_client()
            resp = await client.get(
                f"{_BASE}/paper/search", params={"query": "test", "limit": 1}, headers=self._headers(), timeout=10.0
            )
            return resp.status_code in (200, 429)  # 429 = reachable but rate-limited
        except httpx.HTTPError:
            return False


def _to_raw_doc(paper: dict[str, Any]) -> RawDocument:
    ext = paper.get("externalIds") or {}
    doi = ext.get("DOI", "") or ""
    arxiv_id = ext.get("ArXiv", "") or ""
    authors = ", ".join(a.get("name", "") for a in paper.get("authors", []) or [])
    return RawDocument(
        source="semantic_scholar",
        external_id=arxiv_id or doi or str(paper.get("paperId", "")),
        title=paper.get("title") or "",
        authors=authors,
        year=paper.get("year") or 0,
        doi=doi,
        url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (f"https://doi.org/{doi}" if doi else ""),
        abstract=paper.get("abstract") or "",
        extra={"citationCount": paper.get("citationCount") or 0, "s2_id": paper.get("paperId")},
    )
