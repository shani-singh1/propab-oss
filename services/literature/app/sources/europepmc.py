"""
Europe PMC source — real full-text search across PMC + bioRxiv + medRxiv,
no API key needed.

Why this exists: measured live, ``bioRxiv``'s public API has no free-text
search endpoint at all (``sources/biorxiv.py::search`` has always returned
``[]``), and PubMed's ``esearch`` implicitly ANDs every bare word together,
making natural-language-ish queries return zero hits (see CHANGELOG.md).
Europe PMC is a genuine relevance-ranked full-text search engine covering
PMC, preprint servers (bioRxiv/medRxiv), and MEDLINE in one index, and its
search results already carry a PMCID directly when one exists — so, unlike
``sources/pubmed.py``, this source skips the PMID→PMCID id-conversion round
trip entirely and hands the PMCID straight to the shared
``sources/_pmc.fetch_pmc_fulltext`` helper.
"""
from __future__ import annotations

from typing import Any

import httpx

from services.literature.app.models import FullTextDocument, RawDocument, TabMatch
from services.literature.app.sources._pmc import fetch_pmc_fulltext
from services.literature.app.sources.base import BaseSource, RateLimiter

_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"


class EuropePmcSource(BaseSource):
    name = "europepmc"
    supported_domains: tuple[str, ...] = ("genomics", "enzyme_kinetics")

    def __init__(self, *, min_interval_sec: float = 0.5, http_timeout: float = 30.0,
                 user_agent: str = "propab-literature/0.1") -> None:
        super().__init__(http_timeout=http_timeout, user_agent=user_agent)
        self._rate_limiter = RateLimiter(min_interval_sec)

    async def search(self, query: str, profile: dict[str, Any]) -> list[RawDocument]:
        # When extra ``search_terms`` are supplied (the /prior path) OR them
        # in as separate quoted clauses. When they're absent (the eval path
        # passes a single already-targeted query string, often containing its
        # own quotes — see CHANGELOG 0.7.0), use the query verbatim; wrapping
        # a quoted query in more quotes breaks Europe PMC's parser and was a
        # silent recall killer.
        extra = [t for t in (profile.get("search_terms") or []) if t.strip()]
        if extra:
            query_string = " OR ".join(f'"{t}"' if " " in t else t for t in ([query] + extra)[:8] if t.strip())
        else:
            query_string = query

        client = await self._get_client()
        await self._rate_limiter.wait()
        try:
            resp = await client.get(
                f"{_BASE}/search",
                params={
                    "query": query_string,
                    "format": "json",
                    "resultType": "core",
                    "pageSize": 30,
                },
            )
        except httpx.HTTPError:
            return []
        if resp.status_code != 200:
            return []
        try:
            results = resp.json().get("resultList", {}).get("result", [])
        except Exception:
            return []

        out = []
        for r in results:
            pmcid = r.get("pmcid", "") or ""
            doi = r.get("doi", "") or ""
            ext_id = pmcid or doi or r.get("id", "")
            if not ext_id:
                continue
            year = int(r.get("pubYear") or 0) if str(r.get("pubYear") or "").isdigit() else 0
            out.append(
                RawDocument(
                    source="europepmc",
                    external_id=ext_id,
                    title=r.get("title", ""),
                    authors=r.get("authorString", ""),
                    year=year,
                    doi=doi,
                    url=f"https://europepmc.org/article/{r.get('source', 'MED')}/{r.get('id', '')}",
                    abstract=r.get("abstractText", "") or "",
                    extra={
                        "pmcid": pmcid,
                        "is_open_access": r.get("isOpenAccess") == "Y",
                        "source_db": r.get("source", ""),
                    },
                )
            )
        return out

    async def fetch_full_text(self, doc: RawDocument) -> FullTextDocument:
        pmcid = doc.extra.get("pmcid")
        if pmcid and doc.extra.get("is_open_access"):
            client = await self._get_client()
            await self._rate_limiter.wait()
            pmc_result = await fetch_pmc_fulltext(client, pmcid)
            if pmc_result is not None:
                return FullTextDocument(
                    source="europepmc",
                    external_id=doc.external_id,
                    title=doc.title or pmc_result["title"],
                    authors=doc.authors,
                    year=doc.year,
                    doi=doc.doi,
                    url=doc.url,
                    body_text=pmc_result["body_text"],
                    captions=pmc_result["captions"],
                    extraction_method="pmc_fulltext",
                    extraction_quality=1.0,
                    is_appendix_included=True,
                )

        return FullTextDocument(
            source="europepmc",
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
            resp = await client.get(f"{_BASE}/search", params={"query": "test", "format": "json", "pageSize": 1}, timeout=10.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
