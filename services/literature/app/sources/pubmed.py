"""PubMed source — E-utilities search + full-text (PMC open access) retrieval,
falling back to abstract-only.

Biology domains only; a generic source that simply does nothing useful if a
domain profile never lists "pubmed" in ``source_priorities`` (the pipeline
skips it via ``is_relevant``).

Full-text path (measured live against real PMC articles, 2026-07-05): most
LitQA2-style questions ask about a specific number that lives in Results/
Methods/a table — never in the abstract. PubMed Central serves full JATS XML
for its "OA subset" via the same E-utilities pattern already used for
abstracts (``db=pmc`` instead of ``db=pubmed``). Parsing (including table
extraction and the "not every PMC article allows this" detection) lives in
``sources/_pmc.py``, shared with ``sources/europepmc.py``, whose search
results already carry a PMCID directly and skip the id-conversion step
this source needs (a bare PubMed search only gives a PMID).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import httpx

from services.literature.app.models import FullTextDocument, RawDocument, TabMatch
from services.literature.app.sources._pmc import fetch_pmc_fulltext
from services.literature.app.sources.base import BaseSource, RateLimiter

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_PMC_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"


class PubmedSource(BaseSource):
    name = "pubmed"
    supported_domains: tuple[str, ...] = ("genomics", "enzyme_kinetics")

    def __init__(self, *, api_key: str = "", min_interval_sec: float = 0.4, http_timeout: float = 30.0,
                 user_agent: str = "propab-literature/0.1") -> None:
        super().__init__(http_timeout=http_timeout, user_agent=user_agent)
        self._api_key = api_key
        self._rate_limiter = RateLimiter(min_interval_sec)

    def _key_params(self) -> dict[str, str]:
        return {"api_key": self._api_key} if self._api_key else {}

    async def search(self, query: str, profile: dict[str, Any]) -> list[RawDocument]:
        # Query construction is the load-bearing decision here (measured live,
        # CHANGELOG 0.7.0): OR-ing every reformulated keyword into one term
        # returned 500K+ papers sorted by recency, burying the specific
        # target paper entirely. Two fixes: (1) if the caller passes extra
        # ``search_terms``, AND them onto the primary query (narrow), never OR
        # (broaden); (2) ``sort=relevance`` so PubMed's own relevance ranking
        # surfaces the on-topic paper instead of just the newest. A targeted
        # query + relevance sort put the exact target paper at rank 1 where
        # the old broad-OR missed it in the top 100.
        extra = list(profile.get("search_terms", []) or [])
        term_query = query
        if extra:
            term_query = f"({query}) AND (" + " OR ".join(extra[:6]) + ")"
        mesh = (profile.get("classification_codes", {}) or {}).get("mesh", [])
        if mesh:
            term_query = f"({term_query}) AND (" + " OR ".join(f'"{m}"[MeSH Terms]' for m in mesh) + ")"

        client = await self._get_client()
        await self._rate_limiter.wait()
        search_resp = await client.get(
            f"{_BASE}/esearch.fcgi",
            params={"db": "pubmed", "term": term_query, "retmax": 50, "retmode": "json",
                    "sort": "relevance", **self._key_params()},
        )
        if search_resp.status_code != 200:
            return []
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        await self._rate_limiter.wait()
        summary_resp = await client.get(
            f"{_BASE}/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json", **self._key_params()},
        )
        if summary_resp.status_code != 200:
            return []
        result = summary_resp.json().get("result", {})
        out = []
        for pmid in result.get("uids", []):
            item = result.get(pmid, {})
            authors = ", ".join(a.get("name", "") for a in item.get("authors", []) or [])
            doi = next((aid.get("value") for aid in item.get("articleids", []) if aid.get("idtype") == "doi"), "")
            year = 0
            pubdate = item.get("pubdate", "")
            if pubdate[:4].isdigit():
                year = int(pubdate[:4])
            out.append(
                RawDocument(
                    source="pubmed",
                    external_id=pmid,
                    title=item.get("title", ""),
                    authors=authors,
                    year=year,
                    doi=doi or "",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                )
            )
        return out

    async def fetch_full_text(self, doc: RawDocument) -> FullTextDocument:
        pmcid = await self._pmid_to_pmcid(doc.external_id)
        if pmcid:
            client = await self._get_client()
            await self._rate_limiter.wait()
            pmc_result = await fetch_pmc_fulltext(client, pmcid, api_key=self._api_key)
            if pmc_result is not None:
                return FullTextDocument(
                    source="pubmed",
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

        client = await self._get_client()
        await self._rate_limiter.wait()
        resp = await client.get(
            f"{_BASE}/efetch.fcgi",
            params={"db": "pubmed", "id": doc.external_id, "retmode": "xml", **self._key_params()},
        )
        abstract = ""
        if resp.status_code == 200:
            try:
                root = ET.fromstring(resp.text)
                parts = [el.text or "" for el in root.iter("AbstractText")]
                abstract = "\n".join(parts)
            except ET.ParseError:
                abstract = ""
        return FullTextDocument(
            source="pubmed",
            external_id=doc.external_id,
            title=doc.title,
            authors=doc.authors,
            year=doc.year,
            doi=doc.doi,
            url=doc.url,
            body_text=abstract,
            extraction_method="abstract_only",
            extraction_quality=0.4,
        )

    async def _pmid_to_pmcid(self, pmid: str) -> str | None:
        client = await self._get_client()
        try:
            await self._rate_limiter.wait()
            resp = await client.get(_PMC_IDCONV_URL, params={"ids": pmid, "format": "json"})
            if resp.status_code != 200:
                return None
            records = resp.json().get("records", [])
            return next((r.get("pmcid") for r in records if r.get("pmcid")), None)
        except Exception:
            return None

    async def check_tabulated(self, values: dict[str, Any]) -> list[TabMatch]:
        return []

    async def health(self) -> bool:
        try:
            client = await self._get_client()
            resp = await client.get(
                f"{_BASE}/esearch.fcgi", params={"db": "pubmed", "term": "test", "retmode": "json"}, timeout=10.0
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
