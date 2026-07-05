import httpx
import pytest

from services.literature.app.models import RawDocument
from services.literature.app.sources.europepmc import EuropePmcSource

_SEARCH_RESPONSE = {
    "resultList": {
        "result": [
            {
                "id": "40000001",
                "source": "PPR",
                "pmcid": "PMC10880268",
                "doi": "10.1101/2024.01.01.000001",
                "title": "A preprint about prime editing efficiency",
                "authorString": "Smith J, Doe A.",
                "pubYear": "2024",
                "abstractText": "We show that MLH1dn increases editing efficiency substantially.",
                "isOpenAccess": "Y",
            },
            {
                "id": "40000002",
                "source": "MED",
                "title": "A closed-access paper with no pmcid",
                "authorString": "Lee K.",
                "pubYear": "2022",
                "abstractText": "Abstract only, no full text available anywhere.",
                "isOpenAccess": "N",
            },
        ]
    }
}

_PMC_FULLTEXT_XML = """<?xml version="1.0"?>
<pmc-articleset><article><front><article-meta>
<title-group><article-title>A preprint about prime editing efficiency</article-title></title-group>
</article-meta></front>
<body><p>MLH1dn expression increases prime editing efficiency by 2.7 fold on average across tested loci, a result that held consistently across every replicate cohort examined in this study, and was independently reproduced in two separate laboratories using different cell lines.</p></body>
</article></pmc-articleset>"""


def make_source(handler) -> EuropePmcSource:
    source = EuropePmcSource(min_interval_sec=0.0)
    source._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return source


class TestEuropePmcSearch:
    @pytest.mark.asyncio
    async def test_parses_results_with_and_without_pmcid(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_SEARCH_RESPONSE)

        source = make_source(handler)
        results = await source.search("prime editing", {})

        assert len(results) == 2
        assert results[0].external_id == "PMC10880268"
        assert results[0].extra["is_open_access"] is True
        # No pmcid or doi in the fixture for the closed-access item — falls back to the raw Europe PMC id.
        assert results[1].external_id == "40000002"
        assert results[1].extra["is_open_access"] is False

    @pytest.mark.asyncio
    async def test_search_failure_returns_empty(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        source = make_source(handler)
        results = await source.search("q", {})
        assert results == []


class TestEuropePmcFetchFullText:
    @pytest.mark.asyncio
    async def test_fetches_pmc_fulltext_when_open_access_with_pmcid(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=_PMC_FULLTEXT_XML)

        source = make_source(handler)
        doc = RawDocument(
            source="europepmc", external_id="PMC10880268", title="T",
            extra={"pmcid": "PMC10880268", "is_open_access": True},
        )
        full = await source.fetch_full_text(doc)

        assert full.extraction_method == "pmc_fulltext"
        assert "2.7 fold" in full.body_text

    @pytest.mark.asyncio
    async def test_falls_back_to_abstract_when_not_open_access(self):
        source = make_source(lambda r: httpx.Response(200))
        doc = RawDocument(
            source="europepmc", external_id="10.1/x", title="T", abstract="Fallback abstract text here.",
            extra={"pmcid": "", "is_open_access": False},
        )
        full = await source.fetch_full_text(doc)

        assert full.extraction_method == "abstract_only"
        assert full.body_text == "Fallback abstract text here."
