import httpx
import pytest

from services.literature.app.models import RawDocument
from services.literature.app.sources.pubmed import PubmedSource

_IDCONV_OK = '{"status":"ok","records":[{"pmid":38353623,"pmcid":"PMC10880268","doi":"10.1/x"}]}'
_IDCONV_NO_PMC = '{"status":"ok","records":[{"pmid":38353623}]}'

_PMC_FULLTEXT_XML = """<?xml version="1.0"?>
<pmc-articleset><article><front><article-meta>
<title-group><article-title>A Real Paper Title</article-title></title-group>
</article-meta></front>
<body>
<sec><title>Results</title>
<p>Knockout mice showed a 2.7 fold increase in tumor growth compared to controls.</p>
<p>This effect was statistically significant across all three replicate cohorts tested here.</p>
</sec>
<fig><caption><p>Figure 1: Tumor growth curves over time for each genotype group.</p></caption></fig>
</body></article></pmc-articleset>"""

_PMC_RESTRICTED_XML = (
    '<?xml version="1.0"?><pmc-articleset><article>'
    '<!--The publisher of this article does not allow downloading of the full text in XML form.-->'
    '<front><article-meta><title-group><article-title>Restricted</article-title></title-group></article-meta></front>'
    "</article></pmc-articleset>"
)

_ABSTRACT_XML = """<?xml version="1.0"?>
<PubmedArticleSet><PubmedArticle><MedlineCitation><Article>
<Abstract><AbstractText>This is the fallback abstract text for the paper.</AbstractText></Abstract>
</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""


def _handler_factory(idconv_body: str, pmc_body: str | None, pmc_status: int = 200):
    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "idconv" in url:
            return httpx.Response(200, text=idconv_body)
        if "db=pmc" in url:
            if pmc_body is None:
                return httpx.Response(404)
            return httpx.Response(pmc_status, text=pmc_body)
        if "db=pubmed" in url and "efetch" in url:
            return httpx.Response(200, text=_ABSTRACT_XML)
        return httpx.Response(404)

    return handler


def make_source(handler) -> PubmedSource:
    source = PubmedSource(min_interval_sec=0.0)
    source._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return source


class TestPmcFullText:
    @pytest.mark.asyncio
    async def test_uses_pmc_fulltext_when_available(self):
        source = make_source(_handler_factory(_IDCONV_OK, _PMC_FULLTEXT_XML))
        doc = RawDocument(source="pubmed", external_id="38353623", title="T")
        full = await source.fetch_full_text(doc)

        assert full.extraction_method == "pmc_fulltext"
        assert "2.7 fold increase in tumor growth" in full.body_text
        assert any("Figure 1" in c for c in full.captions)

    @pytest.mark.asyncio
    async def test_preserves_paragraph_boundaries_not_one_giant_blob(self):
        # Regression: itertext() over the whole <body> concatenates every
        # <p> into one undifferentiated string, destroying the paragraph
        # breaks chunking/sentence-splitting downstream depend on.
        source = make_source(_handler_factory(_IDCONV_OK, _PMC_FULLTEXT_XML))
        doc = RawDocument(source="pubmed", external_id="38353623", title="T")
        full = await source.fetch_full_text(doc)

        paragraphs = full.body_text.split("\n\n")
        assert len(paragraphs) >= 2
        assert "tumor growth" in paragraphs[0]
        assert "replicate cohorts" in paragraphs[1]

    @pytest.mark.asyncio
    async def test_falls_back_to_abstract_when_publisher_restricts_fulltext(self):
        source = make_source(_handler_factory(_IDCONV_OK, _PMC_RESTRICTED_XML))
        doc = RawDocument(source="pubmed", external_id="38353623", title="T")
        full = await source.fetch_full_text(doc)

        assert full.extraction_method == "abstract_only"
        assert "fallback abstract" in full.body_text

    @pytest.mark.asyncio
    async def test_falls_back_to_abstract_when_no_pmcid(self):
        source = make_source(_handler_factory(_IDCONV_NO_PMC, _PMC_FULLTEXT_XML))
        doc = RawDocument(source="pubmed", external_id="38353623", title="T")
        full = await source.fetch_full_text(doc)

        assert full.extraction_method == "abstract_only"

    @pytest.mark.asyncio
    async def test_falls_back_when_pmc_fetch_fails(self):
        source = make_source(_handler_factory(_IDCONV_OK, None))
        doc = RawDocument(source="pubmed", external_id="38353623", title="T")
        full = await source.fetch_full_text(doc)

        assert full.extraction_method == "abstract_only"

    @pytest.mark.asyncio
    async def test_falls_back_when_body_too_short(self):
        thin_xml = (
            '<?xml version="1.0"?><pmc-articleset><article><body><p>Too short.</p></body></article></pmc-articleset>'
        )
        source = make_source(_handler_factory(_IDCONV_OK, thin_xml))
        doc = RawDocument(source="pubmed", external_id="38353623", title="T")
        full = await source.fetch_full_text(doc)

        assert full.extraction_method == "abstract_only"

    @pytest.mark.asyncio
    async def test_idconv_network_error_falls_back_gracefully(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            if "idconv" in str(request.url):
                raise httpx.ConnectError("simulated network failure", request=request)
            return httpx.Response(200, text=_ABSTRACT_XML)

        source = make_source(handler)
        doc = RawDocument(source="pubmed", external_id="38353623", title="T")
        full = await source.fetch_full_text(doc)

        assert full.extraction_method == "abstract_only"
