import pytest

from services.literature.app.evaluator.metrics import _external_id_for, citation_verification_rate, verify_claim
from services.literature.app.models import ExtractedClaim, FullTextDocument, RawDocument
from services.literature.app.sources.base import BaseSource


def _claim(source: str, verbatim: str = "x", source_url: str = "", source_doi: str = "") -> ExtractedClaim:
    return ExtractedClaim(
        text=verbatim, claim_type="observation", status="proven", verbatim=verbatim,
        source=source, source_url=source_url, source_doi=source_doi, claim_id="1",
    )


class TestExternalIdFor:
    def test_arxiv(self):
        claim = _claim("arxiv", source_url="https://arxiv.org/abs/2401.00001v1")
        assert _external_id_for(claim) == "2401.00001"

    def test_oeis(self):
        claim = _claim("oeis", source_url="https://oeis.org/A003023")
        assert _external_id_for(claim) == "A003023"

    def test_mathoverflow(self):
        claim = _claim("mathoverflow", source_url="https://mathoverflow.net/questions/354687/well-spread")
        assert _external_id_for(claim) == "354687"

    def test_pubmed_extracts_pmid_from_url(self):
        claim = _claim("pubmed", source_url="https://pubmed.ncbi.nlm.nih.gov/38353623/")
        assert _external_id_for(claim) == "38353623"

    def test_pubmed_without_matching_url_returns_empty(self):
        claim = _claim("pubmed", source_url="https://doi.org/10.1016/j.cell.2021.09.018")
        assert _external_id_for(claim) == ""

    def test_unknown_source_falls_back_to_doi_or_url(self):
        claim = _claim("crossref", source_doi="10.1/xyz")
        assert _external_id_for(claim) == "10.1/xyz"


class _FakePubmedSource(BaseSource):
    name = "pubmed"

    def __init__(self, body_text: str) -> None:
        super().__init__()
        self._body_text = body_text

    async def fetch_full_text(self, doc: RawDocument) -> FullTextDocument:
        return FullTextDocument(source="pubmed", external_id=doc.external_id, body_text=self._body_text)


class TestVerifyClaim:
    @pytest.mark.asyncio
    async def test_verified_when_verbatim_found_in_refetched_source(self):
        claim = _claim(
            "pubmed", verbatim="The knockout mice showed reduced tumor growth.",
            source_url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        )
        sources = {"pubmed": _FakePubmedSource("Background text. The knockout mice showed reduced tumor growth. More text.")}
        assert await verify_claim(sources, claim) is True

    @pytest.mark.asyncio
    async def test_not_verified_when_verbatim_absent(self):
        claim = _claim(
            "pubmed", verbatim="This exact sentence never appears anywhere in the source.",
            source_url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        )
        sources = {"pubmed": _FakePubmedSource("Completely unrelated abstract content here.")}
        assert await verify_claim(sources, claim) is False

    @pytest.mark.asyncio
    async def test_none_when_pmid_unresolvable(self):
        claim = _claim("pubmed", verbatim="x", source_url="not-a-pubmed-url")
        sources = {"pubmed": _FakePubmedSource("anything")}
        assert await verify_claim(sources, claim) is None

    @pytest.mark.asyncio
    async def test_none_when_source_missing(self):
        claim = _claim("pubmed", verbatim="x", source_url="https://pubmed.ncbi.nlm.nih.gov/1/")
        assert await verify_claim({}, claim) is None


class TestCitationVerificationRate:
    @pytest.mark.asyncio
    async def test_empty_claims_returns_none_rate(self):
        result = await citation_verification_rate({}, [])
        assert result == {"sampled": 0, "verifiable": 0, "verified": 0, "rate": None, "unverifiable_sources": []}

    @pytest.mark.asyncio
    async def test_mixed_verifiable_and_unverifiable(self):
        good = _claim(
            "pubmed", verbatim="Confirmed finding here.", source_url="https://pubmed.ncbi.nlm.nih.gov/1/",
        )
        unresolvable = _claim("pubmed", verbatim="x", source_url="bad-url")
        sources = {"pubmed": _FakePubmedSource("Confirmed finding here.")}
        result = await citation_verification_rate(sources, [good, unresolvable], sample_size=2)
        assert result["sampled"] == 2
        assert result["verifiable"] == 1
        assert result["verified"] == 1
        assert result["rate"] == 1.0
