import asyncio

import pytest

from services.literature.app.extractors.claims import ClaimsExtractor, extract_bibliography_annotations
from services.literature.app.extractors.contradictions import ContradictionsExtractor
from services.literature.app.extractors.gaps import GapsExtractor
from services.literature.app.extractors.open_problems import OpenProblemsExtractor
from services.literature.app.extractors.tables import TablesExtractor
from services.literature.app.indexer.embeddings import EmbeddingClient
from services.literature.app.models import ExtractedClaim, FullTextDocument


def make_doc(**kwargs) -> FullTextDocument:
    defaults = dict(source="arxiv", external_id="2401.00001", title="Test Paper", authors="A. Author", year=2024)
    defaults.update(kwargs)
    return FullTextDocument(**defaults)


class TestClaimsExtractor:
    @pytest.mark.asyncio
    async def test_extracts_theorem_environment_as_proven(self):
        doc = make_doc(
            latex_environments=[
                {"env": "theorem", "content": "For every prime p, F(p) >= sqrt(p) - 1.", "location": 'section "Main"'}
            ]
        )
        claims = await ClaimsExtractor().extract(doc)
        assert len(claims) == 1
        assert claims[0].claim_type == "theorem"
        assert claims[0].status == "proven"
        assert claims[0].verbatim == "For every prime p, F(p) >= sqrt(p) - 1."
        assert claims[0].source == "arxiv"

    @pytest.mark.asyncio
    async def test_conjecture_environment_is_conjectured(self):
        doc = make_doc(latex_environments=[{"env": "conjecture", "content": "F(n) ~ n^{1/2}.", "location": "body"}])
        claims = await ClaimsExtractor().extract(doc)
        assert claims[0].claim_type == "conjecture"
        assert claims[0].status == "conjectured"

    @pytest.mark.asyncio
    async def test_open_signal_inside_theorem_overrides_status(self):
        doc = make_doc(
            latex_environments=[
                {"env": "theorem", "content": "It remains open whether F(n) converges.", "location": "body"}
            ]
        )
        claims = await ClaimsExtractor().extract(doc)
        assert claims[0].status == "open"

    @pytest.mark.asyncio
    async def test_proof_body_is_not_a_claim_but_quant_sentence_is(self):
        doc = make_doc(
            latex_environments=[
                {
                    "env": "proof",
                    "content": "We proceed by induction. This gives us that F(n) > n/2 for all n > 10.",
                    "location": "section 3",
                }
            ]
        )
        claims = await ClaimsExtractor().extract(doc)
        assert len(claims) == 1
        assert claims[0].claim_type == "proof_intermediate"
        assert "F(n) > n/2" in claims[0].verbatim

    @pytest.mark.asyncio
    async def test_footnote_with_assertion_is_extracted(self):
        doc = make_doc(footnotes=["Note that this bound is tight for n > 100.", "See also [3] for related work."])
        claims = await ClaimsExtractor().extract(doc)
        footnote_claims = [c for c in claims if c.claim_type == "footnote_claim"]
        assert len(footnote_claims) == 1
        assert footnote_claims[0].location == "footnote 1"

    @pytest.mark.asyncio
    async def test_caption_with_assertion_is_extracted(self):
        doc = make_doc(captions=["Density decreases monotonically for n > 100.", "An unrelated diagram."])
        claims = await ClaimsExtractor().extract(doc)
        caption_claims = [c for c in claims if c.claim_type == "caption_claim"]
        assert len(caption_claims) == 1
        assert caption_claims[0].location == "caption 1"

    @pytest.mark.asyncio
    async def test_linguistic_scan_over_body_text(self):
        doc = make_doc(
            body_text=(
                "This is unrelated filler text with nothing notable in it at all so it should not match. "
                "We prove that the maximum density is exactly one half. "
                "In particular, the bound above is not known to be tight."
            )
        )
        claims = await ClaimsExtractor().extract(doc)
        texts = [c.text for c in claims]
        assert any("We prove that the maximum density" in t for t in texts)
        assert any(c.claim_type == "remark" for c in claims)

    @pytest.mark.asyncio
    async def test_no_duplicate_claims_for_identical_verbatim(self):
        doc = make_doc(
            latex_environments=[
                {"env": "theorem", "content": "F(n) > 0.5 for all n.", "location": "body"},
                {"env": "theorem", "content": "F(n) > 0.5 for all n.", "location": "body"},
            ]
        )
        claims = await ClaimsExtractor().extract(doc)
        assert len(claims) == 1

    @pytest.mark.asyncio
    async def test_bibliography_annotations(self):
        doc = make_doc(
            cite_sentences=[{"text": "By the result of [cite], the density is at least 0.5.", "keys": ["X1"]}]
        )
        claims = await extract_bibliography_annotations(doc)
        assert len(claims) == 1
        assert claims[0].claim_type == "bibliography_annotation"


class TestTablesExtractor:
    @pytest.mark.asyncio
    async def test_parses_tabular_environment(self):
        raw = r"{c|c} n & F(n) \\ 1 & 2 \\ 2 & 3 \\ 3 & 5 \\"
        doc = make_doc(tables_raw=[{"raw": raw, "location": 'appendix A (table)', "is_appendix": True}])
        tables = await TablesExtractor().extract(doc)
        assert len(tables) == 1
        t = tables[0]
        assert t.values == {"1": 2, "2": 3, "3": 5}
        assert t.max_index == 3.0
        assert t.min_index == 1.0
        assert t.is_in_appendix is True

    @pytest.mark.asyncio
    async def test_parses_enumerated_list_in_body(self):
        doc = make_doc(
            body_text="For n = 1, 2, 3, 4 the maximum sizes are 1, 2, 3, 5. This is unrelated to anything else."
        )
        tables = await TablesExtractor().extract(doc)
        assert len(tables) == 1
        assert tables[0].values == {"1": 1, "2": 2, "3": 3, "4": 5}

    @pytest.mark.asyncio
    async def test_ignores_malformed_table(self):
        doc = make_doc(tables_raw=[{"raw": "{c} justoneheader \\\\", "location": "body"}])
        tables = await TablesExtractor().extract(doc)
        assert tables == []


class TestOpenProblemsExtractor:
    @pytest.mark.asyncio
    async def test_explicit_marker(self):
        doc = make_doc(
            body_text="Some intro text.\n\nOpen problem: Determine whether F(n) is unbounded as n grows large.\n\nMore text follows here."
        )
        problems = await OpenProblemsExtractor().extract(doc)
        assert len(problems) == 1
        assert "unbounded" in problems[0].statement

    @pytest.mark.asyncio
    async def test_computationally_approachable_flag(self):
        doc = make_doc(
            body_text="Open problem: Compute F(n) numerically for n up to 10^6 to test the conjecture."
        )
        problems = await OpenProblemsExtractor().extract(doc)
        assert problems[0].computationally_approachable is True
        assert problems[0].approachable_angle != ""

    @pytest.mark.asyncio
    async def test_conjecture_environment_is_an_open_problem(self):
        doc = make_doc(latex_environments=[{"env": "conjecture", "content": "F(n) tends to 1 as n grows.", "location": "body"}])
        problems = await OpenProblemsExtractor().extract(doc)
        assert len(problems) == 1


def _claim(text, status="proven", year=2020, embedding=None, doi="", location="body"):
    return ExtractedClaim(
        text=text, claim_type="theorem", status=status, verbatim=text, source_doi=doi,
        source_year=year, location=location, embedding=embedding or [], claim_id=text,
    )


class TestContradictionsExtractor:
    @pytest.mark.asyncio
    async def test_detects_direct_numeric_contradiction(self):
        a = _claim("F(n) > 0.7 for all n.", year=2015, doi="10.1/a")
        b = _claim("F(n) < 0.4 for all n.", year=2020, doi="10.1/b")
        contradictions = await ContradictionsExtractor().find_contradictions([a, b])
        assert len(contradictions) == 1
        assert contradictions[0].contradiction_type == "superseded"

    @pytest.mark.asyncio
    async def test_compatible_bounds_are_not_contradictions(self):
        a = _claim("F(n) > 0.5 for all n.")
        b = _claim("F(n) > 0.6 for all n.")
        contradictions = await ContradictionsExtractor().find_contradictions([a, b])
        assert contradictions == []

    @pytest.mark.asyncio
    async def test_same_source_same_location_not_flagged(self):
        a = _claim("F(n) > 0.7.", doi="10.1/a", location="section 2")
        b = _claim("F(n) < 0.4.", doi="10.1/a", location="section 2")
        contradictions = await ContradictionsExtractor().find_contradictions([a, b])
        assert contradictions == []


class TestGapsExtractor:
    @pytest.mark.asyncio
    async def test_gap_between_proven_and_conjectured_bound(self):
        proven = _claim("F(n) > 0.5 for all n.", status="proven", year=2010)
        conjectured = _claim("F(n) > 0.7 for all n.", status="conjectured", year=2022)
        gaps = await GapsExtractor().find_gaps([proven, conjectured], [])
        assert len(gaps) == 1
        assert gaps[0].last_progress == 2022

    @pytest.mark.asyncio
    async def test_untethered_open_problem_still_becomes_a_gap(self):
        from services.literature.app.models import OpenProblem

        op = OpenProblem(statement="Is X true for all n?", computationally_approachable=True, year=2018)
        gaps = await GapsExtractor().find_gaps([], [op])
        assert len(gaps) == 1
        assert gaps[0].computationally_approachable is True


@pytest.mark.asyncio
async def test_embeddings_are_deterministic_and_similar_for_similar_text():
    client = EmbeddingClient(provider="offline", dim=64)
    e1 = await client.embed_one("F(n) is greater than 0.5 for all n")
    e2 = await client.embed_one("F(n) is greater than 0.5 for all n")
    e3 = await client.embed_one("completely unrelated statement about biology genes")
    from services.literature.app.indexer.embeddings import cosine_similarity

    assert e1 == e2
    assert cosine_similarity(e1, e2) > 0.99
    assert cosine_similarity(e1, e3) < cosine_similarity(e1, e2)
