from unittest.mock import patch

import pytest

from services.literature.app.extractors.llm_claim_locator import (
    build_location_prompt,
    locate_claims,
    split_sentences,
)
from services.literature.app.models import FullTextDocument


def make_doc(body_text: str, **kwargs) -> FullTextDocument:
    defaults = dict(source="pubmed", external_id="123", title="Test Paper", year=2024)
    defaults.update(kwargs)
    return FullTextDocument(body_text=body_text, **defaults)


class TestSplitSentences:
    def test_splits_on_sentence_boundaries(self):
        sents = split_sentences("The protein binds ATP with high affinity. It also shows GTP hydrolysis activity.")
        assert sents == [
            "The protein binds ATP with high affinity.",
            "It also shows GTP hydrolysis activity.",
        ]

    def test_never_crosses_paragraph_boundary(self):
        # Mirrors the exact bug found in claims.py: a sentence-terminator
        # lookbehind can miss (".)" not followed by whitespace), and without
        # paragraph scoping this would splice the next paragraph's text in.
        text = "We found a strong effect (p < 0.01.)\n\nAnswer: this replicates prior work on the topic clearly."
        sents = split_sentences(text)
        assert not any("Answer:" in s and "p < 0.01" in s for s in sents)

    def test_drops_short_fragments(self):
        sents = split_sentences("Yes. This is a long enough sentence to count as real content here.")
        assert "Yes." not in sents


class TestBuildLocationPrompt:
    def test_numbers_sentences_and_forbids_paraphrase(self):
        prompt = build_location_prompt(["First sentence here.", "Second sentence here."])
        assert "[0] First sentence here." in prompt
        assert "[1] Second sentence here." in prompt
        assert "Do NOT restate" in prompt or "index only" in prompt


class TestLocateClaimsSafety:
    """The load-bearing property: verbatim is always a code-side lookup by
    index, never text the LLM supplies — even when the LLM ignores
    instructions and echoes a paraphrase back."""

    @pytest.mark.asyncio
    async def test_ignores_llm_supplied_text_field(self):
        doc = make_doc(
            "The maximum Sidon set size grows like the square root of n. "
            "This is an unrelated filler sentence with no claim content at all."
        )

        async def fake_gemini_generate(prompt, **kwargs):
            # The model disobeys instructions and includes a paraphrased
            # "text" field alongside the index — this must be discarded.
            return (
                '{"claims": [{"index": 0, "claim_type": "theorem", "status": "proven", '
                '"text": "F(n) is approximately sqrt(n), a totally different wording"}]}'
            )

        with patch(
            "services.literature.app.extractors.llm_claim_locator.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            claims = await locate_claims(doc, api_key="k", model="m")

        assert len(claims) == 1
        # verbatim must be the ORIGINAL sentence 0, not the LLM's paraphrase.
        assert claims[0].verbatim == "The maximum Sidon set size grows like the square root of n."
        assert "totally different wording" not in claims[0].verbatim

    @pytest.mark.asyncio
    async def test_out_of_range_index_is_skipped_not_crashed(self):
        doc = make_doc("Only one real sentence exists in this short document right here.")

        async def fake_gemini_generate(prompt, **kwargs):
            return '{"claims": [{"index": 0, "claim_type": "observation", "status": "proven"}, {"index": 99, "claim_type": "observation", "status": "proven"}]}'

        with patch(
            "services.literature.app.extractors.llm_claim_locator.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            claims = await locate_claims(doc, api_key="k", model="m")

        assert len(claims) == 1

    @pytest.mark.asyncio
    async def test_negative_index_is_skipped(self):
        doc = make_doc("Only one real sentence exists in this short document right here.")

        async def fake_gemini_generate(prompt, **kwargs):
            return '{"claims": [{"index": -1, "claim_type": "observation", "status": "proven"}]}'

        with patch(
            "services.literature.app.extractors.llm_claim_locator.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            claims = await locate_claims(doc, api_key="k", model="m")

        assert claims == []

    @pytest.mark.asyncio
    async def test_invalid_claim_type_falls_back_to_observation(self):
        doc = make_doc("The enzyme shows a tenfold increase in catalytic efficiency under these conditions.")

        async def fake_gemini_generate(prompt, **kwargs):
            return '{"claims": [{"index": 0, "claim_type": "not_a_real_type", "status": "not_a_real_status"}]}'

        with patch(
            "services.literature.app.extractors.llm_claim_locator.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            claims = await locate_claims(doc, api_key="k", model="m")

        assert claims[0].claim_type == "observation"
        assert claims[0].status == "unknown"

    @pytest.mark.asyncio
    async def test_duplicate_indices_deduped(self):
        doc = make_doc("A single clear factual claim appears exactly once in this test document.")

        async def fake_gemini_generate(prompt, **kwargs):
            return '{"claims": [{"index": 0, "claim_type": "observation", "status": "proven"}, {"index": 0, "claim_type": "observation", "status": "proven"}]}'

        with patch(
            "services.literature.app.extractors.llm_claim_locator.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            claims = await locate_claims(doc, api_key="k", model="m")

        assert len(claims) == 1

    @pytest.mark.asyncio
    async def test_unparseable_response_returns_empty(self):
        doc = make_doc("A single clear factual claim appears exactly once in this test document.")

        async def fake_gemini_generate(prompt, **kwargs):
            return "I cannot process this request."

        with patch(
            "services.literature.app.extractors.llm_claim_locator.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            claims = await locate_claims(doc, api_key="k", model="m")

        assert claims == []

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty_not_raises(self):
        doc = make_doc("A single clear factual claim appears exactly once in this test document.")

        async def failing_gemini_generate(prompt, **kwargs):
            raise RuntimeError("simulated API failure")

        with patch(
            "services.literature.app.extractors.llm_claim_locator.gemini_generate",
            side_effect=failing_gemini_generate,
        ):
            claims = await locate_claims(doc, api_key="k", model="m")

        assert claims == []

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty_without_calling_llm(self):
        doc = make_doc("A single clear factual claim appears exactly once in this test document.")

        with patch(
            "services.literature.app.extractors.llm_claim_locator.gemini_generate",
        ) as mock_gen:
            claims = await locate_claims(doc, api_key="", model="m")

        assert claims == []
        mock_gen.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_document_returns_empty(self):
        doc = make_doc("")
        claims = await locate_claims(doc, api_key="k", model="m")
        assert claims == []

    @pytest.mark.asyncio
    async def test_claim_carries_source_metadata(self):
        doc = make_doc(
            "The knockout mice showed a significant reduction in tumor growth rate.",
            source="pubmed", doi="10.1/xyz", title="Some Paper", authors="A. Author", year=2022,
            url="https://pubmed.ncbi.nlm.nih.gov/123/",
        )

        async def fake_gemini_generate(prompt, **kwargs):
            return '{"claims": [{"index": 0, "claim_type": "observation", "status": "proven"}]}'

        with patch(
            "services.literature.app.extractors.llm_claim_locator.gemini_generate",
            side_effect=fake_gemini_generate,
        ):
            claims = await locate_claims(doc, api_key="k", model="m")

        assert claims[0].source == "pubmed"
        assert claims[0].source_doi == "10.1/xyz"
        assert claims[0].source_title == "Some Paper"
        assert claims[0].source_year == 2022
        assert "LLM-located" in claims[0].location
