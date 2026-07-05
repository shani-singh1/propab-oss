import pytest

from services.literature.app.indexer.embeddings import EmbeddingClient
from services.literature.app.models import FullTextDocument, RawDocument
from services.literature.app.retriever.chunk_rag import (
    Chunk,
    chunk_document,
    rank_chunks_bm25,
    retrieve_relevant_chunks,
)
from services.literature.app.sources.base import BaseSource


def make_doc(body_text: str, **kwargs) -> FullTextDocument:
    defaults = dict(source="pubmed", external_id="1", title="A Paper", year=2023)
    defaults.update(kwargs)
    return FullTextDocument(body_text=body_text, **defaults)


def _chunk(text: str) -> Chunk:
    return Chunk(text=text, source="pubmed", title="T", year=2023, url="")


class TestRankChunksBm25:
    def test_ranks_lexically_relevant_chunk_first(self):
        chunks = [
            _chunk("Gene expression varies across human tissues in large cohort studies."),
            _chunk("The MLH1dn dominant negative increases prime editing efficiency by 2.7 fold."),
            _chunk("Unrelated background on cell culture protocols and reagents used."),
        ]
        ranked = rank_chunks_bm25("By what factor does MLH1dn increase prime editing efficiency?", chunks, top_k=3)
        assert "MLH1dn" in ranked[0].text

    def test_empty_chunks_returns_empty(self):
        assert rank_chunks_bm25("q", [], top_k=5) == []

    def test_respects_top_k(self):
        chunks = [_chunk(f"chunk number {i} about genes and proteins") for i in range(10)]
        ranked = rank_chunks_bm25("genes proteins", chunks, top_k=3)
        assert len(ranked) == 3

    def test_chunks_with_no_tokens_do_not_crash(self):
        chunks = [_chunk("!!! ???"), _chunk("real content about editing efficiency here")]
        ranked = rank_chunks_bm25("editing efficiency", chunks, top_k=2)
        assert ranked  # does not raise, returns something


class TestChunkDocument:
    def test_splits_on_paragraph_boundaries(self):
        doc = make_doc(
            "First paragraph with enough content to count as a real chunk on its own merit.\n\n"
            "Second paragraph with enough content to count as a real chunk on its own merit too."
        )
        chunks = chunk_document(doc, max_chars=50, min_chars=10)
        assert len(chunks) == 2
        assert "First paragraph" in chunks[0].text
        assert "Second paragraph" in chunks[1].text

    def test_merges_short_paragraphs_up_to_max_chars(self):
        doc = make_doc("Short one.\n\nShort two.\n\nShort three.")
        chunks = chunk_document(doc, max_chars=500, min_chars=5)
        assert len(chunks) == 1
        assert "Short one" in chunks[0].text
        assert "Short three" in chunks[0].text

    def test_drops_chunks_below_min_chars(self):
        doc = make_doc("ok")
        chunks = chunk_document(doc, min_chars=80)
        assert chunks == []

    def test_empty_document_returns_no_chunks(self):
        assert chunk_document(make_doc("")) == []

    def test_oversized_single_paragraph_kept_whole_not_truncated(self):
        long_para = "word " * 500
        doc = make_doc(long_para.strip())
        chunks = chunk_document(doc, max_chars=100, min_chars=10)
        assert len(chunks) == 1
        assert chunks[0].text == long_para.strip()

    def test_chunk_carries_source_metadata(self):
        doc = make_doc("A sufficiently long paragraph to survive the min_chars filter here.", source="arxiv", title="T", year=2020, url="https://arxiv.org/abs/1")
        chunks = chunk_document(doc, min_chars=10)
        assert chunks[0].source == "arxiv"
        assert chunks[0].title == "T"
        assert chunks[0].year == 2020
        assert "arxiv" in chunks[0].format()


class _FakeSource(BaseSource):
    name = "fake"

    def __init__(self, raw_docs: list[RawDocument], full_texts: dict[str, FullTextDocument]) -> None:
        super().__init__()
        self._raw_docs = raw_docs
        self._full_texts = full_texts

    def is_relevant(self, profile) -> bool:
        return True

    async def search(self, query, profile):
        return list(self._raw_docs)

    async def fetch_full_text(self, doc):
        return self._full_texts[doc.external_id]


class TestRetrieveRelevantChunks:
    @pytest.mark.asyncio
    async def test_ranks_chunks_by_similarity_to_question_not_search_query(self):
        raw = [
            RawDocument(source="fake", external_id="1"),
            RawDocument(source="fake", external_id="2"),
        ]
        full_texts = {
            "1": make_doc(
                "Sidon sets have applications in coding theory and combinatorics broadly speaking here.\n\n"
                "The maximum Sidon set size in the integers up to n grows like the square root of n exactly.",
                source="fake",
            ),
            "2": make_doc(
                "Gene expression varies substantially across human tissues in large cohort studies overall.\n\n"
                "GTEx provides expression data across dozens of tissue types for thousands of donors nationwide.",
                source="fake",
            ),
        }
        sources = {"fake": _FakeSource(raw, full_texts)}
        embedder = EmbeddingClient(provider="offline", dim=64)

        chunks, sources_with_hits = await retrieve_relevant_chunks(
            sources, embedder,
            question="What is the growth rate of the maximum Sidon set size?",
            search_terms=["Sidon set"],
            profile={},
            top_k=2,
        )

        assert sources_with_hits == ["fake"]
        assert chunks
        # The top-ranked chunk should be about Sidon sets, not gene expression.
        assert "Sidon" in chunks[0].text

    @pytest.mark.asyncio
    async def test_empty_when_no_documents_found(self):
        sources = {"fake": _FakeSource([], {})}
        embedder = EmbeddingClient(provider="offline", dim=64)
        chunks, sources_with_hits = await retrieve_relevant_chunks(
            sources, embedder, question="q?", search_terms=["x"], profile={},
        )
        assert chunks == []
        assert sources_with_hits == []

    @pytest.mark.asyncio
    async def test_source_search_failure_does_not_crash(self):
        class _FailingSource(BaseSource):
            name = "failing"

            def is_relevant(self, profile):
                return True

            async def search(self, query, profile):
                raise RuntimeError("simulated failure")

        sources = {"failing": _FailingSource()}
        embedder = EmbeddingClient(provider="offline", dim=64)
        chunks, sources_with_hits = await retrieve_relevant_chunks(
            sources, embedder, question="q?", search_terms=["x"], profile={},
        )
        assert chunks == []
        assert sources_with_hits == []
