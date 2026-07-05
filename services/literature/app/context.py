"""
Dependency container passed through the retriever layer.

Kept separate from ``pipeline.py`` (which constructs one) purely to avoid a
circular import: ``pipeline.py`` imports the ``retriever`` modules, and the
retriever modules need this type for their function signatures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.literature.app.extractors.claims import ClaimsExtractor
from services.literature.app.extractors.contradictions import ContradictionsExtractor
from services.literature.app.extractors.gaps import GapsExtractor
from services.literature.app.extractors.open_problems import OpenProblemsExtractor
from services.literature.app.extractors.tables import TablesExtractor
from services.literature.app.indexer.embeddings import EmbeddingClient
from services.literature.app.sources.base import BaseSource


@dataclass
class PipelineContext:
    sources: dict[str, BaseSource]
    embedder: EmbeddingClient
    vector_store: Any
    structured_store: Any
    claims_extractor: ClaimsExtractor = field(default_factory=ClaimsExtractor)
    tables_extractor: TablesExtractor = field(default_factory=TablesExtractor)
    open_problems_extractor: OpenProblemsExtractor = field(default_factory=OpenProblemsExtractor)
    contradictions_extractor: ContradictionsExtractor = field(default_factory=ContradictionsExtractor)
    gaps_extractor: GapsExtractor = field(default_factory=GapsExtractor)
    novelty_similarity_floor: float = 0.70
    novelty_top_k: int = 30
    novelty_confidence_verdict_floor: float = 0.85
    dedup_similarity_threshold: float = 0.95
    depth_docs: dict[str, int] = field(
        default_factory=lambda: {"standard": 8, "deep": 20, "exhaustive": 50}
    )
    # Used only by extractors/llm_claim_locator.py, and only for documents
    # the regex-based ClaimsExtractor finds nothing in (see
    # retriever/query.py::process_document). Empty api_key disables it —
    # the rest of the pipeline has no LLM step at all.
    llm_api_key: str = ""
    llm_model: str = ""
