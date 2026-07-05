"""
Shared data contracts for the literature intelligence service.

Every source, extractor, and retriever component speaks these types. Domain
knowledge never appears here — this module only knows about papers, claims,
tables, and citations in the abstract.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Documents (what sources produce)
# --------------------------------------------------------------------------


class RawDocument(BaseModel):
    """A search hit before full text has been fetched."""

    source: str
    external_id: str  # arxiv_id, doi, question_id, sequence id, etc.
    title: str = ""
    authors: str = ""
    year: int = 0
    doi: str = ""
    url: str = ""
    abstract: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class FullTextDocument(BaseModel):
    """A document with retrievable full text, ready for extraction."""

    source: str
    external_id: str
    title: str = ""
    authors: str = ""
    year: int = 0
    doi: str = ""
    url: str = ""
    sections: dict[str, str] = Field(default_factory=dict)  # section title -> text
    body_text: str = ""  # full concatenated text, used for linguistic scanning
    latex_environments: list[dict[str, Any]] = Field(default_factory=list)
    tables_raw: list[dict[str, Any]] = Field(default_factory=list)
    footnotes: list[str] = Field(default_factory=list)
    captions: list[str] = Field(default_factory=list)
    bibliography: list[dict[str, Any]] = Field(default_factory=list)
    cite_sentences: list[dict[str, Any]] = Field(default_factory=list)
    extraction_method: str = ""  # "latex" | "pdf_nougat" | "pdf_text" | "abstract_only"
    extraction_quality: float = 0.0  # environments found / estimated paragraphs
    is_appendix_included: bool = False


class TabMatch(BaseModel):
    """Result of checking candidate values against a source's tabulations."""

    source: str
    identifier: str
    matched: bool
    matched_index: Any = None
    matched_value: Any = None
    url: str = ""


# --------------------------------------------------------------------------
# Extracted knowledge (what extractors produce)
# --------------------------------------------------------------------------

ClaimType = Literal[
    "theorem", "lemma", "conjecture", "open_problem", "observation", "remark",
    "corollary", "proposition", "definition", "example", "claim", "footnote_claim",
    "caption_claim", "bibliography_annotation", "proof_intermediate",
]
ClaimStatus = Literal["proven", "conjectured", "open", "disproven", "unknown"]


class ExtractedClaim(BaseModel):
    text: str
    claim_type: ClaimType
    status: ClaimStatus
    verbatim: str
    source: str = ""  # which BaseSource produced this ("arxiv", "oeis", ...) — used for citation re-verification
    source_doi: str = ""
    source_title: str = ""
    source_authors: str = ""
    source_year: int = 0
    source_url: str = ""
    location: str = ""
    embedding: list[float] = Field(default_factory=list)
    claim_id: str = ""


class TabulatedSequence(BaseModel):
    description: str
    index_variable: str = "n"
    value_variable: str = ""
    values: dict[str, Any] = Field(default_factory=dict)  # keys are str(index) — JSON-safe
    max_index: float | None = None
    min_index: float | None = None
    source_doi: str = ""
    source_title: str = ""
    source_year: int = 0
    location: str = ""
    is_in_appendix: bool = False
    is_in_supplementary: bool = False
    oeis_match: str | None = None
    extraction_confidence: float = 1.0


class OpenProblem(BaseModel):
    statement: str
    source_doi: str = ""
    source_title: str = ""
    stated_by: str = ""
    year: int = 0
    context: str = ""
    known_partial_results: list[str] = Field(default_factory=list)
    computationally_approachable: bool = False
    approachable_angle: str = ""


class Contradiction(BaseModel):
    claim_a: ExtractedClaim
    claim_b: ExtractedClaim
    contradiction_type: Literal["direct", "implied", "superseded"] = "direct"
    resolution: str | None = None
    requires_investigation: bool = True


class KnowledgeGap(BaseModel):
    description: str
    what_is_known: str = ""
    what_is_open: str = ""
    best_known_bound: str = ""
    last_progress: int = 0
    computationally_approachable: bool = False
    approachable_angle: str = ""


# --------------------------------------------------------------------------
# API contract (Output contract per agent3.md)
# --------------------------------------------------------------------------


class NoveltyBar(BaseModel):
    criteria: str
    tabulated_ceiling: dict[str, Any] = Field(default_factory=dict)
    established_bounds: list[str] = Field(default_factory=list)


class PriorRequest(BaseModel):
    research_question: str
    domain_id: str
    literature_profile: dict[str, Any] = Field(default_factory=dict)
    depth: Literal["standard", "deep", "exhaustive"] = "standard"


class PriorResponse(BaseModel):
    established_facts: list[ExtractedClaim] = Field(default_factory=list)
    open_gaps: list[KnowledgeGap] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    dead_ends: list[ExtractedClaim] = Field(default_factory=list)
    tabulated_values: list[TabulatedSequence] = Field(default_factory=list)
    novelty_bar: NoveltyBar
    sources_consulted: list[str] = Field(default_factory=list)
    papers_indexed: int = 0
    citation_verification_rate: float | None = None


class Finding(BaseModel):
    claim: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    domain_id: str = ""


class NoveltyRequest(BaseModel):
    finding: Finding
    literature_profile: dict[str, Any] = Field(default_factory=dict)


class NoveltyResponse(BaseModel):
    verdict: Literal["known", "novel", "uncertain"]
    confidence: float
    explanation: str
    matching_sources: list[dict[str, Any]] = Field(default_factory=list)
    recommendation: str = ""


class GapsRequest(BaseModel):
    domain_id: str
    literature_profile: dict[str, Any] = Field(default_factory=dict)


class GapsResponse(BaseModel):
    frontier_questions: list[OpenProblem] = Field(default_factory=list)


class IngestRequest(BaseModel):
    doi_or_arxiv_id: str
    domain_id: str = ""
    literature_profile: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    claims_extracted: int
    tables_extracted: int
    open_problems_found: int
    extraction_quality: float = 0.0
    extraction_method: str = ""


class CoverageDomain(BaseModel):
    domain_id: str
    papers_indexed: int
    claims_indexed: int
    last_updated: str = ""


class CoverageResponse(BaseModel):
    domains: list[CoverageDomain] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    citation_verification_rate: float | None
    sources_healthy: dict[str, bool] = Field(default_factory=dict)
