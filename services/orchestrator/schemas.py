from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Prior:
    """Structured prior aligned with ARCHITECTURE §5.4."""

    established_facts: list[dict]
    contested_claims: list[dict]
    open_gaps: list[dict]
    dead_ends: list[dict]
    key_papers: list[dict]
    evidence_status: str = "READY"
    evidence_coverage: float = 0.0
    retrieval_diagnostics: dict | None = None

    def to_dict(self) -> dict:
        out = {
            "established_facts": self.established_facts,
            "contested_claims": self.contested_claims,
            "open_gaps": self.open_gaps,
            "dead_ends": self.dead_ends,
            "key_papers": self.key_papers,
            "evidence_status": self.evidence_status,
            "evidence_coverage": self.evidence_coverage,
        }
        if self.retrieval_diagnostics is not None:
            out["retrieval_diagnostics"] = self.retrieval_diagnostics
        return out


@dataclass(slots=True)
class RankedHypothesis:
    """Hypothesis with §6.2 ranking dimensions."""

    id: str
    text: str
    test_methodology: str
    scores: dict[str, float]
    rank: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "test_methodology": self.test_methodology,
            "scores": self.scores,
            "rank": self.rank,
        }
