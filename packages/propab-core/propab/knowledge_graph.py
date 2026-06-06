"""
Phase A — reusable knowledge objects that survive across campaigns.

Claim, Mechanism, Failure, Theory, Question nodes in a persisted graph.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from propab.config import settings

KNOWLEDGE_VERSION = 1


def knowledge_store_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "knowledge_graph.json"


@dataclass
class Claim:
    id: str
    text: str
    verdict: str
    theme: str
    confidence: float = 0.0
    replication_level: str = "T1"
    campaign_id: str | None = None
    claim_type: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MechanismRecord:
    id: str
    claim_id: str
    cause: str
    effect: str
    conditions: str
    failure_modes: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    campaign_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FailureRecord:
    id: str
    text: str
    reason: str
    failure_signature: str | None
    theme: str
    verdict: str
    campaign_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Theory:
    id: str
    name: str
    assumptions: list[str]
    mechanism_summary: str
    predictions: list[str]
    failure_regions: list[str]
    supporting_claim_ids: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchQuestion:
    id: str
    text: str
    source_campaign_id: str | None = None
    priority: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeGraph:
    """Cross-campaign knowledge store."""

    version: int = KNOWLEDGE_VERSION
    claims: dict[str, Claim] = field(default_factory=dict)
    mechanisms: dict[str, MechanismRecord] = field(default_factory=dict)
    failures: dict[str, FailureRecord] = field(default_factory=dict)
    theories: dict[str, Theory] = field(default_factory=dict)
    questions: dict[str, ResearchQuestion] = field(default_factory=dict)
    links: list[dict[str, str]] = field(default_factory=list)
    campaign_ids: list[str] = field(default_factory=list)

    def add_claim(self, claim: Claim) -> None:
        self.claims[claim.id] = claim

    def add_failure(self, failure: FailureRecord) -> None:
        self.failures[failure.id] = failure

    def add_mechanism(self, mech: MechanismRecord) -> None:
        self.mechanisms[mech.id] = mech

    def add_theory(self, theory: Theory) -> None:
        self.theories[theory.id] = theory

    def link(self, src: str, dst: str, relation: str) -> None:
        self.links.append({"src": src, "dst": dst, "relation": relation})

    def dead_end_texts(self, *, limit: int = 40) -> list[str]:
        out: list[str] = []
        for f in self.failures.values():
            out.append(f"{f.text} [{f.reason}]")
            if len(out) >= limit:
                break
        return out

    def established_fact_texts(self, *, limit: int = 30) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        for c in sorted(self.claims.values(), key=lambda x: -x.confidence):
            if c.verdict != "confirmed":
                continue
            facts.append({
                "text": c.text[:500],
                "confidence": c.confidence,
                "paper_ids": [],
                "theme": c.theme,
                "replication_level": c.replication_level,
            })
            if len(facts) >= limit:
                break
        return facts

    def theme_success_rates(
        self,
        *,
        campaign_ids: set[str] | None = None,
    ) -> dict[str, float]:
        """Fraction confirmed among tested claims per theme (optional bucket filter)."""
        by_theme: dict[str, list[str]] = {}
        for c in self.claims.values():
            if campaign_ids is not None and c.campaign_id and c.campaign_id not in campaign_ids:
                continue
            by_theme.setdefault(c.theme, []).append(c.verdict)
        rates: dict[str, float] = {}
        for theme, verdicts in by_theme.items():
            if not verdicts:
                continue
            rates[theme] = sum(1 for v in verdicts if v == "confirmed") / len(verdicts)
        return rates

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "claims": {k: v.to_dict() for k, v in self.claims.items()},
            "mechanisms": {k: v.to_dict() for k, v in self.mechanisms.items()},
            "failures": {k: v.to_dict() for k, v in self.failures.items()},
            "theories": {k: v.to_dict() for k, v in self.theories.items()},
            "questions": {k: v.to_dict() for k, v in self.questions.items()},
            "links": self.links,
            "campaign_ids": self.campaign_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        g = cls(version=int(data.get("version") or KNOWLEDGE_VERSION))
        for k, v in (data.get("claims") or {}).items():
            g.claims[k] = Claim(**v)
        for k, v in (data.get("mechanisms") or {}).items():
            g.mechanisms[k] = MechanismRecord(**v)
        for k, v in (data.get("failures") or {}).items():
            g.failures[k] = FailureRecord(**v)
        for k, v in (data.get("theories") or {}).items():
            g.theories[k] = Theory(**v)
        for k, v in (data.get("questions") or {}).items():
            g.questions[k] = ResearchQuestion(**v)
        g.links = list(data.get("links") or [])
        g.campaign_ids = list(data.get("campaign_ids") or [])
        return g

    def save(self, path: Path | None = None) -> Path:
        p = path or knowledge_store_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> KnowledgeGraph:
        p = path or knowledge_store_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return cls()


def new_id(prefix: str = "kg") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
