"""
Phase A — reusable knowledge objects that survive across campaigns.

Claim, Mechanism, Failure, Theory, Question nodes in a persisted graph.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from propab.config import settings

logger = logging.getLogger(__name__)

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
    numerical_seeds: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    diversity_by_domain: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)

    def store_numerical_seeds(
        self,
        domain: str,
        campaign_id: str,
        seeds: list[dict[str, Any]],
    ) -> None:
        bucket = list(self.numerical_seeds.get(domain) or [])
        for seed in seeds:
            entry = dict(seed)
            entry["campaign_id"] = campaign_id
            bucket.append(entry)
        self.numerical_seeds[domain] = bucket[-200:]

    def get_numerical_seeds(self, domain: str, *, limit: int = 30) -> list[dict[str, Any]]:
        return list((self.numerical_seeds.get(domain) or [])[-limit:])

    def store_diversity_distribution(
        self,
        domain: str,
        distribution: dict[str, dict[str, int]],
    ) -> None:
        self.diversity_by_domain[domain] = distribution

    def get_diversity_distribution(self, domain: str) -> dict[str, dict[str, int]]:
        return dict(self.diversity_by_domain.get(domain) or {})

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

    # LL6 promotion gate: a confirmed claim only becomes an unqualified
    # "established fact" (surfaced to every future campaign as settled prior)
    # if it is either replicated (T2+) OR carries a solid confidence floor.
    # A single-campaign, unreplicated, confidence-0 claim (T1) stays in the
    # graph but is NOT promoted — it is not settled knowledge.
    MIN_ESTABLISHED_CONFIDENCE = 0.6
    _REPLICATED_LEVELS = frozenset({"T2", "T3"})

    def _is_established(self, claim: Claim) -> bool:
        if claim.verdict != "confirmed":
            return False
        level = str(claim.replication_level or "").strip().upper()
        if level in self._REPLICATED_LEVELS:
            return True
        return float(claim.confidence or 0.0) >= self.MIN_ESTABLISHED_CONFIDENCE

    def established_fact_texts(self, *, limit: int = 30) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        for c in sorted(self.claims.values(), key=lambda x: -x.confidence):
            if not self._is_established(c):
                continue
            # Carry provenance so a promoted fact is traceable back to its
            # source campaign/claim instead of the untraceable `[]` (LL6).
            provenance = [p for p in (c.campaign_id, c.id) if p]
            facts.append({
                "text": c.text[:500],
                "confidence": c.confidence,
                "paper_ids": provenance,
                "campaign_id": c.campaign_id,
                "claim_id": c.id,
                "theme": c.theme,
                "replication_level": c.replication_level,
            })
            if len(facts) >= limit:
                break
        return facts

    def theme_counts(
        self,
        *,
        campaign_ids: set[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Confirmed vs failed counts per theme (optional bucket filter).

        ``self.claims`` holds ONLY confirmed claims — refuted/inconclusive
        outcomes are persisted as ``FailureRecord``s in ``self.failures``.
        A theme's honest denominator is therefore confirmed claims PLUS the
        failures recorded against that same theme. Both confirmed claims and
        failures are themed from the node's ``primary_theme`` (see
        ``negative_knowledge.extract_confirmed_claims`` /
        ``extract_failures_from_campaign``), so they share one theme namespace.
        """
        counts: dict[str, dict[str, int]] = {}

        def _bucket(theme: str) -> dict[str, int]:
            return counts.setdefault(theme, {"confirmed": 0, "failed": 0})

        for c in self.claims.values():
            if campaign_ids is not None and c.campaign_id and c.campaign_id not in campaign_ids:
                continue
            b = _bucket(c.theme)
            if c.verdict == "confirmed":
                b["confirmed"] += 1
            else:
                b["failed"] += 1
        for f in self.failures.values():
            if campaign_ids is not None and f.campaign_id and f.campaign_id not in campaign_ids:
                continue
            _bucket(f.theme)["failed"] += 1
        return counts

    def theme_success_rates(
        self,
        *,
        campaign_ids: set[str] | None = None,
    ) -> dict[str, float]:
        """Fraction confirmed among tested outcomes per theme (optional bucket filter).

        Denominator is confirmed claims + matching failures for the theme, so a
        theme that mostly fails yields a rate well below 1.0 (the previously
        structurally-1.0 statistic that made the penalty branch dead code).
        """
        rates: dict[str, float] = {}
        for theme, c in self.theme_counts(campaign_ids=campaign_ids).items():
            total = c["confirmed"] + c["failed"]
            if total <= 0:
                continue
            rates[theme] = c["confirmed"] / total
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
            "numerical_seeds": self.numerical_seeds,
            "diversity_by_domain": self.diversity_by_domain,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        g = cls(version=int(data.get("version") or KNOWLEDGE_VERSION))

        def _fields(record_cls: type, rec: dict[str, Any]) -> dict[str, Any]:
            # Drop keys not in the current dataclass so a drifted/renamed field
            # on one persisted record can never TypeError and wipe the store.
            return {k: v for k, v in rec.items() if k in record_cls.__dataclass_fields__}

        for k, v in (data.get("claims") or {}).items():
            g.claims[k] = Claim(**_fields(Claim, v))
        for k, v in (data.get("mechanisms") or {}).items():
            g.mechanisms[k] = MechanismRecord(**_fields(MechanismRecord, v))
        for k, v in (data.get("failures") or {}).items():
            g.failures[k] = FailureRecord(**_fields(FailureRecord, v))
        for k, v in (data.get("theories") or {}).items():
            g.theories[k] = Theory(**_fields(Theory, v))
        for k, v in (data.get("questions") or {}).items():
            g.questions[k] = ResearchQuestion(**_fields(ResearchQuestion, v))
        g.links = list(data.get("links") or [])
        g.campaign_ids = list(data.get("campaign_ids") or [])
        g.numerical_seeds = dict(data.get("numerical_seeds") or {})
        g.diversity_by_domain = dict(data.get("diversity_by_domain") or {})
        return g

    def save(self, path: Path | None = None) -> Path:
        from propab.lifetime_postgres import lifetime_postgres_enabled, save_knowledge_graph

        if lifetime_postgres_enabled():
            save_knowledge_graph(self)
            p = knowledge_store_path()
            return p
        p = path or knowledge_store_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> KnowledgeGraph:
        from propab.lifetime_postgres import lifetime_postgres_enabled, load_knowledge_graph

        if lifetime_postgres_enabled():
            try:
                return load_knowledge_graph()
            except Exception:
                pass
        p = path or knowledge_store_path()
        if not p.is_file():
            # Legitimate first run: no store yet. An empty graph here is safe
            # because there is nothing on disk to clobber.
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            # An EXISTING store failed to load. Returning an empty graph would
            # let the caller's next save() overwrite accumulated cross-campaign
            # knowledge with nothing — a silent, permanent wipe (LL3). Fail
            # closed: log loudly and re-raise so callers abort before saving,
            # leaving the on-disk file intact for inspection/recovery.
            #
            # Drifted/renamed record fields no longer reach here — from_dict
            # field-filters them — so this path now means genuine corruption
            # (unreadable file or invalid JSON), which must not be masked.
            logger.error(
                "Refusing to load lifetime knowledge graph from %s: %s. "
                "Not returning an empty graph (would clobber accumulated "
                "knowledge on next save); re-raising to fail closed.",
                p,
                exc,
                exc_info=True,
            )
            raise


def new_id(prefix: str = "kg") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
