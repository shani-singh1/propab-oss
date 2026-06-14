"""Policy objects as hypotheses — never auto-promoted."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from propab.search_policy import SearchPolicy


class PolicyStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


@dataclass
class PredictedEffects:
    closure_ratio_delta: float = 0.0
    compute_efficiency_delta: float = 0.0
    refute_ratio_delta: float = 0.0
    # V2 entropy dynamics (replaces theme_entropy_delta per fixes.md P1)
    start_H: float = 0.0
    growth_rate: float = 0.0
    saturation_H: float = 0.0
    cross_H_1_5_at_tested: float = 0.0
    cross_H_2_0_at_tested: float = 0.0
    theme_entropy_delta: float = 0.0  # legacy — ignored when V2 fields are set

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def uses_entropy_dynamics(self) -> bool:
        return any(
            getattr(self, k, 0) != 0
            for k in (
                "start_H",
                "growth_rate",
                "saturation_H",
                "cross_H_1_5_at_tested",
                "cross_H_2_0_at_tested",
            )
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PredictedEffects:
        if not data:
            return cls()
        fields = {k: float(data.get(k) or 0) for k in cls.__dataclass_fields__}
        return cls(**fields)


@dataclass
class PolicyRecord:
    """Full policy schema — parameters mutated only by deterministic engine."""

    id: str
    generation: int
    parent_policy_id: str | None
    budget_bucket: str
    domain_bucket: str
    boosts: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    blocked_failures: list[str] = field(default_factory=list)
    saturated_themes: list[str] = field(default_factory=list)
    rationale: str = ""
    predicted_effects: PredictedEffects = field(default_factory=PredictedEffects)
    falsification_conditions: list[str] = field(default_factory=list)
    status: PolicyStatus = PolicyStatus.CANDIDATE
    prefer_replication_t2_plus: bool = True
    closure_target: float = 0.35
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["predicted_effects"] = self.predicted_effects.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyRecord:
        status = PolicyStatus(data.get("status") or PolicyStatus.CANDIDATE.value)
        return cls(
            id=str(data["id"]),
            generation=int(data.get("generation") or 0),
            parent_policy_id=data.get("parent_policy_id"),
            budget_bucket=str(data.get("budget_bucket") or "general"),
            domain_bucket=str(data.get("domain_bucket") or "general"),
            boosts=dict(data.get("boosts") or data.get("theme_boost") or {}),
            penalties=dict(data.get("penalties") or data.get("theme_penalty") or {}),
            blocked_failures=list(data.get("blocked_failures") or data.get("blocked_failure_signatures") or []),
            saturated_themes=list(data.get("saturated_themes") or []),
            rationale=str(data.get("rationale") or ""),
            predicted_effects=PredictedEffects.from_dict(data.get("predicted_effects")),
            falsification_conditions=list(data.get("falsification_conditions") or []),
            status=status,
            prefer_replication_t2_plus=bool(data.get("prefer_replication_t2_plus", True)),
            closure_target=float(data.get("closure_target") or 0.35),
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
        )

    def to_search_policy(self) -> SearchPolicy:
        """Runtime adapter for existing frontier / prior wiring."""
        return SearchPolicy(
            version=2,
            generation=self.generation,
            theme_boost=dict(self.boosts),
            theme_penalty=dict(self.penalties),
            blocked_failure_signatures=list(self.blocked_failures),
            saturated_themes=list(self.saturated_themes),
            prefer_replication_t2_plus=self.prefer_replication_t2_plus,
            closure_target=self.closure_target,
            notes=[f"policy_id={self.id} status={self.status.value}"],
        )

    @classmethod
    def empty_accepted(cls, *, budget_bucket: str, domain_bucket: str) -> PolicyRecord:
        from propab.knowledge_graph import new_id

        return cls(
            id=new_id("pol"),
            generation=0,
            parent_policy_id=None,
            budget_bucket=budget_bucket,
            domain_bucket=domain_bucket,
            status=PolicyStatus.ACCEPTED,
            rationale="Initial accepted policy (empty).",
        )
