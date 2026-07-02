"""Belief objects for campaign-level synthesis (fixes.md §3.1b–3.1c)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ConfidenceLevel = Literal["strong", "weak", "unclear"]
BeliefStatus = Literal["active", "strengthened", "weakened", "abandoned"]

MAX_ACTIVE_BELIEFS = 3
RIVAL_MAX_ACTIVE_BELIEFS = 2
EXHAUSTION_ROUNDS_REQUIRED = 3


@dataclass
class BeliefObject:
    statement: str
    confidence: ConfidenceLevel = "unclear"
    supporting_nodes: list[str] = field(default_factory=list)
    contradicting_nodes: list[str] = field(default_factory=list)
    status: BeliefStatus = "active"
    exhaustion_rounds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "confidence": self.confidence,
            "supporting_nodes": list(self.supporting_nodes),
            "contradicting_nodes": list(self.contradicting_nodes),
            "status": self.status,
            "exhaustion_rounds": self.exhaustion_rounds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BeliefObject:
        return cls(
            statement=str(data.get("statement") or ""),
            confidence=data.get("confidence") or "unclear",  # type: ignore[arg-type]
            supporting_nodes=list(data.get("supporting_nodes") or []),
            contradicting_nodes=list(data.get("contradicting_nodes") or []),
            status=data.get("status") or "active",  # type: ignore[arg-type]
            exhaustion_rounds=int(data.get("exhaustion_rounds") or 0),
        )


@dataclass
class ClosedBelief:
    statement: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"statement": self.statement, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClosedBelief:
        return cls(
            statement=str(data.get("statement") or ""),
            reason=str(data.get("reason") or ""),
        )


@dataclass
class CampaignBeliefState:
    """Pinned belief context for a campaign branch (never summarized away)."""

    active_beliefs: list[BeliefObject] = field(default_factory=list)
    closed_beliefs: list[ClosedBelief] = field(default_factory=list)
    human_messages: list[str] = field(default_factory=list)
    recent_activity_summary: str = ""
    results_since_last_synthesis: int = 0
    exhaustion_rounds: int = 0
    branch_exhausted: bool = False
    rival_exhaustion_mode: bool = False
    last_synthesis_node_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_beliefs": [b.to_dict() for b in self.active_beliefs],
            "closed_beliefs": [c.to_dict() for c in self.closed_beliefs],
            "human_messages": list(self.human_messages[-10:]),
            "recent_activity_summary": self.recent_activity_summary,
            "results_since_last_synthesis": self.results_since_last_synthesis,
            "exhaustion_rounds": self.exhaustion_rounds,
            "branch_exhausted": self.branch_exhausted,
            "rival_exhaustion_mode": self.rival_exhaustion_mode,
            "last_synthesis_node_ids": list(self.last_synthesis_node_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CampaignBeliefState:
        if not data:
            return cls()
        return cls(
            active_beliefs=[BeliefObject.from_dict(b) for b in (data.get("active_beliefs") or [])],
            closed_beliefs=[ClosedBelief.from_dict(c) for c in (data.get("closed_beliefs") or [])],
            human_messages=list(data.get("human_messages") or [])[-10:],
            recent_activity_summary=str(data.get("recent_activity_summary") or ""),
            results_since_last_synthesis=int(data.get("results_since_last_synthesis") or 0),
            exhaustion_rounds=int(data.get("exhaustion_rounds") or 0),
            branch_exhausted=bool(data.get("branch_exhausted")),
            rival_exhaustion_mode=bool(data.get("rival_exhaustion_mode")),
            last_synthesis_node_ids=list(data.get("last_synthesis_node_ids") or []),
        )

    def add_human_message(self, message: str) -> None:
        msg = (message or "").strip()
        if not msg:
            return
        self.human_messages.append(msg)
        if len(self.human_messages) > 10:
            self.human_messages = self.human_messages[-10:]

    def abandon_belief(self, belief: BeliefObject, reason: str) -> None:
        belief.status = "abandoned"
        self.active_beliefs = [b for b in self.active_beliefs if b.statement != belief.statement]
        self.closed_beliefs.append(ClosedBelief(statement=belief.statement, reason=reason))

    def apply_synthesis_beliefs(
        self,
        raw_beliefs: list[dict[str, Any]],
        *,
        tree_nodes: dict[str, Any] | None = None,
        dataset_feature_count: int = 98,
        dataset_n_samples: int = 56,
        metrics: Any | None = None,
    ) -> None:
        """Merge synthesis output into active beliefs, with Evidence Binding + admission gates."""
        from propab.evidence_binding import (
            RIVAL_MAX_ACTIVE_BELIEFS,
            belief_falsifiable_in_dataset,
            filter_node_citations,
        )

        cap = RIVAL_MAX_ACTIVE_BELIEFS if self.rival_exhaustion_mode else MAX_ACTIVE_BELIEFS
        nodes = tree_nodes or {}
        updated: list[BeliefObject] = []
        for item in raw_beliefs:
            if not isinstance(item, dict):
                continue
            stmt = str(item.get("statement") or "").strip()
            if not stmt:
                continue
            conf = item.get("confidence") or "unclear"
            if conf not in ("strong", "weak", "unclear"):
                conf = "unclear"
            status = item.get("status") or "active"
            if status not in ("active", "strengthened", "weakened", "abandoned"):
                status = "active"
            if status == "abandoned":
                self.closed_beliefs.append(ClosedBelief(
                    statement=stmt,
                    reason=str(item.get("abandon_reason") or "abandoned by synthesis"),
                ))
                continue

            falsifiable, f_reason = belief_falsifiable_in_dataset(
                stmt,
                feature_count=dataset_feature_count,
                n_samples=dataset_n_samples,
            )
            if not falsifiable:
                if metrics is not None:
                    metrics.falsifiability_rejected_count = int(
                        getattr(metrics, "falsifiability_rejected_count", 0) or 0,
                    ) + 1
                    metrics.rejection_reasons.append(f"falsifiability:{f_reason}")
                continue

            if len(updated) >= cap:
                if metrics is not None:
                    metrics.belief_cap_rejected_count = int(
                        getattr(metrics, "belief_cap_rejected_count", 0) or 0,
                    ) + 1
                    metrics.rejection_reasons.append(f"belief_cap:{stmt[:80]}")
                continue

            raw_sup = [str(x) for x in (item.get("supporting_nodes") or [])]
            raw_con = [str(x) for x in (item.get("contradicting_nodes") or [])]
            if nodes:
                supporting = filter_node_citations(stmt, raw_sup, nodes, metrics=metrics)
                contradicting = filter_node_citations(stmt, raw_con, nodes, metrics=metrics)
            else:
                supporting = raw_sup
                contradicting = raw_con

            updated.append(BeliefObject(
                statement=stmt,
                confidence=conf,  # type: ignore[arg-type]
                supporting_nodes=supporting,
                contradicting_nodes=contradicting,
                status=status,  # type: ignore[arg-type]
            ))
        self.active_beliefs = updated[:cap]

    def check_exhaustion(self) -> bool:
        """Section 4.2: no belief above unclear and no new belief introduced."""
        if not self.active_beliefs:
            return True
        return all(b.confidence == "unclear" for b in self.active_beliefs)

    def record_exhaustion_round(self, exhausted_this_round: bool) -> None:
        if exhausted_this_round:
            self.exhaustion_rounds += 1
        else:
            self.exhaustion_rounds = 0
        if self.exhaustion_rounds >= EXHAUSTION_ROUNDS_REQUIRED:
            self.branch_exhausted = True
            for b in list(self.active_beliefs):
                if b.confidence == "unclear":
                    self.abandon_belief(
                        b,
                        "exhausted — no belief above unclear after three rounds",
                    )

    def record_synthesis_exhaustion(self, direction_exhausted: bool) -> None:
        """Section 4.2 — global or per-rival exhaustion after a synthesis pass."""
        if self.rival_exhaustion_mode and len(self.active_beliefs) >= 2:
            rivals = self.active_beliefs[:2]
            for b in rivals:
                if direction_exhausted and b.confidence == "unclear":
                    b.exhaustion_rounds += 1
                else:
                    b.exhaustion_rounds = 0
            if all(b.exhaustion_rounds >= EXHAUSTION_ROUNDS_REQUIRED for b in rivals):
                self.branch_exhausted = True
                for b in rivals:
                    if b.exhaustion_rounds >= EXHAUSTION_ROUNDS_REQUIRED:
                        self.abandon_belief(
                            b,
                            "exhausted — no discriminating progress after three rounds",
                        )
            return

        exhausted = self.check_exhaustion() and direction_exhausted
        self.record_exhaustion_round(exhausted)
