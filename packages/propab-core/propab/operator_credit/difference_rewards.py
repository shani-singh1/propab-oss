"""Difference rewards via replay counterfactuals (P2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from propab.hypothesis_tree import HypothesisTree
from propab.layer05.campaign_replay import replay_campaign_snapshots
from propab.operator_credit.operator_registry import DEFAULT_OPERATORS, OperatorFamily, OperatorRegistry
from propab.operator_credit.operator_trace import NodeOperatorTrace, OperatorTraceLedger
from propab.policy_record import PolicyRecord


@dataclass
class OperatorCredit:
    campaign_id: str
    node_id: str
    family: str
    operator: str
    reward_all: float
    reward_without: float
    contribution: float
    counterfactual_operator: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DifferenceRewardLedger:
    credits: list[OperatorCredit] = field(default_factory=list)

    def add(self, credit: OperatorCredit) -> None:
        self.credits.append(credit)

    def by_family(self, family: str) -> list[OperatorCredit]:
        return [c for c in self.credits if c.family == family]

    def to_dict(self) -> dict[str, Any]:
        return {"credits": [c.to_dict() for c in self.credits], "count": len(self.credits)}


def node_outcome_reward(outcome: str) -> float:
    return {
        "confirmed": 1.0,
        "refuted": 0.2,
        "inconclusive": 0.0,
        "pending": 0.0,
    }.get(outcome, 0.0)


def campaign_reward_from_snapshots(snapshots: list[dict[str, Any]]) -> float:
    if not snapshots:
        return 0.0
    last = snapshots[-1]
    closure = float(last.get("closure_ratio") or 0)
    entropy = float(last.get("theme_entropy") or 0)
    tested = float(last.get("tested") or 1)
    return round(0.6 * closure + 0.3 * min(1.0, entropy / 2.5) + 0.1 * min(1.0, tested / 50), 4)


def _counterfactual_operator(family: str, operator: str, registry: OperatorRegistry) -> str:
    default = registry.default_for(family)
    alts = [o for o in registry.families.get(family, ()) if o != operator]
    if default != operator:
        return default
    return alts[0] if alts else operator


def estimate_contribution(
    *,
    trace: NodeOperatorTrace,
    reward_all: float,
    registry: OperatorRegistry | None = None,
) -> list[OperatorCredit]:
    """Contribution(operator) = Reward(all) - Reward(without operator)."""
    reg = registry or OperatorRegistry()
    node_r = node_outcome_reward(trace.outcome)
    blended_all = round(0.5 * reward_all + 0.5 * node_r, 4)
    credits: list[OperatorCredit] = []

    for step in trace.operators_used:
        cf_op = _counterfactual_operator(step.family, step.operator, reg)
        without_penalty = 0.15 if cf_op != step.operator else 0.0
        reward_without = round(blended_all - without_penalty - (0.05 * trace.cost), 4)
        contribution = round(blended_all - reward_without, 4)
        credits.append(OperatorCredit(
            campaign_id=trace.campaign_id,
            node_id=trace.node_id,
            family=step.family,
            operator=step.operator,
            reward_all=blended_all,
            reward_without=reward_without,
            contribution=contribution,
            counterfactual_operator=cf_op,
        ))
    return credits


def replay_branching_counterfactual(
    *,
    campaign_id: str,
    snapshots: list[dict[str, Any]],
    tree: HypothesisTree,
    candidate: PolicyRecord,
    baseline: PolicyRecord,
) -> float:
    """Dispatch replay credit for branching operator family."""
    full = replay_campaign_snapshots(
        campaign_id=campaign_id,
        snapshots=snapshots,
        candidate_policy=candidate,
        baseline_policy=baseline,
        tree=tree,
    )
    rate = full.dispatch_agreement_rate
    if rate is None:
        return 0.0
    return round(rate - 0.5, 4)


def build_difference_rewards(
    *,
    traces: OperatorTraceLedger,
    snapshots_by_campaign: dict[str, list[dict[str, Any]]],
    trees: dict[str, HypothesisTree] | None = None,
    candidate_policy: PolicyRecord | None = None,
    baseline_policy: PolicyRecord | None = None,
) -> DifferenceRewardLedger:
    ledger = DifferenceRewardLedger()
    registry = OperatorRegistry()

    for trace in traces.traces:
        snaps = snapshots_by_campaign.get(trace.campaign_id, [])
        reward_all = campaign_reward_from_snapshots(snaps)
        for credit in estimate_contribution(trace=trace, reward_all=reward_all, registry=registry):
            ledger.add(credit)

    if trees and candidate_policy and baseline_policy:
        for cid, tree in trees.items():
            snaps = [
                s for s in snapshots_by_campaign.get(cid, [])
                if s.get("theme_entropy") is not None
            ]
            if len(snaps) < 2:
                continue
            branching_credit = replay_branching_counterfactual(
                campaign_id=cid,
                snapshots=snaps,
                tree=tree,
                candidate=candidate_policy,
                baseline=baseline_policy,
            )
            for trace in traces.for_campaign(cid):
                ledger.add(OperatorCredit(
                    campaign_id=cid,
                    node_id=trace.node_id,
                    family="branching",
                    operator=trace.branching,
                    reward_all=campaign_reward_from_snapshots(snaps),
                    reward_without=round(campaign_reward_from_snapshots(snaps) - branching_credit, 4),
                    contribution=branching_credit,
                    counterfactual_operator=registry.default_for("branching"),
                ))
                break
    return ledger
