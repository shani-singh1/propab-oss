"""Hierarchical credit assignment — micro / meso / macro / mega (fixes.md #4)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from propab.operator_credit.difference_rewards import (
    DifferenceRewardLedger,
    campaign_reward_from_snapshots,
    node_outcome_reward,
)
from propab.operator_credit.operator_trace import NodeOperatorTrace, OperatorTraceLedger


class CreditLevel(str, Enum):
    MICRO = "micro"
    MESO = "meso"
    MACRO = "macro"
    MEGA = "mega"


@dataclass
class HierarchicalCredit:
    level: str
    entity_id: str
    campaign_id: str
    family: str
    operator: str
    contribution: float
    reward: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HierarchicalCreditLedger:
    credits: list[HierarchicalCredit] = field(default_factory=list)

    def add(self, credit: HierarchicalCredit) -> None:
        self.credits.append(credit)

    def by_level(self, level: CreditLevel) -> list[HierarchicalCredit]:
        return [c for c in self.credits if c.level == level.value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "credits": [c.to_dict() for c in self.credits],
            "summary": {
                lvl.value: len(self.by_level(lvl))
                for lvl in CreditLevel
            },
        }


def _micro_credits(trace: NodeOperatorTrace) -> list[HierarchicalCredit]:
    out: list[HierarchicalCredit] = []
    for i, tc in enumerate(trace.tool_calls):
        success = bool(tc.get("success"))
        dur = int(tc.get("duration_ms") or 0)
        reward = 0.3 if success else -0.1
        out.append(HierarchicalCredit(
            level=CreditLevel.MICRO.value,
            entity_id=f"{trace.node_id}:tool:{i}",
            campaign_id=trace.campaign_id,
            family="tool",
            operator=str(tc.get("tool_name") or "unknown"),
            contribution=round(reward - dur / 10000.0, 4),
            reward=round(reward, 4),
            detail={"duration_ms": dur, "success": success},
        ))
    return out


def _meso_credits(
    trace: NodeOperatorTrace,
    diff_ledger: DifferenceRewardLedger | None,
) -> list[HierarchicalCredit]:
    out: list[HierarchicalCredit] = []
    node_r = node_outcome_reward(trace.outcome)
    if diff_ledger:
        for c in diff_ledger.credits:
            if c.node_id == trace.node_id and c.campaign_id == trace.campaign_id:
                out.append(HierarchicalCredit(
                    level=CreditLevel.MESO.value,
                    entity_id=trace.node_id,
                    campaign_id=trace.campaign_id,
                    family=c.family,
                    operator=c.operator,
                    contribution=c.contribution,
                    reward=c.reward_all,
                    detail={"counterfactual": c.counterfactual_operator},
                ))
    if not out:
        for step in trace.operators_used:
            out.append(HierarchicalCredit(
                level=CreditLevel.MESO.value,
                entity_id=trace.node_id,
                campaign_id=trace.campaign_id,
                family=step.family,
                operator=step.operator,
                contribution=round(node_r / max(1, len(trace.operators_used)), 4),
                reward=node_r,
            ))
    return out


def _macro_credits(
    campaign_id: str,
    traces: list[NodeOperatorTrace],
    snapshots: list[dict[str, Any]],
) -> list[HierarchicalCredit]:
    reward = campaign_reward_from_snapshots(snapshots)
    family_totals: dict[tuple[str, str], float] = {}
    for t in traces:
        for step in t.operators_used:
            key = (step.family, step.operator)
            family_totals[key] = family_totals.get(key, 0) + node_outcome_reward(t.outcome)
    n = max(1, len(traces))
    return [
        HierarchicalCredit(
            level=CreditLevel.MACRO.value,
            entity_id=campaign_id,
            campaign_id=campaign_id,
            family=fam,
            operator=op,
            contribution=round(total / n, 4),
            reward=reward,
        )
        for (fam, op), total in family_totals.items()
    ]


def _mega_credits(
    family_id: str,
    campaign_ids: list[str],
    macro: list[HierarchicalCredit],
) -> list[HierarchicalCredit]:
    by_key: dict[tuple[str, str], list[float]] = {}
    for c in macro:
        if c.campaign_id in campaign_ids:
            by_key.setdefault((c.family, c.operator), []).append(c.contribution)
    return [
        HierarchicalCredit(
            level=CreditLevel.MEGA.value,
            entity_id=family_id,
            campaign_id=campaign_ids[0] if campaign_ids else "",
            family=fam,
            operator=op,
            contribution=round(sum(vals) / len(vals), 4),
            reward=round(sum(vals), 4),
            detail={"n_campaigns": len(campaign_ids)},
        )
        for (fam, op), vals in by_key.items()
    ]


def build_hierarchical_credits(
    *,
    traces: OperatorTraceLedger,
    snapshots_by_campaign: dict[str, list[dict[str, Any]]],
    diff_ledger: DifferenceRewardLedger | None = None,
    campaign_families: dict[str, list[str]] | None = None,
) -> HierarchicalCreditLedger:
    ledger = HierarchicalCreditLedger()
    macro_all: list[HierarchicalCredit] = []

    for cid in {t.campaign_id for t in traces.traces}:
        camp_traces = traces.for_campaign(cid)
        snaps = snapshots_by_campaign.get(cid, [])
        for trace in camp_traces:
            for c in _micro_credits(trace):
                ledger.add(c)
            for c in _meso_credits(trace, diff_ledger):
                ledger.add(c)
        macro = _macro_credits(cid, camp_traces, snaps)
        macro_all.extend(macro)
        for c in macro:
            ledger.add(c)

    if campaign_families:
        for fam_id, cids in campaign_families.items():
            for c in _mega_credits(fam_id, cids, macro_all):
                ledger.add(c)

    return ledger
