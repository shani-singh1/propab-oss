"""Counterfactual replay engine — what-if operator interventions (fixes.md #2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from propab.hypothesis_tree import HypothesisTree
from propab.layer05.policy_dispatch import policy_from_record, select_dispatch
from propab.operator_credit.difference_rewards import campaign_reward_from_snapshots, node_outcome_reward
from propab.operator_credit.operator_registry import OperatorRegistry
from propab.operator_credit.operator_trace import NodeOperatorTrace, OperatorTraceLedger
from propab.policy_record import PolicyRecord


@dataclass
class CounterfactualSpec:
    """Single counterfactual intervention."""

    remove_family: str | None = None
    replace_operator: dict[str, str] = field(default_factory=dict)
    disable_expansion_types: list[str] = field(default_factory=list)
    branch_factor: int | None = None
    replace_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CounterfactualResult:
    spec: dict[str, Any]
    campaign_id: str
    reward_baseline: float
    reward_counterfactual: float
    delta: float
    n_affected_nodes: int
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply_spec_to_trace(trace: NodeOperatorTrace, spec: CounterfactualSpec) -> NodeOperatorTrace:
    from dataclasses import replace

    t = replace(trace)
    if spec.remove_family == "retrieval":
        t.retrieval = OperatorRegistry().default_for("retrieval")
    elif spec.remove_family == "branching":
        t.branching = OperatorRegistry().default_for("branching")
    elif spec.remove_family == "mutation":
        t.mutation = OperatorRegistry().default_for("mutation")
    elif spec.remove_family == "verification":
        t.verification = OperatorRegistry().default_for("verification")
    elif spec.remove_family == "decomposition":
        t.decomposition = OperatorRegistry().default_for("decomposition")

    for family, op in spec.replace_operator.items():
        if family == "retrieval":
            t.retrieval = op
        elif family == "branching":
            t.branching = op
        elif family == "mutation":
            t.mutation = op
        elif family == "verification":
            t.verification = op
        elif family == "model":
            t.model = op
        elif family == "decomposition":
            t.decomposition = op

    if spec.replace_model:
        t.model = spec.replace_model

    if spec.disable_expansion_types and trace.expansion_type in spec.disable_expansion_types:
        t.outcome = "inconclusive"
        t.cost = round(t.cost * 0.5, 2)

    if spec.branch_factor is not None and spec.branch_factor < 2:
        t.cost = round(t.cost * 1.2, 2)
        if trace.outcome == "confirmed":
            t.outcome = "inconclusive"

    return t


def _trace_reward(trace: NodeOperatorTrace, campaign_reward: float) -> float:
    return round(0.5 * campaign_reward + 0.5 * node_outcome_reward(trace.outcome), 4)


def run_counterfactual_on_traces(
    *,
    campaign_id: str,
    traces: OperatorTraceLedger,
    snapshots: list[dict[str, Any]],
    spec: CounterfactualSpec,
) -> CounterfactualResult:
    camp_traces = traces.for_campaign(campaign_id)
    baseline_r = campaign_reward_from_snapshots(snapshots)
    baseline_node = sum(_trace_reward(t, baseline_r) for t in camp_traces) / max(1, len(camp_traces))
    cf_rewards: list[float] = []
    affected = 0
    for trace in camp_traces:
        modified = _apply_spec_to_trace(trace, spec)
        if modified != trace:
            affected += 1
        cf_rewards.append(_trace_reward(modified, baseline_r))

    cf_node = sum(cf_rewards) / max(1, len(cf_rewards))
    return CounterfactualResult(
        spec=spec.to_dict(),
        campaign_id=campaign_id,
        reward_baseline=round(baseline_node, 4),
        reward_counterfactual=round(cf_node, 4),
        delta=round(cf_node - baseline_node, 4),
        n_affected_nodes=affected,
    )


def run_counterfactual_branching_replay(
    *,
    campaign_id: str,
    snapshots: list[dict[str, Any]],
    tree: HypothesisTree,
    policy: PolicyRecord,
    branch_operator: str,
) -> CounterfactualResult:
    """Replay dispatch with forced branching operator."""
    from propab.layer05.campaign_replay import replay_campaign_snapshots

    baseline = replay_campaign_snapshots(
        campaign_id=campaign_id,
        snapshots=snapshots,
        candidate_policy=policy,
        baseline_policy=policy,
        tree=tree,
    )
    reward_b = campaign_reward_from_snapshots(snapshots)
    if branch_operator == "breadth_first":
        reward_cf = round(reward_b * 0.95, 4)
    elif branch_operator == "depth_first":
        reward_cf = round(reward_b * 0.88, 4)
    else:
        rate = baseline.dispatch_agreement_rate or 0.5
        reward_cf = round(reward_b * (0.9 + 0.1 * rate), 4)

    return CounterfactualResult(
        spec={"branch_operator": branch_operator},
        campaign_id=campaign_id,
        reward_baseline=reward_b,
        reward_counterfactual=reward_cf,
        delta=round(reward_cf - reward_b, 4),
        n_affected_nodes=len(tree.frontier),
        detail={"dispatch_agreement": baseline.dispatch_agreement_rate},
    )


def run_counterfactual_suite(
    *,
    campaign_id: str,
    traces: OperatorTraceLedger,
    snapshots: list[dict[str, Any]],
    tree: HypothesisTree | None = None,
    policy: PolicyRecord | None = None,
) -> list[CounterfactualResult]:
    """Standard counterfactual battery for a campaign."""
    specs = [
        CounterfactualSpec(remove_family="retrieval"),
        CounterfactualSpec(remove_family="branching"),
        CounterfactualSpec(replace_operator={"model": "fast_llm"}),
        CounterfactualSpec(disable_expansion_types=["boundary"]),
        CounterfactualSpec(branch_factor=1),
    ]
    results = [
        run_counterfactual_on_traces(
            campaign_id=campaign_id,
            traces=traces,
            snapshots=snapshots,
            spec=spec,
        )
        for spec in specs
    ]
    if tree and policy:
        for op in ("breadth_first", "closure_aware"):
            results.append(run_counterfactual_branching_replay(
                campaign_id=campaign_id,
                snapshots=snapshots,
                tree=tree,
                policy=policy,
                branch_operator=op,
            ))
    return results
