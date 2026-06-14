"""Campaign replay engine — deterministic, no LLM (fixes.md Component 1)."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from propab.entropy_trajectory import summarize_entropy_trajectory, trajectory_point_from_snapshot
from propab.hypothesis_tree import HypothesisTree
from propab.layer05.policy_dispatch import (
    policy_from_record,
    select_dispatch,
    select_dispatch_baseline,
)
from propab.layer05.replay_state import ReplayState
from propab.policy_record import PolicyRecord

@dataclass
class CampaignReplayResult:
    campaign_id: str
    n_snapshots: int
    elapsed_ms: float
    entropy_trajectory: dict[str, Any]
    closure_start: float
    closure_end: float
    branching_factor: float
    dispatch_agreement_rate: float | None
    policy_vs_baseline_disagreements: int
    replay_states: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _branching_factor(snapshots: list[dict[str, Any]]) -> float:
    if len(snapshots) < 2:
        return 0.0
    deltas = []
    for i in range(1, len(snapshots)):
        prev = int(snapshots[i - 1].get("generated") or snapshots[i - 1].get("tested") or 0)
        cur = int(snapshots[i].get("generated") or snapshots[i].get("tested") or 0)
        deltas.append(max(0, cur - prev))
    return round(sum(deltas) / len(deltas), 4) if deltas else 0.0


def _compare_dispatch_policies(
    tree: HypothesisTree,
    candidate: PolicyRecord,
    baseline: PolicyRecord | None,
) -> tuple[float | None, int]:
    cand_policy = policy_from_record(candidate)
    base_policy = policy_from_record(baseline) if baseline else policy_from_record(candidate)
    agreements = 0
    disagreements = 0
    steps = 0
    for _ in range(min(50, len(tree.frontier) + 10)):
        cand_node = select_dispatch(tree, cand_policy)
        base_node = select_dispatch_baseline(tree)
        if cand_node is None or base_node is None:
            break
        steps += 1
        if cand_node.id == base_node.id:
            agreements += 1
        else:
            disagreements += 1
        tree.nodes[cand_node.id].verdict = "inconclusive"
        tree.frontier = [nid for nid in tree.frontier if nid != cand_node.id]
    if steps == 0:
        return None, 0
    return round(agreements / steps, 4), disagreements


def replay_campaign_snapshots(
    *,
    campaign_id: str,
    snapshots: list[dict[str, Any]],
    candidate_policy: PolicyRecord,
    baseline_policy: PolicyRecord | None = None,
    tree: HypothesisTree | None = None,
) -> CampaignReplayResult:
    """Replay snapshot sequence and compare policy dispatch vs baseline scoring."""
    t0 = time.perf_counter()
    states = [ReplayState.from_snapshot(s).to_dict() for s in snapshots]
    points = [trajectory_point_from_snapshot(s) for s in snapshots]
    traj = summarize_entropy_trajectory(points)

    agreement: float | None = None
    disagreements = 0
    if tree is not None and tree.frontier:
        tree_copy = HypothesisTree.from_dict(tree.to_dict())
        agreement, disagreements = _compare_dispatch_policies(
            tree_copy, candidate_policy, baseline_policy
        )

    elapsed = (time.perf_counter() - t0) * 1000
    return CampaignReplayResult(
        campaign_id=campaign_id,
        n_snapshots=len(snapshots),
        elapsed_ms=round(elapsed, 2),
        entropy_trajectory=traj.to_dict(),
        closure_start=float(snapshots[0].get("closure_ratio") or 0) if snapshots else 0.0,
        closure_end=float(snapshots[-1].get("closure_ratio") or 0) if snapshots else 0.0,
        branching_factor=_branching_factor(snapshots),
        dispatch_agreement_rate=agreement,
        policy_vs_baseline_disagreements=disagreements,
        replay_states=states[:5] + (states[-1:] if len(states) > 6 else []),
    )
