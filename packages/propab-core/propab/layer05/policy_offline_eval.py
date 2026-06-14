"""Policy offline evaluation workflow (fixes.md P3)."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from propab.layer05.campaign_replay import replay_campaign_snapshots
from propab.layer05.replay_state import SearchState
from propab.layer05.simulation_fitness_ledger import (
    SimulationFitnessLedger,
    SimulationFitnessRecord,
)
from propab.layer05.simulator_acceptance import evaluate_simulator_acceptance
from propab.layer05.simulator_bench import run_simulator_bench
from propab.layer05.trajectories import CampaignTrajectories, ClosureTrajectory, EntropyTrajectory
from propab.layer05.simulator_dispatch import (
    resolve_simulator_version,
    simulate_for_version,
)
from propab.layer05.simulator_registry import SimulatorRegistry
from propab.policy_record import PolicyRecord
from propab.policy_store import PolicyStore


@dataclass
class OfflinePolicyEvalResult:
    policy_id: str
    campaign_id: str
    accept_or_reject: str
    elapsed_ms: float
    replay: dict[str, Any]
    simulation: dict[str, Any]
    simulator_bench: dict[str, Any]
    trajectories: dict[str, Any]
    simulator_acceptance: dict[str, Any]
    recommendation: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_policy_offline(
    *,
    candidate: PolicyRecord,
    campaign_id: str,
    snapshots: list[dict[str, Any]],
    baseline: PolicyRecord | None = None,
    ledger: SimulationFitnessLedger | None = None,
    persist: bool = True,
    budget_bucket: str = "3h",
    domain_bucket: str = "graphs",
    simulator_version: str | None = None,
    trajectory_path: str | None = None,
) -> OfflinePolicyEvalResult:
    """
    candidate → replay → simulator → SimulatorBench → accept/reject.

    Reject obvious failures before expensive campaigns.
    """
    t0 = time.perf_counter()
    fitness = ledger or SimulationFitnessLedger.load()
    store = PolicyStore.load()
    baseline = baseline or store.accepted_policy(
        domain_bucket=domain_bucket,
        budget_bucket=budget_bucket,
    )

    replay = replay_campaign_snapshots(
        campaign_id=campaign_id,
        snapshots=snapshots,
        candidate_policy=candidate,
        baseline_policy=baseline,
    )

    registry = SimulatorRegistry.load()
    sim_version = resolve_simulator_version(version=simulator_version, registry=registry)
    state = SearchState.from_snapshot(snapshots[0] if snapshots else {})
    sim = simulate_for_version(
        version=sim_version,
        state=state,
        policy=candidate,
        steps=max(10, len(snapshots)),
        trajectory_path=trajectory_path,
        exclude_campaign_id=campaign_id,
    )

    bench = run_simulator_bench(
        simulated_entropy_points=sim.entropy_points,
        simulated_closure_values=sim.closure_trajectory,
        observed_snapshots=snapshots,
    )

    observed_traj = CampaignTrajectories.from_snapshots(snapshots)
    predicted_traj = CampaignTrajectories(
        entropy=EntropyTrajectory.from_points(sim.entropy_points),
        closure=ClosureTrajectory.from_values(sim.closure_trajectory),
        branching=observed_traj.branching,
        theme_saturation=observed_traj.theme_saturation,
    )

    prev_records = [
        r for r in fitness.records
        if r.simulator_version == sim_version
    ]
    prev_metrics = None
    if prev_records:
        last = prev_records[-1].detail.get("simulator_bench") or {}
        prev_metrics = {
            "mae_entropy": last.get("mae_entropy"),
            "mae_closure": last.get("mae_closure"),
            "directional_agreement": last.get("directional_agreement"),
        }

    acceptance = evaluate_simulator_acceptance(
        current={
            "mae_entropy": bench.mae_entropy,
            "mae_closure": bench.mae_closure,
            "directional_agreement": bench.directional_agreement,
        },
        previous=prev_metrics,
        version=sim_version,
    )

    accept_or_reject = "ACCEPT" if bench.passed else "REJECT"
    residuals = {
        "mae_entropy": bench.mae_entropy,
        "rmse_entropy": bench.rmse_entropy,
        "mae_closure": bench.mae_closure,
        "directional_agreement": bench.directional_agreement,
        "threshold_crossing_error": bench.threshold_crossing_error,
    }

    if persist:
        fitness.record(SimulationFitnessRecord(
            simulator_version=sim_version,
            replay_campaign_id=campaign_id,
            policy_id=candidate.id,
            predicted_trajectory=predicted_traj.to_dict(),
            observed_trajectory=observed_traj.to_dict(),
            residuals=residuals,
            accept_or_reject=accept_or_reject,
            budget_bucket=budget_bucket,
            domain_bucket=domain_bucket,
            detail={
                "simulator_bench": bench.to_dict(),
                "replay": replay.to_dict(),
            },
        ))
        if acceptance.accepted:
            fitness.accepted_simulator_version = sim_version
        fitness.save()

    elapsed = (time.perf_counter() - t0) * 1000
    recommendation = (
        "proceed_to_30min_campaign"
        if accept_or_reject == "ACCEPT"
        else "reject_before_expensive_campaign"
    )

    return OfflinePolicyEvalResult(
        policy_id=candidate.id,
        campaign_id=campaign_id,
        accept_or_reject=accept_or_reject,
        elapsed_ms=round(elapsed, 2),
        replay=replay.to_dict(),
        simulation=sim.to_dict(),
        simulator_bench=bench.to_dict(),
        trajectories=observed_traj.to_dict(),
        simulator_acceptance=acceptance.to_dict(),
        recommendation=recommendation,
        detail={"simulator_version": sim_version},
    )
