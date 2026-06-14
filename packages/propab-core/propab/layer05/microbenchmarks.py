"""Microbenchmarks — FrontierBench, EntropyBench, ClosureBench, PolicyBench, AnalystBench."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from propab.entropy_trajectory import summarize_entropy_trajectory, trajectory_point_from_snapshot
from propab.layer05.analyst_replay import replay_analyst_on_history
from propab.layer05.campaign_replay import replay_campaign_snapshots
from propab.layer05.replay_state import SearchState
from propab.layer05.policy_offline_eval import evaluate_policy_offline
from propab.layer05.search_simulator import simulate_search
from propab.layer05.simulation_fitness_ledger import SimulationFitnessLedger
from propab.layer05.simulator_bench import run_simulator_bench
from propab.policy_fitness_ledger import PolicyFitnessLedger
from propab.policy_store import PolicyStore


@dataclass
class BenchmarkReport:
    name: str
    elapsed_ms: float
    metrics: dict[str, Any]
    passed: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Layer05Report:
    benchmarks: list[BenchmarkReport] = field(default_factory=list)
    total_elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_elapsed_ms": self.total_elapsed_ms,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
        }


def frontier_bench(
    *,
    campaign_id: str,
    snapshots: list[dict[str, Any]],
    candidate_policy_id: str | None = None,
) -> BenchmarkReport:
    t0 = time.perf_counter()
    store = PolicyStore.load()
    acc = store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")
    cand = (
        store.get_policy(candidate_policy_id)
        if candidate_policy_id
        else store.latest_candidate(domain_bucket="graphs", budget_bucket="3h")
    )
    policy = cand or acc
    result = replay_campaign_snapshots(
        campaign_id=campaign_id,
        snapshots=snapshots,
        candidate_policy=policy,
        baseline_policy=acc,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return BenchmarkReport(
        name="FrontierBench",
        elapsed_ms=round(elapsed, 2),
        metrics={
            "dispatch_agreement_rate": result.dispatch_agreement_rate,
            "branching_factor": result.branching_factor,
            "n_snapshots": result.n_snapshots,
            "replay_elapsed_ms": result.elapsed_ms,
        },
        passed=result.elapsed_ms < 5000,
        notes="Dispatch quality + branching from snapshot replay",
    )


def entropy_bench(
    *,
    campaign_id: str,
    snapshots: list[dict[str, Any]],
    policy_id: str | None = None,
) -> BenchmarkReport:
    t0 = time.perf_counter()
    store = PolicyStore.load()
    policy = (
        store.get_policy(policy_id)
        or store.latest_candidate(domain_bucket="graphs", budget_bucket="3h")
        or store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")
    )
    points = [trajectory_point_from_snapshot(s) for s in snapshots]
    obs = summarize_entropy_trajectory(points)
    state = SearchState.from_snapshot(snapshots[0] if snapshots else {})
    sim = simulate_search(state=state, policy=policy, steps=max(10, len(points)))
    bench = run_simulator_bench(
        simulated_entropy_points=sim.entropy_points,
        simulated_closure_values=sim.closure_trajectory,
        observed_snapshots=snapshots,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return BenchmarkReport(
        name="EntropyBench",
        elapsed_ms=round(elapsed, 2),
        metrics={
            "observed": obs.to_dict(),
            "simulated": sim.entropy_trajectory,
            "simulator_bench": bench.to_dict(),
        },
        passed=elapsed < 2000,
        notes="Simulated vs observed entropy trajectory",
    )


def simulator_bench(
    *,
    campaign_id: str,
    snapshots: list[dict[str, Any]],
    policy_id: str | None = None,
) -> BenchmarkReport:
    t0 = time.perf_counter()
    store = PolicyStore.load()
    policy = (
        store.get_policy(policy_id)
        or store.latest_candidate(domain_bucket="graphs", budget_bucket="3h")
        or store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")
    )
    state = SearchState.from_snapshot(snapshots[0] if snapshots else {})
    sim = simulate_search(state=state, policy=policy, steps=max(10, len(snapshots)))
    bench = run_simulator_bench(
        simulated_entropy_points=sim.entropy_points,
        simulated_closure_values=sim.closure_trajectory,
        observed_snapshots=snapshots,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return BenchmarkReport(
        name="SimulatorBench",
        elapsed_ms=round(elapsed, 2),
        metrics=bench.to_dict(),
        passed=bench.passed,
        notes="MAE/RMSE/directional agreement vs historical trajectories",
    )


def offline_policy_eval_bench(
    *,
    campaign_id: str,
    snapshots: list[dict[str, Any]],
    policy_id: str | None = None,
    sim_ledger: SimulationFitnessLedger | None = None,
) -> BenchmarkReport:
    t0 = time.perf_counter()
    store = PolicyStore.load()
    policy = (
        store.get_policy(policy_id)
        or store.latest_candidate(domain_bucket="graphs", budget_bucket="3h")
        or store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")
    )
    ledger = sim_ledger if sim_ledger is not None else SimulationFitnessLedger()
    result = evaluate_policy_offline(
        candidate=policy,
        campaign_id=campaign_id,
        snapshots=snapshots,
        ledger=ledger,
        persist=sim_ledger is not None,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return BenchmarkReport(
        name="OfflinePolicyEval",
        elapsed_ms=round(elapsed, 2),
        metrics=result.to_dict(),
        passed=result.accept_or_reject == "ACCEPT",
        notes="Full P3 workflow: replay → simulate → SimulatorBench",
    )


def closure_bench(
    fitness: PolicyFitnessLedger | None = None,
) -> BenchmarkReport:
    t0 = time.perf_counter()
    ledger = fitness or PolicyFitnessLedger.load()
    closure_res: list[float] = []
    refute_res: list[float] = []
    for rec in ledger.records:
        closure_res.append(float((rec.residuals or {}).get("closure_ratio", 0)))
        refute_res.append(float((rec.residuals or {}).get("refute_ratio", 0)))
    elapsed = (time.perf_counter() - t0) * 1000
    return BenchmarkReport(
        name="ClosureBench",
        elapsed_ms=round(elapsed, 2),
        metrics={
            "n_records": len(ledger.records),
            "closure_residuals": closure_res[-10:],
            "refute_residuals": refute_res[-10:],
            "mean_abs_closure": round(
                sum(abs(x) for x in closure_res) / max(1, len(closure_res)), 4
            ),
        },
        passed=True,
        notes="Predicted vs actual closure/refute from fitness ledger",
    )


def policy_bench(
    fitness: PolicyFitnessLedger | None = None,
) -> BenchmarkReport:
    t0 = time.perf_counter()
    ledger = fitness or PolicyFitnessLedger.load()
    rows = []
    for rec in ledger.records[-20:]:
        rows.append({
            "campaign_id": rec.campaign_id[:8],
            "policy_id": rec.policy_id[:12],
            "residuals": rec.residuals,
            "accept_or_reject": rec.accept_or_reject,
        })
    elapsed = (time.perf_counter() - t0) * 1000
    return BenchmarkReport(
        name="PolicyBench",
        elapsed_ms=round(elapsed, 2),
        metrics={"records": rows, "n": len(rows)},
        passed=True,
        notes="Historical policy prediction residuals",
    )


def analyst_bench(
    fitness: PolicyFitnessLedger | None = None,
) -> BenchmarkReport:
    t0 = time.perf_counter()
    result = replay_analyst_on_history(fitness)
    elapsed = (time.perf_counter() - t0) * 1000
    return BenchmarkReport(
        name="AnalystBench",
        elapsed_ms=round(elapsed, 2),
        metrics=result.to_dict(),
        passed=elapsed < 3000,
        notes="Analyst bias/variance from residual history",
    )


def run_all_benchmarks(
    *,
    campaign_id: str | None = None,
    snapshots: list[dict[str, Any]] | None = None,
    fitness: PolicyFitnessLedger | None = None,
    sim_ledger: SimulationFitnessLedger | None = None,
) -> Layer05Report:
    t0 = time.perf_counter()
    reports: list[BenchmarkReport] = []
    if campaign_id and snapshots:
        reports.append(frontier_bench(campaign_id=campaign_id, snapshots=snapshots))
        reports.append(entropy_bench(campaign_id=campaign_id, snapshots=snapshots))
        reports.append(simulator_bench(campaign_id=campaign_id, snapshots=snapshots))
        reports.append(offline_policy_eval_bench(
            campaign_id=campaign_id,
            snapshots=snapshots,
            sim_ledger=sim_ledger,
        ))
    reports.append(closure_bench(fitness))
    reports.append(policy_bench(fitness))
    reports.append(analyst_bench(fitness))
    total = (time.perf_counter() - t0) * 1000
    return Layer05Report(benchmarks=reports, total_elapsed_ms=round(total, 2))
