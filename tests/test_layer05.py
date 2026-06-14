"""Tests for Layer 0.5 — replay, simulation, microbenchmarks."""
from __future__ import annotations

from propab.hypothesis_tree import HypothesisTree
from propab.layer05.campaign_replay import replay_campaign_snapshots
from propab.layer05.microbenchmarks import run_all_benchmarks
from propab.layer05.replay_state import ReplayState, SearchState
from propab.layer05.search_simulator import simulate_search
from propab.layer05.simulator_bench import run_simulator_bench
from propab.policy_fitness_ledger import FitnessRecord, PolicyFitnessLedger
from propab.policy_record import PolicyRecord, PolicyStatus, PredictedEffects


def _synthetic_snapshots(n: int = 8) -> list[dict]:
    snaps = []
    hist = {"diffusion_dynamics": 1}
    for i in range(1, n + 1):
        if i > 2:
            hist["spectral"] = hist.get("spectral", 0) + 1
        if i > 5:
            hist["modularity"] = hist.get("modularity", 0) + 1
        tested = i
        decisive = max(1, i - 1)
        confirmed = max(0, i // 4)
        refuted = max(0, i // 5)
        snaps.append({
            "tested": tested,
            "executed": tested,
            "generated": tested + 2,
            "pending": 2,
            "theme_histogram": dict(hist),
            "theme_entropy": 0.5 + i * 0.15,
            "closure_ratio": round((confirmed + refuted) / tested, 4),
            "frontier_size": 2,
        })
    return snaps


def _policy(**boosts) -> PolicyRecord:
    return PolicyRecord(
        id="pol-test",
        generation=2,
        parent_policy_id="pol-parent",
        budget_bucket="3h",
        domain_bucket="graphs",
        boosts=boosts or {"diffusion_dynamics": 0.35},
        status=PolicyStatus.CANDIDATE,
    )


def test_replay_state_from_snapshot():
    snap = _synthetic_snapshots(1)[0]
    state = ReplayState.from_snapshot(snap)
    assert state.tested_count == 1
    assert state.entropy > 0
    assert "diffusion_dynamics" in state.theme_histogram


def test_campaign_replay_finishes_quickly():
    snaps = _synthetic_snapshots(10)
    result = replay_campaign_snapshots(
        campaign_id="test-camp",
        snapshots=snaps,
        candidate_policy=_policy(),
        baseline_policy=_policy(diffusion_dynamics=0.1),
    )
    assert result.n_snapshots == 10
    assert result.elapsed_ms < 500
    assert result.entropy_trajectory["H_end"] > result.entropy_trajectory["H_start"]


def test_simulator_produces_trajectory():
    state = SearchState.from_snapshot(_synthetic_snapshots(3)[2])
    sim = simulate_search(state=state, policy=_policy(), steps=20)
    assert sim.steps == 20
    assert sim.entropy_trajectory["n_snapshots"] == 21  # initial + 20 steps
    assert len(sim.closure_trajectory) == 21


def test_simulation_residuals_vs_observed():
    snaps = _synthetic_snapshots(12)
    points = [{"tested": s["tested"], "theme_entropy": s["theme_entropy"]} for s in snaps]
    state = SearchState.from_snapshot(snaps[0])
    sim = simulate_search(state=state, policy=_policy(), steps=len(snaps))
    bench = run_simulator_bench(
        simulated_entropy_points=sim.entropy_points,
        simulated_closure_values=sim.closure_trajectory,
        observed_snapshots=snaps,
    )
    assert bench.mae_entropy >= 0
    assert "directional_agreement" in bench.to_dict()


def test_replay_with_tree_dispatch():
    tree = HypothesisTree()
    tree.add_seeds([
        {"text": "A", "primary_theme": "diffusion_dynamics"},
        {"text": "B", "primary_theme": "spectral"},
        {"text": "C", "primary_theme": "general"},
    ])
    snaps = _synthetic_snapshots(5)
    result = replay_campaign_snapshots(
        campaign_id="tree-camp",
        snapshots=snaps,
        candidate_policy=_policy(),
        tree=tree,
    )
    assert result.dispatch_agreement_rate is not None


def test_microbenchmarks_all():
    snaps = _synthetic_snapshots(15)
    fitness = PolicyFitnessLedger()
    fitness.record(FitnessRecord(
        policy_id="pol-1",
        campaign_id="camp-1",
        budget_bucket="3h",
        domain_bucket="graphs",
        predictions={
            "start_H": 0.7,
            "growth_rate": 0.02,
            "saturation_H": 1.8,
            "cross_H_1_5_at_tested": 25,
            "cross_H_2_0_at_tested": 50,
        },
        observations={"closure_ratio": -0.1},
        residuals={"closure_ratio": -0.12, "growth_rate": 0.01},
        accept_or_reject="REJECT",
        detail={
            "observed_trajectory": {
                "H_start": 0.72,
                "H_end": 1.9,
                "growth_rate": 0.025,
                "saturation_H": 1.9,
                "cross_H_1_5_at_tested": 27,
                "cross_H_2_0_at_tested": 60,
            },
        },
    ))
    report = run_all_benchmarks(
        campaign_id="bench-camp",
        snapshots=snaps,
        fitness=fitness,
    )
    assert report.total_elapsed_ms < 5000
    names = {b.name for b in report.benchmarks}
    assert "FrontierBench" in names
    assert "EntropyBench" in names
    assert "AnalystBench" in names
    assert "SimulatorBench" in names
    assert report.total_elapsed_ms < 10000
