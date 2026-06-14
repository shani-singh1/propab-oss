"""Tests for Layer 0.5 P0–P5 — simulation ledger, trajectories, SimulatorBench."""
from __future__ import annotations

from propab.layer05.policy_offline_eval import evaluate_policy_offline
from propab.layer05.simulation_fitness_ledger import (
    SimulationFitnessLedger,
    SimulationFitnessRecord,
)
from propab.layer05.simulator_acceptance import evaluate_simulator_acceptance
from propab.layer05.simulator_bench import run_simulator_bench
from propab.layer05.simulator_residual_history import simulator_residual_history_for_bucket
from propab.layer05.trajectories import CampaignTrajectories, EntropyTrajectory
from propab.policy_record import PolicyRecord, PolicyStatus
from services.orchestrator.policy_analyst import _build_analyst_prompt


def _snapshots(n: int = 10) -> list[dict]:
    hist = {"diffusion_dynamics": 1}
    snaps = []
    for i in range(1, n + 1):
        if i > 3:
            hist["spectral"] = hist.get("spectral", 0) + 1
        snaps.append({
            "tested": i,
            "executed": i,
            "generated": i + 1,
            "theme_histogram": dict(hist),
            "theme_entropy": 0.5 + i * 0.12,
            "closure_ratio": max(0.05, 0.4 - i * 0.02),
        })
    return snaps


def _policy(**boosts) -> PolicyRecord:
    return PolicyRecord(
        id="pol-offline",
        generation=2,
        parent_policy_id="pol-parent",
        budget_bucket="3h",
        domain_bucket="graphs",
        boosts=boosts or {"diffusion_dynamics": 0.35},
        status=PolicyStatus.CANDIDATE,
    )


def test_trajectory_objects_from_snapshots():
    traj = CampaignTrajectories.from_snapshots(_snapshots(12))
    assert traj.entropy.start > 0
    assert traj.entropy.end > traj.entropy.start
    assert traj.closure.n_points == 12
    assert traj.branching.n_points >= 1


def test_simulator_bench_metrics():
    snaps = _snapshots(15)
    sim_pts = [{"tested": i, "theme_entropy": 0.5 + i * 0.1} for i in range(16)]
    sim_closure = [max(0.05, 0.4 - i * 0.015) for i in range(16)]
    bench = run_simulator_bench(
        simulated_entropy_points=sim_pts,
        simulated_closure_values=sim_closure,
        observed_snapshots=snaps,
    )
    assert bench.mae_entropy >= 0
    assert bench.rmse_entropy >= 0
    assert 0 <= bench.directional_agreement <= 1


def test_simulation_fitness_ledger_roundtrip(tmp_path, monkeypatch):
    from propab import config

    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))
    ledger = SimulationFitnessLedger()
    ledger.record(SimulationFitnessRecord(
        simulator_version="v1",
        replay_campaign_id="camp-1",
        policy_id="pol-1",
        predicted_trajectory={"entropy": {"start": 0.7}},
        observed_trajectory={"entropy": {"start": 0.72}},
        residuals={"mae_entropy": 0.05},
        accept_or_reject="ACCEPT",
    ))
    ledger.save()
    loaded = SimulationFitnessLedger.load()
    assert len(loaded.records) == 1
    assert loaded.records[0].residuals["mae_entropy"] == 0.05


def test_simulator_residual_history_for_analyst():
    ledger = SimulationFitnessLedger()
    ledger.record(SimulationFitnessRecord(
        simulator_version="v1",
        replay_campaign_id="camp-abc",
        policy_id="pol-xyz",
        predicted_trajectory={"entropy": {"end": 1.8}},
        observed_trajectory={"entropy": {"end": 1.9}},
        residuals={"mae_entropy": 0.1},
        accept_or_reject="REJECT",
        budget_bucket="3h",
        domain_bucket="graphs",
        detail={"simulator_bench": {"mae_entropy": 0.1}},
    ))
    hist = simulator_residual_history_for_bucket(
        ledger, budget_bucket="3h", domain_bucket="graphs",
    )
    assert len(hist) == 1
    assert hist[0]["residuals"]["mae_entropy"] == 0.1


def test_analyst_prompt_includes_simulator_history():
    prompt = _build_analyst_prompt(
        domain_bucket="graphs",
        budget_bucket="3h",
        parent=None,
        params={"boosts": {}, "penalties": {}, "blocked_failures": []},
        campaign_metrics={},
        trajectory_summary={"H_start": 0.7},
        residual_history=[{"residuals": {"growth_rate": -0.01}}],
        simulator_residual_history=[{"residuals": {"mae_entropy": 0.12}}],
        trajectory_history=[{"entropy": {"end": 1.9}}],
    )
    assert "simulator residual history" in prompt.lower()
    assert "trajectory summaries" in prompt.lower()


def test_simulator_acceptance_thresholds():
    ok = evaluate_simulator_acceptance(
        current={"directional_agreement": 0.85, "mae_entropy": 0.2, "mae_closure": 0.05},
        previous={"directional_agreement": 0.7, "mae_entropy": 0.3, "mae_closure": 0.08},
    )
    assert ok.accepted
    bad = evaluate_simulator_acceptance(
        current={"directional_agreement": 0.6, "mae_entropy": 0.2, "mae_closure": 0.05},
        previous={"directional_agreement": 0.7, "mae_entropy": 0.3, "mae_closure": 0.08},
    )
    assert not bad.accepted


def test_offline_policy_eval_fast():
    ledger = SimulationFitnessLedger()
    result = evaluate_policy_offline(
        candidate=_policy(),
        campaign_id="bench-camp",
        snapshots=_snapshots(12),
        ledger=ledger,
        persist=False,
    )
    assert result.elapsed_ms < 2000
    assert result.simulator_bench
    assert result.recommendation in (
        "proceed_to_30min_campaign",
        "reject_before_expensive_campaign",
    )
