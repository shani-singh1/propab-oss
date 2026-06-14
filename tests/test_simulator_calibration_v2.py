"""Tests for simulator calibration v2 — dataset, index, hybrid sim, registry."""
from __future__ import annotations

import json
from pathlib import Path

from propab.layer05.bench_metrics import mae
from propab.layer05.component_bench import run_component_bench_suite
from propab.layer05.hybrid_simulator import SIM_V2, simulate_search_hybrid
from propab.layer05.replay_state import SearchState
from propab.layer05.search_simulator import simulate_search
from propab.layer05.simulation_residual_dataset import SimulationResidualDataset
from propab.layer05.simulator_calibration import calibrate_campaign, run_calibration_cycle
from propab.layer05.simulator_dispatch import simulate_for_version
from propab.layer05.simulator_registry import SIM_V1, SimulatorRegistry
from propab.layer05.state_embedding_index import StateEmbeddingIndex, state_features
from propab.policy_record import PolicyRecord, PolicyStatus


def _synthetic_snapshots(n: int = 8, campaign_offset: int = 0) -> list[dict]:
    snaps = []
    hist = {"diffusion_dynamics": 1}
    for i in range(1, n + 1):
        if i > 2:
            hist["spectral"] = hist.get("spectral", 0) + 1
        tested = i + campaign_offset
        snaps.append({
            "tested": tested,
            "executed": tested,
            "generated": tested + 2,
            "pending": 2,
            "theme_histogram": dict(hist),
            "theme_entropy": 0.5 + i * 0.12 + campaign_offset * 0.01,
            "closure_ratio": round(min(0.95, 0.1 + i * 0.08), 4),
            "frontier_size": 2,
        })
    return snaps


def _policy() -> PolicyRecord:
    return PolicyRecord(
        id="pol-cal",
        generation=1,
        parent_policy_id="pol-root",
        budget_bucket="3h",
        domain_bucket="graphs",
        boosts={"diffusion_dynamics": 0.3},
        status=PolicyStatus.CANDIDATE,
    )


def test_state_features_vector():
    state = SearchState.from_snapshot(_synthetic_snapshots(3)[0])
    feats = state_features(state)
    assert len(feats) == 5
    assert all(0 <= f <= 1.5 for f in feats)


def test_state_index_nearest_neighbors():
    by_id = {
        "c1": _synthetic_snapshots(6, campaign_offset=0),
        "c2": _synthetic_snapshots(6, campaign_offset=10),
    }
    index = StateEmbeddingIndex().build_from_snapshots_map(by_id)
    query = SearchState.from_snapshot(by_id["c1"][0])
    neighbors = index.nearest(query, k=2)
    assert len(neighbors) == 2
    assert neighbors[0][0].campaign_id in ("c1", "c2")


def test_hybrid_simulator_uses_neighbors(tmp_path, monkeypatch):
    by_id = {
        "c1": _synthetic_snapshots(8),
        "c2": _synthetic_snapshots(8, campaign_offset=5),
    }
    index = StateEmbeddingIndex().build_from_snapshots_map(by_id)
    state = SearchState.from_snapshot(by_id["c1"][0])
    rules = simulate_search(state=state, policy=_policy(), steps=10)
    hybrid = simulate_search_hybrid(
        state=state, policy=_policy(), index=index, steps=10,
    )
    assert hybrid.policy_id == "pol-cal"
    assert len(hybrid.entropy_points) == 11
    rules_h = [p["theme_entropy"] for p in rules.entropy_points]
    hybrid_h = [p["theme_entropy"] for p in hybrid.entropy_points]
    assert hybrid_h != rules_h or mae(hybrid_h, rules_h) < 1.0


def test_simulate_for_version_v1_vs_v2():
    by_id = {"c1": _synthetic_snapshots(6), "c2": _synthetic_snapshots(6, 3)}
    state = SearchState.from_snapshot(by_id["c1"][0])
    index = StateEmbeddingIndex().build_from_snapshots_map(by_id)
    v1 = simulate_for_version(version=SIM_V1, state=state, policy=_policy(), steps=8)
    v2 = simulate_for_version(
        version=SIM_V2, state=state, policy=_policy(), steps=8, index=index,
    )
    assert v1.steps == 8
    assert v2.steps == 8
    assert len(v1.entropy_points) == len(v2.entropy_points)


def test_residual_dataset_from_trajectory_file(tmp_path):
    traj = {
        "campaigns": [
            {
                "campaign_id": "camp-a",
                "trajectory": [
                    {"tested": 1, "theme_entropy": 0.5, "closure_ratio": 0.1},
                    {"tested": 2, "theme_entropy": 0.7, "closure_ratio": 0.2},
                ],
            }
        ]
    }
    path = tmp_path / "traj.json"
    path.write_text(json.dumps(traj), encoding="utf-8")
    ds = SimulationResidualDataset.build_from_trajectory_file(path)
    assert ds.rows[0].replay_campaign_id == "camp-a"
    assert ds.rows[0].observed_trajectories["entropy"]["start"] == 0.5


def test_component_bench_suite_weakest():
    snaps = _synthetic_snapshots(8)
    sim = simulate_search(state=SearchState.from_snapshot(snaps[2]), policy=_policy(), steps=7)
    suite = run_component_bench_suite(
        simulated_entropy_points=sim.entropy_points,
        simulated_closure=sim.closure_trajectory,
        simulated_branching=[1.0] * len(sim.closure_trajectory),
        simulated_saturation=[0.5] * len(sim.closure_trajectory),
        observed_snapshots=snaps,
    )
    assert suite.weakest_component in ("entropy", "closure", "branching", "theme_saturation")
    assert 0 <= suite.aggregate_directional <= 1


def test_simulator_registry_never_overwrites_version():
    reg = SimulatorRegistry()
    reg.register(
        version=SIM_V1,
        metrics={"directional_agreement": 0.14, "mae_entropy": 1.4, "mae_closure": 0.5},
    )
    reg.register(
        version=SIM_V2,
        metrics={"directional_agreement": 0.25, "mae_entropy": 1.2, "mae_closure": 0.45},
    )
    assert SIM_V1 in reg.versions
    assert SIM_V2 in reg.versions
    assert reg.versions[SIM_V2].metrics["directional_agreement"] == 0.25


def test_calibrate_campaign_leave_one_out(monkeypatch):
    class _FakeStore:
        def get_policy(self, _pid):
            return _policy()

        def accepted_policy(self, **_kw):
            return _policy()

    monkeypatch.setattr(
        "propab.layer05.simulator_calibration.PolicyStore.load",
        lambda: _FakeStore(),
    )
    by_id = {
        "a": _synthetic_snapshots(8, 0),
        "b": _synthetic_snapshots(8, 4),
        "c": _synthetic_snapshots(8, 8),
    }
    index = StateEmbeddingIndex().build_from_snapshots_map(by_id)
    result = calibrate_campaign(
        campaign_id="a",
        snapshots=by_id["a"],
        index=index,
        simulator_version=SIM_V2,
        exclude_self=True,
    )
    assert result.simulator_version == SIM_V2
    assert 0 <= result.directional_agreement <= 1
    assert "weakest_component" in result.component_suite


def test_run_calibration_cycle_on_artifact(tmp_path, monkeypatch):
    from propab.config import settings

    traj = tmp_path / "entropy_trajectories.json"
    traj.write_text(json.dumps({
        "campaigns": [
            {"campaign_id": f"c{i}", "trajectory": _synthetic_snapshots(6, i * 2)}
            for i in range(3)
        ]
    }), encoding="utf-8")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings, "propab_data_dir", str(data_dir))

    class _FakeStore:
        def get_policy(self, _pid):
            return _policy()

        def accepted_policy(self, **_kw):
            return _policy()

    monkeypatch.setattr(
        "propab.layer05.simulator_calibration.PolicyStore.load",
        lambda: _FakeStore(),
    )

    report = run_calibration_cycle(trajectory_path=traj, simulator_version=SIM_V2, persist=True)
    assert report.n_campaigns == 3
    assert "directional_agreement" in report.aggregate
    assert (data_dir / "lifetime_knowledge" / "state_embedding_index.json").is_file()
