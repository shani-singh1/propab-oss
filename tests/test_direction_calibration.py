"""Tests for direction error reduction (fixes.md P0–P5)."""
from __future__ import annotations

import json

from propab.layer05.direction_error_dataset import DirectionErrorDataset
from propab.layer05.direction_loss import (
    count_direction_errors,
    direction_accuracy_weighted_loss,
    direction_weighted_score,
)
from propab.layer05.neighbor_attribution import build_neighbor_attribution
from propab.layer05.replay_state import SearchState
from propab.layer05.stage_aware_predictors import StageAwareEntropyPredictor
from propab.layer05.stage_simulators import SIM_V4, simulate_search_stage_aware
from propab.layer05.state_embedding_index import StateEmbeddingIndex
from propab.layer05.trajectory_stages import stage_at_index
from propab.layer05.weak_campaign_analysis import analyze_campaign_stages
from propab.policy_record import PolicyRecord, PolicyStatus


def _snaps(n: int = 8, off: int = 0) -> list[dict]:
    out = []
    for i in range(1, n + 1):
        out.append({
            "tested": i + off,
            "theme_entropy": 0.5 + i * 0.15,
            "closure_ratio": 0.1 + i * 0.05,
            "theme_histogram": {"a": i, "b": 1},
            "frontier_size": 2,
            "pending": 1,
        })
    return out


def _policy() -> PolicyRecord:
    return PolicyRecord(
        id="pol-dir",
        generation=1,
        parent_policy_id="p0",
        budget_bucket="3h",
        domain_bucket="graphs",
        boosts={"diffusion_dynamics": 0.3},
        status=PolicyStatus.CANDIDATE,
    )


def test_stage_at_index():
    assert stage_at_index(0, 30) == "cold_start"
    assert stage_at_index(22, 30) == "plateau"


def test_direction_weighted_score_prefers_direction():
    high_dir = direction_weighted_score({"directional_agreement": 0.8, "mae_entropy": 0.5})
    low_dir = direction_weighted_score({"directional_agreement": 0.4, "mae_entropy": 0.1})
    assert high_dir > low_dir


def test_direction_error_dataset_rows():
    by_id = {"a": _snaps(6), "b": _snaps(6, 2)}
    index = StateEmbeddingIndex().build_from_snapshots_map(by_id)
    sim_pts = [{"tested": i, "theme_entropy": 2.0 - i * 0.3} for i in range(6)]
    ds = DirectionErrorDataset.build_from_simulation(
        campaign_id="a",
        simulator_version=SIM_V4,
        simulated_entropy_points=sim_pts,
        observed_snapshots=by_id["a"],
        index=index,
    )
    assert ds.rows
    assert ds.rows[0].stage in ("cold_start", "growth", "plateau")
    assert ds.rows[0].neighbors


def test_neighbor_attribution():
    by_id = {"a": _snaps(5), "b": _snaps(5, 1)}
    index = StateEmbeddingIndex().build_from_snapshots_map(by_id)
    neighbors = index.nearest(SearchState.from_snapshot(by_id["a"][0]), k=2)
    attrs = build_neighbor_attribution(neighbors)
    assert attrs[0].weight > 0
    assert attrs[0].similarity > 0


def test_stage_aware_predictor():
    by_id = {"a": _snaps(8), "b": _snaps(8, 3)}
    index = StateEmbeddingIndex().build_from_snapshots_map(by_id)
    neighbors = index.nearest(SearchState.from_snapshot(by_id["a"][0]), k=3)
    pred = StageAwareEntropyPredictor()
    series = pred.predict(neighbors, steps=7, rules_series=[0.5] * 8, query_snapshots=by_id["a"])
    assert len(series) == 8


def test_stage_simulator_v4():
    by_id = {"a": _snaps(8), "b": _snaps(8, 4)}
    index = StateEmbeddingIndex().build_from_snapshots_map(by_id)
    sim = simulate_search_stage_aware(
        state=SearchState.from_snapshot(by_id["a"][0]),
        policy=_policy(),
        index=index,
        steps=7,
        query_snapshots=by_id["a"],
    )
    assert len(sim.entropy_points) == 8


def test_weak_campaign_stage_breakdown():
    snaps = _snaps(8)
    sim_pts = [{"tested": i, "theme_entropy": 0.5 + i * 0.1} for i in range(8)]
    report = analyze_campaign_stages(
        campaign_id="test",
        simulated_entropy_points=sim_pts,
        observed_snapshots=snaps,
    )
    assert report.worst_stage in ("cold_start", "growth", "plateau")
    assert 0 <= report.overall_directional <= 1


def test_asymmetric_direction_loss():
    sim = [0.0, 0.5, 1.0, 1.5]
    obs = [0.0, 0.4, 0.9, 1.4]
    good = direction_accuracy_weighted_loss(sim, obs)
    bad_sim = [0.0, 1.0, 0.5, 1.5]
    bad = direction_accuracy_weighted_loss(bad_sim, obs)
    assert bad > good
