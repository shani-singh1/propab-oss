"""Hybrid simulator — rules + retrieval-based trajectory priors (fixes.md P3)."""
from __future__ import annotations

from typing import Any

from propab.entropy_trajectory import summarize_entropy_trajectory
from propab.layer05.replay_state import SearchState
from propab.layer05.search_simulator import SimulationResult, simulate_search
from propab.layer05.simulator_hyperparams import SimulatorHyperparams
from propab.layer05.state_embedding_index import StateEmbeddingIndex, StateIndexEntry
from propab.layer05.trajectories import CampaignTrajectories
from propab.policy_record import PolicyRecord
from propab.search_policy import SearchPolicy

SIM_V2 = "sim_v2"
RETRIEVAL_WEIGHT = 0.7
RULES_WEIGHT = 0.3


def _resample_series(
    snapshots: list[dict[str, Any]],
    key: str,
    steps: int,
) -> list[float]:
    if not snapshots:
        return [0.0] * (steps + 1)
    values = [float(s.get(key) or 0) for s in snapshots]
    if len(values) == 1:
        return values * (steps + 1)
    out: list[float] = []
    for i in range(steps + 1):
        frac = i / max(1, steps)
        idx = min(len(values) - 1, int(frac * (len(values) - 1)))
        out.append(values[idx])
    return out


def _branching_series(snapshots: list[dict[str, Any]], steps: int) -> list[float]:
    deltas: list[float] = [0.0]
    for i in range(1, len(snapshots)):
        prev = int(snapshots[i - 1].get("generated") or snapshots[i - 1].get("tested") or 0)
        cur = int(snapshots[i].get("generated") or snapshots[i].get("tested") or 0)
        deltas.append(float(max(0, cur - prev)))
    if not deltas:
        return [0.0] * (steps + 1)
    out: list[float] = []
    for i in range(steps + 1):
        frac = i / max(1, steps)
        idx = min(len(deltas) - 1, int(frac * (len(deltas) - 1)))
        out.append(deltas[idx])
    return out


def _saturation_series(snapshots: list[dict[str, Any]], steps: int) -> list[float]:
    fracs: list[float] = []
    for s in snapshots:
        hist = s.get("theme_histogram") or {}
        total = sum(hist.values()) or 1
        fracs.append(max(hist.values()) / total if hist else 0.5)
    if not fracs:
        return [0.5] * (steps + 1)
    out: list[float] = []
    for i in range(steps + 1):
        frac = i / max(1, steps)
        idx = min(len(fracs) - 1, int(frac * (len(fracs) - 1)))
        out.append(fracs[idx])
    return out


def _blend_series(
    series_list: list[tuple[list[float], float]],
) -> list[float]:
    if not series_list:
        return []
    n = max(len(s) for s, _ in series_list)
    out = [0.0] * n
    wsum = 0.0
    for series, w in series_list:
        for i in range(n):
            out[i] += w * (series[i] if i < len(series) else series[-1])
        wsum += w
    if wsum <= 0:
        return out
    return [round(v / wsum, 4) for v in out]


def _retrieve_component_series(
    neighbors: list[tuple[StateIndexEntry, float]],
    *,
    steps: int,
) -> dict[str, list[float]]:
    weights = [1.0 / (d + 0.01) for _, d in neighbors]
    entropy_parts: list[tuple[list[float], float]] = []
    closure_parts: list[tuple[list[float], float]] = []
    branch_parts: list[tuple[list[float], float]] = []
    sat_parts: list[tuple[list[float], float]] = []

    for (entry, dist), w in zip(neighbors, weights):
        snaps = entry.snapshots
        entropy_parts.append((_resample_series(snaps, "theme_entropy", steps), w))
        closure_parts.append((_resample_series(snaps, "closure_ratio", steps), w))
        branch_parts.append((_branching_series(snaps, steps), w))
        sat_parts.append((_saturation_series(snaps, steps), w))

    return {
        "entropy": _blend_series(entropy_parts),
        "closure": _blend_series(closure_parts),
        "branching": _blend_series(branch_parts),
        "saturation": _blend_series(sat_parts),
    }


def simulate_search_hybrid(
    *,
    state: SearchState,
    policy: PolicyRecord | SearchPolicy,
    index: StateEmbeddingIndex,
    steps: int = 30,
    k_neighbors: int = 3,
    hyperparams: SimulatorHyperparams | None = None,
    query_snapshots: list[dict[str, Any]] | None = None,
) -> SimulationResult:
    """
    Hybrid: retrieval-based trajectory prior + rule-based v1 correction.
    Components calibrated separately then merged.
    """
    hp = (hyperparams or SimulatorHyperparams()).for_v2()
    retrieval_w = hp.retrieval_weight
    rules_w = hp.rules_weight
    rules = simulate_search(state=state, policy=policy, steps=steps)
    neighbors = index.nearest(
        state,
        k=hp.k_neighbors,
        hyperparams=hp,
        query_snapshots=query_snapshots,
    )

    if not neighbors:
        rules.policy_id = getattr(policy, "id", "search_policy")
        return rules

    retrieved = _retrieve_component_series(neighbors, steps=steps)
    r_ent = retrieved["entropy"]
    r_clo = retrieved["closure"]
    rules_h = [float(p.get("theme_entropy") or 0) for p in rules.entropy_points]
    rules_c = rules.closure_trajectory

    n = steps + 1
    while len(rules_h) < n:
        rules_h.append(rules_h[-1] if rules_h else 0)
    while len(rules_c) < n:
        rules_c.append(rules_c[-1] if rules_c else 0)
    while len(r_ent) < n:
        r_ent.append(r_ent[-1] if r_ent else 0)
    while len(r_clo) < n:
        r_clo.append(r_clo[-1] if r_clo else 0)

    blended_h = [
        round(retrieval_w * r_ent[i] + rules_w * rules_h[i], 4)
        for i in range(n)
    ]
    blended_c = [
        round(retrieval_w * r_clo[i] + rules_w * rules_c[i], 4)
        for i in range(n)
    ]

    tested0 = state.tested_count
    entropy_points = [
        {"tested": tested0 + i, "theme_entropy": blended_h[i]}
        for i in range(n)
    ]
    summary = summarize_entropy_trajectory(entropy_points)
    total = sum(state.theme_histogram.values()) or 1
    sat_end = retrieved["saturation"][-1] if retrieved["saturation"] else 0.5

    return SimulationResult(
        policy_id=getattr(policy, "id", "search_policy"),
        steps=steps,
        entropy_trajectory=summary.to_dict(),
        entropy_points=entropy_points,
        closure_trajectory=blended_c,
        branching_factor=round(
            sum(retrieved["branching"]) / max(1, len(retrieved["branching"])), 4
        ),
        theme_saturation={"dominant": round(sat_end, 4)},
        expected_compute_cost=round((tested0 + steps) * 45.0, 1),
    )


def trajectories_from_simulation(sim: SimulationResult) -> dict[str, Any]:
    entropy_pts = sim.entropy_points
    closure_pts = [{"tested": i, "closure_ratio": v} for i, v in enumerate(sim.closure_trajectory)]
    snaps = [
        {
            "tested": entropy_pts[i].get("tested") if i < len(entropy_pts) else i,
            "theme_entropy": entropy_pts[i].get("theme_entropy") if i < len(entropy_pts) else 0,
            "closure_ratio": sim.closure_trajectory[i] if i < len(sim.closure_trajectory) else 0,
            "generated": i + 1,
        }
        for i in range(len(sim.closure_trajectory))
    ]
    return CampaignTrajectories.from_snapshots(snaps).to_dict()
