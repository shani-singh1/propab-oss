"""Search policy simulator — trajectory prediction (fixes.md Component 2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from propab.entropy_trajectory import summarize_entropy_trajectory
from propab.layer05.replay_state import SearchState
from propab.policy_record import PolicyRecord
from propab.research_quality import compute_theme_entropy
from propab.search_policy import SearchPolicy


@dataclass
class SimulationResult:
    policy_id: str
    steps: int
    entropy_trajectory: dict[str, Any]
    entropy_points: list[dict[str, Any]]
    closure_trajectory: list[float]
    branching_factor: float
    theme_saturation: dict[str, float]
    expected_compute_cost: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _theme_weights(policy: SearchPolicy, themes: list[str]) -> dict[str, float]:
    if not themes:
        themes = ["general"]
    weights = {t: policy.theme_weight(t) for t in themes}
    total = sum(weights.values()) or 1.0
    return {t: w / total for t, w in weights.items()}


def _weighted_theme_pick(weights: dict[str, float], rng_idx: int) -> str:
    themes = list(weights.keys())
    acc = 0.0
    target = (rng_idx % 1000) / 1000.0
    for t in themes:
        acc += weights[t]
        if target <= acc:
            return t
    return themes[-1]


def simulate_search(
    *,
    state: SearchState,
    policy: PolicyRecord | SearchPolicy,
    steps: int = 30,
) -> SimulationResult:
    """
    Deterministic forward simulation from a search state.

    Each step adds one node to the theme histogram using policy theme weights.
    No LLM, no sandbox — pure trajectory projection.
    """
    sp = policy if isinstance(policy, SearchPolicy) else policy.to_search_policy()
    hist = dict(state.theme_histogram)
    if not hist:
        hist = {"general": max(1, state.pending_nodes)}

    themes = list(hist.keys())
    weights = _theme_weights(sp, themes)
    closure = state.closure_ratio
    tested = max(0, state.tested_count)
    initial_entropy = state.entropy or compute_theme_entropy(hist)
    traj_points: list[dict[str, Any]] = [
        {"tested": tested, "theme_entropy": initial_entropy},
    ]
    closure_traj: list[float] = [closure]

    boost_strength = sum(sp.theme_boost.values())
    concentration = 1.0 / (1.0 + boost_strength * 2.0)

    for i in range(steps):
        theme = _weighted_theme_pick(weights, tested + i)
        hist[theme] = hist.get(theme, 0) + 1
        tested += 1
        entropy = compute_theme_entropy(hist)
        traj_points.append({"tested": tested, "theme_entropy": entropy})
        closure = max(0.05, closure * (0.98 + 0.01 * concentration))
        closure_traj.append(round(closure, 4))
        if tested % 5 == 0:
            weights = _theme_weights(sp, list(hist.keys()))

    summary = summarize_entropy_trajectory(traj_points)
    total = sum(hist.values()) or 1
    saturation = {k: round(v / total, 4) for k, v in hist.items()}

    return SimulationResult(
        policy_id=getattr(policy, "id", "search_policy"),
        steps=steps,
        entropy_trajectory=summary.to_dict(),
        entropy_points=traj_points,
        closure_trajectory=closure_traj,
        branching_factor=1.0,
        theme_saturation=saturation,
        expected_compute_cost=round(tested * 45.0, 1),
    )


def compare_simulation_to_observed(
    simulated: SimulationResult,
    observed_points: list[dict[str, Any]],
) -> dict[str, float]:
    """Residuals: simulated vs observed entropy trajectory."""
    obs = summarize_entropy_trajectory(observed_points)
    sim = simulated.entropy_trajectory
    return {
        "start_H": float(sim.get("H_start", 0)) - obs.H_start,
        "growth_rate": float(sim.get("growth_rate", 0)) - obs.growth_rate,
        "saturation_H": float(sim.get("H_end", 0)) - obs.H_end,
        "cross_H_1_5": float(sim.get("cross_H_1_5_at_tested") or 0)
        - float(obs.cross_H_1_5_at_tested or 0),
    }
