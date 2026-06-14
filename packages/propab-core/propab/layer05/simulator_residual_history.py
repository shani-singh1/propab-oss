"""Simulator residual history for analyst context (fixes.md P4)."""
from __future__ import annotations

from typing import Any

from propab.layer05.simulation_fitness_ledger import SimulationFitnessLedger


def simulator_residual_history_for_bucket(
    ledger: SimulationFitnessLedger,
    *,
    budget_bucket: str,
    domain_bucket: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    rows = ledger.for_bucket(
        budget_bucket=budget_bucket,
        domain_bucket=domain_bucket,
        limit=limit,
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "simulator_version": r.simulator_version,
            "campaign_id": r.replay_campaign_id[:8],
            "policy_id": r.policy_id[:16],
            "predicted_trajectory": r.predicted_trajectory,
            "observed_trajectory": r.observed_trajectory,
            "residuals": r.residuals,
            "accept_or_reject": r.accept_or_reject,
            "bench": (r.detail or {}).get("simulator_bench"),
        })
    return out
