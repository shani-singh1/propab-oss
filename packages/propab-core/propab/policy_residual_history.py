"""Residual history for policy analyst context (fixes.md P0)."""
from __future__ import annotations

from typing import Any

from propab.policy_fitness_ledger import PolicyFitnessLedger


def residual_history_for_bucket(
    fitness: PolicyFitnessLedger,
    *,
    budget_bucket: str,
    domain_bucket: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Prior (predicted, observed, residual) tuples for the same bucket."""
    rows = [
        r
        for r in fitness.records
        if r.budget_bucket == budget_bucket and r.domain_bucket == domain_bucket
    ]
    rows = rows[-limit:]
    out: list[dict[str, Any]] = []
    for r in rows:
        detail = r.detail or {}
        traj = detail.get("observed_trajectory") or detail.get("trajectory_summary")
        entry: dict[str, Any] = {
            "policy_id": r.policy_id[:16],
            "campaign_id": r.campaign_id[:8],
            "predictions": r.predictions,
            "observations": r.observations,
            "residuals": r.residuals,
            "accept_or_reject": r.accept_or_reject,
        }
        if traj:
            entry["observed_trajectory"] = traj
        out.append(entry)
    return out
