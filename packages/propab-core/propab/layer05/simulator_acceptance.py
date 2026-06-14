"""Simulator version acceptance thresholds (fixes.md P5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from propab.layer05.simulation_fitness_ledger import SIMULATOR_VERSION


@dataclass
class SimulatorAcceptanceResult:
    accepted: bool
    version: str
    directional_agreement: float
    entropy_mae: float
    closure_mae: float
    entropy_mae_improved: bool
    closure_mae_improved: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "version": self.version,
            "directional_agreement": self.directional_agreement,
            "entropy_mae": self.entropy_mae,
            "closure_mae": self.closure_mae,
            "entropy_mae_improved": self.entropy_mae_improved,
            "closure_mae_improved": self.closure_mae_improved,
            "reason": self.reason,
        }


def evaluate_simulator_acceptance(
    *,
    current: dict[str, float],
    previous: dict[str, float] | None,
    version: str = SIMULATOR_VERSION,
    directional_min: float = 0.80,
) -> SimulatorAcceptanceResult:
    """
    Accept simulator version only if:
    - directional agreement > 80%
    - entropy MAE decreases vs previous (or no previous baseline)
    - closure MAE decreases vs previous (or no previous baseline)
    """
    direction = float(current.get("directional_agreement") or 0)
    mae_e = float(current.get("mae_entropy") or 999)
    mae_c = float(current.get("mae_closure") or 999)

    if previous is None:
        accepted = direction >= directional_min
        return SimulatorAcceptanceResult(
            accepted=accepted,
            version=version,
            directional_agreement=direction,
            entropy_mae=mae_e,
            closure_mae=mae_c,
            entropy_mae_improved=True,
            closure_mae_improved=True,
            reason="first_baseline" if accepted else "directional_below_threshold",
        )

    prev_e = float(previous.get("mae_entropy") or mae_e)
    prev_c = float(previous.get("mae_closure") or mae_c)
    ent_improved = mae_e <= prev_e
    clo_improved = mae_c <= prev_c
    accepted = direction >= directional_min and ent_improved and clo_improved

    reason = "accepted"
    if direction < directional_min:
        reason = "directional_below_80pct"
    elif not ent_improved:
        reason = "entropy_mae_not_improved"
    elif not clo_improved:
        reason = "closure_mae_not_improved"

    return SimulatorAcceptanceResult(
        accepted=accepted,
        version=version,
        directional_agreement=direction,
        entropy_mae=mae_e,
        closure_mae=mae_c,
        entropy_mae_improved=ent_improved,
        closure_mae_improved=clo_improved,
        reason=reason,
    )


def evaluate_v2_acceptance(
    *,
    current: dict[str, float],
    baseline_v1: dict[str, float],
) -> SimulatorAcceptanceResult:
    """sim_v2: beat v1 on directional + both MAEs (fixes.md P6)."""
    direction = float(current.get("directional_agreement") or 0)
    mae_e = float(current.get("mae_entropy") or 999)
    mae_c = float(current.get("mae_closure") or 999)
    v1_dir = float(baseline_v1.get("directional_agreement") or 0)
    v1_e = float(baseline_v1.get("mae_entropy") or mae_e)
    v1_c = float(baseline_v1.get("mae_closure") or mae_c)
    dir_ok = direction > v1_dir
    ent_ok = mae_e < v1_e
    clo_ok = mae_c < v1_c
    accepted = dir_ok and ent_ok and clo_ok
    reason = "beats_v1" if accepted else "v2_not_better_than_v1"
    if not dir_ok:
        reason = "directional_not_above_v1"
    elif not ent_ok:
        reason = "entropy_mae_not_below_v1"
    elif not clo_ok:
        reason = "closure_mae_not_below_v1"
    return SimulatorAcceptanceResult(
        accepted=accepted,
        version="sim_v2",
        directional_agreement=direction,
        entropy_mae=mae_e,
        closure_mae=mae_c,
        entropy_mae_improved=ent_ok,
        closure_mae_improved=clo_ok,
        reason=reason,
    )


def evaluate_v3_acceptance(
    *,
    aggregate: dict[str, float],
    loo_aggregate: dict[str, float],
) -> SimulatorAcceptanceResult:
    """sim_v3: aggregate and LOO directional both > 80% (fixes.md P6)."""
    direction = float(aggregate.get("directional_agreement") or 0)
    loo_dir = float(loo_aggregate.get("directional_agreement") or 0)
    mae_e = float(aggregate.get("mae_entropy") or 999)
    mae_c = float(aggregate.get("mae_closure") or 999)
    accepted = direction >= 0.80 and loo_dir >= 0.80
    reason = "accepted" if accepted else "v3_below_80pct_loo_gate"
    if direction < 0.80:
        reason = "aggregate_directional_below_80pct"
    elif loo_dir < 0.80:
        reason = "loo_directional_below_80pct"
    return SimulatorAcceptanceResult(
        accepted=accepted,
        version="sim_v3",
        directional_agreement=direction,
        entropy_mae=mae_e,
        closure_mae=mae_c,
        entropy_mae_improved=True,
        closure_mae_improved=True,
        reason=reason,
    )
