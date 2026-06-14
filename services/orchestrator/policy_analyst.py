"""
LLM Policy Analyst — rationale and predictions only.

The LLM never edits boosts, penalties, or blocked_failures.
Those are produced solely by the deterministic mutation engine.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from propab.entropy_trajectory import EntropyTrajectorySummary
from propab.knowledge_graph import KnowledgeGraph
from propab.policy_mutation import mutate_policy_params
from propab.policy_record import PolicyRecord, PredictedEffects
from propab.meta_science import MetaScienceLedger
from propab.policy_fitness_ledger import PolicyFitnessLedger
from propab.policy_residual_history import residual_history_for_bucket
from propab.layer05.simulation_fitness_ledger import SimulationFitnessLedger
from propab.layer05.simulator_residual_history import simulator_residual_history_for_bucket

logger = logging.getLogger(__name__)

_ANALYST_SCHEMA = {
    "rationale": "string",
    "predicted_effects": {
        "closure_ratio_delta": "float",
        "compute_efficiency_delta": "float",
        "refute_ratio_delta": "float",
        "start_H": "float",
        "growth_rate": "float",
        "saturation_H": "float",
        "cross_H_1_5_at_tested": "float",
        "cross_H_2_0_at_tested": "float",
    },
    "falsification_conditions": ["string"],
}


def _trajectory_dict(
    trajectory_summary: EntropyTrajectorySummary | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if trajectory_summary is None:
        return None
    if isinstance(trajectory_summary, EntropyTrajectorySummary):
        return trajectory_summary.to_dict()
    return dict(trajectory_summary)


def _deterministic_entropy_prediction(
    trajectory: dict[str, Any] | None,
    *,
    has_boosts: bool,
) -> dict[str, float]:
    """Conservative V2 entropy dynamics from observed trajectory shape."""
    if trajectory and trajectory.get("n_snapshots", 0) >= 2:
        cross15 = trajectory.get("cross_H_1_5_at_tested")
        cross20 = trajectory.get("cross_H_2_0_at_tested")
        return {
            "start_H": float(trajectory.get("H_start") or 0.7),
            "growth_rate": float(trajectory.get("growth_rate") or 0.02),
            "saturation_H": float(trajectory.get("H_end") or trajectory.get("saturation_H") or 1.8),
            "cross_H_1_5_at_tested": float(cross15 if cross15 is not None else 30),
            "cross_H_2_0_at_tested": float(cross20 if cross20 is not None else 50),
        }
    if has_boosts:
        return {
            "start_H": 0.72,
            "growth_rate": 0.025,
            "saturation_H": 1.9,
            "cross_H_1_5_at_tested": 27.0,
            "cross_H_2_0_at_tested": 60.0,
        }
    return {
        "start_H": 0.7,
        "growth_rate": 0.02,
        "saturation_H": 1.5,
        "cross_H_1_5_at_tested": 35.0,
        "cross_H_2_0_at_tested": 70.0,
    }


def _deterministic_narrative(
    *,
    parent: PolicyRecord | None,
    params: dict[str, Any],
    budget_bucket: str,
    domain_bucket: str,
    trajectory_summary: dict[str, Any] | None = None,
) -> tuple[str, PredictedEffects, list[str]]:
    boosts = list((params.get("boosts") or {}).keys())[:4]
    penalties = list((params.get("penalties") or {}).keys())[:4]
    blocked = (params.get("blocked_failures") or [])[:4]
    traj_note = ""
    if trajectory_summary:
        traj_note = (
            f" Prior run entropy: H_start={trajectory_summary.get('H_start')}, "
            f"H_end={trajectory_summary.get('H_end')}, "
            f"pattern={trajectory_summary.get('growth_pattern')}."
        )
    rationale = (
        f"Bucket {domain_bucket}/{budget_bucket}: "
        f"boost themes {boosts or ['none']}, "
        f"penalize {penalties or ['none']}, "
        f"block signatures {blocked or ['none']}. "
        f"Parent generation {parent.generation if parent else 0}.{traj_note}"
    )
    entropy = _deterministic_entropy_prediction(trajectory_summary, has_boosts=bool(boosts))
    predicted = PredictedEffects(
        closure_ratio_delta=0.02 if boosts else 0.0,
        compute_efficiency_delta=0.0001 if boosts else 0.0,
        refute_ratio_delta=0.01 if blocked else 0.0,
        **entropy,
    )
    falsification = [
        f"closure_ratio drops more than 50% vs bucket baseline in {budget_bucket}",
        "cross_H_1_5_at_tested occurs before tested=10 (premature concentration)",
        "saturation_H below 1.2 without confirm rate improvement",
    ]
    if blocked:
        falsification.append(f"blocked signature {blocked[0]} appears in >20% of new failures")
    return rationale, predicted, falsification


def _parse_analyst_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _build_analyst_prompt(
    *,
    domain_bucket: str,
    budget_bucket: str,
    parent: PolicyRecord | None,
    params: dict[str, Any],
    campaign_metrics: dict[str, Any] | None,
    trajectory_summary: dict[str, Any] | None,
    residual_history: list[dict[str, Any]] | None,
    simulator_residual_history: list[dict[str, Any]] | None = None,
    trajectory_history: list[dict[str, Any]] | None = None,
) -> str:
    sections = [
        "You are a meta-science policy analyst. Given READ-ONLY mutation summary, "
        "prior residual history, and the just-finished campaign's entropy trajectory, "
        "output JSON only with keys: rationale, predicted_effects, falsification_conditions. "
        "Do NOT propose theme boosts or penalties — those are already fixed.\n",
        "Predict entropy DYNAMICS (not theme_entropy_delta): start_H, growth_rate, "
        "saturation_H, cross_H_1_5_at_tested, cross_H_2_0_at_tested. "
        "These describe within-campaign rise/plateau shape.\n",
        f"Bucket: {domain_bucket}/{budget_bucket}",
        f"Parent generation: {parent.generation if parent else 0}",
        f"Mutation summary (read-only): {json.dumps({k: params[k] for k in ('boosts', 'penalties', 'blocked_failures')})}",
        f"Campaign metrics: {json.dumps(campaign_metrics or {})}",
    ]
    if trajectory_summary:
        sections.append(
            "Just-finished campaign trajectory summary: "
            + json.dumps(trajectory_summary)
        )
    if residual_history:
        sections.append(
            "Prior campaign residual history (same bucket): "
            + json.dumps(residual_history[-8:])
        )
    if simulator_residual_history:
        sections.append(
            "Prior simulator residual history (replay+simulation): "
            + json.dumps(simulator_residual_history[-8:])
        )
    if trajectory_history:
        sections.append(
            "Prior observed trajectory summaries: "
            + json.dumps(trajectory_history[-6:])
        )
    sections.append(f"Schema: {json.dumps(_ANALYST_SCHEMA)}")
    return "\n\n".join(sections)


async def llm_policy_analyst(
    *,
    parent: PolicyRecord | None,
    graph: KnowledgeGraph,
    meta: MetaScienceLedger,
    budget_bucket: str,
    domain_bucket: str,
    campaign_metrics: dict[str, Any] | None,
    llm: Any,
    session_id: str,
    trajectory_summary: EntropyTrajectorySummary | dict[str, Any] | None = None,
    fitness: PolicyFitnessLedger | None = None,
    simulation_fitness: SimulationFitnessLedger | None = None,
) -> tuple[str, PredictedEffects, list[str], dict[str, Any]]:
    """
    LLM generates rationale, predicted_effects, falsification_conditions only.
    Mutation params are computed deterministically and passed as read-only context.
    """
    params = mutate_policy_params(
        graph,
        meta,
        budget_bucket=budget_bucket,
        domain_bucket=domain_bucket,
        parent=parent,
        campaign_metrics=campaign_metrics,
    )
    traj = _trajectory_dict(trajectory_summary)
    history = (
        residual_history_for_bucket(
            fitness,
            budget_bucket=budget_bucket,
            domain_bucket=domain_bucket,
        )
        if fitness is not None
        else []
    )
    sim_history = (
        simulator_residual_history_for_bucket(
            simulation_fitness,
            budget_bucket=budget_bucket,
            domain_bucket=domain_bucket,
        )
        if simulation_fitness is not None
        else []
    )
    trajectory_history = [
        r.get("observed_trajectory")
        for r in sim_history
        if r.get("observed_trajectory")
    ]
    fallback = _deterministic_narrative(
        parent=parent,
        params=params,
        budget_bucket=budget_bucket,
        domain_bucket=domain_bucket,
        trajectory_summary=traj,
    )
    if llm is None:
        return (*fallback, params)

    prompt = _build_analyst_prompt(
        domain_bucket=domain_bucket,
        budget_bucket=budget_bucket,
        parent=parent,
        params=params,
        campaign_metrics=campaign_metrics,
        trajectory_summary=traj,
        residual_history=history or None,
        simulator_residual_history=sim_history or None,
        trajectory_history=trajectory_history or None,
    )
    try:
        raw = await llm.call(
            prompt=prompt,
            purpose="policy_analyst",
            session_id=session_id,
        )
        parsed = _parse_analyst_json(raw)
        if not parsed:
            return (*fallback, params)
        pe = PredictedEffects.from_dict(parsed.get("predicted_effects"))
        rationale = str(parsed.get("rationale") or fallback[0])[:2000]
        fals = [str(x) for x in (parsed.get("falsification_conditions") or [])][:8]
        if not fals:
            fals = fallback[2]
        return rationale, pe, fals, params
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM policy analyst failed (%s); using deterministic narrative.", exc)
        return (*fallback, params)


def propose_policy_narrative_sync(
    *,
    parent: PolicyRecord | None,
    graph: KnowledgeGraph,
    meta: MetaScienceLedger,
    budget_bucket: str,
    domain_bucket: str,
    campaign_metrics: dict[str, Any] | None = None,
    trajectory_summary: EntropyTrajectorySummary | dict[str, Any] | None = None,
    fitness: PolicyFitnessLedger | None = None,
    simulation_fitness: SimulationFitnessLedger | None = None,
) -> tuple[str, PredictedEffects, list[str], dict[str, Any]]:
    """Sync path for tests and ingest fallback — deterministic only."""
    params = mutate_policy_params(
        graph,
        meta,
        budget_bucket=budget_bucket,
        domain_bucket=domain_bucket,
        parent=parent,
        campaign_metrics=campaign_metrics,
    )
    traj = _trajectory_dict(trajectory_summary)
    rationale, predicted, fals = _deterministic_narrative(
        parent=parent,
        params=params,
        budget_bucket=budget_bucket,
        domain_bucket=domain_bucket,
        trajectory_summary=traj,
    )
    return rationale, predicted, fals, params
