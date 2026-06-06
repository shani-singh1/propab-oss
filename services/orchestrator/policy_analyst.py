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

from propab.knowledge_graph import KnowledgeGraph
from propab.policy_mutation import mutate_policy_params
from propab.policy_record import PolicyRecord, PredictedEffects
from propab.meta_science import MetaScienceLedger

logger = logging.getLogger(__name__)

_ANALYST_SCHEMA = {
    "rationale": "string",
    "predicted_effects": {
        "closure_ratio_delta": "float",
        "theme_entropy_delta": "float",
        "compute_efficiency_delta": "float",
        "refute_ratio_delta": "float",
    },
    "falsification_conditions": ["string"],
}


def _deterministic_narrative(
    *,
    parent: PolicyRecord | None,
    params: dict[str, Any],
    budget_bucket: str,
    domain_bucket: str,
) -> tuple[str, PredictedEffects, list[str]]:
    boosts = list((params.get("boosts") or {}).keys())[:4]
    penalties = list((params.get("penalties") or {}).keys())[:4]
    blocked = (params.get("blocked_failures") or [])[:4]
    rationale = (
        f"Bucket {domain_bucket}/{budget_bucket}: "
        f"boost themes {boosts or ['none']}, "
        f"penalize {penalties or ['none']}, "
        f"block signatures {blocked or ['none']}. "
        f"Parent generation {parent.generation if parent else 0}."
    )
    # Conservative predictions — evaluation uses tolerance bands
    predicted = PredictedEffects(
        closure_ratio_delta=0.02 if boosts else 0.0,
        theme_entropy_delta=-0.05 if penalties else 0.0,
        compute_efficiency_delta=0.0001 if boosts else 0.0,
        refute_ratio_delta=0.01 if blocked else 0.0,
    )
    falsification = [
        f"closure_ratio drops more than 50% vs bucket baseline in {budget_bucket}",
        f"theme_entropy rises above 1.5 without confirm rate improvement",
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
    fallback = _deterministic_narrative(
        parent=parent,
        params=params,
        budget_bucket=budget_bucket,
        domain_bucket=domain_bucket,
    )
    if llm is None:
        return (*fallback, params)

    prompt = (
        "You are a meta-science policy analyst. Given READ-ONLY mutation summary, "
        "output JSON only with keys: rationale, predicted_effects "
        "(closure_ratio_delta, theme_entropy_delta, compute_efficiency_delta, refute_ratio_delta), "
        "falsification_conditions (list of strings). "
        "Do NOT propose theme boosts or penalties — those are already fixed.\n\n"
        f"Bucket: {domain_bucket}/{budget_bucket}\n"
        f"Parent generation: {parent.generation if parent else 0}\n"
        f"Mutation summary (read-only): {json.dumps({k: params[k] for k in ('boosts', 'penalties', 'blocked_failures')})}\n"
        f"Campaign metrics: {json.dumps(campaign_metrics or {})}\n"
        f"Schema: {json.dumps(_ANALYST_SCHEMA)}"
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
    rationale, predicted, fals = _deterministic_narrative(
        parent=parent,
        params=params,
        budget_bucket=budget_bucket,
        domain_bucket=domain_bucket,
    )
    return rationale, predicted, fals, params
