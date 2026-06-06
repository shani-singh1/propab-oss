"""Deterministic policy mutation — code only, never LLM."""
from __future__ import annotations

from typing import Any

from propab.knowledge_graph import KnowledgeGraph
from propab.meta_science import MetaScienceLedger
from propab.policy_record import PolicyRecord


def campaign_ids_in_bucket(
    meta: MetaScienceLedger,
    *,
    budget_bucket: str,
    domain_bucket: str,
) -> set[str]:
    return {
        o.campaign_id
        for o in meta.observations
        if o.budget_bucket == budget_bucket and o.domain_bucket == domain_bucket
    }


def mutate_policy_params(
    graph: KnowledgeGraph,
    meta: MetaScienceLedger,
    *,
    budget_bucket: str,
    domain_bucket: str,
    parent: PolicyRecord | None = None,
    campaign_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Derive boosts, penalties, blocked failures from bucket-local statistics only.
    """
    cids = campaign_ids_in_bucket(meta, budget_bucket=budget_bucket, domain_bucket=domain_bucket)
    rates = graph.theme_success_rates(campaign_ids=cids or None)

    boosts: dict[str, float] = {}
    penalties: dict[str, float] = {}
    saturated: list[str] = list(parent.saturated_themes if parent else [])

    for theme, rate in rates.items():
        if rate >= 0.4:
            boosts[theme] = round(min(0.35, rate * 0.5), 3)
        elif rate < 0.15:
            n = sum(
                1 for c in graph.claims.values()
                if c.theme == theme and (not cids or c.campaign_id in cids)
            )
            if n >= 5:
                penalties[theme] = 0.25
                if theme not in saturated:
                    saturated.append(theme)

    sig_counts: dict[str, int] = {}
    for f in graph.failures.values():
        if cids and f.campaign_id and f.campaign_id not in cids:
            continue
        if f.failure_signature:
            sig_counts[f.failure_signature] = sig_counts.get(f.failure_signature, 0) + 1
    blocked = [s for s, n in sig_counts.items() if n >= 3][:12]

    prefer_t2 = parent.prefer_replication_t2_plus if parent else True
    if campaign_metrics:
        cr = float(campaign_metrics.get("closure_ratio") or 0)
        if cr < (parent.closure_target if parent else 0.35):
            prefer_t2 = True

    return {
        "boosts": boosts,
        "penalties": penalties,
        "blocked_failures": blocked,
        "saturated_themes": saturated[:20],
        "prefer_replication_t2_plus": prefer_t2,
    }
