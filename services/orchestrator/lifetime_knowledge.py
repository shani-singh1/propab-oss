"""
Wire roadmap Phases A/D/E/G into campaign start and end.

Campaign N+1 loads knowledge + policy from Campaign N.
"""
from __future__ import annotations

import logging
from typing import Any

from propab.campaign import ResearchCampaign
from propab.knowledge_graph import KnowledgeGraph, new_id
from propab.mechanism_extractor import extract_mechanism_from_finding
from propab.meta_science import CampaignObservation, MetaScienceLedger
from propab.negative_knowledge import (
    extract_confirmed_claims,
    extract_failures_from_campaign,
    merge_failures_into_graph,
)
from propab.search_policy import SearchPolicy, update_policy_from_graph
from propab.theory_objects import form_theories_from_claims, merge_theories_into_graph

logger = logging.getLogger(__name__)


def load_lifetime_state() -> tuple[KnowledgeGraph, SearchPolicy, MetaScienceLedger]:
    return KnowledgeGraph.load(), SearchPolicy.load(), MetaScienceLedger.load()


def enrich_prior_from_lifetime(
    prior_dict: dict[str, Any],
    graph: KnowledgeGraph,
    policy: SearchPolicy,
) -> dict[str, Any]:
    """Inject cross-campaign facts and dead ends into prior (Campaign N+1 differs)."""
    out = dict(prior_dict or {})
    facts = list(out.get("established_facts") or [])
    seen = {str(f.get("text", ""))[:120] for f in facts if isinstance(f, dict)}
    for fact in graph.established_fact_texts(limit=20):
        key = fact["text"][:120]
        if key not in seen:
            facts.append(fact)
            seen.add(key)
    out["established_facts"] = facts

    dead = list(out.get("dead_ends") or [])
    seen_dead = {str(d.get("text", ""))[:120] for d in dead if isinstance(d, dict)}
    for text in graph.dead_end_texts(limit=25):
        if text[:120] not in seen_dead:
            dead.append({"text": text, "paper_ids": [], "source": "lifetime_knowledge"})
            seen_dead.add(text[:120])
    out["dead_ends"] = dead

    theories = [t.to_dict() for t in graph.theories.values()]
    if theories:
        out["lifetime_theories"] = theories[:8]
    out["search_policy_generation"] = policy.generation
    out["theme_policy"] = {
        "boost": dict(policy.theme_boost),
        "penalty": dict(policy.theme_penalty),
        "saturated": list(policy.saturated_themes),
    }
    return out


def apply_policy_to_tree_scoring(
    campaign: ResearchCampaign,
    policy: SearchPolicy,
) -> None:
    """Adjust frontier theme saturation from learned policy."""
    penalty = 0.15
    if policy.saturated_themes:
        penalty = min(0.35, 0.15 + 0.05 * len(policy.saturated_themes))
    campaign.hypothesis_tree.set_scoring_context(
        campaign.question,
        theme_saturation_penalty=penalty,
    )


def lifetime_context_for_seeds(graph: KnowledgeGraph, policy: SearchPolicy) -> str:
    lines: list[str] = []
    if graph.theories:
        lines.append("Established theories from prior campaigns:")
        for t in list(graph.theories.values())[:5]:
            lines.append(f"- [{t.name}] {t.mechanism_summary[:200]}")
    if policy.theme_boost:
        lines.append(f"High-yield themes to prioritize: {list(policy.theme_boost.keys())[:6]}")
    if policy.saturated_themes:
        lines.append(f"Saturated themes to avoid repeating: {policy.saturated_themes[:6]}")
    if policy.blocked_failure_signatures:
        lines.append(f"Failure patterns to avoid: {policy.blocked_failure_signatures[:6]}")
    return "\n".join(lines)


def ingest_campaign(
    campaign: ResearchCampaign,
    *,
    snapshot_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist campaign outcomes into lifetime knowledge + policy + meta-science."""
    graph, policy, meta = load_lifetime_state()
    cid = campaign.id
    if cid not in graph.campaign_ids:
        graph.campaign_ids.append(cid)

    tree = campaign.hypothesis_tree
    nodes = {k: v.to_dict() for k, v in tree.nodes.items()}
    ledger = list(tree.finding_ledger or [])

    claims = extract_confirmed_claims(cid, ledger)
    for c in claims:
        graph.add_claim(c)

    for entry in ledger:
        mech = extract_mechanism_from_finding(entry, campaign_id=cid)
        if mech and mech.claim_id:
            graph.add_mechanism(mech)

    failures = extract_failures_from_campaign(cid, nodes, ledger=ledger)
    n_fail = merge_failures_into_graph(graph, failures)

    theories = form_theories_from_claims(claims, min_support=2)
    n_theory = merge_theories_into_graph(graph, theories)

    summary = campaign.summary()
    tree_sum = tree.summary()
    metrics = {
        "closure_ratio": tree_sum.get("closure_ratio", 0),
        **(snapshot_metrics or {}),
    }
    policy = update_policy_from_graph(policy, graph, campaign_metrics=metrics)

    theme_hist = tree.theme_counts()
    total = max(1, len(tree.nodes))
    general_frac = theme_hist.get("general", 0) / total
    from propab.research_quality import compute_theme_entropy

    obs = CampaignObservation(
        campaign_id=cid,
        question=campaign.question[:300],
        tested=int(summary.get("total_hypotheses") or tree_sum.get("total_nodes") or 0),
        confirmed=int(summary.get("total_confirmed") or tree_sum.get("confirmed_count") or 0),
        refuted=int((tree_sum.get("verdict_counts") or {}).get("refuted", 0)),
        inconclusive=int((tree_sum.get("verdict_counts") or {}).get("inconclusive", 0)),
        closure_ratio=float(tree_sum.get("closure_ratio") or 0),
        theme_entropy=compute_theme_entropy(theme_hist),
        general_theme_fraction=round(general_frac, 4),
        compute_seconds=int(summary.get("compute_seconds_used") or 0),
        policy_generation=policy.generation,
        knowledge_claims=len(graph.claims),
        knowledge_failures=len(graph.failures),
    )
    meta.record(obs)

    graph.save()
    policy.save()
    meta.save()

    report = {
        "campaign_id": cid,
        "claims_added": len(claims),
        "failures_added": n_fail,
        "theories_added": n_theory,
        "policy_generation": policy.generation,
        "theme_boost": policy.theme_boost,
        "theme_penalty": policy.theme_penalty,
        "observations": len(meta.observations),
    }
    logger.info("[lifetime] ingested campaign %s: %s", cid, report)
    return report
