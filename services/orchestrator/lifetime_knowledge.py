"""
Wire roadmap Phases A/D/E/G into campaign start and end.

Campaign N+1 loads ACCEPTED policy for its bucket.
Campaign end proposes CANDIDATE only — never auto-promotes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from propab.campaign import ResearchCampaign
from propab.knowledge_graph import KnowledgeGraph
from propab.mechanism_extractor import extract_mechanism_from_finding
from propab.meta_science import CampaignObservation, MetaScienceLedger
from propab.negative_knowledge import (
    extract_confirmed_claims,
    extract_failures_from_campaign,
    merge_failures_into_graph,
)
from propab.policy_buckets import budget_bucket, domain_bucket
from propab.entropy_trajectory import EntropyTrajectorySummary, summarize_entropy_trajectory
from propab.policy_evaluation import evaluate_candidate_policy
from propab.policy_fitness_ledger import FitnessRecord, PolicyFitnessLedger
from propab.policy_record import PolicyRecord, PolicyStatus
from propab.policy_store import PolicyStore
from propab.search_policy import SearchPolicy
from propab.theory_objects import form_theories_from_claims, merge_theories_into_graph
from services.orchestrator.policy_analyst import propose_policy_narrative_sync

logger = logging.getLogger(__name__)


@dataclass
class LifetimeState:
    graph: KnowledgeGraph
    policy: SearchPolicy
    policy_record: PolicyRecord
    meta: MetaScienceLedger
    store: PolicyStore
    budget_bucket: str
    domain_bucket: str


def load_lifetime_state(
    campaign: ResearchCampaign,
    *,
    session_domain: str = "",
    policy_mode: Literal["accepted", "candidate"] | None = None,
) -> LifetimeState:
    """Load knowledge + bucket-local policy; bind campaign to active policy."""
    graph = KnowledgeGraph.load()
    meta = MetaScienceLedger.load()
    store = PolicyStore.load()
    bb = budget_bucket(campaign.compute_budget_seconds)
    db = domain_bucket(campaign.question, session_domain)
    mode: Literal["accepted", "candidate"] = (
        "candidate" if (policy_mode or campaign.policy_mode) == "candidate" else "accepted"
    )
    baseline_id: str | None = None
    if mode == "candidate":
        base_obs = meta.baseline_observation(budget_bucket=bb, domain_bucket=db)
        if base_obs:
            baseline_id = base_obs.campaign_id
    record = store.bind_campaign(
        campaign_id=campaign.id,
        policy_mode=mode,
        domain_bucket=db,
        budget_bucket=bb,
        baseline_campaign_id=baseline_id,
    )
    store.save()
    return LifetimeState(
        graph=graph,
        policy=record.to_search_policy(),
        policy_record=record,
        meta=meta,
        store=store,
        budget_bucket=bb,
        domain_bucket=db,
    )


def enrich_prior_from_lifetime(
    prior_dict: dict[str, Any],
    graph: KnowledgeGraph,
    policy: SearchPolicy,
    *,
    policy_record: PolicyRecord | None = None,
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
    out["policy_id"] = policy_record.id if policy_record else None
    out["policy_status"] = policy_record.status.value if policy_record else "ACCEPTED"
    out["theme_policy"] = {
        "boost": dict(policy.theme_boost),
        "penalty": dict(policy.theme_penalty),
        "saturated": list(policy.saturated_themes),
    }
    return out


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
    session_domain: str = "",
    analyst_overlay: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Persist knowledge; evaluate candidate if calibration run; propose new candidate.
    Never overwrites ACCEPTED policy without evaluation.
    """
    graph = KnowledgeGraph.load()
    meta = MetaScienceLedger.load()
    store = PolicyStore.load()
    fitness = PolicyFitnessLedger.load()
    bb = budget_bucket(campaign.compute_budget_seconds)
    db = domain_bucket(campaign.question, session_domain)
    binding = store.active_bindings.get(campaign.id)

    cid = campaign.id
    if cid not in graph.campaign_ids:
        graph.campaign_ids.append(cid)

    tree = campaign.hypothesis_tree
    nodes = {k: v.to_dict() for k, v in tree.nodes.items()}
    ledger = list(tree.finding_ledger or [])

    claims = extract_confirmed_claims(cid, ledger)
    for c in claims:
        graph.add_claim(c)

    from propab.domain_modules.registry import get_domain_plugin, resolve_domain_plugin
    from propab.numerical_seeds import classify_hypothesis_bucket
    from propab.synthesis_diversity import aggregate_diversity_distribution

    plugin = resolve_domain_plugin(
        question=campaign.question,
        payload={"domain_profile": getattr(campaign, "domain_profile", None)},
    )
    if plugin is None:
        plugin = get_domain_plugin(session_domain or db)
    seed_domain = plugin.domain_id if plugin is not None else (session_domain or db)
    confirmed_nodes = [
        nodes[nid] for nid, n in nodes.items()
        if isinstance(n, dict) and n.get("verdict") == "confirmed"
    ]
    if plugin is not None and confirmed_nodes:
        seeds = plugin.extract_numerical_seeds(confirmed_nodes)
        if seeds:
            graph.store_numerical_seeds(seed_domain, cid, seeds)
    buckets = [
        classify_hypothesis_bucket(
            str(n.get("text") or ""),
            str(n.get("test_methodology") or ""),
        )
        for n in nodes.values()
        if isinstance(n, dict) and n.get("verdict") in ("confirmed", "refuted", "inconclusive")
    ]
    if buckets:
        graph.store_diversity_distribution(session_domain or db, aggregate_diversity_distribution(buckets))

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
    raw_traj = (snapshot_metrics or {}).get("entropy_trajectory")
    trajectory_summary: EntropyTrajectorySummary | None = None
    if isinstance(raw_traj, dict) and raw_traj.get("n_snapshots", 0) > 0:
        trajectory_summary = EntropyTrajectorySummary(**{
            k: raw_traj[k]
            for k in EntropyTrajectorySummary.__dataclass_fields__
            if k in raw_traj
        })
    metrics = {
        "closure_ratio": tree_sum.get("closure_ratio", 0),
        **(snapshot_metrics or {}),
    }

    active_record = (
        store.get_policy(binding.policy_id)
        if binding
        else store.accepted_policy(domain_bucket=db, budget_bucket=bb)
    )
    policy_mode = binding.policy_mode if binding else "accepted"

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
        policy_generation=active_record.generation if active_record else 0,
        knowledge_claims=len(graph.claims),
        knowledge_failures=len(graph.failures),
        budget_bucket=bb,
        domain_bucket=db,
        policy_id=active_record.id if active_record else None,
        policy_mode=policy_mode,
    )
    meta.record(obs)

    evaluation_report: dict[str, Any] | None = None
    if binding and binding.policy_mode == "candidate" and active_record:
        baseline = meta.baseline_observation(
            budget_bucket=bb,
            domain_bucket=db,
            exclude_campaign_id=cid,
            pin_campaign_id=binding.baseline_campaign_id,
        )
        if baseline:
            accepted, detail = evaluate_candidate_policy(
                predicted=active_record.predicted_effects,
                baseline_obs=baseline,
                current_obs=obs,
                budget_bucket=bb,
                trajectory_summary=trajectory_summary,
            )
            detail = {
                **detail,
                "baseline_campaign_id": baseline.campaign_id,
                "accepted_policy_id": store.accepted.get(f"{db}:{bb}"),
            }
            fitness.record(FitnessRecord(
                policy_id=active_record.id,
                campaign_id=cid,
                budget_bucket=bb,
                domain_bucket=db,
                predictions=detail["predicted"],
                observations=detail["observed"],
                residuals=detail["residuals"],
                accept_or_reject=detail["accept_or_reject"],
                detail=detail,
            ))
            # Residual batch: always record evaluation; promote/reject at most once.
            if active_record.status == PolicyStatus.CANDIDATE:
                if accepted:
                    store.accept_policy(active_record.id)
                else:
                    store.reject_policy(active_record.id)
            evaluation_report = {
                "policy_id": active_record.id,
                "policy_status_at_eval": active_record.status.value,
                "accepted": accepted,
                **detail,
            }

    parent = store.accepted_policy(domain_bucket=db, budget_bucket=bb)
    if analyst_overlay:
        rationale = str(analyst_overlay.get("rationale") or "")
        from propab.policy_record import PredictedEffects
        predicted = PredictedEffects.from_dict(analyst_overlay.get("predicted_effects"))
        fals = list(analyst_overlay.get("falsification_conditions") or [])
        params = analyst_overlay.get("mutation_params") or {}
    else:
        rationale, predicted, fals, params = propose_policy_narrative_sync(
            parent=parent,
            graph=graph,
            meta=meta,
            budget_bucket=bb,
            domain_bucket=db,
            campaign_metrics=metrics,
            trajectory_summary=trajectory_summary,
            fitness=fitness,
        )

    candidate = store.add_candidate(
        parent=parent,
        params=params,
        rationale=rationale,
        predicted=predicted,
        falsification=fals,
    )

    store.unbind_campaign(cid)
    graph.save()
    store.save()
    meta.save()
    fitness.save()

    report = {
        "campaign_id": cid,
        "claims_added": len(claims),
        "failures_added": n_fail,
        "theories_added": n_theory,
        "budget_bucket": bb,
        "domain_bucket": db,
        "active_policy_id": active_record.id if active_record else None,
        "active_policy_status": active_record.status.value if active_record else None,
        "candidate_policy_id": candidate.id,
        "candidate_status": candidate.status.value,
        "evaluation": evaluation_report,
        "theme_boost": candidate.boosts,
        "theme_penalty": candidate.penalties,
        "observations": len(meta.observations),
        "rejected_policies": len(store.rejected_ids),
    }
    logger.info("[lifetime] ingested campaign %s: %s", cid, report)
    return report
