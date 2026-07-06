"""
Campaign loop — the long-running research driver.

A campaign wraps the existing multi-round session loop with:
1. A HypothesisTree that grows as findings accumulate
2. A measured baseline (never assumed)
3. BreakthroughCriteria to define success
4. Persistent checkpointing to Postgres after every batch

Usage:
    campaign = ResearchCampaign(
        id=str(uuid4()),
        question="Find optimal MLP for MNIST under 50k params",
        breakthrough_criteria=BreakthroughCriteria(
            metric_name="val_accuracy",
            improvement_threshold=0.05,
            direction="higher_is_better",
        ),
        compute_budget_seconds=14400,
    )
    await run_campaign_loop(campaign, session_factory=..., emitter=...)
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
import json
import logging
import re
import time
import traceback
from datetime import UTC, datetime
from uuid import UUID, uuid4, uuid5

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.campaign import (
    BreakthroughCriteria,
    ResearchCampaign,
    STATUS_ACTIVE,
    STATUS_BREAKTHROUGH,
    STATUS_BUDGET_EXHAUSTED,
    STOP_REASON_ALL_BRANCHES_EXHAUSTED,
    STOP_REASON_BREAKTHROUGH,
    STOP_REASON_DOMAIN_PREFLIGHT_FAILED,
    STOP_REASON_FATAL_ERROR,
    STOP_REASON_FRONTIER_REFILL_FAILED,
    STOP_REASON_HYPOTHESIS_CAP_REACHED,
    STOP_REASON_NO_BOOTSTRAP_SEEDS,
    STOP_REASON_NO_DISPATCHABLE_NODES,
    STOP_REASON_NO_SEEDS_GENERATED,
    STOP_REASON_SALVAGED_AFTER_ERROR,
    STOP_REASON_SYNTHESIS_EMPTY,
    STOP_REASON_TIME_BUDGET_EXHAUSTED,
)
from propab.campaign_db import (  # persistence lives in core so API/worker/orchestrator share it
    db_load_campaign,
    db_load_session_events_tail,
    db_save_campaign,
)
from propab.campaign_resume import backfill_belief_state_if_empty
from propab.campaign_snapshot import write_campaign_snapshot
from propab.campaign_synthesis import (
    _is_duplicate_frontier_candidate,
    run_campaign_synthesis_pass,
    should_trigger_synthesis,
)
from propab.research_quality import (
    NODE_ROLE_CONTROL,
    NODE_ROLE_DISCOVERY,
    build_canonical_finding,
    build_mechanism_object,
    build_refutation_mechanism,
    build_verification_escalation,
    classify_claim_strength,
    classify_inconclusive_reason,
    compute_claim_dedup_key,
    compute_evidence_hash,
    compute_replication_level,
    compute_verification_hash,
    extract_theme_vector,
    failure_signature_from_reason,
    infer_finding_links,
    infer_node_role,
    is_discovery_node,
    is_valid_evidence_for_hash,
    paper_eligible_finding,
    should_retest_inconclusive,
)
from propab.config import settings
from propab.db import create_redis
from propab.events import EventEmitter
from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.llm import LLMClient
from propab.types import EventType
from services.orchestrator.campaign_diagnostics import (
    classify_verification_method,
    frontier_snapshot,
    infer_hypothesis_theme,
    parse_evidence_obj,
)
from services.orchestrator.hypotheses import generate_ranked_hypotheses
from services.orchestrator.hypothesis_ranking import (
    compute_question_relevance_score_lexical,
    strip_question_suffix,
)
from services.orchestrator.hypotheses import _is_ml_template_hypothesis
from services.orchestrator.intake import parse_question
from services.orchestrator.lifetime_knowledge import (
    enrich_prior_from_lifetime,
    ingest_campaign,
    lifetime_context_for_seeds,
    load_lifetime_state,
)
from propab.entropy_trajectory import summarize_entropy_trajectory, trajectory_point_from_snapshot
from propab.layer05.simulation_fitness_ledger import SimulationFitnessLedger
from propab.policy_fitness_ledger import PolicyFitnessLedger
from services.orchestrator.policy_analyst import llm_policy_analyst
from services.orchestrator.literature import build_prior
from services.orchestrator.paper import _session_experiment_step_count, write_paper_minimal
from services.orchestrator.schemas import Prior
from services.worker.peer_findings import build_peer_finding_payload, publish_peer_finding
from services.worker.tasks import run_sub_agent_task

logger = logging.getLogger(__name__)

_NODE_ID_NAMESPACE = UUID("c3e7a1f0-4b2d-4e8a-95cf-0d1e3f5a7c9b")


def _campaign_ledger_from_tree(tree: HypothesisTree) -> dict[str, Any]:
    """Ledger shape aligned with AccumulatedLedger.summary() for paper_sections / gates."""
    confirmed = [
        nid for nid in tree.confirmed
        if nid in tree.nodes and is_discovery_node(tree.nodes[nid])
    ]
    refuted = [
        nid for nid, n in tree.nodes.items()
        if n.verdict == "refuted" and is_discovery_node(n)
    ]
    inconclusive = [nid for nid, n in tree.nodes.items() if n.verdict == "inconclusive"]
    return {
        "confirmed": confirmed,
        "refuted": refuted,
        "inconclusive": inconclusive,
        "total_confirmed": len(confirmed),
        "total_refuted": len(refuted),
        "total_inconclusive": len(inconclusive),
        "rounds_completed": 0,
    }


def _campaign_reasoning_trace(campaign: ResearchCampaign) -> dict[str, Any]:
    """Trace-derived reasoning context for the paper's Chain-of-Reasoning section.

    Emits ONLY structural facts read straight off the hypothesis tree and belief state
    (parent->child lineage, per-node verdict/generation, and the belief history). No
    finding is *claimed* here: verdicts are the tree's own labels and are re-gated against
    the authoritative DB rows at render time. Controls are excluded from lineage edges.
    """
    tree = campaign.hypothesis_tree
    nodes_out: dict[str, dict[str, Any]] = {}
    for nid, n in tree.nodes.items():
        nodes_out[str(nid)] = {
            "id": str(nid),
            "text": (n.text or "")[:400],
            "parent_id": str(n.parent_id) if n.parent_id else None,
            "depth": int(n.depth or 0),
            "generation": int(n.generation or 0),
            "verdict": str(n.verdict or "pending"),
            "expansion_type": n.expansion_type,
            "node_role": n.node_role,
            "primary_theme": n.primary_theme or n.theme_id,
            "mechanism": (n.mechanism or "")[:300] or None,
            "confidence": float(n.confidence or 0.0),
        }
    # Lineage edges (parent -> child) among discovery nodes only.
    edges: list[dict[str, str]] = []
    for nid, meta in nodes_out.items():
        pid = meta["parent_id"]
        if pid and pid in nodes_out and meta["node_role"] != NODE_ROLE_CONTROL:
            edges.append({"parent": pid, "child": nid, "expansion_type": meta["expansion_type"] or "expansion"})

    bs = campaign.belief_state
    beliefs = {
        "active": [b.to_dict() for b in bs.active_beliefs],
        "closed": [c.to_dict() for c in bs.closed_beliefs],
        "recent_activity": (bs.recent_activity_summary or "")[:600],
        "branch_exhausted": bool(bs.branch_exhausted),
    }
    # Per-generation verdict histogram: how beliefs/outcomes evolved across rounds.
    gen_hist: dict[int, dict[str, int]] = {}
    for meta in nodes_out.values():
        if meta["node_role"] == NODE_ROLE_CONTROL:
            continue
        g = int(meta["generation"])
        row = gen_hist.setdefault(g, {"confirmed": 0, "refuted": 0, "inconclusive": 0, "pending": 0})
        v = meta["verdict"] if meta["verdict"] in row else "pending"
        row[v] += 1
    generations = [
        {"generation": g, **gen_hist[g]} for g in sorted(gen_hist)
    ]
    return {
        "nodes": nodes_out,
        "lineage_edges": edges[:400],
        "beliefs": beliefs,
        "generations": generations,
        "max_depth": max((m["depth"] for m in nodes_out.values()), default=0),
        "max_generation": max((m["generation"] for m in nodes_out.values()), default=0),
    }


def _prior_snippets(prior_dict: dict[str, Any]) -> list[str]:
    snippets: list[str] = []
    for f in prior_dict.get("established_facts") or []:
        if isinstance(f, dict):
            t = str(f.get("text") or "").strip()
            if t:
                snippets.append(t[:600])
    return snippets


def _resolve_synthesis_domain_id(campaign: ResearchCampaign, parsed_domain: str = "") -> str:
    """Resolve domain plugin id from campaign tag/payload — not infer_session_domain()."""
    from propab.domain_modules.registry import resolve_domain_plugin

    plugin = resolve_domain_plugin(
        question=campaign.question,
        payload={"domain_profile": getattr(campaign, "domain_profile", None)},
    )
    if plugin is not None:
        return plugin.domain_id
    explicit = str(getattr(campaign, "domain_profile", None) or "").strip()
    if explicit:
        return explicit
    m = re.search(r"\[domain_profile:([a-z0-9_]+)\]", campaign.question or "", re.I)
    if m:
        return m.group(1).lower()
    return parsed_domain


def _apply_result_diagnostics(
    tree: HypothesisTree,
    node_id: str,
    verdict: str,
    confidence: float,
    evidence: str,
    *,
    failure_reason: str | None = None,
) -> None:
    """Persist quality metadata on tree nodes (fixes.md P0–P4 post-contagion)."""
    node = tree.nodes.get(node_id)
    if node is None:
        return
    node.node_role = infer_node_role(node.text)
    node.lineage_length = tree.lineage_length(node_id)

    evidence_obj = parse_evidence_obj(evidence)
    if verdict == "inconclusive" and not is_valid_evidence_for_hash(evidence_obj):
        fr_blob = (failure_reason or evidence or "").lower()
        if "timeout" in fr_blob or "revoke" in fr_blob:
            evidence_obj.setdefault("verdict_reason", "code timeout")
        elif not evidence_obj.get("verdict_reason"):
            evidence_obj["verdict_reason"] = "no metric-bearing steps executed"

    ev_hash = compute_evidence_hash(evidence_obj)
    ver_hash = compute_verification_hash(evidence_obj)
    node.evidence_hash = ev_hash
    node.verification_hash = ver_hash

    primary, secondary, theme_conf = extract_theme_vector(node.text)
    node.primary_theme = primary
    node.secondary_themes = secondary
    node.theme_id = primary
    node.theme_confidence = theme_conf

    claim_strength = classify_claim_strength(evidence_obj, verdict, hypothesis_text=node.text)
    node.claim_type = claim_strength
    verification_method = classify_verification_method(evidence)

    sibling_confirmed = 0
    if node.parent_id and node.parent_id in tree.nodes:
        parent = tree.nodes[node.parent_id]
        sibling_confirmed = sum(
            1 for c in parent.children
            if c in tree.nodes and tree.nodes[c].verdict == "confirmed"
        )
    node.replication_level = compute_replication_level(
        evidence_obj, hypothesis_text=node.text, sibling_confirmed=sibling_confirmed,
    )

    vr = str(evidence_obj.get("verdict_reason") or "")
    if verdict == "inconclusive":
        node.inconclusive_reason = classify_inconclusive_reason(
            evidence_obj, failure_reason=failure_reason, verdict_reason=vr,
        )
        node.failure_signature = failure_signature_from_reason(node.inconclusive_reason, verdict_reason=vr)

    effective_verdict = verdict
    if node.node_role == NODE_ROLE_CONTROL and verdict == "confirmed":
        effective_verdict = "inconclusive"
        node.verdict = "inconclusive"
        node.inconclusive_reason = "control_calibration"
        if node_id in tree.confirmed:
            tree.confirmed.remove(node_id)

    elif verdict == "confirmed" and is_discovery_node(node):
        if ev_hash is None:
            effective_verdict = "inconclusive"
            node.verdict = "inconclusive"
            node.inconclusive_reason = "metric_missing"
            node.failure_signature = failure_signature_from_reason(node.inconclusive_reason, verdict_reason=vr)
            if node_id in tree.confirmed:
                tree.confirmed.remove(node_id)
        elif not tree.register_confirmed_claim(compute_claim_dedup_key(node.text)):
            effective_verdict = "inconclusive"
            node.verdict = "inconclusive"
            node.inconclusive_reason = "duplicate_evidence"
            node.confidence = min(confidence, 0.4)
            if node_id in tree.confirmed:
                tree.confirmed.remove(node_id)

    mechanism_obj = None
    if effective_verdict == "confirmed" and is_discovery_node(node):
        mechanism_obj = build_mechanism_object(
            claim=node.text,
            mechanism=vr or None,
            evidence=evidence_obj,
            verdict="confirmed",
        )
        if mechanism_obj:
            node.mechanism = mechanism_obj.get("effect") or mechanism_obj.get("mechanism")
    elif effective_verdict == "refuted" and is_discovery_node(node):
        mechanism_obj = build_refutation_mechanism(
            claim=node.text,
            evidence=evidence_obj,
            verdict_reason=vr or str(failure_reason or "hypothesis refuted"),
        )
        node.mechanism = mechanism_obj.get("effect")

    node.verification_method = verification_method

    if effective_verdict in ("confirmed", "refuted") and claim_strength and is_discovery_node(node):
        links = infer_finding_links(tree.finding_ledger, {"claim_id": node_id, "primary_theme": primary, "verdict": effective_verdict, "secondary_themes": secondary}, parent_id=node.parent_id)
        node.finding = build_canonical_finding(
            claim_id=node_id,
            claim=node.text,
            claim_type=claim_strength,
            replication_level=node.replication_level or "T1",
            confidence=confidence,
            verification_method=verification_method,
            primary_theme=primary,
            secondary_themes=secondary,
            mechanism_obj=mechanism_obj,
            evidence_hash=ev_hash,
            verification_hash=ver_hash,
            node_role=node.node_role,
            verdict=effective_verdict,
            theme_confidence=theme_conf,
            links=links,
            failure_signature=node.failure_signature,
        )
        tree.finding_ledger.append(node.finding)


def build_campaign_synthesis_payload(campaign: ResearchCampaign) -> dict[str, Any]:
    """Synthesis dict for paper generation — single source of truth with tree ledger."""
    ledger = _campaign_ledger_from_tree(campaign.hypothesis_tree)
    ledger_findings = [
        f for f in campaign.hypothesis_tree.finding_ledger if paper_eligible_finding(f)
    ][:10]
    if not ledger_findings:
        ledger_findings = [
            {
                "node_id": nid,
                "claim": campaign.hypothesis_tree.nodes[nid].text,
                "text": campaign.hypothesis_tree.nodes[nid].text,
                "claim_type": campaign.hypothesis_tree.nodes[nid].claim_type,
                "replication_level": campaign.hypothesis_tree.nodes[nid].replication_level,
                "verification_method": campaign.hypothesis_tree.nodes[nid].verification_method,
                "confidence": campaign.hypothesis_tree.nodes[nid].confidence,
            }
            for nid in campaign.hypothesis_tree.confirmed[:10]
            if nid in campaign.hypothesis_tree.nodes
        ]
    return {
        "campaign_id": campaign.id,
        "status": campaign.status,
        "total_hypotheses_tested": campaign.total_hypotheses,
        "total_confirmed": ledger["total_confirmed"],
        "total_refuted": ledger["total_refuted"],
        "total_inconclusive": ledger["total_inconclusive"],
        "ledger": ledger,
        "best_finding": campaign.best_finding,
        "baseline_metric": campaign.baseline_metric,
        "best_metric": campaign.best_metric,
        "improvement_pct_over_baseline": round(campaign.improvement_pct * 100, 2),
        "metric_name": campaign.breakthrough_criteria.metric_name,
        "tree_summary": campaign.hypothesis_tree.summary(),
        "theme_histogram": campaign.hypothesis_tree.theme_counts(),
        "claim_histogram": {
            ct: sum(1 for n in campaign.hypothesis_tree.nodes.values() if n.claim_type == ct)
            for ct in {
                n.claim_type for n in campaign.hypothesis_tree.nodes.values() if n.claim_type
            }
        },
        "counts_source": "tree_preview_db_authoritative_at_paper_time",
        "confirmed_findings": ledger_findings,
        "reasoning_trace": _campaign_reasoning_trace(campaign),
    }


def _coerce_scalar_float(val: Any) -> float | None:
    """Accept int/float, numeric strings, and numpy scalars (item())."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip().replace("%", "")
        try:
            return float(s)
        except ValueError:
            return None
    item = getattr(val, "item", None)
    if callable(item):
        try:
            x = item()
            if isinstance(x, bool):
                return None
            if isinstance(x, (int, float)):
                return float(x)
        except Exception:
            return None
    return None


def _sanity_check_metric(metric_name: str, val: float | None) -> float | None:
    """Drop obviously wrong scalars when the campaign metric is classification accuracy."""
    if val is None:
        return None
    fv = float(val)
    mn = (metric_name or "").lower()
    if "acc" in mn or mn in ("metric_value", ""):
        # Treat 0–100 percentage reporting as a fraction
        if fv > 1.001 and fv <= 100.0:
            fv = fv / 100.0
        if fv < 0.01 or fv > 1.001:
            return None
    return float(fv)


def _deep_find_accuracy_metric(obj: Any, *, depth: int = 0) -> float | None:
    """Walk nested dict/list from tool outputs for common accuracy keys."""
    if depth > 14 or obj is None:
        return None
    if isinstance(obj, dict):
        for k in ("metric_value", "val_accuracy", "test_accuracy", "validation_accuracy", "accuracy"):
            v = _coerce_scalar_float(obj.get(k))
            if v is not None:
                return float(v)
        for v in obj.values():
            got = _deep_find_accuracy_metric(v, depth=depth + 1)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj[-24:]:
            got = _deep_find_accuracy_metric(v, depth=depth + 1)
            if got is not None:
                return got
    return None


def _normalize_baseline_dataset(raw: str | None, question: str) -> str:
    """Map noisy LLM dataset strings to train_model's expected name ('mnist' | 'synthetic')."""
    q = (question or "").lower()
    r = (str(raw or "").strip().lower())
    compact = r.replace(" ", "").replace("_", "").replace("-", "")
    if "mnist" in q and "fashion" not in q:
        return "mnist"
    if "mnist" in compact and "fashion" not in compact:
        return "mnist"
    return r or "synthetic"


def _normalize_metric_name(raw: str | None, campaign: ResearchCampaign) -> str:
    """Align LLM metric labels with worker result keys."""
    fallback = campaign.breakthrough_criteria.metric_name
    if not raw or not str(raw).strip():
        return fallback
    s = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if s in ("accuracy", "acc", "classification_accuracy", "top1", "top_1"):
        return "val_accuracy"
    if s in ("validation_accuracy", "validation_acc", "valid_acc"):
        return "val_accuracy"
    if "accuracy" in s and "test" in s:
        return "test_accuracy"
    if "val" in s and "accuracy" in s:
        return "val_accuracy"
    if "validation" in s and "accuracy" in s:
        return "val_accuracy"
    if "accuracy" in s and "loss" not in s:
        return "val_accuracy"
    return str(raw).strip()


def _deep_scan_accuracy_candidates(obj: Any, *, depth: int = 0) -> list[float]:
    """Collect sane accuracy scalars from arbitrarily nested worker / tool JSON."""
    found: list[float] = []
    if depth > 18 or obj is None:
        return found
    mn = "val_accuracy"
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in ("val_accuracy", "test_accuracy", "validation_accuracy", "accuracy"):
                fv = _coerce_scalar_float(v)
                ok = _sanity_check_metric(mn, fv)
                if ok is not None:
                    found.append(ok)
            found.extend(_deep_scan_accuracy_candidates(v, depth=depth + 1))
    elif isinstance(obj, list):
        for x in obj[-200:]:
            found.extend(_deep_scan_accuracy_candidates(x, depth=depth + 1))
    return found


def _extract_primary_metric_from_worker_result(result: dict, metric_name: str) -> float | None:
    """Parse val_accuracy (etc.) from Celery sub-agent return dict or embedded evidence JSON."""
    if not isinstance(result, dict):
        return None

    for key in (
        metric_name,
        "metric_value",
        "val_accuracy",
        "test_accuracy",
        "accuracy",
        "validation_accuracy",
    ):
        v = _coerce_scalar_float(result.get(key))
        if v is not None:
            got = _sanity_check_metric(metric_name, v)
            if got is not None:
                return got

    nested = _deep_find_accuracy_metric(result)
    if nested is not None:
        got = _sanity_check_metric(metric_name, nested)
        if got is not None:
            return got

    deep_vals = _deep_scan_accuracy_candidates(result)
    if deep_vals:
        got = _sanity_check_metric(metric_name, max(deep_vals))
        if got is not None:
            return got

    text_blob = str(result.get("key_finding") or "") + " " + str(result.get("evidence_summary") or "")
    m = re.search(r'"metric_value"\s*:\s*([0-9.eE+-]+)', text_blob)
    if m:
        try:
            got = _sanity_check_metric(metric_name, float(m.group(1)))
            if got is not None:
                return got
        except ValueError:
            pass
    matches = re.findall(r"(?:val_accuracy|accuracy|test_accuracy)[^\d]*([0-9]+\.?[0-9]*)", text_blob)
    if matches:
        try:
            got = _sanity_check_metric(metric_name, float(matches[0]))
            if got is not None:
                return got
        except ValueError:
            pass
    return None


def _parse_llm_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"_parse_error": True}
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {"_parse_error": True}
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else {"_parse_error": True}
        except json.JSONDecodeError:
            return {"_parse_error": True}


def _node_row_id(campaign_id: str, node_id: str) -> str:
    return str(uuid5(_NODE_ID_NAMESPACE, f"{campaign_id}:{node_id}"))


# ── DB helpers ───────────────────────────────────────────────────────────────
# db_save_campaign / db_load_campaign / db_load_session_events_tail and their
# meta helpers now live in propab.campaign_db (imported above) so that the API,
# worker, and orchestrator share one persistence path with no cross-service imports.


async def db_set_research_session_prior(
    session_id: str,
    session_factory: async_sessionmaker,
    prior_json: str,
) -> None:
    async with session_factory() as db:
        await db.execute(
            text("""
                UPDATE research_sessions
                SET prior_json = CAST(:prior_json AS jsonb)
                WHERE id = CAST(:id AS uuid)
            """),
            {"id": session_id, "prior_json": prior_json},
        )
        await db.commit()


async def db_set_research_session_stage(
    session_id: str,
    session_factory: async_sessionmaker,
    *,
    stage: str,
) -> None:
    """Update ``research_sessions.stage`` for UI / monitors (campaign phases)."""
    async with session_factory() as db:
        await db.execute(
            text("""
                UPDATE research_sessions
                SET stage = :stage
                WHERE id = CAST(:id AS uuid)
            """),
            {"id": session_id, "stage": stage},
        )
        await db.commit()


async def db_mark_research_session_completed(
    session_id: str,
    session_factory: async_sessionmaker,
) -> None:
    """
    Campaign shares ``research_sessions.id`` with the orchestrator session row.
    Keeps GET /campaigns and monitors aligned with the multi-round research loop.
    """
    async with session_factory() as db:
        await db.execute(
            text("""
                UPDATE research_sessions
                SET status = 'completed', stage = 'completed', completed_at = NOW()
                WHERE id = CAST(:id AS uuid)
            """),
            {"id": session_id},
        )
        await db.commit()


async def db_mark_research_session_failed(
    session_id: str,
    session_factory: async_sessionmaker,
) -> None:
    async with session_factory() as db:
        await db.execute(
            text("""
                UPDATE research_sessions
                SET status = 'failed', stage = COALESCE(stage, 'campaign'), completed_at = NOW()
                WHERE id = CAST(:id AS uuid)
            """),
            {"id": session_id},
        )
        await db.commit()


async def db_load_session_prior_json(
    session_id: str,
    session_factory: async_sessionmaker,
) -> dict[str, Any] | None:
    async with session_factory() as db:
        row = (await db.execute(
            text("SELECT prior_json FROM research_sessions WHERE id = CAST(:id AS uuid)"),
            {"id": session_id},
        )).mappings().one_or_none()
    if row is None:
        return None
    raw = row.get("prior_json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# ── Baseline measurement ─────────────────────────────────────────────────────

_ML_METRIC_TOKENS = (
    "accuracy", "loss", "error", "f1", "auc", "perplexity", "flops", "bleu", "mse", "rmse", "recall", "precision",
)
_ML_QUESTION_TOKENS = (
    "mnist", "cifar", "imagenet", "dataset", "neural", "network", " mlp", "transformer", "epoch", "optimizer",
    "classifier", "classification", "regression", "embedding", "architecture", "train a", "training a model",
)


def _is_ml_campaign(campaign: ResearchCampaign) -> bool:
    """Heuristic: does this campaign have an empirical ML training baseline to measure?

    Verification/math/combinatorics campaigns (Erdős-style problems) have no MLP to train,
    so the baseline step must be skipped rather than recording a meaningless trained metric.
    """
    metric = str(campaign.breakthrough_criteria.metric_name or "").lower()
    if any(tok in metric for tok in _ML_METRIC_TOKENS):
        return True
    q = str(campaign.question or "").lower()
    return any(tok in q for tok in _ML_QUESTION_TOKENS)


async def measure_baseline(
    campaign: ResearchCampaign,
    llm: LLMClient,
    emitter: EventEmitter,
    session_factory: async_sessionmaker,
) -> float:
    """
    Run the baseline experiment and return the primary metric value.
    Uses the campaign question to infer what "standard approach" means.
    """
    baseline_mode = str(getattr(settings, "campaign_baseline_mode", "sub_agent") or "sub_agent").strip().lower()
    if baseline_mode in {"skip", "none", "disabled"}:
        logger.warning("Campaign baseline measurement skipped by campaign_baseline_mode=%s", baseline_mode)
        return 0.0

    # Non-empirical (verification / math / combinatorics) campaigns have no ML training
    # baseline to measure — forcing an MLP train here would record a meaningless number.
    if not _is_ml_campaign(campaign):
        logger.info("Non-ML campaign detected; skipping ML baseline measurement.")
        await emitter.emit(
            session_id=campaign.id,
            event_type=EventType.BASELINE_MEASURED,
            step="campaign.baseline_measured",
            payload={
                "baseline_metric": 0.0,
                "metric_name": campaign.breakthrough_criteria.metric_name,
                "note": (
                    "Verification/computational campaign: no numeric training baseline. "
                    "Hypotheses are judged by deterministic verification, not metric improvement."
                ),
            },
        )
        return 0.0

    if baseline_mode == "fast_tool":
        config = {
            "dataset": "mnist" if "mnist" in campaign.question.lower() else "synthetic",
            "architecture": "auto MLP",
            "metric": campaign.breakthrough_criteria.metric_name,
            "n_steps": int(getattr(settings, "campaign_baseline_max_train_steps", 40)),
            "description": "Fast development baseline.",
        }
    else:
        baseline_prompt = f"""
Given this research campaign question:
"{campaign.question}"

What is the standard/baseline approach to measure?  Extract:
1. dataset (e.g. "mnist")
2. architecture description (e.g. "784-60-10 MLP")
3. metric to track (e.g. "val_accuracy")
4. training steps (e.g. 150)

Return JSON only:
{{"dataset": "...", "architecture": "...", "metric": "...", "n_steps": 150, "description": "..."}}
"""
        # Use a synthetic session_id for the baseline measurement LLM call
        baseline_session_id = f"campaign-{campaign.id}-baseline"
        try:
            raw = await llm.call(
                prompt=baseline_prompt,
                purpose="campaign.baseline_config",
                session_id=baseline_session_id,
            )
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            config = json.loads(json_match.group()) if json_match else {}
        except Exception:
            config = {}

    metric_name = _normalize_metric_name(
        config.get("metric") or campaign.breakthrough_criteria.metric_name,
        campaign,
    )
    dataset_name = _normalize_baseline_dataset(config.get("dataset"), campaign.question)
    n_steps = int(config.get("n_steps") or 150)

    # Run a minimal baseline experiment via the sub-agent infrastructure
    baseline_hypothesis = {
        "id": "baseline",
        "text": (
            f"Baseline: standard approach for '{campaign.question[:200]}'. "
            f"Metric to record: {metric_name}. "
            f"First call train_model with model_id='auto', dataset={dataset_name!r}, "
            f"n_steps from campaign config, task='classification'; use only schema-valid keys."
        ),
        "test_methodology": (
            f"Train {config.get('architecture', 'standard MLP')} on "
            f"{dataset_name} for {n_steps} steps. "
            f"Record {metric_name} (val_accuracy from train_model output)."
        ),
        "scores": {},
        "rank": 1,
        "gap_reference": "",
        "expected_result": f"Baseline {metric_name} recorded.",
        "refinement_of": None,
    }

    min_baseline_steps = 20 if baseline_mode == "fast_tool" else 35
    task_payload = {
        "session_id": campaign.id,
        "hypothesis_id": _node_row_id(campaign.id, "baseline"),
        "campaign_node_id": "baseline",
        "hypothesis": baseline_hypothesis,
        "baseline": {"metric_name": metric_name, "metric_value": 0.0, "description": "baseline measurement"},
        "prior": {"established_facts": [], "contested_claims": [], "open_gaps": [], "dead_ends": [], "key_papers": []},
        "domain": "deep_learning",
        "question": campaign.question,
        "agent_limits": {
            "max_steps": int(getattr(settings, "campaign_baseline_agent_max_steps", 6)),
            "max_seconds": int(getattr(settings, "campaign_baseline_agent_max_seconds", 480)),
            "max_tool_calls": int(getattr(settings, "campaign_baseline_agent_max_tool_calls", 14)),
        },
        # Worker: if the LLM/tool trace omits val_accuracy, run train_model once deterministically.
        "baseline_measurement": {
            "dataset": dataset_name,
            "n_steps": max(
                min_baseline_steps,
                min(
                    int(n_steps),
                    int(getattr(settings, "campaign_baseline_max_train_steps", 150)),
                ),
            ),
        },
    }
    if baseline_mode == "fast_tool":
        task_payload["fast_path"] = "baseline_measurement"

    try:
        async with session_factory() as db:
            await db.execute(
                text(
                    """
                    INSERT INTO hypotheses (
                        id, session_id, round_id, text, test_methodology, scores_json,
                        rank, status, verdict, confidence, evidence_summary, key_finding, created_at
                    ) VALUES (
                        CAST(:id AS uuid), CAST(:session_id AS uuid), NULL,
                        :text, :test_methodology, CAST(:scores_json AS jsonb),
                        :rank, :status, NULL, NULL, NULL, NULL, NOW()
                    )
                    ON CONFLICT (id) DO NOTHING
                    """
                ),
                {
                    "id": task_payload["hypothesis_id"],
                    "session_id": campaign.id,
                    "text": baseline_hypothesis["text"],
                    "test_methodology": baseline_hypothesis["test_methodology"],
                    "scores_json": "{}",
                    "rank": 1,
                    "status": "pending",
                },
            )
            await db.commit()

        ar = run_sub_agent_task.delay(task_payload)
        _bl_to = max(15, int(getattr(settings, "campaign_baseline_worker_timeout_sec", 600)))
        result = await asyncio.to_thread(lambda: ar.get(timeout=_bl_to))
        metric_val = _extract_primary_metric_from_worker_result(result, metric_name)
        if metric_val is None and metric_name != "val_accuracy":
            metric_val = _extract_primary_metric_from_worker_result(result, "val_accuracy")
        if metric_val is None:
            logger.warning(
                "Baseline measurement could not read metric %r from worker result; using 0.0",
                metric_name,
            )
            return 0.0
        return float(metric_val)
    except Exception as exc:
        logger.warning("Baseline measurement failed (%s); using 0.0", exc)
        return 0.0


# ── Campaign synthesis (fixes.md redesign) ───────────────────────────────────

def _count_dispatchable_candidates(tree: HypothesisTree, exclude_ids: set[str] | frozenset[str]) -> int:
    ex = frozenset(exclude_ids or ())
    return sum(
        1
        for nid in tree.frontier
        if nid in tree.nodes
        and tree.nodes[nid].verdict == "pending"
        and nid not in ex
    )


async def _maybe_run_campaign_synthesis(
    *,
    campaign: ResearchCampaign,
    llm: LLMClient,
    emitter: EventEmitter,
    generation: int,
    max_concurrent: int,
    inflight_ids: set[str],
    prior_snippets: list[str] | None,
    session_factory: async_sessionmaker | None = None,
    lifetime_context: str = "",
    lifetime_context_ref: list[str] | None = None,
) -> bool:
    """Tier-2 synthesis if triggered. Returns True if candidates were added."""
    ctx = lifetime_context_ref[0] if lifetime_context_ref else lifetime_context
    if not bool(getattr(settings, "campaign_synthesis_enabled", True)):
        return False
    if campaign.belief_state.branch_exhausted:
        return False
    queued = _count_dispatchable_candidates(campaign.hypothesis_tree, inflight_ids)
    if not should_trigger_synthesis(
        campaign.belief_state,
        results_since=campaign.belief_state.results_since_last_synthesis,
        max_concurrent=max_concurrent,
        queued_candidates=queued,
        threshold_multiplier=float(getattr(settings, "campaign_synthesis_trigger_multiplier", 1.0) or 1.0),
    ):
        return False
    added, metrics = await run_campaign_synthesis_pass(
        campaign_id=campaign.id,
        question=campaign.question,
        tree=campaign.hypothesis_tree,
        belief_state=campaign.belief_state,
        llm=llm,
        generation=generation,
        prior_snippets=prior_snippets,
        emitter=emitter,
        session_factory=session_factory,
        domain_id=_resolve_synthesis_domain_id(campaign),
        lifetime_context=ctx,
    )
    return len(added) > 0


async def generate_seed_hypotheses(
    campaign: ResearchCampaign,
    prior: object,
    parsed_question: object,
    llm: LLMClient,
    emitter: EventEmitter,
    session_factory: async_sessionmaker,
    generation: int,
    *,
    lifetime_context: str = "",
) -> list[HypothesisNode]:
    """
    Generate seed hypotheses when the frontier is empty (new campaign or exhausted tree).
    Reuses the existing hypothesis generator but plants results in the tree.
    When seed_source=anomaly, seeds come from mechanism-inducer output (Phase 7).
    """
    from propab.seed_source import SeedSource

    prior_context = campaign.hypothesis_tree.confirmed_findings_text(max_n=10)
    if lifetime_context.strip():
        prior_context = f"{lifetime_context.strip()}\n\n{prior_context}".strip()

    batch = settings.campaign_batch_size
    if campaign.seed_source == SeedSource.ANOMALY.value:
        from pathlib import Path

        from services.orchestrator.anomaly_seeds import (
            generate_anomaly_seed_hypotheses,
            load_mechanisms_from_artifacts,
            resolve_artifacts_dir,
        )

        art_dir = resolve_artifacts_dir(campaign.anomaly_artifacts_dir or "artifacts")
        mechanisms = load_mechanisms_from_artifacts(art_dir)
        raw_hyps = await generate_anomaly_seed_hypotheses(
            parsed_question,
            prior,
            mechanisms,
            max_hypotheses=batch,
            llm=llm,
            session_id=campaign.id,
            emitter=emitter,
            prior_round_findings=prior_context,
            artifacts_dir=art_dir,
        )
    else:
        raw_hyps = await generate_ranked_hypotheses(
            parsed_question,
            prior,
            max_hypotheses=batch,
            llm=llm,
            session_id=campaign.id,
            emitter=emitter,
            prior_round_findings=prior_context,
        )
    seed_dicts = []
    same_round_texts: list[str] = []
    from propab.domain_modules.registry import hypothesis_is_on_topic
    from propab.scoped_claim import enrich_entry_with_scope, parse_scope_from_methodology, validate_scoped_claim

    for h in raw_hyps:
        raw_text = h.text
        if not hypothesis_is_on_topic(
            raw_text,
            question=campaign.question,
            test_methodology=str(h.test_methodology or ""),
        ):
            continue
        dup, _ = _is_duplicate_frontier_candidate(
            raw_text,
            campaign.hypothesis_tree,
            campaign.belief_state,
            same_round_texts=same_round_texts,
        )
        if dup:
            continue
        entry = enrich_entry_with_scope(
            {"id": h.id, "text": raw_text, "test_methodology": h.test_methodology},
            parsed_question.text,
        )
        scope = parse_scope_from_methodology(entry["text"], entry["test_methodology"])
        ok, _ = validate_scoped_claim(scope)
        if not ok:
            continue
        same_round_texts.append(raw_text)
        seed_dicts.append({
            "id": h.id,
            "text": entry["text"],
            "test_methodology": entry["test_methodology"],
            "claim_scope": entry.get("claim_scope"),
            "theme_id": (h.scores or {}).get("theme_id") or infer_hypothesis_theme(entry["text"]),
            "question_relevance_score": (h.scores or {}).get("question_relevance"),
            "feature_subset": (h.scores or {}).get("feature_subset") or [],
            "mechanism_id": (h.scores or {}).get("mechanism_id"),
        })
    if not seed_dicts and raw_hyps:
        for h in raw_hyps:
            if not hypothesis_is_on_topic(
                h.text,
                question=campaign.question,
                test_methodology=str(h.test_methodology or ""),
            ):
                continue
            entry = enrich_entry_with_scope(
                {"id": h.id, "text": h.text, "test_methodology": h.test_methodology},
                parsed_question.text,
            )
            seed_dicts.append({
                "id": h.id,
                "text": entry["text"],
                "test_methodology": entry["test_methodology"],
                "claim_scope": entry.get("claim_scope"),
                "theme_id": infer_hypothesis_theme(entry["text"]),
                "question_relevance_score": (h.scores or {}).get("question_relevance"),
            })
            break
    from propab.synthesis_diversity import filter_seed_dicts_for_diversity

    node_dicts = {
        nid: (n.to_dict() if hasattr(n, "to_dict") else n)
        for nid, n in campaign.hypothesis_tree.nodes.items()
    }
    seed_dicts = filter_seed_dicts_for_diversity(
        seed_dicts,
        tree_nodes=node_dicts,
        active_belief_statements=[
            b.statement for b in campaign.belief_state.active_beliefs
        ],
        question=campaign.question,
        generation=generation,
    )
    return campaign.hypothesis_tree.add_seeds(seed_dicts, generation=generation)


# ── Sub-agent dispatch (pipelined campaign pool) ────────────────────────────


async def _campaign_dispatch_sub_agent(
    *,
    campaign: ResearchCampaign,
    node: HypothesisNode,
    prior_dict: dict,
    domain: str,
    emitter: EventEmitter,
    session_factory: async_sessionmaker,
) -> dict:
    """Insert hypothesis row, enqueue Celery sub-agent, emit dispatch; return pending handle dict."""
    db_hid = _node_row_id(campaign.id, node.id)
    async with session_factory() as db:
        await db.execute(
            text(
                """
                INSERT INTO hypotheses (
                    id, session_id, round_id, text, test_methodology, scores_json,
                    rank, status, verdict, confidence, evidence_summary, key_finding, created_at
                ) VALUES (
                    CAST(:id AS uuid), CAST(:session_id AS uuid), NULL,
                    :text, :test_methodology, CAST(:scores_json AS jsonb),
                    :rank, :status, NULL, NULL, NULL, NULL, NOW()
                )
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {
                "id": db_hid,
                "session_id": campaign.id,
                "text": node.text,
                "test_methodology": "",
                "scores_json": "{}",
                "rank": 1,
                "status": "pending",
            },
        )
        await db.commit()

    hypothesis_dict = {
        "id": node.id,
        "text": node.text,
        "test_methodology": "",
        "scores": {},
        "rank": 1,
        "gap_reference": "",
        "expected_result": "",
        "refinement_of": node.parent_id,
    }
    await emitter.emit(
        session_id=campaign.id,
        event_type=EventType.HYPO_DISPATCHED,
        step="hypothesis.dispatch",
        payload={"hypothesis_id": node.id, "depth": node.depth, "generation": node.generation},
        hypothesis_id=db_hid,
    )
    parent = campaign.hypothesis_tree.nodes.get(node.parent_id) if node.parent_id else None
    verification_escalation = build_verification_escalation(node, parent=parent)
    agent_limits: dict[str, Any] = {}
    if verification_escalation.get("prefer_smaller_experiment"):
        scale = float(verification_escalation.get("max_steps_scale") or 0.75)
        agent_limits["max_steps"] = max(5, int(int(settings.agent_max_steps) * scale))
        agent_limits["max_tool_calls"] = max(3, int(int(getattr(settings, "agent_max_tool_calls", 12) or 12) * scale))

    ar = run_sub_agent_task.delay({
        "session_id": campaign.id,
        "hypothesis_id": db_hid,
        "campaign_node_id": node.id,
        "hypothesis": hypothesis_dict,
        "baseline": {
            "metric_name": campaign.breakthrough_criteria.metric_name,
            "metric_value": campaign.baseline_metric,
            "description": "Campaign baseline measured at start.",
            "lit_compare_safe": abs(float(campaign.baseline_metric)) >= 1e-12,
        },
        "prior": prior_dict,
        "domain": domain,
        "question": campaign.question,
        "seed_source": campaign.seed_source,
        "verification_escalation": verification_escalation,
        "agent_limits": agent_limits or None,
    })
    return {"ar": ar, "nid": node.id, "db_hid": db_hid, "enq_mono": time.monotonic()}


def _peer_db_hids_inflight(pending: list[dict], *, exclude_nid: str | None = None) -> list[str]:
    return [p["db_hid"] for p in pending if exclude_nid is None or p["nid"] != exclude_nid]


async def _iter_campaign_pipelined_results(
    *,
    campaign: ResearchCampaign,
    max_concurrent: int,
    prior_dict: dict,
    domain: str,
    emitter: EventEmitter,
    session_factory: async_sessionmaker,
    llm: LLMClient | None = None,
    generation: int = 0,
    prior_snippets: list[str] | None = None,
    lifetime_context: str = "",
    lifetime_context_ref: list[str] | None = None,
) -> AsyncIterator[dict]:
    """
    Pipelined dispatch: keep up to ``max_concurrent`` Celery sub-agents running.

    Whenever a slot frees (completion, eviction, or deadline straggler), pull the next
    best frontier node (excluding in-flight ids) so tree expansions mid-wave are picked
    up without waiting for an entire fixed batch to drain.
    """
    pending: list[dict] = []
    completed_ids: set[str] = set()
    last_progress_mono = time.monotonic()

    remaining = max(60.0, float(campaign.remaining_seconds()))
    batch_max_wait = max(0, int(getattr(settings, "campaign_batch_max_wait_sec", 0) or 0))
    if batch_max_wait > 0:
        batch_wait = min(remaining, float(batch_max_wait))
    else:
        batch_wait = min(
            remaining * 0.5,
            max(1800.0, float(max_concurrent) * 2700.0),
        )
    batch_deadline = time.monotonic() + max(5.0, batch_wait)
    evict_after = max(0, int(getattr(settings, "campaign_frontier_evict_idle_sec", 0)))

    _redis = None
    try:
        _redis = await create_redis(settings.redis_url)
    except Exception as exc:
        logger.warning("Redis unavailable for peer broadcast: %s", exc)

    async def refill_slots() -> bool:
        """Dispatch from frontier until at cap or no candidates. Returns True if any dispatch."""
        nonlocal last_progress_mono
        ctx = lifetime_context_ref[0] if lifetime_context_ref else lifetime_context
        progressed_local = False
        while len(pending) < max_concurrent and not campaign.should_stop():
            if campaign.hypothesis_cap_reached():
                break
            inflight = {p["nid"] for p in pending}
            if llm is not None:
                await _maybe_run_campaign_synthesis(
                    campaign=campaign,
                    llm=llm,
                    emitter=emitter,
                    generation=generation,
                    max_concurrent=max_concurrent,
                    inflight_ids=inflight,
                    prior_snippets=prior_snippets,
                    session_factory=session_factory,
                    lifetime_context=ctx,
                    lifetime_context_ref=lifetime_context_ref,
                )
            node = campaign.hypothesis_tree.next_dispatch_candidate(inflight)
            if node is None:
                break
            item = await _campaign_dispatch_sub_agent(
                campaign=campaign,
                node=node,
                prior_dict=prior_dict,
                domain=domain,
                emitter=emitter,
                session_factory=session_factory,
            )
            pending.append(item)
            last_progress_mono = time.monotonic()
            progressed_local = True
        return progressed_local

    await refill_slots()

    while time.monotonic() < batch_deadline:
        if not pending and not await refill_slots():
            if campaign.hypothesis_tree.next_dispatch_candidate(frozenset()) is None:
                break
            if campaign.should_stop():
                break
            await asyncio.sleep(0.05)
            continue

        progressed = False
        max_wall = max(0, int(getattr(settings, "campaign_sub_agent_max_wall_sec", 0) or 0))
        if max_wall > 0 and pending:
            now_mono = time.monotonic()
            for victim in list(pending):
                ar_w = victim["ar"]
                if await asyncio.to_thread(ar_w.ready):
                    continue
                age = now_mono - float(victim.get("enq_mono", now_mono))
                if age <= float(max_wall):
                    continue
                pending.remove(victim)
                try:
                    await asyncio.to_thread(lambda ar_=ar_w: ar_.revoke(terminate=True))
                except Exception as exc:
                    logger.warning("[campaign %s] Celery revoke (per-task wall) failed: %s", campaign.id, exc)
                ev_note = (
                    f"Sub-agent exceeded campaign_sub_agent_max_wall_sec={max_wall}s since dispatch; "
                    "revoked to unblock the frontier."
                )
                wall_result = {
                    "hypothesis_id": victim["db_hid"],
                    "campaign_node_id": victim["nid"],
                    "verdict": "inconclusive",
                    "confidence": 0.0,
                    "evidence_summary": ev_note,
                    "key_finding": None,
                    "metric_value": None,
                    "failure_reason": "sub_agent_wall_exceeded",
                }
                completed_ids.add(victim["nid"])
                last_progress_mono = time.monotonic()
                progressed = True
                await emitter.emit(
                    session_id=campaign.id,
                    event_type=EventType.CAMPAIGN_SUB_AGENT_EVICTED,
                    step="campaign.sub_agent_wall",
                    payload={
                        "node_id": victim["nid"],
                        "max_wall_sec": max_wall,
                        "reason": ev_note,
                    },
                    hypothesis_id=victim["db_hid"],
                )
                yield wall_result
                still_running_db = _peer_db_hids_inflight(pending)
                if still_running_db and _redis is not None:
                    finding_payload = build_peer_finding_payload(wall_result)
                    await publish_peer_finding(
                        _redis, target_hypothesis_ids=still_running_db, finding=finding_payload
                    )
                await refill_slots()
            if progressed:
                continue

        for item in list(pending):
            ar, nid, db_hid = item["ar"], item["nid"], item["db_hid"]
            ready = await asyncio.to_thread(lambda: ar.ready())
            if not ready:
                continue
            progressed = True
            still_running_db = _peer_db_hids_inflight(pending, exclude_nid=nid)
            pending.remove(item)
            try:
                result = await asyncio.to_thread(lambda: ar.get(timeout=30))
            except Exception as exc:
                result = {
                    "hypothesis_id": db_hid,
                    "verdict": "inconclusive",
                    "confidence": 0.0,
                    "evidence_summary": f"Sub-agent failed: {exc}",
                    "key_finding": None,
                    "metric_value": None,
                    "failure_reason": str(exc),
                }
            result["campaign_node_id"] = str(result.get("campaign_node_id") or nid)
            completed_ids.add(nid)
            last_progress_mono = time.monotonic()
            campaign.belief_state.results_since_last_synthesis += 1
            if campaign.hypothesis_cap_reached():
                yield result
                return
            yield result
            if still_running_db and _redis is not None:
                finding_payload = build_peer_finding_payload(result)
                await publish_peer_finding(
                    _redis, target_hypothesis_ids=still_running_db, finding=finding_payload
                )
            await refill_slots()

        stalled = pending and evict_after > 0 and (time.monotonic() - last_progress_mono) >= evict_after
        if stalled:
            idle_candidates = []
            for item in pending:
                ar_live = item["ar"]
                rdy_live = await asyncio.to_thread(ar_live.ready)
                if not rdy_live:
                    idle_candidates.append(item)
            idle_candidates.sort(key=lambda x: float(x.get("enq_mono", time.monotonic())))
            if idle_candidates:
                victim = idle_candidates[0]
                pending.remove(victim)
                try:
                    await asyncio.to_thread(lambda ar_=victim["ar"]: ar_.revoke(terminate=True))
                except Exception as exc:
                    logger.warning("[campaign %s] Celery revoke failed for %s: %s", campaign.id, victim["nid"], exc)
                ev_note = (
                    f"Frontier idle exceeded {evict_after}s — sub-agent revoked (timeout_eviction). "
                    "Another hypothesis was blocking the scheduled batch."
                )
                eviction_result = {
                    "hypothesis_id": victim["db_hid"],
                    "campaign_node_id": victim["nid"],
                    "verdict": "inconclusive",
                    "confidence": 0.0,
                    "evidence_summary": ev_note,
                    "key_finding": None,
                    "metric_value": None,
                    "failure_reason": "timeout_eviction",
                }
                completed_ids.add(victim["nid"])
                last_progress_mono = time.monotonic()
                progressed = True
                await emitter.emit(
                    session_id=campaign.id,
                    event_type=EventType.CAMPAIGN_SUB_AGENT_EVICTED,
                    step="campaign.sub_agent_evict",
                    payload={
                        "node_id": victim["nid"],
                        "idle_sec_threshold": evict_after,
                        "reason": ev_note,
                    },
                    hypothesis_id=victim["db_hid"],
                )
                yield eviction_result
                still_running_db = _peer_db_hids_inflight(pending)
                if still_running_db and _redis is not None:
                    finding_payload = build_peer_finding_payload(eviction_result)
                    await publish_peer_finding(
                        _redis, target_hypothesis_ids=still_running_db, finding=finding_payload
                    )
                await refill_slots()
                continue

        if not progressed and pending:
            await asyncio.sleep(0.15)
        elif not progressed and not pending:
            if not await refill_slots():
                if campaign.hypothesis_tree.next_dispatch_candidate(frozenset()) is None:
                    break
                await asyncio.sleep(0.05)

    if _redis is not None:
        try:
            await _redis.close()
        except Exception:
            pass

    for item in pending:
        yield {
            "hypothesis_id": item["db_hid"],
            "campaign_node_id": item["nid"],
            "verdict": "inconclusive",
            "confidence": 0.0,
            "evidence_summary": "Did not finish within batch deadline.",
            "key_finding": None,
            "metric_value": None,
            "failure_reason": "batch_timeout",
        }


async def _try_salvage_paper(
    campaign: ResearchCampaign,
    prior_dict: dict[str, Any] | None,
    emitter: EventEmitter,
    llm: LLMClient,
    session_factory: async_sessionmaker,
) -> bool:
    """Best-effort paper write after a fatal error. Returns True iff a paper was produced.

    A campaign that ran real experiments should not be thrown away just because a late
    LLM/DB hiccup escaped the loop — if there are experiment steps on disk, compile the
    paper from the accumulated trace so the run still ships a deliverable.
    """
    try:
        steps = await _session_experiment_step_count(session_factory, campaign.id)
        if steps <= 0:
            return False
        try:
            campaign.recount_from_tree()
        except Exception:
            pass
        await db_set_research_session_stage(campaign.id, session_factory, stage="campaign.paper")
        synthesis = build_campaign_synthesis_payload(campaign)
        await write_paper_minimal(
            session_id=campaign.id,
            session_factory=session_factory,
            emitter=emitter,
            llm=llm,
            question=campaign.question,
            prior=prior_dict,
            synthesis=synthesis,
        )
        logger.info("[campaign %s] Salvaged paper from %d experiment steps after error.", campaign.id, steps)
        return True
    except Exception as exc:  # noqa: BLE001 — salvage is best-effort
        logger.warning("[campaign %s] Paper salvage failed: %s", campaign.id, exc)
        return False


# ── Main campaign loop ───────────────────────────────────────────────────────

async def _enforce_domain_preflight(
    campaign: ResearchCampaign,
    session_factory: async_sessionmaker,
    emitter: EventEmitter,
) -> bool:
    """Run the owning domain's preflight before a fresh campaign starts.

    Returns True if the campaign may proceed, False if it was blocked (in which
    case this function has already finalized the campaign with a
    ``DOMAIN_PREFLIGHT_FAILED`` stop reason and marked the session).

    Fail-open on plugin errors (a buggy preflight must not block a campaign) but
    fail-closed on an explicit ``passed=False`` — that is the whole point of the gate.
    """
    from propab.domain_modules.registry import resolve_domain_plugin

    plugin = resolve_domain_plugin(question=campaign.question)
    if plugin is None:
        return True  # no scientific-domain owner → no domain-specific power gate

    try:
        result = plugin.preflight()
    except Exception as exc:  # noqa: BLE001 — a broken preflight must not block launch
        logger.warning("[campaign %s] preflight raised for domain %s (fail-open): %s",
                       campaign.id, plugin.domain_id, exc)
        return True

    if result.passed:
        await emitter.emit(
            session_id=campaign.id,
            event_type=EventType.CAMPAIGN_PROGRESS,
            step="campaign.preflight_ok",
            payload={"domain": plugin.domain_id, "reason": result.reason, "details": result.details},
        )
        return True

    logger.warning(
        "[campaign %s] domain preflight FAILED (%s): %s — refusing to launch",
        campaign.id, plugin.domain_id, result.reason,
    )
    campaign.finalize_stop(STOP_REASON_DOMAIN_PREFLIGHT_FAILED)
    try:
        await db_save_campaign(campaign, session_factory)
    except Exception:
        logger.exception("[campaign %s] db_save after preflight-fail failed", campaign.id)
    try:
        await db_mark_research_session_failed(campaign.id, session_factory)
    except Exception:
        pass
    await emitter.emit(
        session_id=campaign.id,
        event_type=EventType.CAMPAIGN_BUDGET_EXHAUSTED,
        step="campaign.preflight_failed",
        payload={
            "domain": plugin.domain_id,
            "stop_reason": STOP_REASON_DOMAIN_PREFLIGHT_FAILED,
            "reason": result.reason,
            "details": result.details,
            **campaign.summary(),
        },
    )
    return False


async def run_campaign_loop(
    campaign: ResearchCampaign,
    *,
    session_factory: async_sessionmaker,
    emitter: EventEmitter,
) -> None:
    """
    Run the campaign loop until breakthrough or budget exhausted.

    This is the entry point called from the API.  The campaign object may be
    freshly created or resumed from a DB checkpoint.
    """
    llm = LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.llm_api_secret,
        emitter=emitter,
        session_factory=session_factory,
    )
    # Defined inside the try below, but referenced by the fatal-error salvage path —
    # initialize so an early crash can still attempt paper salvage without NameError.
    prior_dict: dict[str, Any] | None = None
    # Health-metric accumulators (worker utilization at campaign end).
    total_agent_seconds: float = 0.0
    max_concurrency_seen: int = 1

    try:
        resume_warm = bool(campaign.hypothesis_tree.nodes)
        if resume_warm:
            logger.info(
                "[campaign %s] Resuming warm checkpoint (%d tree nodes, %d tested)",
                campaign.id,
                len(campaign.hypothesis_tree.nodes),
                campaign.total_hypotheses,
            )
            try:
                ev_rows = await db_load_session_events_tail(campaign.id, session_factory, limit=2000)
                restored, changed = backfill_belief_state_if_empty(
                    campaign.belief_state,
                    events=ev_rows,
                    tree_nodes={
                        nid: n.to_dict() for nid, n in campaign.hypothesis_tree.nodes.items()
                    },
                )
                if changed:
                    campaign.belief_state = restored
                    await db_save_campaign(campaign, session_factory)
                    logger.info(
                        "[campaign %s] Backfilled belief state from synthesis events (%d active)",
                        campaign.id,
                        len(restored.active_beliefs),
                    )
            except Exception as bf_exc:
                logger.warning("[campaign %s] Belief backfill skipped: %s", campaign.id, bf_exc)
            await emitter.emit(
                session_id=campaign.id,
                event_type=EventType.CAMPAIGN_RESUMED,
                step="campaign.resume",
                payload={
                    "tree_nodes": len(campaign.hypothesis_tree.nodes),
                    "total_hypotheses": campaign.total_hypotheses,
                    "stop_reason": campaign.stop_reason,
                },
            )

        await emitter.emit(
            session_id=campaign.id,
            event_type=EventType.CAMPAIGN_STARTED,
            step="campaign.start",
            payload=campaign.summary(),
        )

        # Domain preflight gate: for a fresh campaign, refuse to launch if the
        # owning domain reports its data/features are underpowered. This fails
        # fast (seconds) instead of burning the whole compute budget on a domain
        # that could never confirm anything. Resumed campaigns skip the gate —
        # they already passed it at first launch.
        if not resume_warm and not await _enforce_domain_preflight(campaign, session_factory, emitter):
            return

        await db_set_research_session_stage(campaign.id, session_factory, stage="campaign.prior")

        parsed = await parse_question(campaign.question)
        prior_timeout = max(5, int(getattr(settings, "campaign_prior_timeout_sec", 180)))
        prior: Prior | None = None
        prior_dict: dict[str, Any] | None = None

        if resume_warm:
            prior_dict = await db_load_session_prior_json(campaign.id, session_factory)
            if prior_dict:
                await emitter.emit(
                    session_id=campaign.id,
                    event_type=EventType.CAMPAIGN_PROGRESS,
                    step="campaign.phase",
                    payload={
                        "phase": "prior_skipped",
                        "detail": "Resumed campaign — reusing prior_json from research_sessions.",
                    },
                )
            else:
                prior = Prior(
                    established_facts=[],
                    contested_claims=[],
                    open_gaps=[],
                    dead_ends=[],
                    key_papers=[],
                    evidence_status="INSUFFICIENT_EVIDENCE",
                )
                prior_dict = prior.to_dict()

        if prior_dict is None:
            injected = await db_load_session_prior_json(campaign.id, session_factory)
            if injected:
                prior_dict = injected
                await emitter.emit(
                    session_id=campaign.id,
                    event_type=EventType.CAMPAIGN_PROGRESS,
                    step="campaign.phase",
                    payload={
                        "phase": "prior_injected",
                        "detail": "Using pre-injected prior_json from research_sessions.",
                    },
                )

        if prior_dict is None:
            await emitter.emit(
                session_id=campaign.id,
                event_type=EventType.CAMPAIGN_PROGRESS,
                step="campaign.phase",
                payload={
                    "phase": "prior_build",
                    "timeout_sec": prior_timeout,
                    "detail": (
                        "Literature prior (LLM + retrieval); hypotheses_tested stays 0 "
                        "until prior and baseline phases finish."
                    ),
                },
            )
            try:
                prior = await asyncio.wait_for(
                    build_prior(
                        parsed, session_id=campaign.id, emitter=emitter,
                        session_factory=session_factory,
                        paper_ttl_days=30,
                        llm=llm,
                    ),
                    timeout=prior_timeout,
                )
            except TimeoutError:
                logger.warning("[campaign %s] Prior build timed out; continuing with empty prior.", campaign.id)
                await emitter.emit(
                    session_id=campaign.id,
                    event_type=EventType.LLM_PARSE_ERROR,
                    step="campaign.prior_timeout",
                    payload={"message": "Prior build timed out; using empty prior."},
                )
                prior = Prior(
                    established_facts=[],
                    contested_claims=[],
                    open_gaps=[],
                    dead_ends=[],
                    key_papers=[],
                    evidence_status="INSUFFICIENT_EVIDENCE",
                )
            except Exception as prior_exc:
                logger.exception("[campaign %s] Prior build failed; continuing with empty prior.", campaign.id)
                await emitter.emit(
                    session_id=campaign.id,
                    event_type=EventType.LLM_PARSE_ERROR,
                    step="campaign.prior_failed",
                    payload={"message": str(prior_exc)[:500]},
                )
                prior = Prior(
                    established_facts=[],
                    contested_claims=[],
                    open_gaps=[],
                    dead_ends=[],
                    key_papers=[],
                    evidence_status="INSUFFICIENT_EVIDENCE",
                )
            prior_dict = prior.to_dict()

        # Ownership-contracts: literature citation verification rate per prior build.
        try:
            from propab.health_metrics import count_established_verified, log_literature_prior_health

            _facts_n, _verified_n = count_established_verified(getattr(prior, "established_facts", None))
            await log_literature_prior_health(
                session_factory,
                campaign_id=campaign.id,
                established_facts_count=_facts_n,
                verified_citation_count=_verified_n,
            )
        except Exception:  # noqa: BLE001 — metric logging must never break a campaign
            logger.exception("[campaign %s] literature prior health logging failed", campaign.id)

        # ── Stage 1: Intake + Literature (lightweight, reuse session infra) ──
        # parsed already loaded above
        domain = parsed.domain
        lifetime = load_lifetime_state(campaign, session_domain=domain)
        prior_dict = enrich_prior_from_lifetime(
            prior_dict,
            lifetime.graph,
            lifetime.policy,
            policy_record=lifetime.policy_record,
        )
        lifetime_seed_context = lifetime_context_for_seeds(lifetime.graph, lifetime.policy)
        from propab.numerical_seeds import format_seeds_for_question

        knowledge_graph = lifetime.graph
        domain_profile_id = _resolve_synthesis_domain_id(campaign, parsed_domain=domain)
        seed_block = format_seeds_for_question(
            knowledge_graph.get_numerical_seeds(domain_profile_id or "math_combinatorics"),
        )
        numerical_seeds_loaded = len(
            knowledge_graph.get_numerical_seeds(domain_profile_id or "math_combinatorics")
        )
        if seed_block:
            lifetime_seed_context = f"{lifetime_seed_context}\n\n{seed_block}".strip()
        synthesis_history_buckets: list[dict[str, str]] = []
        diversity_reset_attempts = 0
        search_policy = lifetime.policy
        prior_snippets = _prior_snippets(prior_dict)
        if prior is None:
            prior = Prior(
                established_facts=list(prior_dict.get("established_facts") or []),
                contested_claims=list(prior_dict.get("contested_claims") or []),
                open_gaps=list(prior_dict.get("open_gaps") or []),
                dead_ends=list(prior_dict.get("dead_ends") or []),
                key_papers=list(prior_dict.get("key_papers") or []),
                evidence_status=str(prior_dict.get("evidence_status") or "READY"),
                evidence_coverage=float(prior_dict.get("evidence_coverage") or 0.0),
                retrieval_diagnostics=prior_dict.get("retrieval_diagnostics"),
            )
        theme_penalty = float(getattr(settings, "campaign_theme_saturation_penalty", 0.15))
        if search_policy.saturated_themes:
            theme_penalty = min(0.35, theme_penalty + 0.05 * len(search_policy.saturated_themes))
        campaign.hypothesis_tree.set_scoring_context(
            campaign.question,
            prior_snippets,
            theme_saturation_penalty=theme_penalty,
        )
        await emitter.emit(
            session_id=campaign.id,
            event_type=EventType.CAMPAIGN_PROGRESS,
            step="lifetime.knowledge_loaded",
            payload={
                "policy_generation": search_policy.generation,
                "policy_id": lifetime.policy_record.id,
                "policy_status": lifetime.policy_record.status.value,
                "policy_mode": campaign.policy_mode,
                "budget_bucket": lifetime.budget_bucket,
                "domain_bucket": lifetime.domain_bucket,
                "claims_in_store": len(knowledge_graph.claims),
                "failures_in_store": len(knowledge_graph.failures),
                "theories_in_store": len(knowledge_graph.theories),
                "numerical_seeds_loaded": numerical_seeds_loaded,
                "theme_boost": search_policy.theme_boost,
                "theme_penalty": search_policy.theme_penalty,
            },
        )
        await db_set_research_session_prior(
            campaign.id, session_factory, json.dumps(prior_dict),
        )
        if prior_dict.get("evidence_status") == "INSUFFICIENT_EVIDENCE":
            await emitter.emit(
                session_id=campaign.id,
                event_type=EventType.CAMPAIGN_PROGRESS,
                step="campaign.prior_insufficient",
                payload={
                    "phase": "prior_build",
                    "evidence_status": prior_dict.get("evidence_status"),
                    "evidence_coverage": prior_dict.get("evidence_coverage"),
                    "detail": (
                        "Literature corpus did not pass quality gates; campaign continues "
                        "with explicit insufficient-evidence prior."
                    ),
                },
            )
        # ── Stage 2: Measure baseline if not yet done ─────────────────────
        if abs(campaign.baseline_metric) < 1e-12 and abs(campaign.breakthrough_criteria.baseline_value) < 1e-12:
            await db_set_research_session_stage(campaign.id, session_factory, stage="campaign.baseline")
            _bl_to = int(getattr(settings, "campaign_baseline_worker_timeout_sec", 600))
            await emitter.emit(
                session_id=campaign.id,
                event_type=EventType.CAMPAIGN_PROGRESS,
                step="campaign.phase",
                payload={
                    "phase": "baseline_measure",
                    "celery_join_timeout_sec": _bl_to,
                    "detail": (
                        "Baseline runs a full sub-agent on Celery (MNIST train_model + tools). "
                        "This often dominates the first 7–20+ minutes; hypotheses_tested remains 0 until it finishes."
                    ),
                },
            )
            logger.info("[campaign %s] Measuring baseline...", campaign.id)
            campaign.baseline_metric = await measure_baseline(
                campaign, llm, emitter, session_factory,
            )
            campaign.breakthrough_criteria.baseline_value = campaign.baseline_metric
            await db_save_campaign(campaign, session_factory)
            await emitter.emit(
                session_id=campaign.id,
                event_type=EventType.BASELINE_MEASURED,
                step="campaign.baseline",
                payload={
                    "metric_name": campaign.breakthrough_criteria.metric_name,
                    "baseline_value": campaign.baseline_metric,
                },
            )
            await asyncio.to_thread(write_campaign_snapshot, "post_baseline", campaign, prior_dict)

        await db_set_research_session_stage(campaign.id, session_factory, stage="campaign.experiments")

        # ── Stage 3: Campaign loop ────────────────────────────────────────
        last_checkpoint = time.monotonic()
        campaign_started_mono = time.monotonic()
        prior_theme_histogram: dict[str, int] | None = None
        entropy_trajectory_points: list[dict] = []
        generation = campaign.hypothesis_tree._generation
        # Recompute counters from any resumed tree so a reloaded campaign reports
        # distinct-node totals from the first wave (not stale persisted increments).
        campaign.recount_from_tree()
        first_confirmed_snapshotted = campaign.total_confirmed > 0
        stop_reason: str | None = None

        while not campaign.should_stop():
            if campaign.hypothesis_cap_reached():
                stop_reason = STOP_REASON_HYPOTHESIS_CAP_REACHED
                logger.info(
                    "[campaign %s] Hypothesis cap reached (%s)",
                    campaign.id,
                    campaign.max_hypotheses_cap,
                )
                break
            # Pipelined pool: refill keeps up to max_concurrent Celery tasks busy until the
            # frontier has no dispatchable pending nodes. Tier-2 synthesis refills the queue.
            if campaign.hypothesis_tree.next_dispatch_candidate(frozenset()) is None:
                await emitter.emit(
                    session_id=campaign.id,
                    event_type=EventType.HYPO_TREE_FRONTIER_EMPTY,
                    step="campaign.frontier_empty",
                    payload={"generation": generation, "tree": campaign.hypothesis_tree.summary()},
                )
                generation += 1
                campaign.hypothesis_tree._generation = generation
                try:
                    if len(campaign.hypothesis_tree.nodes) == 0:
                        seeds = await generate_seed_hypotheses(
                            campaign, prior, parsed, llm, emitter, session_factory, generation,
                            lifetime_context=lifetime_seed_context,
                        )
                        if not seeds:
                            logger.info("[campaign %s] No bootstrap seeds; stopping.", campaign.id)
                            stop_reason = STOP_REASON_NO_BOOTSTRAP_SEEDS
                            break
                    elif campaign.belief_state.branch_exhausted:
                        logger.info("[campaign %s] Branch exhausted; stopping.", campaign.id)
                        stop_reason = STOP_REASON_ALL_BRANCHES_EXHAUSTED
                        break
                    elif bool(getattr(settings, "campaign_synthesis_enabled", True)):
                        from propab.synthesis_diversity import diversity_reset_instruction

                        reset_prompt = None
                        if (
                            campaign.hypothesis_tree.next_dispatch_candidate(frozenset()) is None
                            and diversity_reset_attempts < 3
                        ):
                            reset_prompt = diversity_reset_instruction(
                                synthesis_history_buckets,
                                attempt=diversity_reset_attempts,
                            )
                            diversity_reset_attempts += 1
                        added, _syn_metrics = await run_campaign_synthesis_pass(
                            campaign_id=campaign.id,
                            question=campaign.question,
                            tree=campaign.hypothesis_tree,
                            belief_state=campaign.belief_state,
                            llm=llm,
                            generation=generation,
                            prior_snippets=prior_snippets,
                            emitter=emitter,
                            session_factory=session_factory,
                            domain_id=domain_profile_id,
                            synthesis_history_buckets=synthesis_history_buckets,
                            diversity_reset_instruction=reset_prompt,
                            lifetime_context=lifetime_seed_context,
                        )
                        from propab.numerical_seeds import classify_hypothesis_bucket
                        from propab.synthesis_diversity import (
                            bootstrap_forced_problem_type,
                            resolve_forced_problem_type,
                            tree_problem_counts_from_nodes,
                        )
                        from propab.campaign_synthesis import apply_diversity_fallback_seeds

                        if not added:
                            node_dicts = {
                                nid: (n.to_dict() if hasattr(n, "to_dict") else n)
                                for nid, n in campaign.hypothesis_tree.nodes.items()
                            }
                            tree_counts = tree_problem_counts_from_nodes(node_dicts)
                            forced = resolve_forced_problem_type(
                                synthesis_history_buckets,
                                [b.statement for b in campaign.belief_state.active_beliefs],
                                streak=3,
                                tree_problem_counts=tree_counts,
                            )
                            if forced is None:
                                forced = bootstrap_forced_problem_type(campaign.question)
                            if forced:
                                added = apply_diversity_fallback_seeds(
                                    campaign.hypothesis_tree,
                                    forced_type=forced,
                                    generation=generation,
                                    question=campaign.question,
                                    prior_snippets=prior_snippets,
                                    belief_state=campaign.belief_state,
                                )
                                if added:
                                    logger.info(
                                        "[campaign %s] Diversity fallback injected %s %s seed(s).",
                                        campaign.id,
                                        len(added),
                                        forced,
                                    )

                        for node in added:
                            synthesis_history_buckets.append(
                                classify_hypothesis_bucket(node.text, node.test_methodology or ""),
                            )
                        if (
                            not added
                            and campaign.hypothesis_tree.next_dispatch_candidate(frozenset()) is None
                            and diversity_reset_attempts < 3
                        ):
                            logger.info(
                                "[campaign %s] Diversity reset attempt %s produced no candidates; retrying.",
                                campaign.id,
                                diversity_reset_attempts,
                            )
                            continue
                        if (
                            not added
                            and campaign.hypothesis_tree.next_dispatch_candidate(frozenset()) is None
                        ):
                            logger.info("[campaign %s] Synthesis produced no candidates; stopping.", campaign.id)
                            stop_reason = STOP_REASON_SYNTHESIS_EMPTY
                            break
                    else:
                        seeds = await generate_seed_hypotheses(
                            campaign, prior, parsed, llm, emitter, session_factory, generation,
                            lifetime_context=lifetime_seed_context,
                        )
                        if not seeds:
                            logger.info("[campaign %s] No seeds generated; stopping.", campaign.id)
                            stop_reason = STOP_REASON_NO_SEEDS_GENERATED
                            break
                except Exception as seed_exc:  # noqa: BLE001 — degrade, don't crash the campaign
                    err_msg = str(seed_exc)
                    transient = any(
                        tok in err_msg.lower()
                        for tok in ("no address associated", "connection", "timeout", "temporarily unavailable")
                    )
                    if transient and diversity_reset_attempts < 5:
                        logger.warning(
                            "[campaign %s] Transient frontier refill error (%s); retrying.",
                            campaign.id,
                            err_msg[:120],
                        )
                        await asyncio.sleep(5)
                        continue
                    logger.warning(
                        "[campaign %s] Frontier refill failed (%s); finalizing with results so far.",
                        campaign.id,
                        seed_exc,
                    )
                    await emitter.emit(
                        session_id=campaign.id,
                        event_type=EventType.CAMPAIGN_PROGRESS,
                        step="campaign.degraded",
                        payload={
                            "phase": "frontier_refill_failed",
                            "error": str(seed_exc),
                            "note": "Frontier refill hit a transient error; writing the paper with accumulated results.",
                        },
                    )
                    stop_reason = STOP_REASON_FRONTIER_REFILL_FAILED
                    break
                await db_save_campaign(campaign, session_factory)

            max_c = int(getattr(settings, "campaign_max_concurrent_sub_agents", 0) or 0)
            if max_c <= 0:
                max_c = max(
                    1,
                    int(settings.campaign_batch_size)
                    * max(1, int(getattr(settings, "campaign_inflight_multiplier", 1) or 1)),
                )
            max_concurrency_seen = max(max_concurrency_seen, max_c)

            breakthrough_found = False
            cap_hit_in_batch = False
            await emitter.emit(
                session_id=campaign.id,
                event_type=EventType.CAMPAIGN_PROGRESS,
                step="campaign.batch_dispatch",
                payload={
                    "phase": "pipelined_sub_agents",
                    "max_concurrent": max_c,
                    "frontier_size": len(campaign.hypothesis_tree.frontier),
                    "note": (
                        "Hybrid loop: Tier-1 mechanical dispatch; Tier-2 campaign synthesis "
                        "refills frontier when batch accumulates or queue runs low."
                    ),
                },
            )
            _lifetime_ctx = [lifetime_seed_context]
            async for result in _iter_campaign_pipelined_results(
                campaign=campaign,
                max_concurrent=max_c,
                prior_dict=prior_dict,
                domain=domain,
                emitter=emitter,
                session_factory=session_factory,
                llm=llm,
                generation=generation,
                prior_snippets=prior_snippets,
                lifetime_context=lifetime_seed_context,
                lifetime_context_ref=_lifetime_ctx,
            ):
                node_id = str(result.get("campaign_node_id") or "")
                verdict = result.get("verdict", "inconclusive")
                confidence = float(result.get("confidence") or 0.0)
                evidence = result.get("evidence_summary") or ""
                total_agent_seconds += float(result.get("duration_sec") or 0.0)

                if not campaign.hypothesis_tree.update_node(
                    node_id, verdict, confidence, evidence,
                ):
                    logger.warning(
                        "[campaign %s] Worker result for unknown tree node_id=%r "
                        "(hypothesis_id=%r); skipping.",
                        campaign.id,
                        node_id,
                        result.get("hypothesis_id"),
                    )
                    continue

                _apply_result_diagnostics(
                    campaign.hypothesis_tree,
                    node_id,
                    verdict,
                    confidence,
                    evidence,
                    failure_reason=str(result.get("failure_reason") or result.get("error") or ""),
                )

                if verdict == "confirmed":
                    from propab.numerical_seeds import refresh_lifetime_context_with_crossings

                    _lifetime_ctx[0] = refresh_lifetime_context_with_crossings(
                        _lifetime_ctx[0],
                        campaign.hypothesis_tree,
                        campaign.question,
                    )
                    lifetime_seed_context = _lifetime_ctx[0]

                # Recount from the tree (distinct nodes) rather than incrementing per
                # result — re-dispatched nodes must not inflate the totals, so the
                # campaign summary stays consistent with the tree and the paper.
                campaign.recount_from_tree()

                node = campaign.hypothesis_tree.nodes.get(node_id)
                parent_theme: str | None = None
                if node and node.parent_id:
                    parent = campaign.hypothesis_tree.nodes.get(node.parent_id)
                    if parent:
                        parent_theme = infer_hypothesis_theme(parent.text)
                evidence_obj = parse_evidence_obj(evidence)
                await emitter.emit(
                    session_id=campaign.id,
                    event_type=EventType.CAMPAIGN_PROGRESS,
                    step="campaign.verification_diagnostic",
                    payload={
                        "node_id": node_id,
                        "verdict": verdict,
                        "theme": infer_hypothesis_theme(node.text if node else ""),
                        "parent_theme": parent_theme,
                        "verification_method": classify_verification_method(evidence),
                        "verified_true_steps": int(evidence_obj.get("verified_true_steps") or 0),
                        "verified_false_steps": int(evidence_obj.get("verified_false_steps") or 0),
                    },
                )

                effective_verdict = node.verdict if node else verdict

                if effective_verdict == "confirmed" and node is not None and is_discovery_node(node):
                    if not first_confirmed_snapshotted:
                        first_confirmed_snapshotted = True
                        await asyncio.to_thread(
                            write_campaign_snapshot, "post_first_confirmed", campaign, prior_dict
                        )

                    campaign.update_best_metric(result)

                    result_with_replications = {
                        **result,
                        "replication_count": campaign.count_replications(node_id),
                    }
                    if campaign.breakthrough_criteria.is_breakthrough(result_with_replications):
                        breakthrough_found = True

                # Tier 1: no per-node LLM expansion — synthesis (Tier 2) produces candidates.

                # Throttle persistence: always checkpoint on state that matters
                # (confirmed / breakthrough), otherwise at most every checkpoint interval.
                # This removes the per-result write amplification that slowed campaigns.
                now_mono = time.monotonic()
                checkpoint_secs = max(5, int(getattr(settings, "campaign_checkpoint_every", 60) or 60))
                if effective_verdict == "confirmed" or breakthrough_found or (now_mono - last_checkpoint) >= checkpoint_secs:
                    await db_save_campaign(campaign, session_factory)
                    last_checkpoint = now_mono
                    snap = frontier_snapshot(
                        campaign.hypothesis_tree,
                        campaign_started_mono=campaign_started_mono,
                        prior_theme_histogram=prior_theme_histogram,
                    )
                    prior_theme_histogram = dict(snap.get("theme_histogram") or {})
                    entropy_trajectory_points.append(trajectory_point_from_snapshot(snap))
                    await emitter.emit(
                        session_id=campaign.id,
                        event_type=EventType.CAMPAIGN_PROGRESS,
                        step="campaign.frontier_snapshot",
                        payload=snap,
                    )

                if campaign.hypothesis_cap_reached():
                    cap_hit_in_batch = True
                    break

            # Persist end-of-wave state so progress survives interruption.
            await db_save_campaign(campaign, session_factory)
            last_checkpoint = time.monotonic()
            await emitter.emit(
                session_id=campaign.id,
                event_type=EventType.CAMPAIGN_PROGRESS,
                step="campaign.progress",
                payload=campaign.summary(),
            )

            await db_save_campaign(campaign, session_factory)

            # Checkpoint if due
            if time.monotonic() - last_checkpoint >= campaign.checkpoint_every:
                await db_save_campaign(campaign, session_factory)
                await emitter.emit(
                    session_id=campaign.id,
                    event_type=EventType.BUDGET_CHECKPOINT,
                    step="campaign.checkpoint",
                    payload=campaign.summary(),
                )
                last_checkpoint = time.monotonic()

            if breakthrough_found:
                stop_reason = STOP_REASON_BREAKTHROUGH
                campaign.finalize_stop(stop_reason)
                await db_save_campaign(campaign, session_factory)
                await emitter.emit(
                    session_id=campaign.id,
                    event_type=EventType.CAMPAIGN_BREAKTHROUGH,
                    step="campaign.breakthrough",
                    payload={
                        "best_finding": campaign.best_finding,
                        "improvement_pct": campaign.improvement_pct,
                        "baseline_metric": campaign.baseline_metric,
                        "best_metric": campaign.best_metric,
                        "metric_name": campaign.breakthrough_criteria.metric_name,
                        "total_hypotheses_tested": campaign.total_hypotheses,
                        "stop_reason": stop_reason,
                    },
                )
                break

            if cap_hit_in_batch:
                stop_reason = STOP_REASON_HYPOTHESIS_CAP_REACHED
                break

        # ── Stage 4: Final checkpoint + paper ────────────────────────────
        if stop_reason is None and campaign.status == STATUS_BREAKTHROUGH:
            stop_reason = STOP_REASON_BREAKTHROUGH
        if stop_reason is None:
            if campaign.elapsed_seconds() >= campaign.compute_budget_seconds:
                stop_reason = STOP_REASON_TIME_BUDGET_EXHAUSTED
            else:
                stop_reason = STOP_REASON_NO_DISPATCHABLE_NODES

        if campaign.status != STATUS_BREAKTHROUGH:
            campaign.finalize_stop(stop_reason)
            await emitter.emit(
                session_id=campaign.id,
                event_type=EventType.CAMPAIGN_BUDGET_EXHAUSTED,
                step="campaign.budget_exhausted",
                payload={**campaign.summary(), "stop_reason": stop_reason},
            )

        await db_save_campaign(campaign, session_factory)

        loaded = await db_load_campaign(campaign.id, session_factory)
        if loaded is not None:
            campaign = loaded

        # Ownership-contracts per-campaign health metrics: worker experiment
        # success rate + worker utilization onto research_campaigns, and the
        # artifact-gate precision (confirmed findings backed by a null test).
        try:
            from propab.health_metrics import (
                compute_confirmed_audit_counts,
                log_campaign_audit,
                log_campaign_end_health,
            )

            await log_campaign_end_health(
                session_factory,
                campaign=campaign,
                total_agent_seconds=total_agent_seconds,
                max_concurrency=max_concurrency_seen,
            )
            _confirmed, _survived = compute_confirmed_audit_counts(campaign.hypothesis_tree)
            if _confirmed > 0:
                await log_campaign_audit(
                    session_factory,
                    campaign_id=campaign.id,
                    confirmed_findings_count=_confirmed,
                    survived_audit_count=_survived,
                )
        except Exception:  # noqa: BLE001 — metric logging must never break a campaign
            logger.exception("[campaign %s] campaign-end health logging failed", campaign.id)

        await asyncio.to_thread(write_campaign_snapshot, "pre_paper", campaign, prior_dict)

        tree_sum = campaign.hypothesis_tree.summary()
        trajectory_summary = summarize_entropy_trajectory(entropy_trajectory_points)
        analyst_overlay = None
        try:
            parent = lifetime.store.accepted_policy(
                domain_bucket=lifetime.domain_bucket,
                budget_bucket=lifetime.budget_bucket,
            )
            fitness = await asyncio.to_thread(PolicyFitnessLedger.load)
            sim_fitness = await asyncio.to_thread(SimulationFitnessLedger.load)
            rationale, predicted, fals, params = await llm_policy_analyst(
                parent=parent,
                graph=lifetime.graph,
                meta=lifetime.meta,
                budget_bucket=lifetime.budget_bucket,
                domain_bucket=lifetime.domain_bucket,
                campaign_metrics={
                    "closure_ratio": tree_sum.get("closure_ratio", 0),
                    "entropy_trajectory": trajectory_summary.to_dict(),
                },
                llm=llm,
                session_id=campaign.id,
                trajectory_summary=trajectory_summary,
                fitness=fitness,
                simulation_fitness=sim_fitness,
            )
            analyst_overlay = {
                "rationale": rationale,
                "predicted_effects": predicted.to_dict(),
                "falsification_conditions": fals,
                "mutation_params": params,
            }
        except Exception as analyst_exc:  # noqa: BLE001
            logger.warning(
                "[campaign %s] Policy analyst skipped (%s); ingest uses deterministic narrative.",
                campaign.id,
                analyst_exc,
            )
        lifetime_report = await asyncio.to_thread(
            ingest_campaign,
            campaign,
            session_domain=domain,
            analyst_overlay=analyst_overlay,
            snapshot_metrics={"entropy_trajectory": trajectory_summary.to_dict()},
        )
        await emitter.emit(
            session_id=campaign.id,
            event_type=EventType.CAMPAIGN_PROGRESS,
            step="lifetime.ingested",
            payload=lifetime_report,
        )

        # Write a paper with whatever we found
        await db_set_research_session_stage(campaign.id, session_factory, stage="campaign.paper")
        synthesis = build_campaign_synthesis_payload(campaign)
        await write_paper_minimal(
            session_id=campaign.id,
            session_factory=session_factory,
            emitter=emitter,
            llm=llm,
            question=campaign.question,
            prior=prior_dict,
            synthesis=synthesis,
        )

        for attempt in range(3):
            try:
                await db_mark_research_session_completed(campaign.id, session_factory)
                break
            except Exception as mark_exc:
                logger.warning(
                    "[campaign %s] db_mark_research_session_completed attempt %s/3: %s",
                    campaign.id,
                    attempt + 1,
                    mark_exc,
                )
                if attempt == 2:
                    logger.exception(
                        "[campaign %s] session row may still be 'running' — fix DB/network and re-run mark",
                        campaign.id,
                    )
                await asyncio.sleep(1.0 * (attempt + 1))
        await emitter.emit(
            session_id=campaign.id,
            event_type=EventType.CAMPAIGN_COMPLETED,
            step="campaign.complete",
            payload=campaign.summary(),
        )

    except Exception as exc:
        logger.exception("[campaign %s] Fatal error: %s", campaign.id, exc)
        # Last-resort salvage: if real experiments ran, a crash should still yield a
        # paper from the accumulated trace rather than discarding the whole campaign.
        salvaged = await _try_salvage_paper(campaign, prior_dict, emitter, llm, session_factory)
        if salvaged:
            campaign.recount_from_tree()
            campaign.finalize_stop(STOP_REASON_SALVAGED_AFTER_ERROR)
            try:
                await db_save_campaign(campaign, session_factory)
            except Exception:
                logger.exception("[campaign %s] salvage succeeded but db_save failed", campaign.id)
            try:
                await db_mark_research_session_completed(campaign.id, session_factory)
            except Exception:
                logger.exception("[campaign %s] salvage paper written but mark-completed failed", campaign.id)
            await emitter.emit(
                session_id=campaign.id,
                event_type=EventType.CAMPAIGN_COMPLETED,
                step="campaign.complete_salvaged",
                payload={
                    "salvaged_after_error": type(exc).__name__,
                    "stop_reason": campaign.stop_reason,
                    "campaign_summary": campaign.summary(),
                },
            )
            return
        try:
            await db_mark_research_session_failed(campaign.id, session_factory)
        except Exception:
            pass
        # Record an explicit terminal reason so the campaign never stays "active"
        # with a null stop_reason after a fatal, unsalvageable error.
        campaign.finalize_stop(STOP_REASON_FATAL_ERROR)
        try:
            await db_save_campaign(campaign, session_factory)
        except Exception:
            pass
        await emitter.emit(
            session_id=campaign.id,
            event_type=EventType.SESSION_FAILED,
            step="campaign.failed",
            payload={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "campaign_summary": campaign.summary(),
            },
        )
        raise
