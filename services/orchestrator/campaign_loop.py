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
from collections.abc import AsyncIterator, Mapping
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
)
from propab.campaign_snapshot import write_campaign_snapshot
from propab.config import settings
from propab.db import create_redis
from propab.events import EventEmitter
from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.llm import LLMClient
from propab.types import EventType
from services.orchestrator.hypotheses import generate_ranked_hypotheses
from services.orchestrator.intake import parse_question
from services.orchestrator.literature import build_prior
from services.orchestrator.paper import _session_experiment_step_count, write_paper_minimal
from services.orchestrator.schemas import Prior
from services.worker.peer_findings import build_peer_finding_payload, publish_peer_finding
from services.worker.tasks import run_sub_agent_task

logger = logging.getLogger(__name__)

_NODE_ID_NAMESPACE = UUID("c3e7a1f0-4b2d-4e8a-95cf-0d1e3f5a7c9b")


def _campaign_ledger_from_tree(tree: HypothesisTree) -> dict[str, Any]:
    """Ledger shape aligned with AccumulatedLedger.summary() for paper_sections / gates."""
    confirmed = list(tree.confirmed)
    refuted = [nid for nid, n in tree.nodes.items() if n.verdict == "refuted"]
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


def build_campaign_synthesis_payload(campaign: ResearchCampaign) -> dict[str, Any]:
    """Synthesis dict for paper generation — single source of truth with tree ledger."""
    ledger = _campaign_ledger_from_tree(campaign.hypothesis_tree)
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
        "confirmed_findings": [
            {
                "node_id": nid,
                "text": campaign.hypothesis_tree.nodes[nid].text,
                "evidence": campaign.hypothesis_tree.nodes[nid].evidence_summary,
                "depth": campaign.hypothesis_tree.nodes[nid].depth,
            }
            for nid in campaign.hypothesis_tree.confirmed[:10]
            if nid in campaign.hypothesis_tree.nodes
        ],
    }


def _db_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Read a float from a DB row without treating 0.0 as missing (unlike ``x or 0.0``)."""
    v = row.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


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


def _node_row_id(campaign_id: str, node_id: str) -> str:
    return str(uuid5(_NODE_ID_NAMESPACE, f"{campaign_id}:{node_id}"))


def _parse_started_at(iso_s: str) -> datetime:
    """asyncpg expects datetime for TIMESTAMPTZ, not an ISO string."""
    if not iso_s:
        return datetime.now(tz=UTC)
    s = iso_s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


# ── DB helpers ───────────────────────────────────────────────────────────────

async def db_save_campaign(campaign: ResearchCampaign, session_factory: async_sessionmaker) -> None:
    """Upsert the full campaign state to research_campaigns."""
    now = datetime.now(tz=UTC).isoformat()
    campaign.last_checkpoint = now
    async with session_factory() as db:
        await db.execute(
            text("""
                INSERT INTO research_campaigns (
                    id, question, status, breakthrough_criteria_json,
                    hypothesis_tree_json, baseline_metric, best_metric,
                    improvement_pct, best_finding_json, total_hypotheses,
                    total_confirmed, compute_seconds_used, compute_budget_seconds,
                    started_at, last_checkpoint_at
                ) VALUES (
                    :id, :question, :status,
                    CAST(:breakthrough_criteria_json AS jsonb),
                    CAST(:hypothesis_tree_json AS jsonb),
                    :baseline_metric, :best_metric, :improvement_pct,
                    CAST(:best_finding_json AS jsonb),
                    :total_hypotheses, :total_confirmed,
                    :compute_seconds_used, :compute_budget_seconds,
                    :started_at, NOW()
                )
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    hypothesis_tree_json = EXCLUDED.hypothesis_tree_json,
                    baseline_metric = EXCLUDED.baseline_metric,
                    best_metric = EXCLUDED.best_metric,
                    improvement_pct = EXCLUDED.improvement_pct,
                    best_finding_json = EXCLUDED.best_finding_json,
                    total_hypotheses = EXCLUDED.total_hypotheses,
                    total_confirmed = EXCLUDED.total_confirmed,
                    compute_seconds_used = EXCLUDED.compute_seconds_used,
                    last_checkpoint_at = NOW()
            """),
            {
                "id": campaign.id,
                "question": campaign.question,
                "status": campaign.status,
                "breakthrough_criteria_json": json.dumps(campaign.breakthrough_criteria.to_dict()),
                "hypothesis_tree_json": json.dumps(campaign.hypothesis_tree.to_dict()),
                "baseline_metric": campaign.baseline_metric,
                "best_metric": campaign.best_metric,
                "improvement_pct": campaign.improvement_pct,
                "best_finding_json": json.dumps(campaign.best_finding or {}),
                "total_hypotheses": campaign.total_hypotheses,
                "total_confirmed": campaign.total_confirmed,
                "compute_seconds_used": int(campaign.elapsed_seconds()),
                "compute_budget_seconds": campaign.compute_budget_seconds,
                "started_at": _parse_started_at(campaign.started_at),
            },
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


async def db_load_campaign(campaign_id: str, session_factory: async_sessionmaker) -> ResearchCampaign | None:
    """Load a campaign from the DB. Returns None if not found."""
    async with session_factory() as db:
        row = (await db.execute(
            text("SELECT * FROM research_campaigns WHERE id = CAST(:id AS uuid)"),
            {"id": campaign_id},
        )).mappings().one_or_none()
    if row is None:
        return None
    data = {
        "id": str(row["id"]),
        "question": row["question"],
        "status": row["status"],
        "breakthrough_criteria": row["breakthrough_criteria_json"] or {},
        "hypothesis_tree": row["hypothesis_tree_json"] or {},
        "baseline_metric": _db_float(row, "baseline_metric", 0.0),
        "best_metric": _db_float(row, "best_metric", 0.0),
        "improvement_pct": _db_float(row, "improvement_pct", 0.0),
        "best_finding": row.get("best_finding_json") or None,
        "total_hypotheses": row.get("total_hypotheses") or 0,
        "total_confirmed": row.get("total_confirmed") or 0,
        "compute_budget_seconds": row.get("compute_budget_seconds") or 14400,
        "compute_seconds_used": row.get("compute_seconds_used") or 0,
        "started_at": str(row.get("started_at") or ""),
        "last_checkpoint": str(row.get("last_checkpoint_at") or ""),
    }
    return ResearchCampaign.from_dict(data)


# ── Baseline measurement ─────────────────────────────────────────────────────

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


# ── Hypothesis expansion via LLM ─────────────────────────────────────────────

async def expand_tree_node(
    node_id: str,
    tree: HypothesisTree,
    llm: LLMClient,
    campaign_id: str,
    generation: int,
) -> list[HypothesisNode]:
    """
    Ask the LLM to generate child hypotheses for a node and add them to the tree.
    Returns the newly created child nodes.
    """
    prompt = tree.build_expand_prompt(node_id)
    if prompt is None:
        return []
    try:
        raw = await llm.call(
            prompt=prompt,
            purpose="campaign.tree_expand",
            session_id=campaign_id,
        )
        children = tree.parse_expanded_nodes(node_id, raw, generation)
        tree.add_to_frontier(children)
        return children
    except Exception as exc:
        logger.warning("Tree expansion failed for node %s: %s", node_id, exc)
        return []


async def generate_seed_hypotheses(
    campaign: ResearchCampaign,
    prior: object,
    parsed_question: object,
    llm: LLMClient,
    emitter: EventEmitter,
    session_factory: async_sessionmaker,
    generation: int,
) -> list[HypothesisNode]:
    """
    Generate seed hypotheses when the frontier is empty (new campaign or exhausted tree).
    Reuses the existing hypothesis generator but plants results in the tree.
    """
    prior_context = campaign.hypothesis_tree.confirmed_findings_text(max_n=10)
    raw_hyps = await generate_ranked_hypotheses(
        parsed_question,
        prior,
        max_hypotheses=settings.campaign_batch_size,
        llm=llm,
        session_id=campaign.id,
        emitter=emitter,
        prior_round_findings=prior_context,
    )
    seed_dicts = [
        {"id": h.id, "text": h.text, "test_methodology": h.test_methodology}
        for h in raw_hyps
    ]
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
        progressed_local = False
        while len(pending) < max_concurrent and not campaign.should_stop():
            inflight = {p["nid"] for p in pending}
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

    try:
        await emitter.emit(
            session_id=campaign.id,
            event_type=EventType.CAMPAIGN_STARTED,
            step="campaign.start",
            payload=campaign.summary(),
        )
        await db_set_research_session_stage(campaign.id, session_factory, stage="campaign.prior")

        # ── Stage 1: Intake + Literature (lightweight, reuse session infra) ──
        parsed = await parse_question(campaign.question)
        prior_timeout = max(5, int(getattr(settings, "campaign_prior_timeout_sec", 180)))
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
            )
        prior_dict = prior.to_dict()
        domain = parsed.domain

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
        generation = campaign.hypothesis_tree._generation
        # Recompute counters from any resumed tree so a reloaded campaign reports
        # distinct-node totals from the first wave (not stale persisted increments).
        campaign.recount_from_tree()
        first_confirmed_snapshotted = campaign.total_confirmed > 0

        while not campaign.should_stop():
            # Pipelined pool: refill keeps up to max_concurrent Celery tasks busy until the
            # frontier has no dispatchable pending nodes (mid-wave tree expansions included).
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
                    seeds = await generate_seed_hypotheses(
                        campaign, prior, parsed, llm, emitter, session_factory, generation,
                    )
                except Exception as seed_exc:  # noqa: BLE001 — degrade, don't crash the campaign
                    logger.warning(
                        "[campaign %s] Seed regeneration failed (%s); finalizing with results so far.",
                        campaign.id,
                        seed_exc,
                    )
                    await emitter.emit(
                        session_id=campaign.id,
                        event_type=EventType.CAMPAIGN_PROGRESS,
                        step="campaign.degraded",
                        payload={
                            "phase": "seed_regeneration_failed",
                            "error": str(seed_exc),
                            "note": "Re-seeding hit a transient error; writing the paper with accumulated results.",
                        },
                    )
                    break
                if not seeds:
                    logger.info("[campaign %s] No seeds generated; stopping.", campaign.id)
                    break
                await db_save_campaign(campaign, session_factory)

            max_c = int(getattr(settings, "campaign_max_concurrent_sub_agents", 0) or 0)
            if max_c <= 0:
                max_c = max(
                    1,
                    int(settings.campaign_batch_size)
                    * max(1, int(getattr(settings, "campaign_inflight_multiplier", 1) or 1)),
                )

            breakthrough_found = False
            await emitter.emit(
                session_id=campaign.id,
                event_type=EventType.CAMPAIGN_PROGRESS,
                step="campaign.batch_dispatch",
                payload={
                    "phase": "pipelined_sub_agents",
                    "max_concurrent": max_c,
                    "frontier_size": len(campaign.hypothesis_tree.frontier),
                    "note": (
                        "Slots refill as workers complete; new frontier children from expansions "
                        "can dispatch without waiting for a fixed batch to drain."
                    ),
                },
            )
            async for result in _iter_campaign_pipelined_results(
                campaign=campaign,
                max_concurrent=max_c,
                prior_dict=prior_dict,
                domain=domain,
                emitter=emitter,
                session_factory=session_factory,
            ):
                node_id = str(result.get("campaign_node_id") or "")
                verdict = result.get("verdict", "inconclusive")
                confidence = float(result.get("confidence") or 0.0)
                evidence = result.get("evidence_summary") or ""

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

                # Recount from the tree (distinct nodes) rather than incrementing per
                # result — re-dispatched nodes must not inflate the totals, so the
                # campaign summary stays consistent with the tree and the paper.
                campaign.recount_from_tree()

                if verdict == "confirmed":
                    if not first_confirmed_snapshotted:
                        first_confirmed_snapshotted = True
                        await asyncio.to_thread(
                            write_campaign_snapshot, "post_first_confirmed", campaign, prior_dict
                        )

                    # Expand the tree from this confirmed node
                    if settings.campaign_expand_on_confirmed:
                        children = await expand_tree_node(
                            node_id, campaign.hypothesis_tree, llm, campaign.id, generation,
                        )
                        if children:
                            await emitter.emit(
                                session_id=campaign.id,
                                event_type=EventType.HYPO_TREE_EXPANDED,
                                step="campaign.tree_expand",
                                payload={
                                    "parent_id": node_id,
                                    "children_count": len(children),
                                    "tree_size": len(campaign.hypothesis_tree.nodes),
                                },
                                # events.hypothesis_id is UUID FK; tree node ids may be short slugs (e.g. "H1").
                                hypothesis_id=_node_row_id(campaign.id, node_id),
                            )

                    # Update best metric
                    campaign.update_best_metric(result)

                    # Check breakthrough — requires replication count
                    result_with_replications = {
                        **result,
                        "replication_count": campaign.count_replications(node_id),
                    }
                    if campaign.breakthrough_criteria.is_breakthrough(result_with_replications):
                        breakthrough_found = True

                elif verdict == "refuted" and settings.campaign_expand_on_refuted:
                    # Generate alternatives from refuted hypotheses
                    children = await expand_tree_node(
                        node_id, campaign.hypothesis_tree, llm, campaign.id, generation,
                    )
                    if children:
                        await emitter.emit(
                            session_id=campaign.id,
                            event_type=EventType.HYPO_TREE_EXPANDED,
                            step="campaign.tree_expand_alt",
                            payload={
                                "parent_id": node_id,
                                "children_count": len(children),
                                "expansion_reason": "refuted_alternative",
                            },
                            hypothesis_id=_node_row_id(campaign.id, node_id),
                        )

                # Throttle persistence: always checkpoint on state that matters
                # (confirmed / breakthrough), otherwise at most every checkpoint interval.
                # This removes the per-result write amplification that slowed campaigns.
                now_mono = time.monotonic()
                checkpoint_secs = max(5, int(getattr(settings, "campaign_checkpoint_every", 60) or 60))
                if verdict == "confirmed" or breakthrough_found or (now_mono - last_checkpoint) >= checkpoint_secs:
                    await db_save_campaign(campaign, session_factory)
                    last_checkpoint = now_mono

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
                campaign.status = STATUS_BREAKTHROUGH
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
                    },
                )
                break

        # ── Stage 4: Final checkpoint + paper ────────────────────────────
        if campaign.status != STATUS_BREAKTHROUGH:
            campaign.status = STATUS_BUDGET_EXHAUSTED
            await emitter.emit(
                session_id=campaign.id,
                event_type=EventType.CAMPAIGN_BUDGET_EXHAUSTED,
                step="campaign.budget_exhausted",
                payload=campaign.summary(),
            )

        await db_save_campaign(campaign, session_factory)

        loaded = await db_load_campaign(campaign.id, session_factory)
        if loaded is not None:
            campaign = loaded

        await asyncio.to_thread(write_campaign_snapshot, "pre_paper", campaign, prior_dict)

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
                    "campaign_summary": campaign.summary(),
                },
            )
            return
        try:
            await db_mark_research_session_failed(campaign.id, session_factory)
        except Exception:
            pass
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
