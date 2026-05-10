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
from propab.config import settings
from propab.db import create_redis
from propab.events import EventEmitter
from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.llm import LLMClient
from propab.types import EventType
from services.orchestrator.hypotheses import generate_ranked_hypotheses
from services.orchestrator.intake import parse_question
from services.orchestrator.literature import build_prior
from services.orchestrator.paper import write_paper_minimal
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
            f"Use train_model with dataset={dataset_name!r} and record val_accuracy."
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

    task_payload = {
        "session_id": campaign.id,
        "hypothesis_id": _node_row_id(campaign.id, "baseline"),
        "campaign_node_id": "baseline",
        "hypothesis": baseline_hypothesis,
        "baseline": {"metric_name": metric_name, "metric_value": 0.0, "description": "baseline measurement"},
        "prior": {"established_facts": [], "contested_claims": [], "open_gaps": [], "dead_ends": [], "key_papers": []},
        "domain": "deep_learning",
        "question": campaign.question,
        # Worker: if the LLM/tool trace omits val_accuracy, run train_model once deterministically.
        "baseline_measurement": {
            "dataset": dataset_name,
            "n_steps": max(80, min(int(n_steps), 400)),
        },
    }

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
        result = await asyncio.to_thread(lambda: ar.get(timeout=1200))
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


# ── Sub-agent dispatch (campaign round) ─────────────────────────────────────

async def _iter_campaign_batch_results(
    *,
    campaign: ResearchCampaign,
    batch: list[HypothesisNode],
    prior_dict: dict,
    domain: str,
    emitter: EventEmitter,
    session_factory: async_sessionmaker,
) -> AsyncIterator[dict]:
    """
    Dispatch a batch of hypothesis nodes to sub-agents and yield each result as
    the corresponding Celery task completes (order may differ from ``batch``).

    Yields incrementally so the campaign loop can persist tree/metrics to Postgres
    while long-running workers are still in flight — otherwise GET /campaigns shows
    ``hypotheses_tested=0`` for the entire multi-sub-agent wall-clock wait.
    """
    pending: list[dict] = []
    all_node_ids = [node.id for node in batch]
    all_db_hids = [_node_row_id(campaign.id, nid) for nid in all_node_ids]

    for node in batch:
        db_hid = _node_row_id(campaign.id, node.id)
        # Ensure hypothesis row exists so worker can write experiment_steps/tool_calls
        # with FK integrity.
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
        pending.append({"ar": ar, "nid": node.id, "db_hid": db_hid, "enq_mono": time.monotonic()})

    # Wait long enough for parallel Celery sub-agents (MNIST runs can be 15–40+ min each).
    remaining = max(60.0, float(campaign.remaining_seconds()))
    batch_deadline = time.monotonic() + min(
        remaining * 0.5,
        max(1800.0, float(len(batch)) * 2700.0),
    )
    completed_ids: set[str] = set()
    last_progress_mono = time.monotonic()
    evict_after = max(0, int(getattr(settings, "campaign_frontier_evict_idle_sec", 0)))

    _redis = None
    try:
        _redis = await create_redis(settings.redis_url)
    except Exception as exc:
        logger.warning("Redis unavailable for peer broadcast: %s", exc)

    while pending and time.monotonic() < batch_deadline:
        progressed = False
        for item in list(pending):
            ar, nid, db_hid = item["ar"], item["nid"], item["db_hid"]
            ready = await asyncio.to_thread(lambda: ar.ready())
            if not ready:
                continue
            progressed = True
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
            result["campaign_node_id"] = str(
                result.get("campaign_node_id") or nid
            )
            completed_ids.add(nid)
            last_progress_mono = time.monotonic()
            yield result

            still_running_db = [
                hid for hid, tree_id in zip(all_db_hids, all_node_ids, strict=True)
                if tree_id not in completed_ids
            ]
            if still_running_db and _redis is not None:
                finding_payload = build_peer_finding_payload(result)
                await publish_peer_finding(
                    _redis, target_hypothesis_ids=still_running_db, finding=finding_payload
                )

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
                    await asyncio.to_thread(
                        lambda ar_=victim["ar"]: ar_.revoke(terminate=True)
                    )
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

                still_running_db = [
                    hid for hid, tree_id in zip(all_db_hids, all_node_ids, strict=True)
                    if tree_id not in completed_ids
                ]
                if still_running_db and _redis is not None:
                    finding_payload = build_peer_finding_payload(eviction_result)
                    await publish_peer_finding(
                        _redis, target_hypothesis_ids=still_running_db, finding=finding_payload
                    )
                continue

        if not progressed and pending:
            await asyncio.sleep(0.15)

    if _redis is not None:
        try:
            await _redis.close()
        except Exception:
            pass

    # Timeout fallback for stragglers
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

    try:
        await emitter.emit(
            session_id=campaign.id,
            event_type=EventType.CAMPAIGN_STARTED,
            step="campaign.start",
            payload=campaign.summary(),
        )

        # ── Stage 1: Intake + Literature (lightweight, reuse session infra) ──
        parsed = await parse_question(campaign.question)
        try:
            prior = await asyncio.wait_for(
                build_prior(
                    parsed, session_id=campaign.id, emitter=emitter,
                    session_factory=session_factory,
                    paper_ttl_days=30,
                    llm=llm,
                ),
                timeout=180,
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

        # ── Stage 3: Campaign loop ────────────────────────────────────────
        last_checkpoint = time.monotonic()
        generation = campaign.hypothesis_tree._generation

        while not campaign.should_stop():
            # Get next batch from frontier
            batch = campaign.hypothesis_tree.next_batch(
                size=settings.campaign_batch_size,
                strategy="highest_expected_value",
            )

            # If frontier is empty, generate new seed hypotheses
            if not batch:
                await emitter.emit(
                    session_id=campaign.id,
                    event_type=EventType.HYPO_TREE_FRONTIER_EMPTY,
                    step="campaign.frontier_empty",
                    payload={"generation": generation, "tree": campaign.hypothesis_tree.summary()},
                )
                generation += 1
                campaign.hypothesis_tree._generation = generation
                seeds = await generate_seed_hypotheses(
                    campaign, prior, parsed, llm, emitter, session_factory, generation,
                )
                if not seeds:
                    logger.info("[campaign %s] No seeds generated; stopping.", campaign.id)
                    break
                batch = seeds
                # Persist tree + frontier before long-running Celery work so GET /campaigns
                # and monitors show real tree size instead of zeros for the whole batch.
                await db_save_campaign(campaign, session_factory)

            # Run experiments — stream each Celery completion so DB / GET /campaigns
            # advance while sibling workers are still running.
            breakthrough_found = False
            async for result in _iter_campaign_batch_results(
                campaign=campaign,
                batch=batch,
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

                campaign.total_hypotheses += 1

                if verdict == "confirmed":
                    campaign.total_confirmed += 1

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

                await db_save_campaign(campaign, session_factory)

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

        # Write a paper with whatever we found
        ledger = _campaign_ledger_from_tree(campaign.hypothesis_tree)
        synthesis = {
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
        await write_paper_minimal(
            session_id=campaign.id,
            session_factory=session_factory,
            emitter=emitter,
            llm=llm,
            question=campaign.question,
            prior=prior_dict,
            synthesis=synthesis,
        )

        await emitter.emit(
            session_id=campaign.id,
            event_type=EventType.CAMPAIGN_COMPLETED,
            step="campaign.complete",
            payload=campaign.summary(),
        )
        await db_mark_research_session_completed(campaign.id, session_factory)

    except Exception as exc:
        logger.exception("[campaign %s] Fatal error: %s", campaign.id, exc)
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
