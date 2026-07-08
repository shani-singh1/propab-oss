from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.belief_state import CampaignBeliefState, ClosedBelief
from propab.campaign import (
    ACC_METRIC_PLAUSIBILITY_MAX,
    BreakthroughCriteria,
    ResearchCampaign,
    STATUS_ACTIVE,
    STATUS_BUDGET_EXHAUSTED,
)
from propab.campaign_resume import (
    CONTRARIAN_QUESTION,
    apply_contrarian_belief_reset,
    backfill_belief_state_if_empty,
    validate_resume_readiness,
)
from propab.campaign_db import (
    db_load_campaign,
    db_load_session_events_tail,
    db_save_campaign,
)
from propab.config import settings
from propab.domain_modules.registry import resolve_domain_plugin
from propab.events import EventEmitter
from propab.types import EventType
from services.api.app.deps import get_emitter, get_session_factory

logger = logging.getLogger(__name__)

router = APIRouter(tags=["research"])


async def _dispatch_campaign(
    *,
    campaign_id: str,
    mode: str,
    campaign: ResearchCampaign,
    background_tasks: BackgroundTasks,
    session_factory: async_sessionmaker,
    emitter: EventEmitter,
) -> None:
    """Run a campaign loop off the API process.

    Delegates to the orchestrator service over HTTP when ``orchestrator_url`` is
    configured (production), so an API restart cannot kill an in-flight campaign.
    Falls back to an in-API BackgroundTask only for local/dev/test where no
    orchestrator is configured.
    """
    orch = (settings.orchestrator_url or "").strip()
    if orch:
        body = {"campaign_id": campaign_id, "mode": mode}
        headers: dict[str, str] = {}
        if (settings.orchestrator_internal_token or "").strip():
            headers["Authorization"] = f"Bearer {settings.orchestrator_internal_token.strip()}"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(f"{orch.rstrip('/')}/internal/campaign", json=body, headers=headers)
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=f"Orchestrator error: {exc.response.text[:500]}") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Orchestrator unreachable: {exc}") from exc
    else:
        # Lazy import: only the dev/test fallback runs the loop in-process, so the
        # API package has no module-level dependency on the orchestrator service.
        from services.orchestrator.campaign_loop import run_campaign_loop

        background_tasks.add_task(
            run_campaign_loop,
            campaign,
            session_factory=session_factory,
            emitter=emitter,
        )


class ResearchConfig(BaseModel):
    max_hypotheses: int = Field(default=5, ge=1, le=10)
    paper_ttl_days: int = Field(default=30, ge=1, le=365)
    llm_model: str = Field(default="gpt-4o")


class ResearchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "What molecular descriptors best predict dielectric constant in perovskites?",
                    "config": {"max_hypotheses": 5, "paper_ttl_days": 30, "llm_model": "gpt-4o"},
                }
            ]
        }
    )

    question: str = Field(min_length=8, description="Research question for a short hypothesis session")
    config: ResearchConfig = Field(default_factory=ResearchConfig)


class ResearchResponse(BaseModel):
    session_id: str
    stream_url: str
    status: str


@router.post(
    "/research",
    response_model=ResearchResponse,
    summary="Start a short research session",
    description=(
        "Runs a bounded hypothesis session (not a full campaign). "
        "Use `POST /campaigns` for long-running tree growth with beliefs and checkpoints."
    ),
    responses={
        502: {"description": "Orchestrator unreachable when ORCHESTRATOR_URL is set"},
    },
)
async def create_research_session(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    emitter: EventEmitter = Depends(get_emitter),
    session_factory: async_sessionmaker = Depends(get_session_factory),
) -> ResearchResponse:
    session_id = str(uuid4())

    async with session_factory() as session:
        await session.execute(
            text(
                """
                INSERT INTO research_sessions (id, question, status, stage)
                VALUES (:id, :question, :status, :stage)
                """
            ),
            {
                "id": session_id,
                "question": request.question,
                "status": "running",
                "stage": "intake",
            },
        )
        await session.commit()

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.SESSION_STARTED,
        step="session.start",
        payload={"question": request.question, "config": request.config.model_dump()},
    )

    orch = (settings.orchestrator_url or "").strip()
    if orch:
        body = {
            "session_id": session_id,
            "question": request.question,
            "max_hypotheses": request.config.max_hypotheses,
            "paper_ttl_days": request.config.paper_ttl_days,
            "llm_model": request.config.llm_model,
        }
        headers: dict[str, str] = {}
        if (settings.orchestrator_internal_token or "").strip():
            headers["Authorization"] = f"Bearer {settings.orchestrator_internal_token.strip()}"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(f"{orch.rstrip('/')}/internal/research", json=body, headers=headers)
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=f"Orchestrator error: {exc.response.text[:500]}") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Orchestrator unreachable: {exc}") from exc
    else:
        # Lazy import keeps the API free of a module-level orchestrator dependency;
        # this in-process fallback only runs when no orchestrator URL is configured.
        from services.orchestrator.research_loop import run_research_loop

        background_tasks.add_task(
            run_research_loop,
            session_id=session_id,
            question=request.question,
            max_hypotheses=request.config.max_hypotheses,
            paper_ttl_days=request.config.paper_ttl_days,
            emitter=emitter,
            session_factory=session_factory,
            llm_model=request.config.llm_model,
        )

    return ResearchResponse(
        session_id=session_id,
        stream_url=f"/stream/{session_id}",
        status="started",
    )


# ── Campaign endpoints ────────────────────────────────────────────────────────

class BreakthroughCriteriaRequest(BaseModel):
    metric_name: str = Field(default="val_accuracy")
    improvement_threshold: float = Field(default=0.05, ge=0.001, le=1.0)
    direction: str = Field(default="higher_is_better")
    min_confidence: float = Field(default=0.85, ge=0.5, le=1.0)
    min_replications: int = Field(default=3, ge=1, le=20)
    # Optional declared instrumentation ceiling for higher_is_better metrics. If omitted,
    # accuracy-style criteria adopt the default accuracy ceiling; other metrics stay unbounded.
    plausibility_max: float | None = Field(default=None, ge=0.0, le=1.0)
    min_confirmed_findings: int | None = Field(default=None, ge=1, le=50)


class CampaignRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": (
                        "Which additive combinatorics constructions maximize Sidon density "
                        "under greedy search? [domain_profile:math_combinatorics]"
                    ),
                    "compute_budget_hours": 3.0,
                    "policy_mode": "accepted",
                    "max_hypotheses": 120,
                },
                {
                    "question": (
                        "Do housekeeping genes show cross-tissue expression conservation "
                        "under leave-one-tissue-out holdout? [domain_profile:genomics]"
                    ),
                    "compute_budget_hours": 2.0,
                },
            ]
        }
    )

    question: str = Field(
        min_length=8,
        description="Campaign question; include `[domain_profile:<id>]` for explicit domain routing",
    )
    compute_budget_hours: float = Field(default=4.0, ge=0.1, le=168.0)
    breakthrough_criteria: BreakthroughCriteriaRequest = Field(
        default_factory=BreakthroughCriteriaRequest
    )
    # "accepted" uses bucket-local accepted policy; "candidate" runs calibration evaluation
    policy_mode: str = Field(default="accepted", pattern="^(accepted|candidate)$")
    # Anomaly engine integration (Phase 7–8)
    seed_source: str = Field(default="default", pattern="^(default|anomaly)$")
    anomaly_artifacts_dir: str | None = Field(default=None)
    max_hypotheses: int | None = Field(default=None, ge=1, le=500)
    literature_prior: dict[str, Any] | None = Field(
        default=None,
        description="Pre-built prior_json for campaign 2 A/B (skips LLM prior build when set)",
    )
    closed_beliefs: list[dict[str, str]] | None = Field(
        default=None,
        description="Belief subspaces to close at launch (statement + reason)",
    )
    orchestrator_directive: str | None = Field(
        default=None,
        description="Pinned human message for belief_state at campaign start",
    )


class CampaignResponse(BaseModel):
    campaign_id: str
    stream_url: str
    status: str


class CampaignResumeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"compute_budget_hours": 4.0, "clear_hypothesis_cap": True},
                {"belief_reset": "contrarian", "orchestrator_directive": "Test the opposite direction."},
            ]
        }
    )

    max_hypotheses_cap: int | None = Field(default=None, ge=1, le=500)
    compute_budget_hours: float | None = Field(default=None, ge=0.1, le=168.0)
    clear_hypothesis_cap: bool = False
    question: str | None = Field(default=None, min_length=8)
    belief_reset: str | None = Field(
        default=None,
        pattern="^(contrarian)$",
        description="Replace active beliefs without discarding tree/evidence (fixes.md contrarian run)",
    )
    orchestrator_directive: str | None = None


@router.post(
    "/campaigns",
    response_model=CampaignResponse,
    summary="Launch a research campaign",
    responses={
        502: {"description": "Orchestrator unreachable when ORCHESTRATOR_URL is set"},
    },
)
async def create_campaign(
    request: CampaignRequest,
    background_tasks: BackgroundTasks,
    emitter: EventEmitter = Depends(get_emitter),
    session_factory: async_sessionmaker = Depends(get_session_factory),
) -> CampaignResponse:
    """
    Start a long-running research campaign.

    A campaign differs from a session in that it:
    - Grows a HypothesisTree (confirms → expand children, refutes → generate alternatives)
    - Measures its own baseline at the start
    - Runs until breakthrough OR compute budget exhausted
    - Checkpoints to DB after every batch so it can be resumed

    The campaign_id is also the session_id for the event stream, so
    GET /stream/{campaign_id} delivers all events in real time.
    """
    campaign_id = str(uuid4())
    bc = request.breakthrough_criteria
    metric_name = bc.metric_name
    direction = bc.direction
    # When the campaign routes to a domain that declares its own objective (math /
    # graph verification domains), adopt the DOMAIN metric + direction instead of the
    # generic ML default (val_accuracy). Without this a computational campaign is
    # scored against a trained MLP baseline it can never beat: campaign 1ae74abd ran
    # math_combinatorics with metric_name=val_accuracy, so _is_ml_campaign() saw
    # "accuracy", measured baseline_metric=0.875, and best_metric stayed 0.0 the whole
    # run — no cap-set/Sidon construction could register a record. See the deep
    # post-mortem. Only override when the caller left the default in place.
    if bc.metric_name == "val_accuracy":
        try:
            _plugin = resolve_domain_plugin(question=request.question)
            _obj = _plugin.objective_spec() if _plugin is not None else None
        except Exception:  # noqa: BLE001 — routing/objective errors must not block launch
            _obj = None
        if _obj and _obj.get("metric_name"):
            # Only adopt the domain metric when the domain ALSO declares its
            # optimization direction. Silently inheriting the ML default
            # ("higher_is_better") would score a minimization metric backwards —
            # a domain that reports a "lower_is_better" metric with no explicit
            # direction would be optimized in the wrong direction. If direction is
            # missing, keep the generic default metric + direction and log.
            _obj_direction = _obj.get("direction")
            if _obj_direction:
                metric_name = str(_obj["metric_name"])
                direction = str(_obj_direction)
            else:
                logger.warning(
                    "Domain objective_spec for question %r declared metric_name=%r "
                    "but no direction; keeping generic metric %r/%r to avoid scoring "
                    "a minimization metric backwards.",
                    request.question,
                    _obj.get("metric_name"),
                    metric_name,
                    direction,
                )
    plausibility_max = bc.plausibility_max
    if (
        plausibility_max is None
        and direction == "higher_is_better"
        and "accuracy" in (metric_name or "").lower()
    ):
        plausibility_max = ACC_METRIC_PLAUSIBILITY_MAX
    criteria = BreakthroughCriteria(
        metric_name=metric_name,
        improvement_threshold=bc.improvement_threshold,
        direction=direction,
        min_confidence=bc.min_confidence,
        min_replications=bc.min_replications,
        plausibility_max=plausibility_max,
        min_confirmed_findings=bc.min_confirmed_findings,
    )
    campaign = ResearchCampaign(
        id=campaign_id,
        question=request.question,
        breakthrough_criteria=criteria,
        compute_budget_seconds=int(request.compute_budget_hours * 3600),
        checkpoint_every=settings.campaign_checkpoint_every,
        policy_mode=request.policy_mode,
        seed_source=request.seed_source,
        anomaly_artifacts_dir=request.anomaly_artifacts_dir,
        max_hypotheses_cap=request.max_hypotheses,
    )
    if request.closed_beliefs:
        campaign.belief_state.closed_beliefs.extend(
            ClosedBelief(
                statement=str(item.get("statement") or ""),
                reason=str(item.get("reason") or ""),
            )
            for item in request.closed_beliefs
            if item.get("statement")
        )
    if request.orchestrator_directive:
        campaign.belief_state.add_human_message(request.orchestrator_directive)

    # Persist initial state so it can be queried immediately
    await db_save_campaign(campaign, session_factory)

    prior_json_str: str | None = None
    if request.literature_prior:
        prior_json_str = json.dumps(request.literature_prior, ensure_ascii=False)

    # Also create a research_sessions row so the SSE stream endpoint can find it
    async with session_factory() as db:
        if prior_json_str:
            await db.execute(
                text("""
                    INSERT INTO research_sessions (id, question, status, stage, prior_json)
                    VALUES (:id, :question, 'running', 'campaign', CAST(:prior_json AS jsonb))
                    ON CONFLICT (id) DO UPDATE SET prior_json = EXCLUDED.prior_json
                """),
                {"id": campaign_id, "question": request.question, "prior_json": prior_json_str},
            )
        else:
            await db.execute(
                text("""
                    INSERT INTO research_sessions (id, question, status, stage)
                    VALUES (:id, :question, 'running', 'campaign')
                    ON CONFLICT (id) DO NOTHING
                """),
                {"id": campaign_id, "question": request.question},
            )
        await db.commit()

    await _dispatch_campaign(
        campaign_id=campaign_id,
        mode="start",
        campaign=campaign,
        background_tasks=background_tasks,
        session_factory=session_factory,
        emitter=emitter,
    )

    return CampaignResponse(
        campaign_id=campaign_id,
        stream_url=f"/stream/{campaign_id}",
        status="started",
    )


@router.get(
    "/campaigns",
    summary="List campaigns",
    description="Newest-first summary rows from `research_campaigns` for dashboards.",
)
async def list_campaigns(
    session_factory: async_sessionmaker = Depends(get_session_factory),
) -> dict:
    async with session_factory() as db:
        rows = (
            await db.execute(
                text(
                    """
                    SELECT id, question, status, baseline_metric, best_metric, improvement_pct,
                           total_hypotheses, total_confirmed, compute_budget_seconds,
                           compute_seconds_used, started_at, completed_at
                    FROM research_campaigns
                    ORDER BY started_at DESC NULLS LAST
                    """
                )
            )
        ).mappings().all()
    return {"campaigns": [dict(r) for r in rows]}


@router.get(
    "/campaigns/{campaign_id}",
    summary="Get campaign snapshot",
    description="Full campaign state: hypothesis tree, belief state, budget, and event counts.",
    responses={404: {"description": "Campaign not found"}},
)
async def get_campaign_state(
    campaign_id: str,
    session_factory: async_sessionmaker = Depends(get_session_factory),
) -> dict:
    campaign = await db_load_campaign(campaign_id, session_factory)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found")

    async with session_factory() as db:
        sess_row = (
            await db.execute(
                text(
                    """
                    SELECT id, question, status, stage, created_at, completed_at
                    FROM research_sessions
                    WHERE id = CAST(:cid AS uuid)
                    """
                ),
                {"cid": campaign_id},
            )
        ).mappings().first()

    event_counts: dict[str, int] = {}
    async with session_factory() as db:
        rows = (
            await db.execute(
                text(
                    """
                    SELECT event_type, COUNT(*) AS n
                    FROM events
                    WHERE session_id = CAST(:cid AS uuid)
                    GROUP BY event_type
                    """
                ),
                {"cid": campaign_id},
            )
        ).mappings().all()
        event_counts = {str(r["event_type"]): int(r["n"]) for r in rows}

    return {
        "campaign_id": campaign.id,
        "campaign": campaign.to_dict(),
        "summary": campaign.summary(),
        "research_session": dict(sess_row) if sess_row else None,
        "event_counts_by_type": event_counts,
    }


@router.post(
    "/campaigns/{campaign_id}/resume",
    response_model=CampaignResponse,
    summary="Resume a stopped campaign",
    description=(
        "Warm-starts from Postgres checkpoint. Tree and beliefs are preserved unless "
        "`belief_reset=contrarian` is requested."
    ),
    responses={
        404: {"description": "Campaign not found"},
        409: {"description": "Campaign is already active"},
        502: {"description": "Orchestrator unreachable"},
    },
)
async def resume_campaign(
    campaign_id: str,
    background_tasks: BackgroundTasks,
    request: CampaignResumeRequest = CampaignResumeRequest(),
    emitter: EventEmitter = Depends(get_emitter),
    session_factory: async_sessionmaker = Depends(get_session_factory),
) -> CampaignResponse:
    req = request

    launch_meta: dict | None = None
    launch_path = Path(__file__).resolve().parents[4] / "artifacts" / "mandrake_contrarian_campaign.json"
    if not launch_path.exists():
        launch_path = Path(__file__).resolve().parents[4] / "artifacts" / "mandrake_campaign_latest.json"
    if launch_path.exists():
        try:
            blob = json.loads(launch_path.read_text(encoding="utf-8"))
            if blob.get("campaign_id") == campaign_id:
                launch_meta = blob
        except Exception:
            launch_meta = None

    campaign = await db_load_campaign(campaign_id, session_factory)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    if campaign.status == STATUS_ACTIVE:
        raise HTTPException(status_code=409, detail="Campaign is already active")

    campaign.status = STATUS_ACTIVE
    campaign.stop_reason = None

    if req.clear_hypothesis_cap or (launch_meta and launch_meta.get("max_hypotheses") is None):
        campaign.max_hypotheses_cap = None
    elif req.max_hypotheses_cap is not None:
        campaign.max_hypotheses_cap = req.max_hypotheses_cap
    elif campaign.max_hypotheses_cap is None and launch_meta and launch_meta.get("max_hypotheses"):
        campaign.max_hypotheses_cap = int(launch_meta["max_hypotheses"])

    budget_hours = req.compute_budget_hours
    if budget_hours is None and launch_meta and launch_meta.get("compute_budget_hours"):
        budget_hours = float(launch_meta["compute_budget_hours"])
    if budget_hours is not None:
        campaign.compute_budget_seconds = int(budget_hours * 3600)
    # ALWAYS rebase the budget clock on resume, even when no new budget is supplied.
    # Budget is enforced by wall clock (should_stop() -> elapsed_seconds() = now -
    # started_at); compute_seconds_used is not consulted. Without this, a plain resume
    # hours/days after launch sees elapsed >> budget, so should_stop() is True on the
    # first loop iteration and the run finalizes as budget_exhausted with zero new
    # hypotheses while the API still returns status="resumed" — a silent no-op.
    campaign.started_at = datetime.now(tz=UTC).isoformat()

    new_question = req.question
    if new_question is None and launch_meta and launch_meta.get("question"):
        new_question = str(launch_meta["question"])
    if new_question:
        campaign.question = new_question

    events = await db_load_session_events_tail(campaign_id, session_factory, limit=2000)
    belief_reset_mode = req.belief_reset or (launch_meta or {}).get("belief_reset")
    if belief_reset_mode == "contrarian":
        directive = req.orchestrator_directive or (launch_meta or {}).get("orchestrator_directive")
        campaign.belief_state = apply_contrarian_belief_reset(
            campaign.belief_state,
            orchestrator_directive=directive,
        )
        campaign.belief_state.last_synthesis_node_ids = [
            nid
            for nid, n in campaign.hypothesis_tree.nodes.items()
            if n.verdict in ("confirmed", "refuted", "inconclusive")
        ]
    else:
        restored, changed = backfill_belief_state_if_empty(
            campaign.belief_state,
            events=events,
            tree_nodes={nid: n.to_dict() for nid, n in campaign.hypothesis_tree.nodes.items()},
        )
        if changed:
            campaign.belief_state = restored

    await db_save_campaign(campaign, session_factory)

    async with session_factory() as db:
        session_updates: dict[str, Any] = {"id": campaign_id}
        set_clauses = ["status = 'running'", "stage = 'campaign'", "completed_at = NULL"]
        if new_question:
            set_clauses.append("question = :question")
            session_updates["question"] = new_question
        await db.execute(
            text(f"""
                UPDATE research_sessions
                SET {", ".join(set_clauses)}
                WHERE id = CAST(:id AS uuid)
            """),
            session_updates,
        )
        await db.commit()

    await _dispatch_campaign(
        campaign_id=campaign_id,
        mode="resume",
        campaign=campaign,
        background_tasks=background_tasks,
        session_factory=session_factory,
        emitter=emitter,
    )

    return CampaignResponse(
        campaign_id=campaign_id,
        stream_url=f"/stream/{campaign_id}",
        status="resumed",
    )


@router.get(
    "/campaigns/{campaign_id}/resume-readiness",
    summary="Pre-resume validation",
    description="Checks belief backfill, cap persistence, and stale status before calling resume.",
    responses={404: {"description": "Campaign not found"}},
)
async def get_resume_readiness(
    campaign_id: str,
    session_factory: async_sessionmaker = Depends(get_session_factory),
) -> dict:
    """Pre-resume validation: beliefs, cap persistence, stale status detection."""
    campaign = await db_load_campaign(campaign_id, session_factory)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    events = await db_load_session_events_tail(campaign_id, session_factory, limit=2000)
    launch_meta = None
    launch_path = Path(__file__).resolve().parents[4] / "artifacts" / "mandrake_campaign_latest.json"
    if launch_path.exists():
        try:
            launch_meta = json.loads(launch_path.read_text(encoding="utf-8"))
            if launch_meta.get("campaign_id") != campaign_id:
                launch_meta = None
        except Exception:
            launch_meta = None
    return validate_resume_readiness(
        campaign.to_dict(),
        events=events,
        launch_meta=launch_meta,
    )
