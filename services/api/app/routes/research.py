from __future__ import annotations

from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.campaign import BreakthroughCriteria, ResearchCampaign
from propab.config import settings
from propab.events import EventEmitter
from propab.types import EventType
from services.api.app.deps import get_emitter, get_session_factory
from services.orchestrator.campaign_loop import db_load_campaign, db_save_campaign, run_campaign_loop
from services.orchestrator.research_loop import run_research_loop

router = APIRouter(tags=["research"])


class ResearchConfig(BaseModel):
    max_hypotheses: int = Field(default=5, ge=1, le=10)
    paper_ttl_days: int = Field(default=30, ge=1, le=365)
    llm_model: str = Field(default="gpt-4o")


class ResearchRequest(BaseModel):
    question: str = Field(min_length=8)
    config: ResearchConfig = Field(default_factory=ResearchConfig)


class ResearchResponse(BaseModel):
    session_id: str
    stream_url: str
    status: str


@router.post("/research", response_model=ResearchResponse)
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
        background_tasks.add_task(
            run_research_loop,
            session_id=session_id,
            question=request.question,
            max_hypotheses=request.config.max_hypotheses,
            paper_ttl_days=request.config.paper_ttl_days,
            emitter=emitter,
            session_factory=session_factory,
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


class CampaignRequest(BaseModel):
    question: str = Field(min_length=8)
    compute_budget_hours: float = Field(default=4.0, ge=0.1, le=168.0)
    breakthrough_criteria: BreakthroughCriteriaRequest = Field(
        default_factory=BreakthroughCriteriaRequest
    )


class CampaignResponse(BaseModel):
    campaign_id: str
    stream_url: str
    status: str


@router.post("/campaigns", response_model=CampaignResponse)
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
    criteria = BreakthroughCriteria(
        metric_name=request.breakthrough_criteria.metric_name,
        improvement_threshold=request.breakthrough_criteria.improvement_threshold,
        direction=request.breakthrough_criteria.direction,
        min_confidence=request.breakthrough_criteria.min_confidence,
        min_replications=request.breakthrough_criteria.min_replications,
    )
    campaign = ResearchCampaign(
        id=campaign_id,
        question=request.question,
        breakthrough_criteria=criteria,
        compute_budget_seconds=int(request.compute_budget_hours * 3600),
        checkpoint_every=settings.campaign_checkpoint_every,
    )

    # Persist initial state so it can be queried immediately
    await db_save_campaign(campaign, session_factory)

    # Also create a research_sessions row so the SSE stream endpoint can find it
    async with session_factory() as db:
        await db.execute(
            text("""
                INSERT INTO research_sessions (id, question, status, stage)
                VALUES (:id, :question, 'running', 'campaign')
                ON CONFLICT (id) DO NOTHING
            """),
            {"id": campaign_id, "question": request.question},
        )
        await db.commit()

    background_tasks.add_task(
        run_campaign_loop,
        campaign,
        session_factory=session_factory,
        emitter=emitter,
    )

    return CampaignResponse(
        campaign_id=campaign_id,
        stream_url=f"/stream/{campaign_id}",
        status="started",
    )


@router.get("/campaigns/{campaign_id}")
async def get_campaign_state(
    campaign_id: str,
    session_factory: async_sessionmaker = Depends(get_session_factory),
) -> dict:
    """Live campaign snapshot from Postgres (tree, metrics, budget)."""
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
