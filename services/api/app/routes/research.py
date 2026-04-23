from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.events import EventEmitter
from propab.types import EventType
from services.api.app.deps import get_emitter, get_session_factory
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
