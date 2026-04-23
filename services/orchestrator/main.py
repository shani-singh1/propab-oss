"""HTTP entrypoint for running ``run_research_loop`` out-of-process (API delegates when configured)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from propab.config import settings
from propab.db import create_engine, create_redis, create_session_factory
from propab.events import EventEmitter


def _verify_internal(authorization: Annotated[str | None, Header(alias="Authorization")] = None) -> None:
    tok = (settings.orchestrator_internal_token or "").strip()
    if not tok:
        return
    if (authorization or "").strip() != f"Bearer {tok}":
        raise HTTPException(status_code=401, detail="Invalid orchestrator internal token.")


class InternalResearchBody(BaseModel):
    session_id: str = Field(min_length=8)
    question: str = Field(min_length=8)
    max_hypotheses: int = Field(default=5, ge=1, le=20)
    paper_ttl_days: int = Field(default=30, ge=1, le=365)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    redis = await create_redis(settings.redis_url)
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.redis = redis
    app.state.emitter = EventEmitter(
        source="orchestrator",
        redis=redis,
        session_factory=session_factory,
    )
    yield
    await redis.close()
    await engine.dispose()


app = FastAPI(title="Propab Orchestrator", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "orchestrator"}


@app.post("/internal/research")
async def internal_research(
    request: Request,
    body: InternalResearchBody,
    background_tasks: BackgroundTasks,
    _: None = Depends(_verify_internal),
) -> dict[str, str]:
    """Accept a session created by the API and run the full research loop in this process."""
    from services.orchestrator.research_loop import run_research_loop

    background_tasks.add_task(
        run_research_loop,
        session_id=body.session_id,
        question=body.question,
        max_hypotheses=body.max_hypotheses,
        paper_ttl_days=body.paper_ttl_days,
        emitter=request.app.state.emitter,
        session_factory=request.app.state.session_factory,
    )
    return {"status": "accepted", "session_id": body.session_id}
