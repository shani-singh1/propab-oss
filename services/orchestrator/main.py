"""Orchestrator HTTP entrypoint. Runs the campaign loop out-of-process (the API
delegates to ``/internal/campaign`` when ``ORCHESTRATOR_URL`` is configured), so
an API restart cannot kill an in-flight campaign."""

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


class InternalCampaignBody(BaseModel):
    campaign_id: str = Field(min_length=8)
    # "start" for a fresh campaign, "resume" for a warm checkpoint. The API has
    # already persisted the (possibly resume-mutated) campaign to Postgres before
    # calling; the orchestrator reloads it by id and runs the loop in-process.
    mode: str = Field(default="start", pattern="^(start|resume)$")


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


@app.post("/internal/campaign")
async def internal_campaign(
    request: Request,
    body: InternalCampaignBody,
    background_tasks: BackgroundTasks,
    _: None = Depends(_verify_internal),
) -> dict[str, str]:
    """Run a campaign loop in the orchestrator process (off the API process).

    The campaign has already been persisted by the API (create) or mutated and
    re-persisted (resume) before this call, so the orchestrator reconstructs the
    full campaign from Postgres by id. Running here means an API restart cannot
    kill an in-flight campaign — the loop's lifecycle is owned by the orchestrator.
    """
    from propab.campaign_db import db_load_campaign
    from services.orchestrator.campaign_loop import run_campaign_loop

    campaign = await db_load_campaign(body.campaign_id, request.app.state.session_factory)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found")

    background_tasks.add_task(
        run_campaign_loop,
        campaign,
        session_factory=request.app.state.session_factory,
        emitter=request.app.state.emitter,
    )
    return {"status": "accepted", "campaign_id": body.campaign_id, "mode": body.mode}
