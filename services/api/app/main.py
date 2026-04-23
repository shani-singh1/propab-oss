from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from propab.config import settings
from propab.db import create_engine, create_redis, create_session_factory
from propab.events import EventEmitter
from services.api.app.routes.health import router as health_router
from services.api.app.routes.research import router as research_router
from services.api.app.routes.sessions import router as sessions_router
from services.api.app.routes.stream import router as stream_router
from services.api.app.routes.tools import router as tools_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    redis = await create_redis(settings.redis_url)

    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.redis = redis
    app.state.emitter = EventEmitter(source="api", redis=redis, session_factory=session_factory)
    yield

    await redis.close()
    await engine.dispose()


app = FastAPI(title="Propab API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(research_router)
app.include_router(sessions_router)
app.include_router(stream_router)
app.include_router(tools_router)
