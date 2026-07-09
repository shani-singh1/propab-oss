from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from sqlalchemy.exc import DisconnectionError, InterfaceError, OperationalError

from propab.config import settings
from propab.db import create_engine, create_redis, create_session_factory
from propab.events import EventEmitter
from services.api.app.routes.health import router as health_router
from services.api.app.routes.research import router as research_router
from services.api.app.routes.sessions import router as sessions_router
from services.api.app.routes.stream import router as stream_router
from services.api.app.routes.tools import router as tools_router

logger = logging.getLogger("propab.api")


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


app = FastAPI(
    title="Propab API",
    version="0.1.0",
    description=(
        "HTTP entrypoint for research sessions and long-running campaigns. "
        "Interactive OpenAPI docs at `/docs`; workflow guide in `docs/api_reference.md`."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



def _error_response(status_code: int, error: str, detail: str, correlation_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "detail": detail, "correlation_id": correlation_id},
    )


async def _dependency_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    """A downstream dependency (Postgres/Redis) is unreachable → clean 503, not a raw 500.

    Connection-level failures (DB down, connection dropped, Redis unreachable) are
    transient infrastructure problems, not client errors, so we surface a 503 that a
    caller can retry. The full exception (with traceback) is logged server-side under
    a correlation id and never returned to the client.
    """
    correlation_id = uuid.uuid4().hex
    logger.error(
        "Dependency unavailable [%s] on %s %s: %s",
        correlation_id, request.method, request.url.path, exc, exc_info=exc,
    )
    return _error_response(
        503,
        "dependency_unavailable",
        "A backend dependency is temporarily unavailable. Please retry shortly.",
        correlation_id,
    )


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: never leak a stack trace or internal detail to the client.

    FastAPI already hides tracebacks by default, but this guarantees a stable JSON
    error envelope and attaches a correlation id that ties the client-visible error
    to the full traceback in the server logs (quote it when reporting a bug).
    """
    correlation_id = uuid.uuid4().hex
    logger.exception(
        "Unhandled error [%s] on %s %s", correlation_id, request.method, request.url.path,
    )
    return _error_response(
        500,
        "internal_error",
        "An internal error occurred. Quote the correlation_id when reporting this.",
        correlation_id,
    )


# Order does not matter here — Starlette dispatches by the most specific exception
# type. The connection-error handlers win for their types; everything else that is
# otherwise unhandled falls through to the catch-all 500 handler.
for _exc_type in (OperationalError, InterfaceError, DisconnectionError, RedisError):
    app.add_exception_handler(_exc_type, _dependency_unavailable_handler)
app.add_exception_handler(Exception, _unhandled_exception_handler)

app.include_router(health_router)
app.include_router(research_router)
app.include_router(sessions_router)
app.include_router(stream_router)
app.include_router(tools_router)
