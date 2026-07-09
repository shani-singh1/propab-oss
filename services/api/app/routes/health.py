from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from services.api.app.deps import get_app_meta, get_redis, get_session_factory

logger = logging.getLogger("propab.api")

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Liveness check",
    description=(
        "Returns `status: ok` plus build metadata whenever the API process is up. "
        "This is a cheap LIVENESS probe (used by Docker healthchecks and load balancers) "
        "and deliberately does NOT touch Postgres/Redis, so a transient dependency outage "
        "does not cause the container to be killed. Use `/ready` for dependency health."
    ),
    responses={200: {"description": "API process is up"}},
)
async def health(meta: dict = Depends(get_app_meta)) -> dict:
    return {"status": "ok", **meta}


@router.get(
    "/ready",
    summary="Readiness check",
    description=(
        "Verifies the API can reach its backing dependencies (Postgres + Redis) with a "
        "cheap `SELECT 1` / `PING`. Returns 200 when all dependencies are reachable, or "
        "503 with a per-dependency breakdown when one is down. Intended for load-balancer "
        "readiness gating and operator diagnostics."
    ),
    responses={
        200: {"description": "All dependencies reachable"},
        503: {"description": "One or more dependencies unavailable"},
    },
)
async def ready(
    meta: dict = Depends(get_app_meta),
    session_factory: async_sessionmaker = Depends(get_session_factory),
    redis: Redis = Depends(get_redis),
):
    checks: dict[str, str] = {}
    all_ok = True

    try:
        async with session_factory() as db:
            await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001 — report as unhealthy, never raise
        all_ok = False
        checks["database"] = "unavailable"
        logger.warning("Readiness: database check failed: %s", exc)

    try:
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001 — report as unhealthy, never raise
        all_ok = False
        checks["redis"] = "unavailable"
        logger.warning("Readiness: redis check failed: %s", exc)

    body = {"status": "ready" if all_ok else "not_ready", "checks": checks, **meta}
    if not all_ok:
        return JSONResponse(status_code=503, content=body)
    return body
