from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import HTTPException, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from propab.events import EventEmitter


def validate_uuid(value: str, *, not_found_detail: str = "Resource not found") -> str:
    """Return ``value`` if it is a well-formed UUID, else raise a clean 404.

    Path params that address rows keyed by a Postgres ``uuid`` column reach the
    query as raw text. A malformed id (e.g. ``/sessions/not-a-uuid``) would then
    raise a driver ``DataError`` deep in the DB layer and surface as a 500. We
    reject it at the route boundary and treat it as "not found" — a malformed id
    cannot name an existing resource, so 404 matches the contract the routes
    already return for well-formed-but-unknown ids.
    """
    try:
        UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=404, detail=not_found_detail) from None
    return value


def get_engine(request: Request) -> AsyncEngine:
    return request.app.state.engine


def get_session_factory(request: Request) -> async_sessionmaker:
    return request.app.state.session_factory


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


def get_emitter(request: Request) -> EventEmitter:
    return request.app.state.emitter


def get_app_meta(request: Request) -> dict[str, Any]:
    return {
        "service": "api",
        "version": "0.1.0",
    }
