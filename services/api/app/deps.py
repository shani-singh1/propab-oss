from __future__ import annotations

from typing import Any

from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from propab.events import EventEmitter


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
