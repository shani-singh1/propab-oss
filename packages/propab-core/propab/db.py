from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from propab.types import PropabEvent


def create_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def create_redis(redis_url: str) -> Redis:
    client = Redis.from_url(redis_url, decode_responses=True)
    await client.ping()
    return client


async def get_session(session_factory: async_sessionmaker[AsyncSession]) -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        yield session


async def load_events_after(
    session_factory: async_sessionmaker[AsyncSession],
    session_id: str,
    after_event_id: str,
    *,
    limit: int = 2000,
) -> list[dict]:
    """Replay events recorded AFTER ``after_event_id`` for SSE reconnection.

    Returns dicts in the same shape as ``PropabEvent.to_dict()`` (so a reconnecting
    client can consume them identically to live frames). If ``after_event_id`` is
    unknown, returns ``[]`` — the caller then simply streams live events.
    """
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT id, session_id, event_type, source, step,
                           hypothesis_id, parent_event_id, payload_json, created_at
                    FROM events
                    WHERE session_id = CAST(:sid AS uuid)
                      AND created_at > (
                          SELECT created_at FROM events WHERE id = CAST(:after AS uuid)
                      )
                    ORDER BY created_at ASC
                    LIMIT :lim
                    """
                ),
                {"sid": session_id, "after": after_event_id, "lim": limit},
            )
        ).mappings().all()
    out: list[dict] = []
    for r in rows:
        payload = r["payload_json"]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        created = r["created_at"]
        out.append(
            {
                "event_id": str(r["id"]),
                "session_id": str(r["session_id"]),
                "timestamp": created.isoformat() if hasattr(created, "isoformat") else str(created),
                "source": r["source"],
                "event_type": r["event_type"],
                "step": r["step"],
                "payload": payload if isinstance(payload, dict) else {},
                "parent_event_id": str(r["parent_event_id"]) if r["parent_event_id"] else None,
                "hypothesis_id": str(r["hypothesis_id"]) if r["hypothesis_id"] else None,
            }
        )
    return out


async def insert_event(session: AsyncSession, event: PropabEvent) -> None:
    payload_json = json.dumps(event.payload)
    await session.execute(
        text(
            """
            INSERT INTO events (id, session_id, event_type, source, step, hypothesis_id, parent_event_id, payload_json)
            VALUES (:id, :session_id, :event_type, :source, :step, :hypothesis_id, :parent_event_id, CAST(:payload_json AS jsonb))
            """
        ),
        {
            "id": event.event_id,
            "session_id": event.session_id,
            "event_type": event.event_type.value,
            "source": event.source,
            "step": event.step,
            "hypothesis_id": event.hypothesis_id,
            "parent_event_id": event.parent_event_id,
            "payload_json": payload_json,
        },
    )
    await session.commit()
