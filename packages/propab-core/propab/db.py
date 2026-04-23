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
