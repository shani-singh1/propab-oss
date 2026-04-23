from __future__ import annotations

import json

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from propab.db import insert_event
from propab.types import EventType, PropabEvent


class EventEmitter:
    def __init__(self, source: str, redis: Redis, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.source = source
        self.redis = redis
        self.session_factory = session_factory

    async def emit(
        self,
        *,
        session_id: str,
        event_type: EventType,
        step: str,
        payload: dict,
        parent_event_id: str | None = None,
        hypothesis_id: str | None = None,
    ) -> PropabEvent:
        event = PropabEvent.create(
            session_id=session_id,
            source=self.source,
            event_type=event_type,
            step=step,
            payload=payload,
            parent_event_id=parent_event_id,
            hypothesis_id=hypothesis_id,
        )

        async with self.session_factory() as session:
            await insert_event(session, event)

        channel = f"propab:{session_id}"
        await self.redis.publish(channel, json.dumps(event.to_dict()))
        return event
