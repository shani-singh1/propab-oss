from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis

from services.api.app.deps import get_redis

router = APIRouter(tags=["stream"])


async def event_stream(redis: Redis, session_id: str) -> AsyncGenerator[str, None]:
    pubsub = redis.pubsub()
    channel = f"propab:{session_id}"
    await pubsub.subscribe(channel)
    try:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            payload = message.get("data")
            yield f"data: {payload}\n\n"
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()


@router.get("/stream/{session_id}")
async def stream_session_events(session_id: str, redis: Redis = Depends(get_redis)) -> StreamingResponse:
    return StreamingResponse(event_stream(redis, session_id), media_type="text/event-stream")
