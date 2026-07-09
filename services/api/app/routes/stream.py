from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.db import load_events_after
from services.api.app.deps import get_redis, get_session_factory, validate_uuid

router = APIRouter(tags=["stream"])


def _extract_event_id(payload: str) -> str | None:
    """Pull the event_id out of a published JSON frame (best-effort)."""
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict):
        eid = data.get("event_id")
        return str(eid) if eid else None
    return None


def _sse_frame(payload: str) -> str:
    """Format one SSE frame, prefixing an ``id:`` line when the event has an id.

    The ``id:`` line lets browsers echo it back as ``Last-Event-ID`` on reconnect
    so we can replay exactly the events missed during the disconnect.
    """
    event_id = _extract_event_id(payload)
    if event_id:
        return f"id: {event_id}\ndata: {payload}\n\n"
    return f"data: {payload}\n\n"


async def event_stream(
    redis: Redis,
    session_id: str,
    *,
    session_factory: async_sessionmaker | None = None,
    last_event_id: str | None = None,
) -> AsyncGenerator[str, None]:
    pubsub = redis.pubsub()
    channel = f"propab:{session_id}"
    await pubsub.subscribe(channel)
    try:
        # Replay backlog first so a reconnecting client recovers events it missed
        # while disconnected. The client de-dupes by event_id, so the small overlap
        # window between the DB backlog and the live subscription is harmless.
        if last_event_id and session_factory is not None:
            try:
                missed = await load_events_after(session_factory, session_id, last_event_id)
            except Exception:  # noqa: BLE001 — replay is best-effort; never break the live stream
                missed = []
            for ev in missed:
                yield _sse_frame(json.dumps(ev))

        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            payload = message.get("data")
            yield _sse_frame(payload)
    finally:
        # Best-effort teardown: if Redis dropped mid-stream, unsubscribe/close can
        # themselves raise — swallow so the generator exits cleanly and the client
        # can reconnect (with Last-Event-ID) rather than seeing a crashed connection.
        try:
            await pubsub.unsubscribe(channel)
        except Exception:  # noqa: BLE001
            pass
        try:
            await pubsub.close()
        except Exception:  # noqa: BLE001
            pass


@router.get(
    "/stream/{session_id}",
    summary="Live campaign event stream (SSE)",
    description=(
        "Server-sent events for a session or campaign. `session_id` is the campaign UUID "
        "returned by `POST /campaigns`. Each event is a JSON payload on a `data:` line, "
        "preceded by an `id:` line (the event_id). On reconnect, send the last id back via "
        "the `Last-Event-ID` header to replay events missed during the disconnect."
    ),
    responses={
        200: {"description": "text/event-stream of JSON event payloads"},
    },
)
async def stream_session_events(
    session_id: str,
    request: Request,
    redis: Redis = Depends(get_redis),
    session_factory: async_sessionmaker = Depends(get_session_factory),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    # Reject a malformed id up front (404) instead of opening an SSE connection that
    # subscribes to a garbage Redis channel and would only ever deliver empty frames.
    session_id = validate_uuid(session_id, not_found_detail="Session not found")
    # A ``?last_event_id=`` query param is honored as a fallback for clients (e.g.
    # EventSource polyfills) that cannot set the standard header.
    resume_from = last_event_id or request.query_params.get("last_event_id")
    return StreamingResponse(
        event_stream(
            redis,
            session_id,
            session_factory=session_factory,
            last_event_id=resume_from,
        ),
        media_type="text/event-stream",
    )
