from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from services.api.app.deps import get_session_factory

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}")
async def get_session_state(
    session_id: str,
    session_factory: async_sessionmaker = Depends(get_session_factory),
) -> dict:
    async with session_factory() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT id, question, status, stage, prior_json, created_at, completed_at
                    FROM research_sessions
                    WHERE id = :session_id
                    """
                ),
                {"session_id": session_id},
            )
        ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return dict(row)


@router.get("/{session_id}/events")
async def get_session_events(
    session_id: str,
    session_factory: async_sessionmaker = Depends(get_session_factory),
) -> dict:
    async with session_factory() as session:
        exists = (
            await session.execute(
                text("SELECT id FROM research_sessions WHERE id = :session_id"),
                {"session_id": session_id},
            )
        ).scalar_one_or_none()
        if exists is None:
            raise HTTPException(status_code=404, detail="Session not found")
        rows = (
            await session.execute(
                text(
                    """
                    SELECT id, event_type, source, step, hypothesis_id, parent_event_id, payload_json, created_at
                    FROM events
                    WHERE session_id = :session_id
                    ORDER BY created_at ASC
                    """
                ),
                {"session_id": session_id},
            )
        ).mappings().all()
    return {"session_id": session_id, "events": [dict(row) for row in rows]}


@router.get("/{session_id}/prior")
async def get_session_prior(
    session_id: str,
    session_factory: async_sessionmaker = Depends(get_session_factory),
) -> dict:
    async with session_factory() as session:
        prior = (
            await session.execute(
                text("SELECT prior_json FROM research_sessions WHERE id = :session_id"),
                {"session_id": session_id},
            )
        ).scalar_one_or_none()
    if prior is None:
        raise HTTPException(status_code=404, detail="Session or prior not found")
    return {"session_id": session_id, "prior": prior}


@router.get("/{session_id}/hypotheses")
async def get_session_hypotheses(
    session_id: str,
    session_factory: async_sessionmaker = Depends(get_session_factory),
) -> dict:
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT id, text, test_methodology, scores_json, rank, status, verdict, confidence, evidence_summary, key_finding, created_at
                    FROM hypotheses
                    WHERE session_id = :session_id
                    ORDER BY rank ASC
                    """
                ),
                {"session_id": session_id},
            )
        ).mappings().all()
    return {"session_id": session_id, "hypotheses": [dict(row) for row in rows]}


@router.get("/{session_id}/trace")
async def get_session_trace(
    session_id: str,
    session_factory: async_sessionmaker = Depends(get_session_factory),
) -> dict:
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT e.*
                    FROM experiment_steps e
                    JOIN hypotheses h ON h.id = e.hypothesis_id
                    WHERE h.session_id = :session_id
                    ORDER BY e.created_at ASC
                    """
                ),
                {"session_id": session_id},
            )
        ).mappings().all()
    return {"session_id": session_id, "trace": [dict(row) for row in rows]}


@router.get("/{session_id}/paper")
async def get_session_paper(
    session_id: str,
    session_factory: async_sessionmaker = Depends(get_session_factory),
) -> dict:
    async with session_factory() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT payload_json
                    FROM events
                    WHERE session_id = :session_id AND event_type = :event_type
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ),
                {"session_id": session_id, "event_type": "paper.ready"},
            )
        ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Paper not ready or session not found")
    return {"session_id": session_id, "paper": row["payload_json"]}


@router.get("/{session_id}/llm-calls")
async def get_session_llm_calls(
    session_id: str,
    session_factory: async_sessionmaker = Depends(get_session_factory),
) -> dict:
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT id, hypothesis_id, call_purpose, model, prompt_text, response_text, input_tokens, output_tokens, duration_ms, created_at
                    FROM llm_calls
                    WHERE session_id = :session_id
                    ORDER BY created_at ASC
                    """
                ),
                {"session_id": session_id},
            )
        ).mappings().all()
    return {"session_id": session_id, "llm_calls": [dict(row) for row in rows]}
