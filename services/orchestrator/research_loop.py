from __future__ import annotations

import json
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.config import settings
from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.types import EventType
from services.orchestrator.hypotheses import generate_ranked_hypotheses
from services.orchestrator.intake import parse_question
from services.orchestrator.literature import build_prior


async def _update_session(session_factory: async_sessionmaker, session_id: str, **fields: str) -> None:
    assignments = ", ".join(f"{name} = :{name}" for name in fields)
    values = {"id": session_id, **fields}
    query = text(f"UPDATE research_sessions SET {assignments} WHERE id = :id")
    async with session_factory() as session:
        await session.execute(query, values)
        await session.commit()


async def _insert_hypothesis_rows(session_factory: async_sessionmaker, session_id: str, hypotheses: list) -> None:
    async with session_factory() as session:
        for hypothesis in hypotheses:
            await session.execute(
                text(
                    """
                    INSERT INTO hypotheses (
                        id, session_id, text, test_methodology, scores_json, rank, status, verdict, confidence, evidence_summary, key_finding
                    )
                    VALUES (
                        :id, :session_id, :text, :test_methodology, CAST(:scores_json AS jsonb), :rank, :status, :verdict, :confidence, :evidence_summary, :key_finding
                    )
                    """
                ),
                {
                    "id": str(uuid4()),
                    "session_id": session_id,
                    "text": hypothesis.text,
                    "test_methodology": hypothesis.test_methodology,
                    "scores_json": json.dumps(hypothesis.scores),
                    "rank": hypothesis.rank,
                    "status": "completed",
                    "verdict": "inconclusive",
                    "confidence": 0.25,
                    "evidence_summary": "Bootstrap run without experiment worker execution.",
                    "key_finding": None,
                },
            )
        await session.commit()


async def run_research_loop(
    *,
    session_id: str,
    question: str,
    max_hypotheses: int,
    paper_ttl_days: int,
    emitter: EventEmitter,
    session_factory: async_sessionmaker,
) -> None:
    llm = LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        emitter=emitter,
        session_factory=session_factory,
    )

    try:
        await _update_session(session_factory, session_id, stage="intake")
        parsed = await parse_question(question)
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.INTAKE_PARSED,
            step="intake.parse",
            payload={"domain": parsed.domain, "sub_questions": parsed.sub_questions},
        )

        await _update_session(session_factory, session_id, stage="literature")
        prior = await build_prior(
            parsed,
            session_id=session_id,
            emitter=emitter,
            session_factory=session_factory,
            paper_ttl_days=paper_ttl_days,
        )
        await _update_session(session_factory, session_id, prior_json=json.dumps(prior.to_dict()))
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.LIT_PRIOR_BUILT,
            step="literature.prior_build",
            payload={"prior": prior.to_dict()},
        )

        await _update_session(session_factory, session_id, stage="hypothesis")
        hypotheses = await generate_ranked_hypotheses(
            parsed,
            prior,
            max_hypotheses=max_hypotheses,
            llm=llm,
            session_id=session_id,
        )
        await _insert_hypothesis_rows(session_factory, session_id, hypotheses)
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.HYPO_RANKED,
            step="hypothesis.rank",
            payload={"hypotheses": [h.to_dict() for h in hypotheses]},
        )

        await _update_session(session_factory, session_id, stage="synthesis")
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.SYNTH_LEDGER_UPDATED,
            step="synthesis.ledger",
            payload={"final_ledger": {"confirmed": [], "refuted": [], "inconclusive": [h.id for h in hypotheses]}},
        )

        await _update_session(session_factory, session_id, status="completed", stage="completed")
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.SESSION_COMPLETED,
            step="session.complete",
            payload={"breakthroughs": [], "dead_ends": [], "note": "Orchestrator foundation stages completed."},
        )
    except Exception as exc:
        await _update_session(session_factory, session_id, status="failed", stage="failed")
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.SESSION_FAILED,
            step="session.failed",
            payload={"error": str(exc)},
        )
        raise
