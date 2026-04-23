from __future__ import annotations

import asyncio
import json
from uuid import UUID, uuid5

from celery import group
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.config import settings
from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.types import EventType
from services.orchestrator.hypotheses import RankedHypothesis, generate_ranked_hypotheses
from services.orchestrator.intake import parse_question
from services.orchestrator.literature import build_prior
from services.orchestrator.paper import write_paper_minimal
from services.worker.tasks import run_sub_agent_task

_HYPOTHESIS_ID_NAMESPACE = UUID("8b4fd0f5-6c2a-40e2-a4da-4d8c1f2e0b1c")


def _hypothesis_row_id(session_id: str, logical_id: str) -> str:
    return str(uuid5(_HYPOTHESIS_ID_NAMESPACE, f"{session_id}:{logical_id}"))


async def _update_session(session_factory: async_sessionmaker, session_id: str, **fields: str) -> None:
    assignments = ", ".join(f"{name} = :{name}" for name in fields)
    values = {"id": session_id, **fields}
    query = text(f"UPDATE research_sessions SET {assignments} WHERE id = :id")
    async with session_factory() as session:
        await session.execute(query, values)
        await session.commit()


async def _insert_hypothesis_rows(
    session_factory: async_sessionmaker,
    session_id: str,
    hypotheses: list[RankedHypothesis],
) -> list[tuple[str, RankedHypothesis]]:
    inserted: list[tuple[str, RankedHypothesis]] = []
    async with session_factory() as session:
        for hypothesis in hypotheses:
            row_id = _hypothesis_row_id(session_id, hypothesis.id)
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
                    "id": row_id,
                    "session_id": session_id,
                    "text": hypothesis.text,
                    "test_methodology": hypothesis.test_methodology,
                    "scores_json": json.dumps(hypothesis.scores),
                    "rank": hypothesis.rank,
                    "status": "pending",
                    "verdict": None,
                    "confidence": None,
                    "evidence_summary": None,
                    "key_finding": None,
                },
            )
            inserted.append((row_id, hypothesis))
        await session.commit()
    return inserted


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
        hypothesis_rows = await _insert_hypothesis_rows(session_factory, session_id, hypotheses)
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.HYPO_RANKED,
            step="hypothesis.rank",
            payload={"hypotheses": [h.to_dict() for h in hypotheses]},
        )

        await _update_session(session_factory, session_id, stage="experiment")
        signatures = []
        for row_id, hypothesis in hypothesis_rows:
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.HYPO_DISPATCHED,
                step="hypothesis.dispatch",
                payload={"hypothesis_id": row_id, "rank": hypothesis.rank},
                hypothesis_id=row_id,
            )
            signatures.append(
                run_sub_agent_task.s(
                    {
                        "session_id": session_id,
                        "hypothesis_id": row_id,
                        "hypothesis": hypothesis.to_dict(),
                        "prior": prior.to_dict(),
                        "domain": parsed.domain,
                    }
                )
            )

        experiment_results: list[dict] = []
        if signatures:

            def _run_group() -> list[dict]:
                return group(signatures).apply_async().get(timeout=600)

            experiment_results = await asyncio.to_thread(_run_group)

        for result in experiment_results:
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.SYNTH_RESULT_RECEIVED,
                step="synthesis.result",
                payload={"result": result},
                hypothesis_id=result.get("hypothesis_id"),
            )
            if result.get("verdict") == "confirmed" and float(result.get("confidence") or 0) > 0.85:
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.SYNTH_BREAKTHROUGH,
                    step="synthesis.breakthrough",
                    payload={"finding": result.get("key_finding")},
                    hypothesis_id=result.get("hypothesis_id"),
                )
            if result.get("verdict") == "refuted":
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.SYNTH_DEAD_END,
                    step="synthesis.dead_end",
                    payload={"hypothesis": result.get("hypothesis_id")},
                    hypothesis_id=result.get("hypothesis_id"),
                )

        ledger = {
            "confirmed": [r["hypothesis_id"] for r in experiment_results if r.get("verdict") == "confirmed"],
            "refuted": [r["hypothesis_id"] for r in experiment_results if r.get("verdict") == "refuted"],
            "inconclusive": [r["hypothesis_id"] for r in experiment_results if r.get("verdict") == "inconclusive"],
        }

        await _update_session(session_factory, session_id, stage="synthesis")
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.SYNTH_LEDGER_UPDATED,
            step="synthesis.ledger",
            payload={"final_ledger": ledger},
        )

        await _update_session(session_factory, session_id, stage="paper")
        paper_payload = await write_paper_minimal(
            session_id=session_id,
            session_factory=session_factory,
            emitter=emitter,
        )

        await _update_session(session_factory, session_id, status="completed", stage="completed")
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.SESSION_COMPLETED,
            step="session.complete",
            payload={
                "paper": paper_payload,
                "breakthroughs": ledger["confirmed"],
                "dead_ends": ledger["refuted"],
                "note": "Session completed with deterministic methods compilation and paper-ready payload.",
            },
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
