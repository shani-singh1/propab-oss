from __future__ import annotations

import asyncio
import json
import time
from uuid import UUID, uuid5
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.config import settings
from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.paper_gate import session_merits_paper, short_circuit_merits_paper
from propab.types import EventType
from services.orchestrator.answer_gate import evaluate_literature_short_circuit
from services.orchestrator.hypotheses import generate_ranked_hypotheses
from services.orchestrator.schemas import RankedHypothesis
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
        api_key=settings.llm_api_secret,
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
        if len(parsed.sub_questions) > 1:
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.INTAKE_DECOMPOSED,
                step="intake.decompose",
                payload={"sub_questions": parsed.sub_questions},
            )

        await _update_session(session_factory, session_id, stage="literature")
        prior = await build_prior(
            parsed,
            session_id=session_id,
            emitter=emitter,
            session_factory=session_factory,
            paper_ttl_days=paper_ttl_days,
            llm=llm,
        )
        await _update_session(session_factory, session_id, prior_json=json.dumps(prior.to_dict()))
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.LIT_PRIOR_BUILT,
            step="literature.prior_build",
            payload={"prior": prior.to_dict()},
        )

        short_answer = await evaluate_literature_short_circuit(
            question=parsed.text,
            prior=prior,
            session_id=session_id,
            emitter=emitter,
        )
        if short_answer is not None:
            sc_merits, sc_reason = short_circuit_merits_paper()
            await _update_session(session_factory, session_id, stage="paper")
            paper_payload: dict | None = None
            if sc_merits:
                paper_payload = await write_paper_minimal(
                    session_id=session_id,
                    session_factory=session_factory,
                    emitter=emitter,
                    llm=llm,
                    question=parsed.text,
                    prior=prior.to_dict(),
                    synthesis={"short_circuit": True, "short_answer": short_answer},
                )
            else:
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.PAPER_SKIPPED,
                    step="paper.skipped",
                    payload={
                        "reason": sc_reason,
                        "literature_answer": short_answer,
                        "note": "No experiment trace; paper suppressed (set PAPER_POLICY=always to allow literature-only PDF).",
                    },
                )
            await _update_session(session_factory, session_id, status="completed", stage="completed")
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.SESSION_COMPLETED,
                step="session.complete",
                payload={
                    "paper": paper_payload,
                    "breakthroughs": [],
                    "dead_ends": [],
                    "existing_answer": short_answer,
                    "paper_skipped_reason": None if paper_payload else sc_reason,
                    "note": (
                        "Literature short-circuit: prior matched an established answer."
                        if paper_payload
                        else "Literature short-circuit: answer returned without experiment paper."
                    ),
                },
            )
            return

        await _update_session(session_factory, session_id, stage="hypothesis")
        hypotheses = await generate_ranked_hypotheses(
            parsed,
            prior,
            max_hypotheses=max_hypotheses,
            llm=llm,
            session_id=session_id,
            emitter=emitter,
        )
        hypothesis_rows = await _insert_hypothesis_rows(session_factory, session_id, hypotheses)
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.HYPO_RANKED,
            step="hypothesis.rank",
            payload={"hypotheses": [h.to_dict() for h in hypotheses]},
        )

        await _update_session(session_factory, session_id, stage="experiment")
        baseline_metric = float(len(prior.established_facts or []))
        baseline = {
            "description": "Session baseline from prior established facts count before hypothesis dispatch.",
            "metric_name": "established_facts_count",
            "metric_value": baseline_metric,
            "how_obtained": "literature",
        }
        pending: list[dict] = []
        for row_id, hypothesis in hypothesis_rows:
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.HYPO_DISPATCHED,
                step="hypothesis.dispatch",
                payload={"hypothesis_id": row_id, "rank": hypothesis.rank},
                hypothesis_id=row_id,
            )
            ar = run_sub_agent_task.delay(
                {
                    "session_id": session_id,
                    "hypothesis_id": row_id,
                    "hypothesis": hypothesis.to_dict(),
                    "baseline": baseline,
                    "prior": prior.to_dict(),
                    "domain": parsed.domain,
                }
            )
            pending.append({"ar": ar, "hid": row_id})

        experiment_results: list[dict] = []
        ledger: dict[str, list[str]] = {"confirmed": [], "refuted": [], "inconclusive": []}
        deadline = time.monotonic() + 600.0

        while pending and time.monotonic() < deadline:
            progressed = False
            for item in list(pending):
                ar = item["ar"]
                hid = item["hid"]
                ready = await asyncio.to_thread(lambda: ar.ready())
                if not ready:
                    continue
                progressed = True
                pending.remove(item)
                try:
                    result = await asyncio.to_thread(lambda: ar.get(timeout=30))
                except Exception as exc:
                    result = {
                        "hypothesis_id": hid,
                        "verdict": "inconclusive",
                        "confidence": 0.0,
                        "evidence_summary": f"Sub-agent task failed: {exc}",
                        "key_finding": None,
                        "tool_trace_id": None,
                        "figures": [],
                        "duration_sec": 0.0,
                        "failure_reason": str(exc),
                    }
                experiment_results.append(result)
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.SYNTH_RESULT_RECEIVED,
                    step="synthesis.result",
                    payload={"result": result},
                    hypothesis_id=result.get("hypothesis_id"),
                )
                rhid = result.get("hypothesis_id")
                if rhid and result.get("verdict") == "confirmed":
                    ledger["confirmed"].append(str(rhid))
                elif rhid and result.get("verdict") == "refuted":
                    ledger["refuted"].append(str(rhid))
                elif rhid:
                    ledger["inconclusive"].append(str(rhid))

                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.SYNTH_LEDGER_UPDATED,
                    step="synthesis.ledger",
                    payload={"ledger": dict(ledger), "last_result_hypothesis_id": rhid},
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

            if not progressed and pending:
                await asyncio.sleep(0.12)

        for item in pending:
            hid = item["hid"]
            result = {
                "hypothesis_id": hid,
                "verdict": "inconclusive",
                "confidence": 0.0,
                "evidence_summary": "Sub-agent did not finish within the orchestrator deadline.",
                "key_finding": None,
                "tool_trace_id": None,
                "figures": [],
                "duration_sec": 0.0,
                "failure_reason": "timeout",
            }
            experiment_results.append(result)
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.SYNTH_RESULT_RECEIVED,
                step="synthesis.result",
                payload={"result": result, "note": "deadline"},
                hypothesis_id=hid,
            )
            ledger["inconclusive"].append(str(hid))
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.SYNTH_LEDGER_UPDATED,
                step="synthesis.ledger",
                payload={"ledger": dict(ledger), "last_result_hypothesis_id": hid},
            )

        await _update_session(session_factory, session_id, stage="synthesis")
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.SYNTH_LEDGER_UPDATED,
            step="synthesis.ledger",
            payload={"final_ledger": dict(ledger)},
        )

        await _update_session(session_factory, session_id, stage="paper")
        synthesis_payload = {
            "ledger": ledger,
            "hypotheses": [h.to_dict() for _, h in hypothesis_rows],
            "experiment_results": [
                {
                    "hypothesis_id": r.get("hypothesis_id"),
                    "verdict": r.get("verdict"),
                    "confidence": r.get("confidence"),
                    "key_finding": (r.get("key_finding") or "")[:800],
                    "evidence_summary": (r.get("evidence_summary") or "")[:1200],
                }
                for r in experiment_results
            ],
        }
        merits, merit_reason = await session_merits_paper(
            session_factory, session_id, ledger=ledger
        )
        paper_payload: dict | None = None
        if merits:
            paper_payload = await write_paper_minimal(
                session_id=session_id,
                session_factory=session_factory,
                emitter=emitter,
                llm=llm,
                question=parsed.text,
                prior=prior.to_dict(),
                synthesis=synthesis_payload,
            )
        else:
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.PAPER_SKIPPED,
                step="paper.skipped",
                payload={
                    "reason": merit_reason,
                    "ledger": dict(ledger),
                    "note": "Paper suppressed until there is confirmed/refuted hypotheses or substantive metric-like tool evidence. Inspect GET /sessions/{id}/trace.",
                },
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
                "paper_skipped_reason": None if paper_payload else merit_reason,
                "note": (
                    "Session completed with paper payload."
                    if paper_payload
                    else "Session completed without paper; experiments did not meet substantive bar."
                ),
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
