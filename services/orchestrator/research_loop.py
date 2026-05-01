from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from uuid import UUID, uuid4, uuid5
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.config import settings
from propab.db import create_redis
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
from services.orchestrator.accumulated_ledger import AccumulatedLedger
from services.orchestrator.budget import ResearchBudget
from services.worker.peer_findings import build_peer_finding_payload, publish_peer_finding
from services.worker.tasks import run_sub_agent_task

logger = logging.getLogger(__name__)

_HYPOTHESIS_ID_NAMESPACE = UUID("8b4fd0f5-6c2a-40e2-a4da-4d8c1f2e0b1c")


def _hypothesis_row_id(session_id: str, logical_id: str, round_id: str = "") -> str:
    # Include round_id so same logical_id in different rounds gets a unique DB key
    key = f"{session_id}:{round_id}:{logical_id}" if round_id else f"{session_id}:{logical_id}"
    return str(uuid5(_HYPOTHESIS_ID_NAMESPACE, key))


async def _update_session(session_factory: async_sessionmaker, session_id: str, **fields: str) -> None:
    assignments = ", ".join(f"{name} = :{name}" for name in fields)
    values = {"id": session_id, **fields}
    query = text(f"UPDATE research_sessions SET {assignments} WHERE id = :id")
    async with session_factory() as session:
        await session.execute(query, values)
        await session.commit()


async def _insert_round(
    session_factory: async_sessionmaker,
    *,
    round_id: str,
    session_id: str,
    round_number: int,
    budget_json: str,
) -> None:
    async with session_factory() as session:
        await session.execute(
            text("""
                INSERT INTO research_rounds
                    (id, session_id, round_number, status, budget_json)
                VALUES
                    (:id, :session_id, :round_number, 'running', CAST(:budget_json AS jsonb))
                ON CONFLICT (session_id, round_number) DO NOTHING
            """),
            {"id": round_id, "session_id": session_id,
             "round_number": round_number, "budget_json": budget_json},
        )
        await session.commit()


async def _complete_round(
    session_factory: async_sessionmaker,
    *,
    round_id: str,
    confirmed: int,
    refuted: int,
    inconclusive: int,
    marginal_return: float | None,
) -> None:
    async with session_factory() as session:
        await session.execute(
            text("""
                UPDATE research_rounds
                SET status='completed', confirmed_count=:c, refuted_count=:r,
                    inconclusive_count=:i, marginal_return=:mr, completed_at=NOW()
                WHERE id = :id
            """),
            {"id": round_id, "c": confirmed, "r": refuted, "i": inconclusive,
             "mr": marginal_return},
        )
        await session.commit()


async def _save_checkpoint(
    session_factory: async_sessionmaker,
    *,
    session_id: str,
    round_number: int,
    ledger: AccumulatedLedger,
    budget: ResearchBudget,
) -> None:
    async with session_factory() as session:
        await session.execute(
            text("""
                INSERT INTO session_checkpoints
                    (id, session_id, round_number, ledger_json, budget_json)
                VALUES
                    (:id, :session_id, :round_number,
                     CAST(:ledger_json AS jsonb), CAST(:budget_json AS jsonb))
                ON CONFLICT (session_id, round_number)
                DO UPDATE SET ledger_json = EXCLUDED.ledger_json,
                              budget_json = EXCLUDED.budget_json
            """),
            {
                "id": str(uuid4()),
                "session_id": session_id,
                "round_number": round_number,
                "ledger_json": json.dumps(ledger.to_dict()),
                "budget_json": json.dumps(budget.to_dict()),
            },
        )
        await session.commit()


async def _insert_hypothesis_rows(
    session_factory: async_sessionmaker,
    session_id: str,
    hypotheses: list[RankedHypothesis],
    round_id: str | None = None,
) -> list[tuple[str, RankedHypothesis]]:
    inserted: list[tuple[str, RankedHypothesis]] = []
    async with session_factory() as session:
        for hypothesis in hypotheses:
            row_id = _hypothesis_row_id(session_id, hypothesis.id, round_id or "")
            await session.execute(
                text("""
                    INSERT INTO hypotheses (
                        id, session_id, round_id, text, test_methodology, scores_json,
                        rank, status, verdict, confidence, evidence_summary, key_finding
                    )
                    VALUES (
                        :id, :session_id, CAST(:round_id AS uuid),
                        :text, :test_methodology,
                        CAST(:scores_json AS jsonb),
                        :rank, :status, :verdict, :confidence, :evidence_summary, :key_finding
                    )
                """),
                {
                    "id": row_id,
                    "session_id": session_id,
                    "round_id": round_id,
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


async def _run_round(
    *,
    session_id: str,
    round_id: str,
    round_number: int,
    hypothesis_rows: list[tuple[str, RankedHypothesis]],
    prior: any,
    parsed_question: any,
    baseline: dict,
    budget: ResearchBudget,
    emitter: EventEmitter,
    session_factory: async_sessionmaker,
) -> list[dict]:
    """
    Dispatch all hypotheses for one round and collect results.
    - Results collected as sub-agents finish (not all-at-once).
    - Each completed result is broadcast as a peer finding to still-running agents.
    - Round deadline enforced.
    """
    all_hypothesis_ids = [row_id for row_id, _ in hypothesis_rows]
    pending: list[dict] = []

    for row_id, hypothesis in hypothesis_rows:
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.HYPO_DISPATCHED,
            step="hypothesis.dispatch",
            payload={"hypothesis_id": row_id, "rank": hypothesis.rank, "round": round_number},
            hypothesis_id=row_id,
        )
        ar = run_sub_agent_task.delay(
            {
                "session_id": session_id,
                "hypothesis_id": row_id,
                "hypothesis": hypothesis.to_dict(),
                "baseline": baseline,
                "prior": prior.to_dict(),
                "domain": parsed_question.domain,
                "question": parsed_question.text,
            }
        )
        pending.append({"ar": ar, "hid": row_id})

    round_budget = budget.round_budget(round_number)
    deadline = time.monotonic() + round_budget.max_seconds
    results: list[dict] = []
    completed_ids: set[str] = set()

    # Open a short-lived Redis connection for peer broadcasts
    _redis = None
    try:
        _redis = await create_redis(settings.redis_url)
    except Exception as exc:
        logger.warning("Could not open Redis for peer broadcast: %s", exc)

    while pending and time.monotonic() < deadline:
        progressed = False
        for item in list(pending):
            ar, hid = item["ar"], item["hid"]
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
                    "learned": None,
                }
            results.append(result)
            completed_ids.add(str(result.get("hypothesis_id", "")))

            await emitter.emit(
                session_id=session_id,
                event_type=EventType.SYNTH_RESULT_RECEIVED,
                step="synthesis.result",
                payload={"result": result, "round": round_number},
                hypothesis_id=result.get("hypothesis_id"),
            )

            # Broadcast to still-running agents
            still_running = [hid for hid in all_hypothesis_ids if hid not in completed_ids]
            if still_running and _redis is not None:
                finding_payload = build_peer_finding_payload(result)
                n_broadcast = await publish_peer_finding(
                    _redis, target_hypothesis_ids=still_running, finding=finding_payload,
                )
                if n_broadcast > 0:
                    await emitter.emit(
                        session_id=session_id,
                        event_type=EventType.MEMORY_LEDGER_BROADCAST,
                        step="memory.broadcast",
                        payload={
                            "from_hypothesis": result.get("hypothesis_id"),
                            "broadcast_to": still_running,
                            "verdict": result.get("verdict"),
                        },
                    )

        if not progressed and pending:
            await asyncio.sleep(0.12)

    if _redis is not None:
        try:
            await _redis.close()
        except Exception:
            pass

    # Handle timed-out pending tasks
    for item in pending:
        hid = item["hid"]
        result = {
            "hypothesis_id": hid,
            "verdict": "inconclusive",
            "confidence": 0.0,
            "evidence_summary": "Sub-agent did not finish within the round deadline.",
            "key_finding": None,
            "tool_trace_id": None,
            "figures": [],
            "duration_sec": 0.0,
            "failure_reason": "round_timeout",
            "learned": None,
        }
        results.append(result)
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.SYNTH_RESULT_RECEIVED,
            step="synthesis.result",
            payload={"result": result, "round": round_number, "note": "deadline"},
            hypothesis_id=hid,
        )

    return results


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
        # ── Stage 1: Intake ──────────────────────────────────────────
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

        # ── Stage 2: Literature ──────────────────────────────────────
        await _update_session(session_factory, session_id, stage="literature")
        prior = await build_prior(
            parsed, session_id=session_id, emitter=emitter,
            session_factory=session_factory, paper_ttl_days=paper_ttl_days, llm=llm,
        )
        await _update_session(session_factory, session_id, prior_json=json.dumps(prior.to_dict()))
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.LIT_PRIOR_BUILT,
            step="literature.prior_build",
            payload={"prior": prior.to_dict()},
        )

        # ── Stage 3: Short-circuit check ────────────────────────────
        short_answer = await evaluate_literature_short_circuit(
            question=parsed.text, prior=prior, session_id=session_id, emitter=emitter,
        )
        if short_answer is not None:
            sc_merits, sc_reason = short_circuit_merits_paper()
            await _update_session(session_factory, session_id, stage="paper")
            paper_payload: dict | None = None
            if sc_merits:
                paper_payload = await write_paper_minimal(
                    session_id=session_id, session_factory=session_factory, emitter=emitter,
                    llm=llm, question=parsed.text, prior=prior.to_dict(),
                    synthesis={"short_circuit": True, "short_answer": short_answer},
                )
            else:
                await emitter.emit(
                    session_id=session_id, event_type=EventType.PAPER_SKIPPED,
                    step="paper.skipped",
                    payload={"reason": sc_reason, "literature_answer": short_answer},
                )
            await _update_session(session_factory, session_id, status="completed", stage="completed")
            await emitter.emit(
                session_id=session_id, event_type=EventType.SESSION_COMPLETED,
                step="session.complete",
                payload={
                    "paper": paper_payload,
                    "breakthroughs": [],
                    "dead_ends": [],
                    "existing_answer": short_answer,
                    "paper_skipped_reason": None if paper_payload else sc_reason,
                },
            )
            return

        # ── Stage 4: Multi-Round Research Loop ───────────────────────
        await _update_session(session_factory, session_id, stage="experiment")

        budget = ResearchBudget.from_settings(settings)
        accumulated_ledger = AccumulatedLedger()
        baseline_metric = float(len(prior.established_facts or []))
        baseline = {
            "description": "Session baseline from prior established facts count.",
            "metric_name": "established_facts_count",
            "metric_value": baseline_metric,
            "how_obtained": "literature",
        }

        round_number = 0
        prior_round_findings = ""

        while not budget.exhausted():
            round_id = str(uuid4())
            await _insert_round(
                session_factory,
                round_id=round_id,
                session_id=session_id,
                round_number=round_number,
                budget_json=json.dumps(budget.summary()),
            )

            await emitter.emit(
                session_id=session_id,
                event_type=EventType.ROUND_STARTED,
                step=f"round.{round_number}.start",
                payload={
                    "round": round_number,
                    "round_id": round_id,
                    "budget": budget.summary(),
                    "prior_round_findings_summary": prior_round_findings[:500],
                },
            )

            # Generate hypotheses for this round (informed by prior rounds)
            hypotheses = await generate_ranked_hypotheses(
                parsed, prior,
                max_hypotheses=max_hypotheses,
                llm=llm,
                session_id=session_id,
                emitter=emitter,
                prior_round_findings=prior_round_findings,
            )
            hypothesis_rows = await _insert_hypothesis_rows(
                session_factory, session_id, hypotheses, round_id=round_id,
            )
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.HYPO_RANKED,
                step="hypothesis.rank",
                payload={
                    "hypotheses": [h.to_dict() for h in hypotheses],
                    "round": round_number,
                },
            )

            # Run experiments for this round
            round_results = await _run_round(
                session_id=session_id,
                round_id=round_id,
                round_number=round_number,
                hypothesis_rows=hypothesis_rows,
                prior=prior,
                parsed_question=parsed,
                baseline=baseline,
                budget=budget,
                emitter=emitter,
                session_factory=session_factory,
            )

            # Merge into accumulated ledger
            round_summary = accumulated_ledger.merge_round(
                round_number=round_number,
                round_id=round_id,
                round_results=round_results,
                hypothesis_texts=[h.text for _, h in hypothesis_rows],
            )

            # Update budget with round results
            budget.record_round(
                confirmed=len(round_summary.confirmed),
                refuted=len(round_summary.refuted),
                inconclusive=len(round_summary.inconclusive),
                n_hypotheses=len(round_results),
            )

            # Compute marginal return for diminishing-returns detection
            marginal = accumulated_ledger.marginal_return(round_number) if round_number > 0 else 1.0

            await _complete_round(
                session_factory,
                round_id=round_id,
                confirmed=len(round_summary.confirmed),
                refuted=len(round_summary.refuted),
                inconclusive=len(round_summary.inconclusive),
                marginal_return=marginal,
            )

            await emitter.emit(
                session_id=session_id,
                event_type=EventType.SYNTH_LEDGER_UPDATED,
                step="synthesis.ledger",
                payload={
                    "ledger": accumulated_ledger.summary(),
                    "round": round_number,
                    "round_id": round_id,
                },
            )

            # Breakthrough / dead end events
            for hid in round_summary.confirmed:
                r = accumulated_ledger.results.get(hid, {})
                if float(r.get("confidence") or 0) > 0.85:
                    await emitter.emit(
                        session_id=session_id, event_type=EventType.SYNTH_BREAKTHROUGH,
                        step="synthesis.breakthrough",
                        payload={"finding": r.get("key_finding"), "round": round_number},
                        hypothesis_id=hid,
                    )
            for hid in round_summary.refuted:
                await emitter.emit(
                    session_id=session_id, event_type=EventType.SYNTH_DEAD_END,
                    step="synthesis.dead_end",
                    payload={"hypothesis": hid, "round": round_number},
                    hypothesis_id=hid,
                )

            # Warn if all results inconclusive
            if round_summary.all_inconclusive():
                await emitter.emit(
                    session_id=session_id, event_type=EventType.SYNTH_ALL_INCONCLUSIVE,
                    step="synthesis.all_inconclusive",
                    payload={
                        "round": round_number,
                        "note": (
                            "All hypotheses inconclusive. Check significance gate logs "
                            "and tool selection for this round."
                        ),
                    },
                )

            await emitter.emit(
                session_id=session_id,
                event_type=EventType.ROUND_COMPLETED,
                step=f"round.{round_number}.complete",
                payload={
                    "round": round_number,
                    "round_id": round_id,
                    "confirmed": len(round_summary.confirmed),
                    "refuted": len(round_summary.refuted),
                    "inconclusive": len(round_summary.inconclusive),
                    "marginal_return": marginal,
                    "budget": budget.summary(),
                },
            )

            # Checkpoint for resumability
            await _save_checkpoint(
                session_factory,
                session_id=session_id,
                round_number=round_number,
                ledger=accumulated_ledger,
                budget=budget,
            )

            # Check diminishing returns (soft stop, only after round 1)
            if round_number >= 1 and marginal < budget.min_marginal_return:
                await emitter.emit(
                    session_id=session_id, event_type=EventType.PROGRESS_DIMINISHING,
                    step="progress.diminishing",
                    payload={
                        "round": round_number,
                        "marginal_return": marginal,
                        "threshold": budget.min_marginal_return,
                    },
                )
                logger.info(
                    "Stopping after round %d: marginal return %.3f < threshold %.3f",
                    round_number, marginal, budget.min_marginal_return,
                )
                break

            # Build prior-round context for next round's hypothesis generator
            prior_round_findings = accumulated_ledger.summary_for_hypothesis_generator()

            round_number += 1

        # ── Stage 5: Paper ────────────────────────────────────────────
        await _update_session(session_factory, session_id, stage="paper")

        # Build synthesis payload from accumulated multi-round ledger
        all_hypothesis_texts = [
            h.text for _, h in []  # will be populated from DB in write_paper_minimal
        ]
        synthesis_payload = {
            "ledger": accumulated_ledger.summary(),
            "total_rounds": round_number + 1,
            "experiment_results": [
                {
                    "hypothesis_id": r.get("hypothesis_id"),
                    "verdict": r.get("verdict"),
                    "confidence": r.get("confidence"),
                    "key_finding": (r.get("key_finding") or "")[:800],
                    "evidence_summary": (r.get("evidence_summary") or "")[:1200],
                }
                for r in accumulated_ledger.results.values()
            ],
        }

        # Use the full multi-round ledger for the paper gate check
        full_ledger_for_gate = {
            "confirmed": accumulated_ledger.confirmed,
            "refuted": accumulated_ledger.refuted,
            "inconclusive": accumulated_ledger.inconclusive,
        }
        merits, merit_reason = await session_merits_paper(
            session_factory, session_id, ledger=full_ledger_for_gate,
        )
        paper_payload = None
        if merits:
            paper_payload = await write_paper_minimal(
                session_id=session_id, session_factory=session_factory,
                emitter=emitter, llm=llm, question=parsed.text,
                prior=prior.to_dict(), synthesis=synthesis_payload,
            )
        else:
            await emitter.emit(
                session_id=session_id, event_type=EventType.PAPER_SKIPPED,
                step="paper.skipped",
                payload={
                    "reason": merit_reason,
                    "ledger": accumulated_ledger.summary(),
                    "note": (
                        "Paper suppressed: experiments did not meet substantive bar after "
                        f"{round_number + 1} round(s). Inspect GET /sessions/{{id}}/trace."
                    ),
                },
            )

        stop_reason = budget.stop_reason() or "research loop completed"
        await _update_session(session_factory, session_id, status="completed", stage="completed")
        await emitter.emit(
            session_id=session_id, event_type=EventType.SESSION_COMPLETED,
            step="session.complete",
            payload={
                "paper": paper_payload,
                "total_rounds": round_number + 1,
                "breakthroughs": accumulated_ledger.confirmed,
                "dead_ends": accumulated_ledger.refuted,
                "paper_skipped_reason": None if paper_payload else merit_reason,
                "stop_reason": stop_reason,
                "budget": budget.summary(),
            },
        )

    except Exception as exc:
        last_events: list[dict] = []
        try:
            async with session_factory() as session:
                rows = (
                    await session.execute(
                        text("""
                            SELECT event_type, step, payload_json, created_at
                            FROM events
                            WHERE session_id = CAST(:sid AS uuid)
                            ORDER BY created_at DESC LIMIT 20
                        """),
                        {"sid": session_id},
                    )
                ).mappings().all()
            last_events = [
                {
                    "event_type": str(r.get("event_type") or ""),
                    "step": str(r.get("step") or ""),
                    "payload": r.get("payload_json"),
                    "created_at": str(r.get("created_at") or ""),
                }
                for r in rows
            ]
        except Exception:
            last_events = []

        await _update_session(session_factory, session_id, status="failed", stage="failed")
        await emitter.emit(
            session_id=session_id, event_type=EventType.SESSION_FAILED,
            step="session.failed",
            payload={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "stage": "failed",
                "last_events": last_events,
            },
        )
        raise
