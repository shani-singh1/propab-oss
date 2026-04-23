from __future__ import annotations

import json
import time
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.config import settings
from propab.db import create_engine, create_redis, create_session_factory
from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.tools.registry import ToolRegistry
from propab.types import EventType
from services.worker.domain_router import route_domain


async def _update_hypothesis(
    session_factory: async_sessionmaker,
    hypothesis_id: str,
    *,
    status: str,
    verdict: str | None = None,
    confidence: float | None = None,
    evidence_summary: str | None = None,
    key_finding: str | None = None,
) -> None:
    fields: list[str] = ["status = :status"]
    params: dict = {"id": hypothesis_id, "status": status}
    if verdict is not None:
        fields.append("verdict = :verdict")
        params["verdict"] = verdict
    if confidence is not None:
        fields.append("confidence = :confidence")
        params["confidence"] = confidence
    if evidence_summary is not None:
        fields.append("evidence_summary = :evidence_summary")
        params["evidence_summary"] = evidence_summary
    if key_finding is not None:
        fields.append("key_finding = :key_finding")
        params["key_finding"] = key_finding
    query = text(f"UPDATE hypotheses SET {', '.join(fields)} WHERE id = :id")
    async with session_factory() as session:
        await session.execute(query, params)
        await session.commit()


async def run_sub_agent_async(payload: dict) -> dict:
    """
    Execute a minimal sub-agent trace: plan, one tool call, verdict.
    Persists experiment_steps and tool_calls; emits full event stream.
    """
    session_id: str = payload["session_id"]
    hypothesis_id: str = payload["hypothesis_id"]
    hypothesis: dict = payload["hypothesis"]

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    redis = await create_redis(settings.redis_url)
    emitter = EventEmitter(source="worker", redis=redis, session_factory=session_factory)
    llm = LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        emitter=emitter,
        session_factory=session_factory,
    )
    registry = ToolRegistry()
    trace_pointer = str(uuid4())
    started = time.perf_counter()
    result = None

    try:
        await _update_hypothesis(session_factory, hypothesis_id, status="running")

        await emitter.emit(
            session_id=session_id,
            event_type=EventType.AGENT_STARTED,
            step=f"experiment.{hypothesis_id}",
            payload={"hypothesis_id": hypothesis_id, "text": hypothesis.get("text")},
            hypothesis_id=hypothesis_id,
        )

        domain, domain_reason = await route_domain(
            hypothesis_text=str(hypothesis.get("text", "")),
            llm=llm,
            session_id=session_id,
            hypothesis_id=hypothesis_id,
        )

        await emitter.emit(
            session_id=session_id,
            event_type=EventType.TOOL_SELECTED,
            step=f"experiment.{hypothesis_id}.domain",
            payload={"domain": domain, "reason": domain_reason},
            hypothesis_id=hypothesis_id,
        )

        specs = registry.get_cluster(domain)
        if not specs:
            specs = registry.get_cluster("general_computation")
        first_spec = specs[0] if specs else None
        if first_spec:
            tool_name = first_spec["name"]
            example = first_spec.get("example") or {}
            params = example.get("params")
            if not params:
                tool_name = "json_extract"
                params = {"data": {"hypothesis_rank": hypothesis.get("rank"), "label": "probe"}, "key": "label"}
        else:
            tool_name = "json_extract"
            params = {"data": {"hypothesis_rank": hypothesis.get("rank"), "label": "probe"}, "key": "label"}

        plan = {"steps": [{"type": "tool", "tool": tool_name, "params": params}]}
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.AGENT_PLAN_CREATED,
            step=f"experiment.{hypothesis_id}.plan",
            payload={"plan": plan, "domain": domain},
            hypothesis_id=hypothesis_id,
        )

        step_index = 0
        for step in plan["steps"]:
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.AGENT_STEP_STARTED,
                step=f"experiment.{hypothesis_id}.step_{step_index}",
                payload={"step": step},
                hypothesis_id=hypothesis_id,
            )

            step_started = time.perf_counter()
            step_id = str(uuid4())
            tool_name = step["tool"]
            params = step["params"]

            await emitter.emit(
                session_id=session_id,
                event_type=EventType.TOOL_CALLED,
                step=f"experiment.{hypothesis_id}.step_{step_index}",
                payload={"tool": tool_name, "params": params},
                hypothesis_id=hypothesis_id,
            )

            result = registry.call(tool_name, params)
            duration_ms = int((time.perf_counter() - step_started) * 1000)

            if result.success:
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.TOOL_RESULT,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={"tool": tool_name, "output": result.output},
                    hypothesis_id=hypothesis_id,
                )
            else:
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.TOOL_ERROR,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={"tool": tool_name, "error": result.error.to_dict() if result.error else {}},
                    hypothesis_id=hypothesis_id,
                )

            async with session_factory() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO experiment_steps (
                            id, hypothesis_id, step_type, step_index, input_json, output_json, error_json, duration_ms
                        )
                        VALUES (
                            :id, :hypothesis_id, 'tool_call', :step_index,
                            CAST(:input_json AS jsonb),
                            CAST(:output_json AS jsonb),
                            CAST(:error_json AS jsonb),
                            :duration_ms
                        )
                        """
                    ),
                    {
                        "id": step_id,
                        "hypothesis_id": hypothesis_id,
                        "step_index": step_index,
                        "input_json": json.dumps({"tool": tool_name, "params": params}),
                        "output_json": json.dumps(result.output) if result.output else "null",
                        "error_json": json.dumps(result.error.to_dict()) if result.error else "null",
                        "duration_ms": duration_ms,
                    },
                )
                await session.execute(
                    text(
                        """
                        INSERT INTO tool_calls (
                            id, step_id, hypothesis_id, tool_name, domain, params_json, result_json, success, duration_ms
                        )
                        VALUES (
                            :id, :step_id, :hypothesis_id, :tool_name, :domain,
                            CAST(:params_json AS jsonb),
                            CAST(:result_json AS jsonb),
                            :success,
                            :duration_ms
                        )
                        """
                    ),
                    {
                        "id": str(uuid4()),
                        "step_id": step_id,
                        "hypothesis_id": hypothesis_id,
                        "tool_name": tool_name,
                        "domain": domain,
                        "params_json": json.dumps(params),
                        "result_json": json.dumps(result.output) if result.output else "null",
                        "success": bool(result.success),
                        "duration_ms": duration_ms,
                    },
                )
                await session.commit()

            await emitter.emit(
                session_id=session_id,
                event_type=EventType.AGENT_STEP_COMPLETED,
                step=f"experiment.{hypothesis_id}.step_{step_index}",
                payload={"step": step, "result": result.output if result.success else None},
                hypothesis_id=hypothesis_id,
            )
            step_index += 1

        verdict = "confirmed" if result.success and result.output and result.output.get("exists") else "inconclusive"
        confidence = 0.55 if verdict == "confirmed" else 0.35
        evidence = (
            "Probe tool executed successfully; key present in payload."
            if verdict == "confirmed"
            else "Tool completed without a positive probe signal."
        )
        key_finding = "Probe extraction succeeded." if verdict == "confirmed" else None

        await _update_hypothesis(
            session_factory,
            hypothesis_id,
            status="completed",
            verdict=verdict,
            confidence=confidence,
            evidence_summary=evidence,
            key_finding=key_finding,
        )

        await emitter.emit(
            session_id=session_id,
            event_type=EventType.AGENT_COMPLETED,
            step=f"experiment.{hypothesis_id}.complete",
            payload={"verdict": verdict, "confidence": confidence},
            hypothesis_id=hypothesis_id,
        )

        duration_sec = time.perf_counter() - started
        tool_ok = result is not None and result.success
        experiment_result = {
            "hypothesis_id": hypothesis_id,
            "verdict": verdict,
            "confidence": confidence,
            "evidence_summary": evidence,
            "key_finding": key_finding,
            "tool_trace_id": trace_pointer,
            "figures": [],
            "duration_sec": round(duration_sec, 3),
            "failure_reason": None if verdict != "inconclusive" or tool_ok else "Tool error",
        }

        await redis.close()
        await engine.dispose()
        return experiment_result

    except Exception as exc:
        await _update_hypothesis(
            session_factory,
            hypothesis_id,
            status="failed",
            verdict="inconclusive",
            confidence=0.0,
            evidence_summary=str(exc),
            key_finding=None,
        )
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.AGENT_FAILED,
            step=f"experiment.{hypothesis_id}.failed",
            payload={"error": str(exc)},
            hypothesis_id=hypothesis_id,
        )
        await redis.close()
        await engine.dispose()
        return {
            "hypothesis_id": hypothesis_id,
            "verdict": "inconclusive",
            "confidence": 0.0,
            "evidence_summary": str(exc),
            "key_finding": None,
            "tool_trace_id": trace_pointer,
            "figures": [],
            "duration_sec": round(time.perf_counter() - started, 3),
            "failure_reason": str(exc),
        }
