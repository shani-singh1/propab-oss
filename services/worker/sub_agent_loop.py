from __future__ import annotations

import asyncio
import json
import re
import time
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.config import settings
from propab.sandbox_profiles import effective_sandbox_timeout_sec
from propab.db import create_engine, create_redis, create_session_factory
from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.sub_agent_plan import build_tool_plan_via_llm
from propab.tool_chain import refine_next_tool_step
from propab.tool_selection import select_tool_steps
from propab.tools.registry import ToolRegistry
from propab.types import EventType
from services.worker.domain_router import route_domain
from services.worker.sandbox import run_sandboxed_python


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{4,}", text.lower())}


def _hypothesis_relevance_score(hypothesis_text: str, successful_outputs: list[dict]) -> float:
    """
    Coarse relevance signal: lexical overlap + evidence-key bonus.
    Keeps confirmation from being based solely on transport-level success.
    """
    if not successful_outputs:
        return 0.0
    hyp_toks = _tokens(hypothesis_text)
    if not hyp_toks:
        return 0.0
    blob = json.dumps(successful_outputs, ensure_ascii=False).lower()
    out_toks = _tokens(blob)
    overlap = len(hyp_toks & out_toks) / float(len(hyp_toks))
    evidence_keys = (
        "conclusion",
        "verdict",
        "significant",
        "p_value",
        "improvement",
        "confidence_interval",
        "summary",
    )
    key_bonus = 0.02 * sum(1 for k in evidence_keys if k in blob)
    return float(overlap + key_bonus)


async def _update_hypothesis(
    session_factory: async_sessionmaker,
    hypothesis_id: str,
    *,
    status: str,
    verdict: str | None = None,
    confidence: float | None = None,
    evidence_summary: str | None = None,
    key_finding: str | None = None,
    tool_trace_id: str | None = None,
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
    if tool_trace_id is not None:
        fields.append("tool_trace_id = :tool_trace_id")
        params["tool_trace_id"] = tool_trace_id
    query = text(f"UPDATE hypotheses SET {', '.join(fields)} WHERE id = :id")
    async with session_factory() as session:
        await session.execute(query, params)
        await session.commit()


async def run_sub_agent_async(payload: dict) -> dict:
    """
    Execute sub-agent trace: up to two ranked tool calls, sandbox probe, verdict.
    Tool failures are non-fatal: later steps still run; verdict uses any successful tool or sandbox.
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
        api_key=settings.llm_api_secret,
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
        sandbox_timeout_sec = effective_sandbox_timeout_sec(domain, settings.sandbox_timeout_sec)

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
        available_tools = [str(s["name"]) for s in specs]
        resource_limits = {
            "memory_mb": settings.sandbox_memory_mb,
            "timeout_sec": sandbox_timeout_sec,
            "domain": domain,
        }
        hyp_text = str(hypothesis.get("text", ""))
        plan_source = (settings.sub_agent_plan_source or "heuristic").strip().lower()
        max_llm = max(1, min(int(settings.sub_agent_max_planned_steps), 12))
        can_llm = settings.llm_provider.strip().lower() == "ollama" or bool(settings.llm_api_secret.strip())

        tool_steps = select_tool_steps(
            specs,
            hypothesis_text=hyp_text,
            hypothesis=hypothesis,
            max_tools=2,
        )
        plan_origin = "heuristic"
        if plan_source in ("llm", "hybrid") and can_llm:
            planned = await build_tool_plan_via_llm(
                llm=llm,
                session_id=session_id,
                hypothesis_id=hypothesis_id,
                hypothesis_text=hyp_text,
                specs=specs,
                max_steps=max_llm,
                emitter=emitter,
            )
            use_llm_plan = planned is not None and (
                plan_source == "llm" or (plan_source == "hybrid" and len(planned) >= 1)
            )
            if use_llm_plan and planned is not None:
                tool_steps = planned
                plan_origin = "llm"

        rank = hypothesis.get("rank")
        sandbox_code = (
            "import json,sys\n"
            "print(json.dumps({\"sandbox\":\"ok\",\"hypothesis_rank\": "
            + json.dumps(rank)
            + "}))\n"
        )

        plan_steps: list[dict] = [
            {"type": "tool", "tool": tn, "params": dict(pr)} for tn, pr in tool_steps
        ] + [{"type": "code", "code": sandbox_code}]
        executable_steps = [s for s in plan_steps if s.get("type") in {"tool", "code"}]
        if not executable_steps:
            raise RuntimeError(
                f"Empty execution plan for hypothesis {hypothesis_id}; refusing silent zero-step trace."
            )
        plan = {
            "steps": plan_steps,
            "available_tools": available_tools,
            "resource_limits": resource_limits,
            "plan_origin": plan_origin,
        }
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.AGENT_PLAN_CREATED,
            step=f"experiment.{hypothesis_id}.plan",
            payload={
                "plan": plan,
                "domain": domain,
                "sandbox_timeout_sec": sandbox_timeout_sec,
                "available_tools": available_tools,
                "resource_limits": resource_limits,
                "plan_origin": plan_origin,
            },
            hypothesis_id=hypothesis_id,
        )

        sandbox_ok = False
        any_tool_success = False
        successful_tool_outputs: list[dict] = []
        for step_index, step in enumerate(plan_steps):
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.AGENT_STEP_STARTED,
                step=f"experiment.{hypothesis_id}.step_{step_index}",
                payload={"step": step},
                hypothesis_id=hypothesis_id,
            )

            step_started = time.perf_counter()
            step_id = str(uuid4())

            if step.get("type") == "code":
                code = str(step.get("code", ""))
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.CODE_GENERATED,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={"code": code},
                    hypothesis_id=hypothesis_id,
                )
                max_attempts = max(1, int(settings.sandbox_code_max_retries))
                sandbox_out: dict = {}
                parsed = None
                for attempt in range(max_attempts):
                    await emitter.emit(
                        session_id=session_id,
                        event_type=EventType.CODE_SUBMITTED,
                        step=f"experiment.{hypothesis_id}.step_{step_index}",
                        payload={
                            "memory_mb": settings.sandbox_memory_mb,
                            "timeout_sec": sandbox_timeout_sec,
                            "domain": domain,
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                        },
                        hypothesis_id=hypothesis_id,
                    )
                    sandbox_out = await asyncio.to_thread(
                        run_sandboxed_python,
                        code,
                        timeout_sec=sandbox_timeout_sec,
                        memory_mb=settings.sandbox_memory_mb,
                    )
                    parsed = sandbox_out.get("parsed") if isinstance(sandbox_out, dict) else None
                    ok_run = bool(
                        sandbox_out.get("ok")
                        and isinstance(parsed, dict)
                        and parsed.get("sandbox") == "ok"
                    )
                    if ok_run:
                        sandbox_ok = True
                        await emitter.emit(
                            session_id=session_id,
                            event_type=EventType.CODE_RESULT,
                            step=f"experiment.{hypothesis_id}.step_{step_index}",
                            payload={
                                "stdout_json": parsed,
                                "stdout": sandbox_out.get("stdout"),
                                "attempt": attempt + 1,
                            },
                            hypothesis_id=hypothesis_id,
                        )
                        break
                    is_timeout = "timeout" in str(sandbox_out.get("message", "")).lower()
                    ev = EventType.CODE_TIMEOUT if is_timeout else EventType.CODE_ERROR
                    await emitter.emit(
                        session_id=session_id,
                        event_type=ev,
                        step=f"experiment.{hypothesis_id}.step_{step_index}",
                        payload={
                            "error": sandbox_out,
                            "retry": attempt + 1,
                            "max_attempts": max_attempts,
                        },
                        hypothesis_id=hypothesis_id,
                    )
                else:
                    sandbox_ok = False

                duration_ms = int((time.perf_counter() - step_started) * 1000)
                async with session_factory() as session:
                    await session.execute(
                        text(
                            """
                            INSERT INTO experiment_steps (
                                id, hypothesis_id, step_type, step_index, input_json, output_json, error_json, duration_ms, memory_mb, timeout_sec
                            )
                            VALUES (
                                :id, :hypothesis_id, 'code_exec', :step_index,
                                CAST(:input_json AS jsonb),
                                CAST(:output_json AS jsonb),
                                CAST(:error_json AS jsonb),
                                :duration_ms, :memory_mb, :timeout_sec
                            )
                            """
                        ),
                        {
                            "id": step_id,
                            "hypothesis_id": hypothesis_id,
                            "step_index": step_index,
                            "input_json": json.dumps({"code": code, "max_attempts": max_attempts}),
                            "output_json": json.dumps({"parsed": parsed, "stdout": sandbox_out.get("stdout")}),
                            "error_json": json.dumps(sandbox_out) if not sandbox_out.get("ok") else "null",
                            "duration_ms": duration_ms,
                            "memory_mb": settings.sandbox_memory_mb,
                            "timeout_sec": sandbox_timeout_sec,
                        },
                    )
                    await session.commit()
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_STEP_COMPLETED,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={"step": step, "sandbox_ok": sandbox_ok},
                    hypothesis_id=hypothesis_id,
                )
            else:
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
                    any_tool_success = True
                    if isinstance(result.output, dict):
                        successful_tool_outputs.append(result.output)
                    await emitter.emit(
                        session_id=session_id,
                        event_type=EventType.TOOL_RESULT,
                        step=f"experiment.{hypothesis_id}.step_{step_index}",
                        payload={"tool": tool_name, "output": result.output},
                        hypothesis_id=hypothesis_id,
                    )
                    if step_index + 1 < len(plan_steps):
                        plan_steps[step_index + 1] = refine_next_tool_step(
                            tool_name,
                            result.output if isinstance(result.output, dict) else None,
                            plan_steps[step_index + 1],
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

                if result.success:
                    await emitter.emit(
                        session_id=session_id,
                        event_type=EventType.AGENT_STEP_COMPLETED,
                        step=f"experiment.{hypothesis_id}.step_{step_index}",
                        payload={"step": step, "tool": tool_name},
                        hypothesis_id=hypothesis_id,
                    )
                else:
                    await emitter.emit(
                        session_id=session_id,
                        event_type=EventType.AGENT_STEP_FAILED,
                        step=f"experiment.{hypothesis_id}.step_{step_index}",
                        payload={"step": step, "non_fatal": True, "tool": tool_name},
                        hypothesis_id=hypothesis_id,
                    )

        tool_hit = any_tool_success
        relevance_score = _hypothesis_relevance_score(hyp_text, successful_tool_outputs)
        supports_hypothesis = tool_hit and relevance_score >= 0.08
        verdict = "confirmed" if supports_hypothesis else "inconclusive"
        confidence = 0.67 if verdict == "confirmed" else 0.35
        n_tool_steps = len([s for s in plan_steps if s.get("type") == "tool"])
        evidence = (
            f"Ran {n_tool_steps} tool step(s); plan_origin={plan_origin}; "
            f"any_success={tool_hit}; relevance_score={relevance_score:.3f}; sandbox_ok={sandbox_ok}."
            if verdict == "confirmed"
            else (
                "No hypothesis-relevant confirming evidence was found. "
                f"any_success={tool_hit}, relevance_score={relevance_score:.3f}, sandbox_ok={sandbox_ok}."
            )
        )
        key_finding = (
            "At least one successful tool output contained hypothesis-relevant evidence."
            if supports_hypothesis
            else ("Sandbox probe completed." if sandbox_ok else None)
        )

        await _update_hypothesis(
            session_factory,
            hypothesis_id,
            status="completed",
            verdict=verdict,
            confidence=confidence,
            evidence_summary=evidence,
            key_finding=key_finding,
            tool_trace_id=trace_pointer,
        )

        await emitter.emit(
            session_id=session_id,
            event_type=EventType.AGENT_COMPLETED,
            step=f"experiment.{hypothesis_id}.complete",
            payload={"verdict": verdict, "confidence": confidence},
            hypothesis_id=hypothesis_id,
        )

        duration_sec = time.perf_counter() - started
        experiment_result = {
            "hypothesis_id": hypothesis_id,
            "verdict": verdict,
            "confidence": confidence,
            "evidence_summary": evidence,
            "key_finding": key_finding,
            "tool_trace_id": trace_pointer,
            "figures": [],
            "duration_sec": round(duration_sec, 3),
            "failure_reason": None if verdict == "confirmed" else "Probes did not meet confirmation criteria.",
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
            tool_trace_id=trace_pointer,
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
