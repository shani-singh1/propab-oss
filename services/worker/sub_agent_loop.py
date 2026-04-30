from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, TypedDict
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.config import settings
from propab.paper_gate import SUBSTANTIVE_TOOL_NAMES
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

logger = logging.getLogger(__name__)

_UTILITY_TOOL_NAMES = frozenset(
    {
        "json_extract",
        "text_stats",
        "format_convert",
    }
)


class HypothesisEvidence(TypedDict):
    metric_value: float | None
    baseline_value: float | None
    delta: float | None
    delta_pct: float | None
    p_value: float | None
    effect_size: float | None
    confidence_interval: list[float] | None
    n_tool_steps: int
    n_metric_steps: int
    relevance_score: float
    verdict_reason: str


def _heuristic_tool_plan_merged(
    specs: list[dict],
    *,
    hypothesis_text: str,
    hypothesis: dict,
) -> list[tuple[str, dict]]:
    """Several selection rounds so the agent chains distinct tools like an iterative researcher."""
    rounds = max(1, int(settings.sub_agent_max_rounds))
    per = max(1, int(settings.sub_agent_tools_per_round))
    initial_ban = _UTILITY_TOOL_NAMES if len(specs) > 1 else frozenset()
    used_names: set[str] = set()
    merged: list[tuple[str, dict]] = []
    for _ in range(rounds):
        batch = select_tool_steps(
            specs,
            hypothesis_text=hypothesis_text,
            hypothesis=hypothesis,
            max_tools=per,
            exclude_tool_names=frozenset(used_names) | initial_ban,
        )
        before = len(merged)
        for tn, pr in batch:
            if tn in used_names:
                continue
            used_names.add(tn)
            merged.append((tn, pr))
        if len(merged) == before:
            break
    if not merged:
        return select_tool_steps(
            specs,
            hypothesis_text=hypothesis_text,
            hypothesis=hypothesis,
            max_tools=per,
        )
    return merged


def _extend_plan_with_heuristic_rounds(
    base: list[tuple[str, dict]],
    specs: list[dict],
    *,
    hypothesis_text: str,
    hypothesis: dict,
) -> list[tuple[str, dict]]:
    """Append extra tool steps (excluding tools already in ``base``)."""
    used_names = {t for t, _ in base}
    extra_rounds = max(0, int(settings.sub_agent_max_rounds) - 1)
    per = max(1, int(settings.sub_agent_tools_per_round))
    initial_ban = _UTILITY_TOOL_NAMES if len(specs) > 1 else frozenset()
    out = list(base)
    for _ in range(extra_rounds):
        batch = select_tool_steps(
            specs,
            hypothesis_text=hypothesis_text,
            hypothesis=hypothesis,
            max_tools=per,
            exclude_tool_names=frozenset(used_names) | initial_ban,
        )
        gained = False
        for tn, pr in batch:
            if tn in used_names:
                continue
            used_names.add(tn)
            out.append((tn, pr))
            gained = True
        if not gained:
            break
    return out


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{4,}", text.lower())}


def _walk_numeric_values(payload: Any, *, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(payload, dict):
        for k, v in payload.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_walk_numeric_values(v, prefix=key))
    elif isinstance(payload, list):
        nums = [x for x in payload if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if nums and len(nums) <= 128:
            out[prefix or "list"] = float(nums[-1])
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        out[prefix or "value"] = float(payload)
    return out


def _extract_ci(payload: dict[str, Any]) -> list[float] | None:
    for k in ("confidence_interval", "ci", "interval"):
        v = payload.get(k)
        if isinstance(v, list) and len(v) >= 2 and all(isinstance(x, (int, float)) for x in v[:2]):
            return [float(v[0]), float(v[1])]
    return None


def _first_key(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        v = payload.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _build_evidence(
    *,
    successful_outputs: list[dict[str, Any]],
    relevance_score: float,
    n_tool_steps: int,
    baseline_value: float | None,
) -> HypothesisEvidence:
    metric_value: float | None = None
    p_value: float | None = None
    effect_size: float | None = None
    ci = None
    metric_steps = 0
    for out in successful_outputs:
        nums = _walk_numeric_values(out)
        if nums:
            metric_steps += 1
            if metric_value is None:
                metric_value = next(iter(nums.values()))
        if p_value is None:
            p_value = _first_key(out, ("p_value", "p", "pvalue"))
        if effect_size is None:
            effect_size = _first_key(out, ("effect_size", "cohens_d", "d"))
        if ci is None:
            ci = _extract_ci(out)
    delta = None if (metric_value is None or baseline_value is None) else (metric_value - baseline_value)
    delta_pct = None
    if delta is not None and baseline_value not in (None, 0.0):
        delta_pct = (delta / baseline_value) * 100.0
    return HypothesisEvidence(
        metric_value=metric_value,
        baseline_value=baseline_value,
        delta=delta,
        delta_pct=delta_pct,
        p_value=p_value,
        effect_size=effect_size,
        confidence_interval=ci,
        n_tool_steps=n_tool_steps,
        n_metric_steps=metric_steps,
        relevance_score=float(relevance_score),
        verdict_reason="",
    )


def _compute_confidence(evidence: HypothesisEvidence) -> float:
    logger.info(
        "compute_confidence called: n_metric_steps=%s, p_value=%s, effect_size=%s, delta_pct=%s",
        evidence.get("n_metric_steps"),
        evidence.get("p_value"),
        evidence.get("effect_size"),
        evidence.get("delta_pct"),
    )
    score = 0.0
    if evidence["metric_value"] is not None:
        score += 0.20
    if evidence["baseline_value"] is not None:
        score += 0.20
    p = evidence["p_value"]
    if p is not None and p < 0.05:
        score += 0.25
    es = evidence["effect_size"]
    if es is not None and abs(es) > 0.2:
        score += 0.15
    if evidence["n_metric_steps"] >= 3:
        score += 0.10
    if evidence["relevance_score"] > 0.30:
        score += 0.10
    return min(max(score, 0.0), 0.95)


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
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    baseline_value = (
        float(baseline.get("metric_value"))
        if isinstance(baseline.get("metric_value"), (int, float))
        else None
    )

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

        tool_steps = _heuristic_tool_plan_merged(
            specs,
            hypothesis_text=hyp_text,
            hypothesis=hypothesis,
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
                tool_steps = _extend_plan_with_heuristic_rounds(
                    list(planned),
                    specs,
                    hypothesis_text=hyp_text,
                    hypothesis=hypothesis,
                )
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
            "heuristic_rounds": int(settings.sub_agent_max_rounds),
            "tools_per_round": int(settings.sub_agent_tools_per_round),
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
        successful_tool_names: list[str] = []
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
                    successful_tool_names.append(tool_name)
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
        substantive_hit = any(n in SUBSTANTIVE_TOOL_NAMES for n in successful_tool_names)
        if substantive_hit:
            relevance_score = min(1.0, relevance_score + 0.06)
        n_tool_steps = len([s for s in plan_steps if s.get("type") == "tool"])
        evidence_obj = _build_evidence(
            successful_outputs=successful_tool_outputs,
            relevance_score=relevance_score,
            n_tool_steps=n_tool_steps,
            baseline_value=baseline_value,
        )
        # Hard gate contract: no metric-bearing steps => no confirmation.
        if evidence_obj["n_metric_steps"] == 0:
            verdict = "inconclusive"
            confidence = 0.0
            evidence_obj["verdict_reason"] = "no metric-bearing steps executed"
        elif evidence_obj["delta"] is None and evidence_obj["p_value"] is None:
            verdict = "inconclusive"
            confidence = 0.30
            evidence_obj["verdict_reason"] = "tools ran but produced no comparable numbers"
        else:
            supports_hypothesis = False
            if evidence_obj["p_value"] is not None:
                supports_hypothesis = bool(
                    evidence_obj["delta"] is not None
                    and evidence_obj["p_value"] < 0.05
                    and evidence_obj["relevance_score"] >= 0.12
                )
            elif evidence_obj["effect_size"] is not None:
                supports_hypothesis = bool(
                    abs(evidence_obj["effect_size"]) > 0.2 and evidence_obj["relevance_score"] >= 0.12
                )
            elif evidence_obj["delta_pct"] is not None:
                supports_hypothesis = bool(
                    abs(evidence_obj["delta_pct"]) >= 2.0 and evidence_obj["relevance_score"] >= 0.12
                )
            confidence = _compute_confidence(evidence_obj)
            verdict = "confirmed" if supports_hypothesis else "inconclusive"
            evidence_obj["verdict_reason"] = (
                "metric crossed confirmation threshold with baseline/statistical support"
                if supports_hypothesis
                else "metric evidence did not cross confirmation threshold"
            )
        evidence = (
            f"evidence={json.dumps(evidence_obj, ensure_ascii=False)}; "
            f"plan_origin={plan_origin}; rounds_config=max{settings.sub_agent_max_rounds}x{settings.sub_agent_tools_per_round}/round; "
            f"any_success={tool_hit}; sandbox_ok={sandbox_ok}."
        )
        key_finding = (
            "At least one metric-bearing tool output crossed the confirmation contract."
            if verdict == "confirmed"
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
            "failure_reason": None if verdict == "confirmed" else evidence_obj["verdict_reason"],
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
