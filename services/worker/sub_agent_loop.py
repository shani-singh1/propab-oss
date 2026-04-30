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
from services.worker.peer_findings import poll_peer_findings
from services.worker.sandbox import run_sandboxed_python
from services.worker.significance import (
    any_significance_tool_ran,
    check_significance,
)
from services.worker.think_act import (
    AgentContext,
    decide_next_action,
    should_stop,
)

logger = logging.getLogger(__name__)

_UTILITY_TOOL_NAMES = frozenset({"json_extract", "text_stats", "format_convert"})

# Tools that produce p_value / effect_size / confidence_interval
_SIGNIFICANCE_TOOL_NAMES = frozenset({
    "statistical_significance",
    "bootstrap_confidence",
    "literature_baseline_compare",
})


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
            specs, hypothesis_text=hypothesis_text, hypothesis=hypothesis, max_tools=per,
        )
    return merged


def _extend_plan_with_heuristic_rounds(
    base: list[tuple[str, dict]],
    specs: list[dict],
    *,
    hypothesis_text: str,
    hypothesis: dict,
) -> list[tuple[str, dict]]:
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


def _tokens(text_: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{4,}", text_.lower())}


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
    lo, hi = payload.get("ci_lower"), payload.get("ci_upper")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        return [float(lo), float(hi)]
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
    if not successful_outputs:
        return 0.0
    hyp_toks = _tokens(hypothesis_text)
    if not hyp_toks:
        return 0.0
    blob = json.dumps(successful_outputs, ensure_ascii=False).lower()
    out_toks = _tokens(blob)
    overlap = len(hyp_toks & out_toks) / float(len(hyp_toks))
    evidence_keys = ("conclusion", "verdict", "significant", "p_value", "improvement",
                     "confidence_interval", "summary")
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


async def _insert_experiment_step_tool(
    session_factory: async_sessionmaker,
    *,
    step_id: str,
    hypothesis_id: str,
    step_index: int,
    tool_name: str,
    params: dict,
    result_output: Any,
    result_error: Any,
    duration_ms: int,
) -> None:
    async with session_factory() as session:
        await session.execute(
            text("""
                INSERT INTO experiment_steps
                    (id, hypothesis_id, step_type, step_index, input_json, output_json, error_json, duration_ms)
                VALUES
                    (:id, :hypothesis_id, 'tool_call', :step_index,
                     CAST(:input_json AS jsonb), CAST(:output_json AS jsonb),
                     CAST(:error_json AS jsonb), :duration_ms)
            """),
            {
                "id": step_id,
                "hypothesis_id": hypothesis_id,
                "step_index": step_index,
                "input_json": json.dumps({"tool": tool_name, "params": params}),
                "output_json": json.dumps(result_output) if result_output is not None else "null",
                "error_json": json.dumps(result_error) if result_error is not None else "null",
                "duration_ms": duration_ms,
            },
        )
        await session.execute(
            text("""
                INSERT INTO tool_calls
                    (id, step_id, hypothesis_id, tool_name, domain, params_json, result_json, success, duration_ms)
                VALUES
                    (:id, :step_id, :hypothesis_id, :tool_name, :domain,
                     CAST(:params_json AS jsonb), CAST(:result_json AS jsonb), :success, :duration_ms)
            """),
            {
                "id": str(uuid4()),
                "step_id": step_id,
                "hypothesis_id": hypothesis_id,
                "tool_name": tool_name,
                "domain": "unknown",
                "params_json": json.dumps(params),
                "result_json": json.dumps(result_output) if result_output is not None else "null",
                "success": result_error is None,
                "duration_ms": duration_ms,
            },
        )
        await session.commit()


async def _insert_experiment_step_code(
    session_factory: async_sessionmaker,
    *,
    step_id: str,
    hypothesis_id: str,
    step_index: int,
    code: str,
    parsed_output: Any,
    sandbox_out: dict,
    duration_ms: int,
    memory_mb: int,
    timeout_sec: int,
) -> None:
    async with session_factory() as session:
        await session.execute(
            text("""
                INSERT INTO experiment_steps
                    (id, hypothesis_id, step_type, step_index, input_json, output_json, error_json,
                     duration_ms, memory_mb, timeout_sec)
                VALUES
                    (:id, :hypothesis_id, 'code_exec', :step_index,
                     CAST(:input_json AS jsonb), CAST(:output_json AS jsonb),
                     CAST(:error_json AS jsonb), :duration_ms, :memory_mb, :timeout_sec)
            """),
            {
                "id": step_id,
                "hypothesis_id": hypothesis_id,
                "step_index": step_index,
                "input_json": json.dumps({"code": code}),
                "output_json": json.dumps({"parsed": parsed_output, "stdout": sandbox_out.get("stdout")}),
                "error_json": json.dumps(sandbox_out) if not sandbox_out.get("ok") else "null",
                "duration_ms": duration_ms,
                "memory_mb": memory_mb,
                "timeout_sec": timeout_sec,
            },
        )
        await session.commit()


async def run_sub_agent_async(payload: dict) -> dict:
    """
    Execute sub-agent trace with think-act or heuristic execution.

    - think-act (SUB_AGENT_PLAN_SOURCE=llm or hybrid): LLM decides each tool call
      after observing accumulated results. Significance gate enforced before verdict.
    - heuristic (SUB_AGENT_PLAN_SOURCE=heuristic): static multi-round tool plan,
      with a significance recovery step appended when no stat tool ran.

    Tool failures are non-fatal. Verdict requires significance evidence.
    """
    session_id: str = payload["session_id"]
    hypothesis_id: str = payload["hypothesis_id"]
    hypothesis: dict = payload["hypothesis"]
    question: str = str(payload.get("question") or "")
    peer_findings: list[dict] = payload.get("peer_findings") or []
    learned_from: str | None = payload.get("learned_from") or None
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
            question=question,
        )
        sandbox_timeout_sec = effective_sandbox_timeout_sec(domain, settings.sandbox_timeout_sec)

        await emitter.emit(
            session_id=session_id,
            event_type=EventType.TOOL_SELECTED,
            step=f"experiment.{hypothesis_id}.domain",
            payload={"domain": domain, "reason": domain_reason},
            hypothesis_id=hypothesis_id,
        )

        # Always include significance tools alongside domain cluster
        specs = registry.get_cluster_with_significance(domain)
        if not specs:
            specs = registry.get_cluster_with_significance("general_computation")
        available_tools = [str(s["name"]) for s in specs]
        resource_limits = {
            "memory_mb": settings.sandbox_memory_mb,
            "timeout_sec": sandbox_timeout_sec,
            "domain": domain,
        }

        hyp_text = str(hypothesis.get("text", ""))
        plan_source = (settings.sub_agent_plan_source or "heuristic").strip().lower()
        can_llm = settings.llm_provider.strip().lower() == "ollama" or bool(settings.llm_api_secret.strip())
        use_think_act = plan_source in ("llm", "hybrid") and can_llm
        agent_max_steps = max(int(settings.agent_max_steps), 5)
        agent_min_steps = max(1, min(int(settings.agent_min_steps), agent_max_steps - 1))

        # Build initial tool steps (heuristic plan used as starting point for both paths)
        heuristic_steps = _heuristic_tool_plan_merged(
            specs, hypothesis_text=hyp_text, hypothesis=hypothesis,
        )
        plan_origin = "think_act" if use_think_act else "heuristic"

        if not use_think_act and plan_source in ("llm", "hybrid") and can_llm:
            max_llm = max(1, min(int(settings.sub_agent_max_planned_steps), 12))
            planned = await build_tool_plan_via_llm(
                llm=llm,
                session_id=session_id,
                hypothesis_id=hypothesis_id,
                hypothesis_text=hyp_text,
                specs=specs,
                max_steps=max_llm,
                emitter=emitter,
            )
            if planned is not None and len(planned) >= 1:
                heuristic_steps = _extend_plan_with_heuristic_rounds(
                    list(planned), specs, hypothesis_text=hyp_text, hypothesis=hypothesis,
                )
                plan_origin = "llm"

        await emitter.emit(
            session_id=session_id,
            event_type=EventType.AGENT_PLAN_CREATED,
            step=f"experiment.{hypothesis_id}.plan",
            payload={
                "domain": domain,
                "sandbox_timeout_sec": sandbox_timeout_sec,
                "available_tools": available_tools,
                "resource_limits": resource_limits,
                "plan_origin": plan_origin,
                "think_act_enabled": use_think_act,
                "agent_max_steps": agent_max_steps,
                "heuristic_steps": [tn for tn, _ in heuristic_steps],
            },
            hypothesis_id=hypothesis_id,
        )

        sandbox_ok = False
        any_tool_success = False
        successful_tool_outputs: list[dict[str, Any]] = []
        successful_tool_names: list[str] = []
        step_counter = 0

        # ── Shared tool execution helper (inline, no external function needed) ─

        async def run_tool_step(tool_name: str, params: dict, step_index: int) -> bool:
            nonlocal any_tool_success
            step_id = str(uuid4())
            t0 = time.perf_counter()

            await emitter.emit(
                session_id=session_id,
                event_type=EventType.TOOL_CALLED,
                step=f"experiment.{hypothesis_id}.step_{step_index}",
                payload={"tool": tool_name, "params": params},
                hypothesis_id=hypothesis_id,
            )

            result = registry.call(tool_name, params)
            duration_ms = int((time.perf_counter() - t0) * 1000)

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
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_STEP_COMPLETED,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={"tool": tool_name},
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
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_STEP_FAILED,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={"tool": tool_name, "non_fatal": True},
                    hypothesis_id=hypothesis_id,
                )

            await _insert_experiment_step_tool(
                session_factory,
                step_id=step_id,
                hypothesis_id=hypothesis_id,
                step_index=step_index,
                tool_name=tool_name,
                params=params,
                result_output=result.output,
                result_error=result.error.to_dict() if result.error else None,
                duration_ms=duration_ms,
            )
            return bool(result.success)

        async def run_code_step(code: str, step_index: int) -> bool:
            nonlocal sandbox_ok
            step_id = str(uuid4())
            t0 = time.perf_counter()
            max_attempts = max(1, int(settings.sandbox_code_max_retries))
            parsed = None
            sandbox_out: dict = {}

            await emitter.emit(
                session_id=session_id,
                event_type=EventType.CODE_GENERATED,
                step=f"experiment.{hypothesis_id}.step_{step_index}",
                payload={"code": code},
                hypothesis_id=hypothesis_id,
            )

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
                    run_sandboxed_python, code,
                    timeout_sec=sandbox_timeout_sec, memory_mb=settings.sandbox_memory_mb,
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
                        payload={"stdout_json": parsed, "stdout": sandbox_out.get("stdout"), "attempt": attempt + 1},
                        hypothesis_id=hypothesis_id,
                    )
                    break
                is_timeout = "timeout" in str(sandbox_out.get("message", "")).lower()
                ev = EventType.CODE_TIMEOUT if is_timeout else EventType.CODE_ERROR
                await emitter.emit(
                    session_id=session_id, event_type=ev,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={"error": sandbox_out, "retry": attempt + 1, "max_attempts": max_attempts},
                    hypothesis_id=hypothesis_id,
                )
            else:
                sandbox_ok = False

            duration_ms = int((time.perf_counter() - t0) * 1000)
            await _insert_experiment_step_code(
                session_factory,
                step_id=step_id,
                hypothesis_id=hypothesis_id,
                step_index=step_index,
                code=code,
                parsed_output=parsed,
                sandbox_out=sandbox_out,
                duration_ms=duration_ms,
                memory_mb=settings.sandbox_memory_mb,
                timeout_sec=sandbox_timeout_sec,
            )
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.AGENT_STEP_COMPLETED,
                step=f"experiment.{hypothesis_id}.step_{step_index}",
                payload={"sandbox_ok": sandbox_ok},
                hypothesis_id=hypothesis_id,
            )
            return sandbox_ok

        # ─────────────────────────────────────────────────────────────────────
        # THINK-ACT PATH: LLM decides each next action from accumulated context
        # ─────────────────────────────────────────────────────────────────────
        if use_think_act:
            agent_ctx = AgentContext(
                hypothesis_text=hyp_text,
                test_methodology=str(hypothesis.get("test_methodology") or ""),
                learned_from=learned_from,
                peer_findings=peer_findings,
                results_so_far=[],
                tool_names_run=[],
                steps_taken=0,
                max_steps=agent_max_steps,
                min_steps=agent_min_steps,
            )

            # Run the first heuristic step immediately to seed the agent with data
            if heuristic_steps:
                first_tool, first_params = heuristic_steps[0]
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_STEP_STARTED,
                    step=f"experiment.{hypothesis_id}.step_{step_counter}",
                    payload={"tool": first_tool, "source": "heuristic_seed"},
                    hypothesis_id=hypothesis_id,
                )
                ok = await run_tool_step(first_tool, dict(first_params), step_counter)
                if ok and successful_tool_outputs:
                    agent_ctx.results_so_far.append(successful_tool_outputs[-1])
                    agent_ctx.tool_names_run.append(first_tool)
                step_counter += 1
                agent_ctx.steps_taken += 1

            # Think-act loop
            while not should_stop(agent_ctx):
                # Poll peer channel non-blocking — inject new findings into context
                try:
                    new_peers = await poll_peer_findings(redis, hypothesis_id=hypothesis_id)
                    if new_peers:
                        agent_ctx.peer_findings.extend(new_peers)
                        logger.debug(
                            "Agent %s received %d peer finding(s).", hypothesis_id, len(new_peers)
                        )
                except Exception:
                    pass  # peer polling is always best-effort

                action = await decide_next_action(
                    context=agent_ctx,
                    specs=specs,
                    llm=llm,
                    session_id=session_id,
                    hypothesis_id=hypothesis_id,
                )

                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_STEP_STARTED,
                    step=f"experiment.{hypothesis_id}.step_{step_counter}",
                    payload={"action": action.action_type, "reasoning": action.reasoning,
                             "expected_outcome": action.expected_outcome},
                    hypothesis_id=hypothesis_id,
                )

                if action.action_type == "stop":
                    await emitter.emit(
                        session_id=session_id,
                        event_type=EventType.AGENT_STEP_COMPLETED,
                        step=f"experiment.{hypothesis_id}.step_{step_counter}",
                        payload={"action": "stop", "reasoning": action.reasoning},
                        hypothesis_id=hypothesis_id,
                    )
                    break

                step_ok = False
                if action.action_type == "tool" and action.tool_name:
                    tool_name_ta = action.tool_name
                    # Tolerate unknown tool names: skip and continue
                    try:
                        step_ok = await run_tool_step(tool_name_ta, action.params, step_counter)
                    except KeyError:
                        logger.warning("Think-act chose unknown tool %s; skipping.", tool_name_ta)
                        await emitter.emit(
                            session_id=session_id,
                            event_type=EventType.TOOL_ERROR,
                            step=f"experiment.{hypothesis_id}.step_{step_counter}",
                            payload={"tool": tool_name_ta, "error": {"type": "unknown_tool"}},
                            hypothesis_id=hypothesis_id,
                        )
                    if step_ok and successful_tool_outputs:
                        agent_ctx.results_so_far.append(successful_tool_outputs[-1])
                        agent_ctx.tool_names_run.append(tool_name_ta)

                elif action.action_type == "code":
                    code_desc = action.code_description or "custom computation"
                    code = (
                        f"import json, sys\n"
                        f"# {code_desc}\n"
                        f"result = {{'computation': {json.dumps(code_desc)}, 'status': 'executed'}}\n"
                        f"print(json.dumps(result))\n"
                    )
                    step_ok = await run_code_step(code, step_counter)

                step_counter += 1
                agent_ctx.steps_taken += 1

        # ─────────────────────────────────────────────────────────────────────
        # HEURISTIC PATH: static multi-round plan + significance recovery
        # ─────────────────────────────────────────────────────────────────────
        else:
            plan_steps: list[dict] = [
                {"type": "tool", "tool": tn, "params": dict(pr)} for tn, pr in heuristic_steps
            ] + [{
                "type": "code",
                "code": (
                    "import json,sys\n"
                    f"print(json.dumps({{\"sandbox\":\"ok\",\"hypothesis_rank\":{json.dumps(hypothesis.get('rank'))}}}))  \n"
                ),
            }]
            if not plan_steps:
                raise RuntimeError(
                    f"Empty execution plan for hypothesis {hypothesis_id}; refusing zero-step trace."
                )

            prev_out: dict | None = None
            for step_index, step in enumerate(plan_steps):
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_STEP_STARTED,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={"step": step},
                    hypothesis_id=hypothesis_id,
                )

                if step.get("type") == "code":
                    await run_code_step(str(step.get("code", "")), step_index)
                else:
                    tool_name = step["tool"]
                    params = step["params"]
                    # Chain prior output into next step if applicable
                    if prev_out is not None and step_index > 0:
                        step = refine_next_tool_step(
                            plan_steps[step_index - 1].get("tool", ""),
                            prev_out,
                            step,
                        )
                        params = step["params"]
                    await run_tool_step(tool_name, params, step_index)
                    prev_out = successful_tool_outputs[-1] if successful_tool_outputs else None

                step_counter = step_index + 1

            # Significance recovery: if no significance tool ran, force one now
            if not any_significance_tool_ran(successful_tool_names) and successful_tool_outputs:
                sig_step_index = step_counter
                logger.info(
                    "Significance recovery: no stat tool ran for %s. Attempting bootstrap_confidence.",
                    hypothesis_id,
                )
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_STEP_STARTED,
                    step=f"experiment.{hypothesis_id}.step_{sig_step_index}",
                    payload={"note": "significance_recovery", "tool": "bootstrap_confidence"},
                    hypothesis_id=hypothesis_id,
                )
                # Extract numeric values from outputs
                values: list[float] = []
                for out in successful_tool_outputs:
                    for v in _walk_numeric_values(out).values():
                        values.append(v)
                        if len(values) >= 20:
                            break
                    if len(values) >= 20:
                        break

                if len(values) >= 2:
                    try:
                        await run_tool_step(
                            "bootstrap_confidence",
                            {"values": values[:20]},
                            sig_step_index,
                        )
                        step_counter += 1
                    except Exception as exc:
                        logger.warning("Significance recovery failed: %s", exc)
                else:
                    logger.info("Significance recovery skipped: not enough numeric values.")

        # ─────────────────────────────────────────────────────────────────────
        # VERDICT COMPUTATION (shared by both paths)
        # ─────────────────────────────────────────────────────────────────────
        relevance_score = _hypothesis_relevance_score(hyp_text, successful_tool_outputs)
        substantive_hit = any(n in SUBSTANTIVE_TOOL_NAMES for n in successful_tool_names)
        if substantive_hit:
            relevance_score = min(1.0, relevance_score + 0.06)

        n_tool_steps = sum(1 for n in successful_tool_names if n not in _SIGNIFICANCE_TOOL_NAMES)
        evidence_obj = _build_evidence(
            successful_outputs=successful_tool_outputs,
            relevance_score=relevance_score,
            n_tool_steps=n_tool_steps,
            baseline_value=baseline_value,
        )

        # Use significance module for the gate check
        sig_result = check_significance(successful_tool_outputs)

        if evidence_obj["n_metric_steps"] == 0:
            verdict = "inconclusive"
            confidence = 0.0
            evidence_obj["verdict_reason"] = "no metric-bearing steps executed"
        elif sig_result.gate_definitively_failed:
            verdict = "refuted"
            confidence = _compute_confidence(evidence_obj)
            evidence_obj["verdict_reason"] = (
                "significance test ran and found no effect (p >= 0.30, negligible effect size)"
            )
        elif not sig_result.gate_passed:
            verdict = "inconclusive"
            confidence = _compute_confidence(evidence_obj)
            evidence_obj["verdict_reason"] = (
                "no significance evidence: p_value/effect_size/CI not produced or not decisive"
            )
        else:
            # Gate passed — check direction
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
                "significance gate passed; metric direction supports hypothesis"
                if supports_hypothesis
                else "significance gate passed but metric direction ambiguous"
            )

        sig_summary = {
            "gate_passed": sig_result.gate_passed,
            "p_value": sig_result.p_value,
            "effect_size": sig_result.effect_size,
            "method": sig_result.method,
        }
        evidence = (
            f"evidence={json.dumps(evidence_obj, ensure_ascii=False)}; "
            f"significance={json.dumps(sig_summary)}; "
            f"plan_origin={plan_origin}; "
            f"any_success={any_tool_success}; sandbox_ok={sandbox_ok}; "
            f"steps={step_counter}."
        )
        key_finding = (
            "Significance gate passed: metric evidence supports hypothesis."
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
            payload={"verdict": verdict, "confidence": confidence, "sig_gate_passed": sig_result.gate_passed},
            hypothesis_id=hypothesis_id,
        )

        duration_sec = time.perf_counter() - started
        result_dict = {
            "hypothesis_id": hypothesis_id,
            "verdict": verdict,
            "confidence": confidence,
            "evidence_summary": evidence,
            "key_finding": key_finding,
            "tool_trace_id": trace_pointer,
            "figures": [],
            "duration_sec": round(duration_sec, 3),
            "failure_reason": None if verdict == "confirmed" else evidence_obj["verdict_reason"],
            "learned": (
                f"Tools run: {', '.join(successful_tool_names[:10])}. "
                f"Significance: {sig_result.method or 'none'}. "
                f"Verdict: {verdict}."
            ),
        }

        await redis.close()
        await engine.dispose()
        return result_dict

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
            "learned": None,
        }
