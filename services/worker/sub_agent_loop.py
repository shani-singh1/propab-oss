from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, TypedDict
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.config import settings
from propab.tools.types import ToolError, ToolResult
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
from services.worker.domain_router import coerce_routed_domain, route_domain
from services.worker.failure_classify import classify_exception
from services.worker.peer_findings import poll_peer_findings
from services.worker.sandbox import run_sandboxed_python
from services.worker.sandbox_code_rewrite import (
    looks_like_heavy_training_code,
    rewrite_sandbox_code_after_timeout,
)
from services.worker.significance import (
    any_significance_tool_ran,
    check_significance,
    classify_verdict,
)
from services.worker.think_act import (
    AgentContext,
    decide_next_action,
    should_stop,
)

logger = logging.getLogger(__name__)


def _sandbox_diag_tails(sandbox_out: dict[str, Any], *, maxlen: int = 1500) -> dict[str, Any]:
    """Short tails for code.timeout / code.error payloads (container stderr is the main signal)."""
    if not isinstance(sandbox_out, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("stderr", "stdout", "message"):
        raw = sandbox_out.get(key)
        if raw is None:
            continue
        s = raw if isinstance(raw, str) else str(raw)
        if not s.strip():
            continue
        out[f"{key}_tail"] = s[-maxlen:] if len(s) > maxlen else s
    et = sandbox_out.get("error_type")
    if et:
        out["sandbox_error_type"] = et
    return out


def _think_act_stub_code(code_desc: str) -> str:
    """
    Deterministic stub: must never embed raw ``code_description`` in a ``#`` line.

    A newline inside the description would end the comment and turn the rest into
    executable Python (LLM pasting a full training script → Docker wall timeouts).
    """
    desc = str(code_desc or "custom computation")
    return (
        "import json, sys\n"
        "result = {"
        f"'computation': {json.dumps(desc)}, "
        "'status': 'executed', 'sandbox': 'ok'"
        "}\n"
        "print(json.dumps(result))\n"
    )


def _is_sandbox_wall_timeout(sandbox_out: dict[str, Any]) -> bool:
    """
    True when Docker or the sandbox wrapper hit a wall-clock limit.

    ``run_sandboxed_python`` historically put everything in ``message``; some
    Docker SDK paths use ``Read timed out`` (no substring ``timeout``), which
    must still count as a wall timeout so we never treat it as a generic error
    and spin identical retries.
    """
    if not isinstance(sandbox_out, dict):
        return False
    et = str(sandbox_out.get("error_type", "") or "").lower()
    if "timeout" in et or et in {"docker_read_timeout", "docker_timeout"}:
        return True
    msg = str(sandbox_out.get("message", "") or "").lower()
    stderr = str(sandbox_out.get("stderr", "") or "").lower()
    blob = f"{msg} {stderr}"
    if "timeout" in blob or "timed out" in blob or "deadline exceeded" in blob:
        return True
    return False


def _is_trusted_inline_sandbox_code(code: str) -> bool:
    """
    Detect agent/heuristic stub snippets that are intentionally tiny JSON printers.
    These must never spend 480s in Docker — they were the dominant source of
    code.generated / code.timeout 1:1 ratios when the worker image/network stalled.

    Think–act stubs that embed ``json.dumps(code_description)`` may contain
    substrings like ``open(`` inside string literals; those are still safe but
    fail this heuristic — callers that built the stub programmatically should pass
    ``force_inline_trusted=True`` to ``run_code_step``.
    """
    s = (code or "").strip()
    if len(s) > 6000:
        return False
    blocked = ("subprocess", "socket", "urllib", "requests", "open(", "Path(", "__import__", "eval(", "exec(")
    if any(b in s for b in blocked):
        return False
    # Think–act stub: import json, sys + result dict + print(json.dumps(result))
    if (
        s.startswith("import json, sys")
        and "result = {" in s
        and "print(json.dumps(result))" in s
        and "'sandbox': 'ok'" in s
    ):
        return True
    # Heuristic tail: one-line print(json.dumps({...sandbox...}))
    if (
        "import json" in s
        and "print(json.dumps(" in s
        and "sandbox" in s
        and s.count("\n") <= 4
    ):
        return True
    return False


def _run_inline_trusted_sandbox_code(code: str) -> dict[str, Any]:
    """Execute trusted stub in-process (no Docker). Returns same shape as run_sandboxed_python."""
    import io
    import contextlib

    buf = io.StringIO()
    g: dict[str, Any] = {"json": json}
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<inline_stub>", "exec"), g, g)
    except Exception as exc:
        return {
            "ok": False,
            "error_type": "inline_stub_error",
            "message": str(exc),
            "stdout": buf.getvalue(),
            "stderr": "",
        }
    out = buf.getvalue()
    parsed = None
    for ln in reversed([x.strip() for x in out.splitlines() if x.strip()]):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                parsed = json.loads(ln)
                break
            except json.JSONDecodeError:
                continue
    if parsed is None and isinstance(g.get("result"), dict):
        parsed = g["result"]
    ok = isinstance(parsed, dict) and parsed.get("sandbox") == "ok"
    return {"ok": ok, "stdout": out, "stderr": "", "parsed": parsed}


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


_ACCURACY_KEY_SUBSTR = (
    "val_accuracy",
    "test_accuracy",
    "validation_accuracy",
    "accuracy",
    "val_acc",
    "test_acc",
)


def _primary_metric_from_tool_output(out: dict[str, Any]) -> float | None:
    """
    Pick a scalar metric from tool JSON without grabbing the first arbitrary number
    (e.g. p_value, learning_rate), which previously drove bogus ~0.01 'accuracies'.
    """
    if not isinstance(out, dict):
        return None
    for k in (
        "val_accuracy",
        "test_accuracy",
        "validation_accuracy",
        "accuracy",
        "metric_value",
        "best_val_accuracy",
        "final_val_accuracy",
    ):
        v = out.get(k)
        fv: float | None = None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            fv = float(v)
        elif isinstance(v, str):
            vs = v.strip().replace("%", "")
            try:
                fv = float(vs)
            except ValueError:
                fv = None
        if fv is not None:
            fv = fv / 100.0 if fv > 1.001 and fv <= 100.0 else fv
            if 0.0 <= fv <= 1.0:
                return fv
    best_hint: float | None = None
    for path, val in _walk_numeric_values(out).items():
        pl = path.lower()
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        fv = float(val)
        if any(s in pl for s in _ACCURACY_KEY_SUBSTR) and fv > 1.01 and fv <= 100.0:
            fv = fv / 100.0
        if fv < 0.0 or fv > 1.0:
            continue
        if any(s in pl for s in _ACCURACY_KEY_SUBSTR):
            return fv
        if 0.2 <= fv <= 1.0:
            best_hint = fv if best_hint is None else max(best_hint, fv)
    return best_hint


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
        cand: float | None = None
        if isinstance(out, dict):
            cand = _primary_metric_from_tool_output(out)
        if cand is not None:
            metric_steps += 1
            metric_value = cand
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
    campaign_node_id: str | None = payload.get("campaign_node_id")
    hypothesis: dict = payload["hypothesis"]
    question: str = str(payload.get("question") or "")
    peer_findings: list[dict] = payload.get("peer_findings") or []
    learned_from: str | None = payload.get("learned_from") or None
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    baseline_lit_compare_safe = bool(baseline.get("lit_compare_safe", False))
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
    last_tool_name: str | None = None
    last_tool_step_index: int | None = None

    try:
        await _update_hypothesis(session_factory, hypothesis_id, status="running")

        await emitter.emit(
            session_id=session_id,
            event_type=EventType.AGENT_STARTED,
            step=f"experiment.{hypothesis_id}",
            payload={"hypothesis_id": hypothesis_id, "text": hypothesis.get("text")},
            hypothesis_id=hypothesis_id,
        )

        logger.info(
            "[sub_agent] startup session_id=%s hypothesis_id=%s PROPAB_PROFILE=%s "
            "sandbox_code_max_retries=%s sandbox_after_timeout_llm_rewrite=%s",
            session_id,
            hypothesis_id,
            (os.environ.get("PROPAB_PROFILE") or "").strip(),
            int(getattr(settings, "sandbox_code_max_retries", 1)),
            bool(getattr(settings, "sandbox_after_timeout_llm_rewrite", True)),
        )

        fast_path = str(payload.get("fast_path") or "").strip().lower()
        if fast_path == "baseline_measurement":
            bm_cfg = payload.get("baseline_measurement") if isinstance(payload.get("baseline_measurement"), dict) else {}
            ds = str(bm_cfg.get("dataset") or "mnist").strip() or "mnist"
            n_bm = max(
                20,
                min(
                    int(bm_cfg.get("n_steps") or settings.campaign_baseline_max_train_steps),
                    int(getattr(settings, "campaign_baseline_max_train_steps", 150)),
                ),
            )
            metric_key = str(baseline.get("metric_name") or "val_accuracy").strip() or "val_accuracy"
            params = {
                "model_id": "auto",
                "dataset": ds,
                "n_steps": n_bm,
                "task": "classification",
            }
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.TOOL_CALLED,
                step=f"experiment.{hypothesis_id}.baseline_fast",
                payload={"tool": "train_model", "params": params, "fast_path": True},
                hypothesis_id=hypothesis_id,
            )
            result = registry.call("train_model", params)
            if result.success and isinstance(result.output, dict):
                metric_val = _primary_metric_from_tool_output(result.output)
                evidence = (
                    "fast baseline measurement via train_model; "
                    f"dataset={ds}; n_steps={n_bm}; metric={metric_key}; "
                    f"metric_value={metric_val}."
                )
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.TOOL_RESULT,
                    step=f"experiment.{hypothesis_id}.baseline_fast",
                    payload={"tool": "train_model", "output": result.output, "fast_path": True},
                    hypothesis_id=hypothesis_id,
                )
                await _update_hypothesis(
                    session_factory,
                    hypothesis_id,
                    status="completed",
                    verdict="inconclusive",
                    confidence=0.0,
                    evidence_summary=evidence,
                    key_finding=None,
                    tool_trace_id=trace_pointer,
                )
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_COMPLETED,
                    step=f"experiment.{hypothesis_id}.complete",
                    payload={"verdict": "inconclusive", "confidence": 0.0, "fast_path": fast_path},
                    hypothesis_id=hypothesis_id,
                )
                await redis.close()
                await engine.dispose()
                out = {
                    "hypothesis_id": hypothesis_id,
                    "campaign_node_id": campaign_node_id,
                    "verdict": "inconclusive",
                    "confidence": 0.0,
                    "evidence_summary": evidence,
                    "key_finding": None,
                    "tool_trace_id": trace_pointer,
                    "figures": [],
                    "duration_sec": round(time.perf_counter() - started, 3),
                    "failure_reason": None,
                    "learned": "Fast baseline measured with train_model.",
                    "metric_value": metric_val,
                    "baseline_value": baseline_value,
                }
                if metric_val is not None:
                    out[metric_key] = metric_val
                return out

            err = result.error.to_dict() if result.error else {"type": "tool_error", "message": "unknown"}
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.TOOL_ERROR,
                step=f"experiment.{hypothesis_id}.baseline_fast",
                payload={"tool": "train_model", "failure_kind": "fast_baseline_failed", "error": err},
                hypothesis_id=hypothesis_id,
            )
            raise RuntimeError(f"fast baseline measurement failed: {err}")

        hyp_text = str(hypothesis.get("text", ""))
        plan_source = (settings.sub_agent_plan_source or "heuristic").strip().lower()
        payload_domain = str(payload.get("domain") or "").strip()
        if plan_source == "heuristic" and payload_domain:
            domain = coerce_routed_domain(payload_domain)
            domain_reason = "Using orchestrator-provided domain in heuristic smoke mode."
        else:
            domain, domain_reason = await route_domain(
                hypothesis_text=hyp_text,
                llm=llm,
                session_id=session_id,
                hypothesis_id=hypothesis_id,
                question=question,
            )
        sandbox_timeout_sec = effective_sandbox_timeout_sec(
            domain,
            settings.sandbox_timeout_sec,
            use_domain_floor=bool(getattr(settings, "sandbox_use_domain_timeout_floor", True)),
        )

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

        can_llm = settings.llm_provider.strip().lower() == "ollama" or bool(settings.llm_api_secret.strip())
        use_think_act = plan_source in ("llm", "hybrid") and can_llm

        _alimits = payload.get("agent_limits") if isinstance(payload.get("agent_limits"), dict) else {}

        def _agent_lim(name: str, fallback: int) -> int:
            raw = _alimits.get(name)
            if raw is None:
                return int(fallback)
            try:
                return int(raw)
            except (TypeError, ValueError):
                return int(fallback)

        agent_max_seconds_eff = max(30, _agent_lim("max_seconds", int(settings.agent_max_seconds)))
        agent_max_steps = max(_agent_lim("max_steps", int(settings.agent_max_steps)), 5)
        agent_min_steps = max(1, min(int(settings.agent_min_steps), agent_max_steps - 1))
        max_tc = max(0, _agent_lim("max_tool_calls", int(getattr(settings, "agent_max_tool_calls", 0) or 0)))

        logger.info(
            "[sub_agent] hypothesis_id=%s sandbox_code_max_retries=%s sandbox_after_timeout_llm_rewrite=%s "
            "agent_max_steps=%s agent_max_seconds=%s agent_max_tool_calls=%s",
            hypothesis_id,
            int(getattr(settings, "sandbox_code_max_retries", 1)),
            bool(getattr(settings, "sandbox_after_timeout_llm_rewrite", True)),
            agent_max_steps,
            agent_max_seconds_eff,
            max_tc,
        )

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
                "agent_max_seconds": agent_max_seconds_eff,
                "agent_max_tool_calls": max_tc,
                "heuristic_steps": [tn for tn, _ in heuristic_steps],
            },
            hypothesis_id=hypothesis_id,
        )

        sandbox_ok = False
        any_tool_success = False
        successful_tool_outputs: list[dict[str, Any]] = []
        successful_tool_names: list[str] = []
        last_code_output: list[dict[str, Any]] = []
        step_counter = 0
        tool_calls_done = 0
        err_box: list[Any] = [None]

        # ── Shared tool execution helper (inline, no external function needed) ─

        async def run_tool_step(tool_name: str, params: dict, step_index: int) -> bool:
            nonlocal any_tool_success, last_tool_name, last_tool_step_index, tool_calls_done
            err_box[0] = None
            last_tool_name = tool_name
            last_tool_step_index = step_index
            step_id = str(uuid4())
            t0 = time.perf_counter()

            if max_tc > 0 and tool_calls_done >= max_tc:
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.TOOL_ERROR,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={
                        "tool": tool_name,
                        "failure_kind": "agent_max_tool_calls",
                        "error": {
                            "type": "budget_exceeded",
                            "detail": (
                                f"agent_max_tool_calls={max_tc} exhausted; refusing further tools."
                            ),
                        },
                    },
                    hypothesis_id=hypothesis_id,
                )
                return False
            tool_calls_done += 1

            call_params = dict(params or {})
            n_steps_cap = max(0, int(getattr(settings, "agent_tool_n_steps_cap", 0) or 0))
            if n_steps_cap > 0:
                for k in ("n_steps", "epochs", "num_steps", "num_epochs", "steps", "n_epochs", "training_steps"):
                    if k in call_params:
                        try:
                            call_params[k] = min(int(call_params[k]), n_steps_cap)
                        except (TypeError, ValueError):
                            pass
            if (
                tool_name == "literature_baseline_compare"
                and call_params.get("baseline_value") is None
                and baseline_lit_compare_safe
                and baseline_value is not None
                and abs(float(baseline_value)) >= 1e-12
            ):
                call_params["baseline_value"] = float(baseline_value)
            if tool_name == "literature_baseline_compare" and call_params.get("baseline_value") is not None:
                try:
                    if abs(float(call_params["baseline_value"])) < 1e-12:
                        call_params.pop("baseline_value", None)
                except (TypeError, ValueError):
                    pass

            await emitter.emit(
                session_id=session_id,
                event_type=EventType.TOOL_CALLED,
                step=f"experiment.{hypothesis_id}.step_{step_index}",
                payload={"tool": tool_name, "params": call_params},
                hypothesis_id=hypothesis_id,
            )
            try:
                result = registry.call(tool_name, call_params)
            except TypeError as exc:
                result = ToolResult(
                    success=False,
                    error=ToolError(type="validation_error", message=str(exc)),
                )
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
                err_d = result.error.to_dict() if result.error else {}
                err_box[0] = err_d
                fk = "tool_execution_error"
                et = str((err_d or {}).get("type") or "")
                if et in ("validation_error", "missing_dependency", "auto_build_error", "zero_variance"):
                    fk = et
                elif "timeout" in str((err_d or {}).get("message", "")).lower():
                    fk = "tool_timeout_or_resource"
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.TOOL_ERROR,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={"tool": tool_name, "failure_kind": fk, "error": err_d},
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
                params=call_params,
                result_output=result.output,
                result_error=result.error.to_dict() if result.error else None,
                duration_ms=duration_ms,
            )
            return bool(result.success)

        async def run_code_step(
            code: str,
            step_index: int,
            *,
            force_inline_trusted: bool = False,
        ) -> bool:
            nonlocal sandbox_ok
            del last_code_output[:]
            step_id = str(uuid4())
            t0 = time.perf_counter()
            code_cur = code
            parsed = None
            sandbox_out: dict = {}
            rewrite_used = False

            await emitter.emit(
                session_id=session_id,
                event_type=EventType.CODE_GENERATED,
                step=f"experiment.{hypothesis_id}.step_{step_index}",
                payload={"code": code_cur, "rewrite_after_timeout": False},
                hypothesis_id=hypothesis_id,
            )

            sandbox_ok = False
            trust_heuristic = _is_trusted_inline_sandbox_code(code_cur)
            use_inline = bool(force_inline_trusted) or trust_heuristic
            if use_inline:
                forced = bool(force_inline_trusted) and not trust_heuristic
                exec_label = "inline_stub_forced" if forced else "inline_stub"
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.CODE_SUBMITTED,
                    step=f"experiment.{hypothesis_id}.step_{step_index}",
                    payload={
                        "memory_mb": settings.sandbox_memory_mb,
                        "timeout_sec": sandbox_timeout_sec,
                        "domain": domain,
                        "attempt": 1,
                        "max_attempts": 1,
                        "llm_rewrite_slot": False,
                        "execution": exec_label,
                        "note": (
                            "Programmatic stub — in-process (no Docker); forced because "
                            "description tripped substring heuristics or bypasses comment injection."
                            if forced
                            else "Trusted agent stub — in-process (no Docker) to avoid spurious sandbox timeouts."
                        ),
                    },
                    hypothesis_id=hypothesis_id,
                )
                sandbox_out = await asyncio.to_thread(_run_inline_trusted_sandbox_code, code_cur)
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
                            "attempt": 1,
                            "rewrite_after_timeout": False,
                            "execution": exec_label,
                        },
                        hypothesis_id=hypothesis_id,
                    )
                else:
                    await emitter.emit(
                        session_id=session_id,
                        event_type=EventType.CODE_ERROR,
                        step=f"experiment.{hypothesis_id}.step_{step_index}",
                        payload={
                            "failure_kind": "inline_stub_error",
                            "error": sandbox_out,
                            "attempt": 1,
                            "max_attempts": 1,
                        },
                        hypothesis_id=hypothesis_id,
                    )
            else:
                # Never burn N × sandbox wall on the same source: the old
                # ``while inv < hard_cap`` loop replayed identical code up to
                # ``sandbox_code_max_retries`` times after each timeout (45 submits /
                # 15 generations in campaigns). We allow at most:
                #   1) one Docker run of the model's code, then
                #   2) one optional second run *only* after an LLM rewrite (different source).
                # Docker path: total executions per ``code.generated`` are capped by
                # ``sandbox_code_max_retries`` (interpreted as max Docker runs, minimum 1).
                # When 1, skip the post-timeout rewrite second run entirely.
                allow_rewrite = bool(getattr(settings, "sandbox_after_timeout_llm_rewrite", True)) and can_llm
                max_docker = max(1, int(getattr(settings, "sandbox_code_max_retries", 1) or 1))
                allow_second = max_docker >= 2 and allow_rewrite
                planned_max = min(max_docker, 1 + (1 if allow_second else 0))
                for exec_n in range(1, planned_max + 1):
                    await emitter.emit(
                        session_id=session_id,
                        event_type=EventType.CODE_SUBMITTED,
                        step=f"experiment.{hypothesis_id}.step_{step_index}",
                        payload={
                            "memory_mb": settings.sandbox_memory_mb,
                            "timeout_sec": sandbox_timeout_sec,
                            "domain": domain,
                            "attempt": exec_n,
                            "max_attempts": planned_max,
                            "llm_rewrite_slot": allow_second,
                        },
                        hypothesis_id=hypothesis_id,
                    )
                    sandbox_out = await asyncio.to_thread(
                        run_sandboxed_python,
                        code_cur,
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
                                "attempt": exec_n,
                                "rewrite_after_timeout": rewrite_used,
                            },
                            hypothesis_id=hypothesis_id,
                        )
                        break
                    is_timeout = _is_sandbox_wall_timeout(sandbox_out)
                    ev = EventType.CODE_TIMEOUT if is_timeout else EventType.CODE_ERROR
                    await emitter.emit(
                        session_id=session_id,
                        event_type=ev,
                        step=f"experiment.{hypothesis_id}.step_{step_index}",
                        payload={
                            "failure_kind": "sandbox_timeout" if is_timeout else "sandbox_error",
                            "timeout_sec": sandbox_timeout_sec,
                            "memory_mb": settings.sandbox_memory_mb,
                            "error": sandbox_out,
                            "attempt": exec_n,
                            "max_attempts": planned_max,
                            **_sandbox_diag_tails(sandbox_out),
                        },
                        hypothesis_id=hypothesis_id,
                    )
                    if (
                        exec_n < planned_max
                        and is_timeout
                        and allow_second
                        and looks_like_heavy_training_code(code_cur)
                    ):
                        new_code = await rewrite_sandbox_code_after_timeout(
                            llm,
                            session_id=session_id,
                            hypothesis_id=hypothesis_id,
                            code=code_cur,
                            sandbox_timeout_sec=int(sandbox_timeout_sec),
                            domain=str(domain or ""),
                        )
                        if new_code and new_code.strip() != code_cur.strip():
                            code_cur = new_code
                            rewrite_used = True
                            await emitter.emit(
                                session_id=session_id,
                                event_type=EventType.CODE_GENERATED,
                                step=f"experiment.{hypothesis_id}.step_{step_index}",
                                payload={
                                    "code": code_cur,
                                    "rewrite_after_timeout": True,
                                    "after_attempt": exec_n,
                                },
                                hypothesis_id=hypothesis_id,
                            )
                            continue
                    break

            duration_ms = int((time.perf_counter() - t0) * 1000)
            await _insert_experiment_step_code(
                session_factory,
                step_id=step_id,
                hypothesis_id=hypothesis_id,
                step_index=step_index,
                code=code_cur,
                parsed_output=parsed,
                sandbox_out=sandbox_out,
                duration_ms=duration_ms,
                memory_mb=settings.sandbox_memory_mb,
                timeout_sec=sandbox_timeout_sec,
            )
            if sandbox_ok and isinstance(parsed, dict):
                last_code_output.append(parsed)
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
            deadline = time.monotonic() + max(30.0, float(agent_max_seconds_eff))
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
                deadline_monotonic=deadline,
                tool_failures=[],
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
                if max_tc > 0 and tool_calls_done >= max_tc:
                    logger.info(
                        "Agent %s hit agent_max_tool_calls=%s before next think step.",
                        hypothesis_id,
                        max_tc,
                    )
                    await emitter.emit(
                        session_id=session_id,
                        event_type=EventType.AGENT_STEP_COMPLETED,
                        step=f"experiment.{hypothesis_id}.budget",
                        payload={"reason": "agent_max_tool_calls", "limit": max_tc},
                        hypothesis_id=hypothesis_id,
                    )
                    break
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
                    sandbox_timeout_sec=int(sandbox_timeout_sec),
                    agent_wall_budget_sec=int(agent_max_seconds_eff),
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
                            payload={
                                "tool": tool_name_ta,
                                "failure_kind": "unknown_tool_name",
                                "error": {
                                    "type": "unknown_tool",
                                    "detail": "Tool name not in registry (bad plan / typo).",
                                },
                            },
                            hypothesis_id=hypothesis_id,
                        )
                    if step_ok and successful_tool_outputs:
                        agent_ctx.results_so_far.append(successful_tool_outputs[-1])
                        agent_ctx.tool_names_run.append(tool_name_ta)
                    elif not step_ok and err_box[0]:
                        agent_ctx.tool_failures.append({"tool": tool_name_ta, "error": err_box[0]})
                        if len(agent_ctx.tool_failures) > 12:
                            agent_ctx.tool_failures.pop(0)

                elif action.action_type == "code":
                    real_code = (action.code or "").strip()
                    if real_code:
                        # Real LLM-authored program → execute in the Docker sandbox.
                        step_ok = await run_code_step(real_code, step_counter)
                        if step_ok and last_code_output:
                            # Feed real computed output back so the LLM can chain on it and
                            # the significance/evidence layer can use real measurements.
                            out = last_code_output[-1]
                            agent_ctx.results_so_far.append(out)
                            successful_tool_outputs.append(out)
                            successful_tool_names.append("__code__")
                    else:
                        # No source supplied (description only) → deterministic no-op stub so the
                        # step is recorded without burning a Docker wall on an empty computation.
                        code_desc = action.code_description or "custom computation"
                        step_ok = await run_code_step(
                            _think_act_stub_code(str(code_desc)),
                            step_counter,
                            force_inline_trusted=True,
                        )
                    if step_ok:
                        agent_ctx.tool_names_run.append("__code__")

                step_counter += 1
                agent_ctx.steps_taken += 1

            if agent_ctx.time_budget_exceeded:
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.AGENT_TIME_BUDGET_EXCEEDED,
                    step=f"experiment.{hypothesis_id}.time_budget",
                    payload={
                        "agent_max_seconds": int(agent_max_seconds_eff),
                        "steps_taken": agent_ctx.steps_taken,
                        "tools_tail": agent_ctx.tool_names_run[-12:],
                        "reason": "Wall clock cap (settings.agent_max_seconds / PROPAB_PROFILE).",
                    },
                    hypothesis_id=hypothesis_id,
                )

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
                    await run_code_step(
                        str(step.get("code", "")),
                        step_index,
                        force_inline_trusted=True,
                    )
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

        bm_cfg = payload.get("baseline_measurement")
        if isinstance(bm_cfg, dict) and evidence_obj.get("metric_value") is None:
            ds = str(bm_cfg.get("dataset") or "mnist").strip() or "mnist"
            n_bm = max(50, min(int(bm_cfg.get("n_steps") or 150), 500))
            try:
                r_bm = registry.call(
                    "train_model",
                    {
                        "model_id": "auto",
                        "dataset": ds,
                        "n_steps": n_bm,
                        "task": "classification",
                    },
                )
                if r_bm.success and isinstance(r_bm.output, dict):
                    cand = _primary_metric_from_tool_output(r_bm.output)
                    if cand is not None:
                        fv_c = float(cand)
                        evidence_obj["metric_value"] = fv_c
                        evidence_obj["n_metric_steps"] = max(1, int(evidence_obj.get("n_metric_steps") or 0))
                        if baseline_value is not None:
                            evidence_obj["delta"] = fv_c - float(baseline_value)
                        logger.info(
                            "baseline_measurement fallback train_model(dataset=%s, n_steps=%s) "
                            "-> val_accuracy=%.4f",
                            ds,
                            n_bm,
                            float(cand),
                        )
            except Exception as exc_bm:
                logger.warning("baseline_measurement train_model fallback failed: %s", exc_bm)

        # Use significance module for the gate check
        sig_result = check_significance(successful_tool_outputs)

        verdict, verdict_reason = classify_verdict(
            evidence_obj,
            sig_result,
            min_metric_steps_for_confirm=int(getattr(settings, "min_metric_steps_for_confirm", 2)),
        )
        evidence_obj["verdict_reason"] = verdict_reason
        confidence = 0.0 if evidence_obj["n_metric_steps"] == 0 else _compute_confidence(evidence_obj)

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
        # The key finding must be the actual claim that was supported — not a generic
        # "significance gate passed" line — or the paper reads like an internal log.
        if verdict == "confirmed":
            claim = re.split(r"\s*\(Question:", str(hyp_text or "").strip())[0].strip()
            claim = re.sub(r"^Hypothesis\s+\d+\s*:\s*", "", claim).strip()
            key_finding = claim or "The hypothesis was supported by statistically significant evidence."
        elif sandbox_ok:
            key_finding = "Sandbox probe completed."
        else:
            key_finding = None

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
        metric_out = evidence_obj.get("metric_value")
        metric_key = str(baseline.get("metric_name") or "").strip()
        result_dict: dict[str, Any] = {
            "hypothesis_id": hypothesis_id,
            "campaign_node_id": campaign_node_id,
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
            "metric_value": metric_out,
            "baseline_value": evidence_obj.get("baseline_value"),
        }
        if metric_key and metric_out is not None:
            result_dict[metric_key] = metric_out

        await redis.close()
        await engine.dispose()
        return result_dict

    except Exception as exc:
        detail = classify_exception(
            exc,
            hint_tool=last_tool_name,
            hint_step_index=last_tool_step_index,
        )
        await _update_hypothesis(
            session_factory,
            hypothesis_id,
            status="failed",
            verdict="inconclusive",
            confidence=0.0,
            evidence_summary=json.dumps(detail, ensure_ascii=False)[:8000],
            key_finding=None,
            tool_trace_id=trace_pointer,
        )
        fail_payload = {
            **detail,
            "error": str(exc)[:1500],
        }
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.AGENT_FAILED,
            step=f"experiment.{hypothesis_id}.failed",
            payload=fail_payload,
            hypothesis_id=hypothesis_id,
        )
        await redis.close()
        await engine.dispose()
        return {
            "hypothesis_id": hypothesis_id,
            "campaign_node_id": campaign_node_id,
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
