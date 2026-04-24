"""Decide whether a session merits a full paper (vs trace-only completion)."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.config import settings

# Tools that typically emit measurable experiment-like outputs (not pure I/O probes).
SUBSTANTIVE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "train_model",
        "evaluate_model",
        "compare_optimizers",
        "compare_gradient_methods",
        "literature_baseline_compare",
        "statistical_significance",
        "run_experiment_grid",
        "convergence_analysis",
        "loss_landscape",
        "lr_range_test",
        "hyperparameter_sweep",
        "compare_implementations",
        "compare_attention_variants",
        "activation_statistics",
        "inspect_gradients",
        "profile_model",
        "compute_flops",
        "hessian_analysis",
        "reproduce_result",
    }
)

_METRIC_MARKERS: frozenset[str] = frozenset(
    (
        "improvement",
        "p_value",
        "baseline",
        "accuracy",
        "loss",
        "epoch",
        "effect",
        "significant",
        "confidence_interval",
        "our_mean",
        "ranking",
        "convergence",
        "final_val",
        "train_loss",
    )
)


def _json_blob(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False).lower()
    except (TypeError, ValueError):
        return str(obj).lower()


def output_suggests_metrics(output: Any) -> bool:
    """True if serialized output mentions several quantitative / comparison signals."""
    blob = _json_blob(output)
    return sum(1 for m in _METRIC_MARKERS if m in blob) >= 2


def merit_from_ledger(ledger: dict[str, Any] | None) -> tuple[bool, str]:
    """Authoritative hypothesis outcomes from the orchestrator ledger."""
    if not isinstance(ledger, dict):
        return False, "missing_ledger"
    if len(ledger.get("confirmed") or []) >= 1:
        return True, "confirmed_hypotheses"
    if len(ledger.get("refuted") or []) >= 1:
        return True, "refuted_hypotheses"
    return False, "ledger_has_no_confirmed_or_refuted"


def _tool_step_failed(error_json: Any) -> bool:
    if error_json is None:
        return False
    if isinstance(error_json, dict):
        return bool(error_json)
    if isinstance(error_json, str):
        t = error_json.strip()
        if t in ("", "null", "{}"):
            return False
        try:
            return bool(json.loads(t))
        except json.JSONDecodeError:
            return True
    return bool(error_json)


def merit_from_trace_rows(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    """
    Inspect persisted experiment steps for substantive tool depth or metric-like outputs.
    ``rows`` are mappings with keys like step_type, input_json, output_json, error_json.
    """
    substantive_hits = 0
    metric_like = False
    for row in rows:
        if (row.get("step_type") or "") != "tool_call":
            continue
        inj = row.get("input_json")
        if isinstance(inj, str):
            try:
                inj = json.loads(inj)
            except json.JSONDecodeError:
                inj = {}
        tool = str((inj or {}).get("tool") or "")
        if _tool_step_failed(row.get("error_json")):
            continue
        out = row.get("output_json")
        if out is None or out == "null":
            continue
        if tool in SUBSTANTIVE_TOOL_NAMES:
            substantive_hits += 1
        if output_suggests_metrics(out):
            metric_like = True
    min_tools = max(1, int(settings.paper_min_substantive_tools))
    if metric_like:
        return True, "trace_contains_metric_like_tool_outputs"
    if substantive_hits >= min_tools:
        return True, f"trace_has_{substantive_hits}_substantive_tool_successes"
    return False, "trace_lacks_substantive_metrics_and_depth"


async def session_merits_paper(
    session_factory: async_sessionmaker,
    session_id: str,
    *,
    ledger: dict[str, Any] | None,
) -> tuple[bool, str]:
    """
    Return (merits, reason_code).

    * ``paper_policy=always`` — any non-empty trace is enough (caller still enforces non-zero steps).
    * ``strict_confirmed`` — only confirmed ledger rows.
    * ``substantive`` (default) — confirmed/refuted ledger, or trace depth / metric signals.
    """
    policy = (settings.paper_policy or "substantive").strip().lower()
    if policy == "always":
        return True, "policy_always"

    if policy == "strict_confirmed":
        if not isinstance(ledger, dict):
            return False, "missing_ledger"
        if len(ledger.get("confirmed") or []) >= 1:
            return True, "confirmed_hypotheses"
        return False, "strict_confirmed_requires_confirmed_hypothesis"

    ok_ledger, reason_ledger = merit_from_ledger(ledger)
    if ok_ledger:
        return True, reason_ledger

    if policy != "substantive":
        return False, f"unknown_paper_policy:{policy}"

    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT e.step_type, e.input_json, e.output_json, e.error_json
                    FROM experiment_steps e
                    JOIN hypotheses h ON h.id = e.hypothesis_id
                    WHERE h.session_id = CAST(:sid AS uuid)
                    ORDER BY e.created_at ASC
                    """
                ),
                {"sid": session_id},
            )
        ).mappings().all()
    trace_rows = [dict(r) for r in rows]
    ok_trace, reason_trace = merit_from_trace_rows(trace_rows)
    if ok_trace:
        return True, reason_trace
    return False, f"{reason_ledger};{reason_trace}"


def short_circuit_merits_paper() -> tuple[bool, str]:
    """Literature-only path has no experiments; never treat as a research paper."""
    policy = (settings.paper_policy or "substantive").strip().lower()
    if policy == "always":
        return True, "policy_always_allows_literature_note"
    return False, "literature_short_circuit_no_experiments"
