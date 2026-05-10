"""Classify worker/sub-agent failures for SSE payloads and observability."""

from __future__ import annotations

import asyncio
from typing import Any


def _chain_parts(exc: BaseException) -> tuple[list[str], list[str], str]:
    """Exception type names, messages, and lowercase blob for substring rules."""
    names: list[str] = []
    msgs: list[str] = []
    cur: BaseException | None = exc
    visited: set[int] = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        names.append(type(cur).__name__)
        msgs.append(str(cur))
        cur = cur.__cause__
    blob = "; ".join(f"{n}: {m}" for n, m in zip(names, msgs)).lower()
    return names, msgs, blob


def classify_exception(
    exc: BaseException,
    *,
    hint_tool: str | None = None,
    hint_step_index: int | None = None,
) -> dict[str, Any]:
    """
    Structured fields for AGENT_FAILED (and similar) events.

    ``failure_kind``: celery_soft_time_limit, celery_hard_time_limit, http_timeout,
    async_timeout, timeout_generic, mnist_dependency, missing_dependency, cancelled, unexpected, ...
    """
    names, msgs, blob = _chain_parts(exc)

    failure_kind = "unexpected"
    if "SoftTimeLimitExceeded" in names:
        failure_kind = "celery_soft_time_limit"
    elif "TimeLimitExceeded" in names:
        failure_kind = "celery_hard_time_limit"
    elif isinstance(exc, asyncio.CancelledError):
        failure_kind = "cancelled"
    elif isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        failure_kind = "async_timeout"
    elif "ReadTimeout" in names:
        failure_kind = "http_timeout"
    else:
        try:
            import httpx

            if isinstance(exc, httpx.TimeoutException):
                failure_kind = "http_timeout"
        except ImportError:
            pass

    if failure_kind == "unexpected" and any(
        n.endswith("TimeoutError") or n.endswith("Timeout") for n in names
    ):
        failure_kind = "http_timeout"

    if failure_kind == "unexpected" and "timeout" in blob:
        failure_kind = "timeout_generic"

    if failure_kind == "unexpected" and conn_err(blob):
        failure_kind = "http_connect"

    if failure_kind == "unexpected" and "mnist load failed" in blob:
        failure_kind = "mnist_dependency"
    if failure_kind == "unexpected" and (
        "pytorch required" in blob or "missing_dependency" in blob
    ):
        failure_kind = "missing_dependency"

    out: dict[str, Any] = {
        "failure_kind": failure_kind,
        "exc_types": ",".join(names[:8]),
        "message": "; ".join(msgs)[:2000],
    }
    if hint_tool:
        out["last_tool"] = hint_tool
    if hint_step_index is not None:
        out["last_step_index"] = hint_step_index
    return out


def conn_err(blob: str) -> bool:
    return (
        "connecterror" in blob
        or "connection refused" in blob
        or "name or service not known" in blob
        or "temporary failure in name resolution" in blob
    )


def compact_failure_summary(payload: Any) -> str:
    """Short suffix from event payload_json dict for textual logs."""
    if not isinstance(payload, dict):
        return ""
    kind = str(payload.get("failure_kind") or "").strip()
    tool = str(payload.get("last_tool") or "").strip()
    ts = (str(payload.get("exc_types") or "").split(",") or [""])[0].strip()
    bits: list[str] = []
    if kind:
        bits.append(kind)
    if ts and ts not in ("Exception", ""):
        bits.append(ts)
    if tool:
        bits.append(f"@{tool}")
    msg = str(payload.get("message") or "").replace("\n", " ").strip()
    if msg and len(bits) < 3:
        bits.append(msg[:100] + ("…" if len(msg) > 100 else ""))
    return "[" + " | ".join(bits) + "]" if bits else ""
