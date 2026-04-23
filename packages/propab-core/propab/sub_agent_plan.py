"""Optional LLM-authored multi-step tool plans for the experiment sub-agent."""

from __future__ import annotations

import json
import re
from typing import Any

from propab.llm import LLMClient
from propab.types import EventType


def _allowed_tool_names(specs: list[dict[str, Any]]) -> set[str]:
    return {str(s.get("name", "")) for s in specs if s.get("name")}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", t)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def parse_llm_plan_steps(
    raw: str,
    *,
    specs: list[dict[str, Any]],
    max_steps: int,
) -> list[tuple[str, dict[str, Any]]] | None:
    """
    Parse ``{\"steps\": [{\"type\":\"tool\",\"tool\":\"...\",\"params\":{...}}, ...]}``.
    Only ``type==\"tool\"`` entries are kept; tools must appear in ``specs``.
    """
    allowed = _allowed_tool_names(specs)
    spec_by_name = {str(s.get("name", "")): s for s in specs if s.get("name")}
    data = _extract_json_object(raw)
    if not data:
        return None
    steps_in = data.get("steps")
    if not isinstance(steps_in, list):
        return None
    out: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()
    for step in steps_in:
        if len(out) >= max(1, max_steps):
            break
        if not isinstance(step, dict) or step.get("type") != "tool":
            continue
        name = str(step.get("tool", "")).strip()
        if not name or name not in allowed or name in used:
            continue
        params = step.get("params")
        if not isinstance(params, dict):
            continue
        spec = spec_by_name.get(name) or {}
        param_spec = spec.get("params") or {}
        ok = True
        for pk, pv in params.items():
            if pk not in param_spec:
                ok = False
                break
            pentry = param_spec.get(pk)
            req = bool(pentry.get("required", False)) if isinstance(pentry, dict) else False
            if req and (pv is None or pv == ""):
                ok = False
                break
        if not ok:
            continue
        used.add(name)
        out.append((name, params))
    return out or None


async def build_tool_plan_via_llm(
    *,
    llm: LLMClient,
    session_id: str,
    hypothesis_id: str,
    hypothesis_text: str,
    specs: list[dict[str, Any]],
    max_steps: int,
    emitter: Any,
) -> list[tuple[str, dict[str, Any]]] | None:
    """Ask the LLM for an ordered tool plan; return None on failure (caller uses heuristic)."""
    names = sorted(_allowed_tool_names(specs))
    if not names:
        return None
    brief = []
    for s in specs[:40]:
        n = str(s.get("name", ""))
        if not n:
            continue
        brief.append(f"- {n}: {str(s.get('description', ''))[:200]}")
    prompt = (
        "You are planning executable tool steps to probe a research hypothesis.\n"
        "Return ONLY a JSON object (no markdown) of the form:\n"
        '{"steps":[{"type":"tool","tool":"<name>","params":{...}}, ...]}\n'
        f"Use between 1 and {max_steps} steps. Each \"tool\" must be one of: {json.dumps(names)}.\n"
        "Each \"params\" object must only use keys declared in that tool's schema; "
        "supply concrete values suitable for a quick experiment.\n\n"
        f"Hypothesis:\n{hypothesis_text[:4000]}\n\n"
        "Tool descriptions:\n" + "\n".join(brief[:40])
    )
    try:
        raw = await llm.call(
            prompt=prompt,
            purpose="experiment_plan",
            session_id=session_id,
            hypothesis_id=hypothesis_id,
        )
    except Exception:
        return None
    parsed = parse_llm_plan_steps(raw, specs=specs, max_steps=max_steps)
    if parsed is None:
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.LLM_PARSE_ERROR,
            step=f"experiment.{hypothesis_id}.plan_llm",
            payload={"purpose": "experiment_plan", "snippet": (raw or "")[:800]},
            hypothesis_id=hypothesis_id,
        )
    return parsed
