"""Pick tool(s) from a domain cluster using hypothesis text overlap (no extra LLM call)."""

from __future__ import annotations

import re
from typing import Any


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", text.lower()) if t}


def score_spec_relevance(hypothesis_text: str, spec: dict[str, Any]) -> float:
    """Higher = better match. Uses name + description + param keys."""
    hyp = _tokens(hypothesis_text)
    if not hyp:
        return 0.0
    blob = " ".join(
        [
            str(spec.get("name", "")),
            str(spec.get("description", "")),
            " ".join(str(k) for k in (spec.get("params") or {}).keys()),
        ]
    ).lower()
    spec_toks = _tokens(blob)
    inter = len(hyp & spec_toks)
    name = str(spec.get("name", ""))
    return float(inter) + 0.01 / (1 + len(name))


def _fallback_params(hypothesis: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return (
        "json_extract",
        {"data": {"hypothesis_rank": hypothesis.get("rank"), "label": "probe"}, "key": "label"},
    )


def select_tool_steps(
    specs: list[dict[str, Any]],
    *,
    hypothesis_text: str,
    hypothesis: dict[str, Any],
    max_tools: int = 2,
) -> list[tuple[str, dict[str, Any]]]:
    """
    Up to ``max_tools`` distinct tools, in relevance order, each with non-empty ``example.params``.
    """
    if max_tools < 1:
        return []
    if not specs:
        return [_fallback_params(hypothesis)][:max_tools]
    ranked = sorted(specs, key=lambda s: score_spec_relevance(hypothesis_text, s), reverse=True)
    out: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()
    for spec in ranked:
        if len(out) >= max_tools:
            break
        name = str(spec.get("name", ""))
        if not name or name in used:
            continue
        params = (spec.get("example") or {}).get("params")
        if not isinstance(params, dict) or not params:
            continue
        used.add(name)
        out.append((name, params))
    if not out:
        return [_fallback_params(hypothesis)][:max_tools]
    return out


def select_tool_and_params(
    specs: list[dict[str, Any]],
    *,
    hypothesis_text: str,
    hypothesis: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Single best tool + params (backward compatible)."""
    steps = select_tool_steps(
        specs,
        hypothesis_text=hypothesis_text,
        hypothesis=hypothesis,
        max_tools=1,
    )
    return steps[0]
