"""Pick a tool from a domain cluster using hypothesis text overlap (no extra LLM call)."""

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
    # Mild preference for shorter tool names (often core utilities) on ties
    name = str(spec.get("name", ""))
    return float(inter) + 0.01 / (1 + len(name))


def select_tool_and_params(
    specs: list[dict[str, Any]],
    *,
    hypothesis_text: str,
    hypothesis: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    Prefer specs with non-empty TOOL_SPEC example.params, ranked by hypothesis relevance.
    Fallback: ``json_extract`` probe (always valid).
    """
    fallback = (
        "json_extract",
        {"data": {"hypothesis_rank": hypothesis.get("rank"), "label": "probe"}, "key": "label"},
    )
    if not specs:
        return fallback
    ranked = sorted(specs, key=lambda s: score_spec_relevance(hypothesis_text, s), reverse=True)
    for spec in ranked[:12]:
        example = spec.get("example") or {}
        params = example.get("params")
        if isinstance(params, dict) and params:
            return str(spec["name"]), params
    return fallback
