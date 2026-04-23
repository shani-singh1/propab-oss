from __future__ import annotations

import json
import re

from propab.llm import LLMClient

DOMAIN_ROUTING_PROMPT = """
Given this hypothesis, return the single most relevant domain for Propab v1 (DL / algorithms / ML focus).

Hypothesis: {hypothesis_text}

Domains (choose exactly one name from this list):
- deep_learning: neural networks, transformers, training, attention, PyTorch workflows
- algorithm_optimization: gradient methods, complexity, benchmarking, convergence, loss surfaces
- ml_research: statistical tests, bootstrap CIs, experiment grids, FLOPs / reproducibility analysis
- mathematics: formal math, symbolic computation, linear algebra proofs
- statistics: classical stats, regression, inference (not deep nets)
- data_analysis: EDA, aggregation, cleaning, tabular summaries
- general_computation: anything else — sandboxed code, parsing, glue logic

Return JSON only: {{"domain": "domain_name", "reason": "one sentence"}}
"""

# v1 routing: DL / ALGO / ML first; coerce legacy or out-of-list labels.
_ALLOWED = frozenset(
    {
        "deep_learning",
        "algorithm_optimization",
        "ml_research",
        "mathematics",
        "statistics",
        "data_analysis",
        "general_computation",
    },
)
_LEGACY_TO_PRIMARY: dict[str, str] = {
    "ml_modeling": "deep_learning",
    "computational_biology": "general_computation",
    "chemistry": "general_computation",
    "physics": "general_computation",
}


def coerce_routed_domain(raw: str) -> str:
    d = (raw or "").strip().lower().replace("-", "_")
    if d in _ALLOWED:
        return d
    return _LEGACY_TO_PRIMARY.get(d, "general_computation")


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}\s*$", text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


async def route_domain(
    *,
    hypothesis_text: str,
    llm: LLMClient,
    session_id: str,
    hypothesis_id: str,
) -> tuple[str, str]:
    prompt = DOMAIN_ROUTING_PROMPT.format(hypothesis_text=hypothesis_text)
    raw = await llm.call(
        prompt=prompt,
        purpose="domain_routing",
        session_id=session_id,
        hypothesis_id=hypothesis_id,
    )
    data = _extract_json(raw) or {}
    domain = coerce_routed_domain(str(data.get("domain", "general_computation")))
    reason = str(data.get("reason", "Default domain after routing.")).strip()
    return domain, reason
