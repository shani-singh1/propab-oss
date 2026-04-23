from __future__ import annotations

import json
import re

from propab.llm import LLMClient

DOMAIN_ROUTING_PROMPT = """
Given this hypothesis, return the single most relevant domain.

Hypothesis: {hypothesis_text}

Domains:
- computational_biology: DNA/RNA/protein, genomics, CRISPR, cell biology
- chemistry: molecular simulation, reaction prediction, spectroscopy
- physics: mechanics, electromagnetism, quantum, thermodynamics
- mathematics: algebra, calculus, linear algebra, number theory, topology
- statistics: hypothesis testing, regression, Bayesian analysis, experimental design
- ml_modeling: training models, evaluation, feature engineering, architecture search
- data_analysis: EDA, visualization, aggregation, cleaning
- general_computation: anything else — code execution, data fetching, parsing

Return JSON only: {{"domain": "domain_name", "reason": "one sentence"}}
"""


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
    domain = str(data.get("domain", "general_computation")).strip() or "general_computation"
    reason = str(data.get("reason", "Default domain after routing.")).strip()
    return domain, reason
