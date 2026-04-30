from __future__ import annotations

import json
import re

from propab.llm import LLMClient

DOMAIN_ROUTING_PROMPT = """
Given this hypothesis and the original research question, return the single most relevant domain.

Original research question: {question}
Hypothesis: {hypothesis_text}

Domains (choose exactly one name from this list):
- deep_learning: neural networks, transformers, training, attention, PyTorch workflows
- algorithm_optimization: gradient methods, complexity, benchmarking, convergence, loss surfaces
- ml_research: statistical tests, bootstrap CIs, experiment grids, FLOPs / reproducibility analysis
- mathematics: formal math, symbolic computation, linear algebra proofs
- statistics: classical stats, regression, inference (not deep nets)
- data_analysis: EDA, aggregation, cleaning, tabular summaries
- general_computation: anything else — sandboxed code, parsing, glue logic

Important: if the research question involves COMPARING or RANKING approaches (e.g. "which optimizer
is better", "does X outperform Y"), prefer ml_research or algorithm_optimization over deep_learning,
even if the hypothesis mentions neural networks.

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


def _keyword_fallback_domain(hypothesis_text: str) -> str:
    t = (hypothesis_text or "").lower()
    if any(k in t for k in ("transformer", "activation", "optimizer", "sgd", "adam", "train", "model", "neural", "mlp", "attention", "learning rate", "warmup")):
        return "deep_learning"
    if any(k in t for k in ("loss surface", "convergence", "complexity", "benchmark", "hessian", "gradient", "ravine", "plateau")):
        return "algorithm_optimization"
    if any(k in t for k in ("baseline", "significance", "p-value", "confidence interval", "reproduc", "flops", "experiment grid")):
        return "ml_research"
    return "general_computation"


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
    question: str = "",
) -> tuple[str, str]:
    prompt = DOMAIN_ROUTING_PROMPT.format(
        hypothesis_text=hypothesis_text,
        question=question or hypothesis_text,
    )
    raw = await llm.call(
        prompt=prompt,
        purpose="domain_routing",
        session_id=session_id,
        hypothesis_id=hypothesis_id,
    )
    data = _extract_json(raw) or {}
    parsed_domain = str(data.get("domain", "")).strip()
    domain = coerce_routed_domain(parsed_domain)
    reason = str(data.get("reason", "")).strip()
    if not parsed_domain:
        domain = _keyword_fallback_domain(hypothesis_text)
        reason = f"Keyword fallback routing to {domain} after non-JSON/empty domain response."
    elif domain == "general_computation":
        kw = _keyword_fallback_domain(hypothesis_text)
        if kw != "general_computation":
            domain = kw
            reason = f"Coerced away from general_computation using keyword fallback to {domain}."
    if not reason:
        reason = "Default domain after routing."
    return domain, reason
