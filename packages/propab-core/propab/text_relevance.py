"""Domain-agnostic lexical text-relevance utilities.

These live in ``propab-core`` (the lower layer) so that core modules such as
``campaign_synthesis`` no longer need to import UP into ``services.*`` — the
dependency-inversion violation recorded as ADR A2. ``services.orchestrator.
hypothesis_ranking`` re-exports the two public helpers for its own callers.

Purely lexical + domain-general: token overlap between a hypothesis and the
question/prior. No domain assumptions, no LLM, no service dependencies.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def strip_question_suffix(text: str) -> str:
    """Remove a trailing ``(Question: ...)`` suffix before relevance scoring."""
    return re.split(r"\s*\(Question:", (text or "").strip(), maxsplit=1)[0].strip()


def compute_question_relevance_score_lexical(
    question: str,
    prior_snippets: list[str],
    hypothesis_text: str,
) -> float:
    """Lexical question↔hypothesis relevance (fixes.md P0.3).

    Domain-agnostic: compares hypothesis tokens to question + prior fact tokens.
    """
    hyp_toks = _token_set(strip_question_suffix(hypothesis_text))
    if not hyp_toks:
        return 0.0
    corpus = " ".join([question] + list(prior_snippets or []))
    corpus_toks = _token_set(corpus)
    if not corpus_toks:
        return 0.0
    overlap = len(hyp_toks & corpus_toks) / float(len(hyp_toks))
    # Reward hypotheses that share rare-ish tokens with the question (not just stopwords).
    q_overlap = len(hyp_toks & _token_set(question)) / float(max(1, len(_token_set(question))))
    return round(min(1.0, max(0.0, 0.65 * overlap + 0.35 * q_overlap)), 4)
