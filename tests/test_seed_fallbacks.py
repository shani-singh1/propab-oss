"""Generation honesty: no template / fallback-seed fabrication.

The template / fallback-seed machinery (``_domain_fallback_options``,
``_fallback_hypothesis_text``, ``_inject_discovery_fallbacks``) was DELETED in
the generation overhaul: Propab must REASON, never expand templates. These tests
pin the replacement behavior — when the LLM yields nothing usable, generation
returns empty rather than fabricating a templated hypothesis.
"""

from __future__ import annotations

import asyncio

from services.orchestrator.hypotheses import (
    _is_ml_template_hypothesis,
    generate_ranked_hypotheses,
)
from services.orchestrator.intake import ParsedQuestion
from services.orchestrator.schemas import Prior


def _empty_prior() -> Prior:
    return Prior(
        established_facts=[],
        contested_claims=[],
        open_gaps=[],
        dead_ends=[],
        key_papers=[],
    )


class _EmptyLLM:
    """LLM double that never returns any hypotheses."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call(self, *, prompt: str, purpose: str, session_id: str, **kwargs) -> str:
        self.calls.append(purpose)
        return ""


class _CollectingEmitter:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def emit(self, **kwargs) -> None:
        self.events.append(kwargs)


def test_ml_template_still_rejects_intervention_wording() -> None:
    hyp = "A targeted intervention measurably improves the primary metric against a baseline."
    assert _is_ml_template_hypothesis(hyp)


def test_empty_llm_returns_no_fabricated_hypothesis() -> None:
    """A mock LLM that returns nothing must yield an EMPTY result — never a
    fabricated/templated hypothesis carrying the round forward."""
    q = "Do coral reef bleaching events correlate with lunar tidal amplitude across Pacific atolls?"
    parsed = ParsedQuestion(text=q, domain="general", sub_questions=[q])
    llm = _EmptyLLM()
    meta: dict = {}

    out = asyncio.run(
        generate_ranked_hypotheses(
            parsed=parsed,
            prior=_empty_prior(),
            max_hypotheses=5,
            llm=llm,
            session_id="s1",
            emitter=_CollectingEmitter(),
            use_llm_ranking=False,
            meta=meta,
        )
    )

    assert out == [], f"empty LLM must produce no fabricated hypotheses, got {[h.text[:60] for h in out]}"
    assert meta.get("llm_empty") is True
    # The honest retry path fires exactly once (novelty-nudged), then stops.
    assert "hypothesis_generation" in llm.calls
    assert "hypothesis_generation_retry" in llm.calls
