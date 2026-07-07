"""Generation overhaul: gap-driven, domain-general, no-template generation.

Replaces the former domain-general SEED-FALLBACK redesign tests. The template /
fallback-seed machinery (``_domain_shape_options``, ``_demo_topic_supplement``,
``_domain_fallback_options``, ``_fallback_hypothesis_text``,
``_inject_discovery_fallbacks``) was DELETED in the generation overhaul.

These tests pin the fixed behaviour:

(a) The generation prompt is DOMAIN-GENERAL and drives gap-targeted NOVELTY:
    it feeds the literature's open gaps, forbids rediscovery, and forbids
    parametric re-tests — asserted on the built prompt string, with zero
    hardcoded domain vocabulary in the core prompt.
(b) Prior open gaps are surfaced generically (``_open_gap_lines`` /
    ``_salient_terms``) regardless of domain.
(c) An empty LLM yields an EMPTY result — never a fabricated/templated seed.
"""

from __future__ import annotations

import asyncio

from services.orchestrator.hypotheses import (
    _build_hypothesis_prompt,
    _open_gap_lines,
    _salient_terms,
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


def _parsed(q: str) -> ParsedQuestion:
    return ParsedQuestion(text=q, domain="general", sub_questions=[q])


# ---------------------------------------------------------------------------
# (a) Prompt drives gap-targeted novelty and forbids rediscovery / retests.
# ---------------------------------------------------------------------------

def test_prompt_contains_novelty_and_no_rediscovery_instructions() -> None:
    prompt = _build_hypothesis_prompt(_parsed("Any research question about X."), _empty_prior(), 5)
    low = prompt.lower()
    # Advance-what-is-known framing.
    assert "advance" in low
    # Explicitly forbids re-deriving known/tabulated values.
    assert "tabulated" in low
    assert "re-deriv" in low or "rederiv" in low or "already" in low
    # Explicitly forbids parametric re-tests of a settled result.
    assert "re-test" in low or "retest" in low or "previously-tested" in low
    assert "parameter sweep" in low
    # Targets OPEN questions.
    assert "open question" in low


def test_prompt_injects_literature_open_gaps() -> None:
    """Domain flavor comes from the injected gaps, not hardcoded keywords."""
    prior = Prior(
        established_facts=[],
        contested_claims=[],
        open_gaps=[
            {
                "what_is_open": "Whether enzyme processivity scales with template GC content",
                "best_known_bound": "no quantitative model below 40% GC",
                "approachable_angle": "in-vitro single-molecule assay",
            }
        ],
        dead_ends=[],
        key_papers=[],
    )
    prompt = _build_hypothesis_prompt(_parsed("Study enzyme processivity."), prior, 5)
    # The literature gap text (domain-specific to THIS question) is surfaced.
    assert "enzyme processivity scales with template GC content" in prompt
    assert "best-known bound" in prompt
    assert "approachable angle" in prompt


def test_prompt_has_no_hardcoded_math_vocabulary() -> None:
    """The CORE generation prompt must be domain-general: no baked-in math terms.

    Domain specificity must arrive via injected gaps only, so a biology/econ
    campaign is not biased toward combinatorics vocabulary."""
    prompt = _build_hypothesis_prompt(_parsed("A neutral question about a process."), _empty_prior(), 5).lower()
    for banned in ("cap set", "cap-set", "sidon", "bose-chowla", "greedy", "f_3", "sqrt(n)"):
        assert banned not in prompt, f"core prompt leaked math vocabulary: {banned!r}"


def test_novelty_nudge_changes_prompt() -> None:
    base = _build_hypothesis_prompt(_parsed("Question Y."), _empty_prior(), 5)
    nudged = _build_hypothesis_prompt(_parsed("Question Y."), _empty_prior(), 5, novelty_nudge=True)
    assert nudged != base
    assert "conceptually distinct" in nudged.lower()


# ---------------------------------------------------------------------------
# (b) Gap surfacing is domain-general.
# ---------------------------------------------------------------------------

def test_open_gap_lines_domain_general() -> None:
    prior = Prior(
        established_facts=[],
        contested_claims=[],
        open_gaps=[
            {"what_is_open": "coral bleaching threshold under variable salinity"},
            {"text": "lunar tidal coupling to reef spawning timing", "best_known_bound": "unquantified"},
        ],
        dead_ends=[],
        key_papers=[],
    )
    lines = _open_gap_lines(prior)
    assert len(lines) == 2
    assert any("coral bleaching threshold" in ln for ln in lines)
    assert any("best-known bound: unquantified" in ln for ln in lines)


def test_salient_terms_ignore_domain_profile_tag_and_stopwords() -> None:
    q = "[domain_profile:ecology] Which coral reefs show bleaching under thermal stress?"
    terms = [t.lower() for t in _salient_terms(q)]
    assert "coral" in terms and "bleaching" in terms
    assert "domain_profile" not in terms and "ecology" not in " ".join(terms)
    assert "which" not in terms  # stopword dropped


def test_prior_context_enriches_salient_terms() -> None:
    q = "Study widget throughput under variable input load."
    prior = Prior(
        established_facts=[],
        contested_claims=[],
        open_gaps=[{"text": "Buffering strategy affects burst tolerance in pipelines."}],
        dead_ends=[],
        key_papers=[{"title": "Queueing amplification in staged pipelines"}],
    )
    terms = [t.lower() for t in _salient_terms(q, prior=prior)]
    assert any(t in terms for t in ("buffering", "burst", "queueing", "pipelines"))


# ---------------------------------------------------------------------------
# (c) Empty LLM -> empty result (no fabrication).
# ---------------------------------------------------------------------------

class _EmptyLLM:
    async def call(self, *, prompt: str, purpose: str, session_id: str, **kwargs) -> str:
        return ""


class _CollectingEmitter:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def emit(self, **kwargs) -> None:
        self.events.append(kwargs)


def test_novel_question_empty_llm_yields_empty_no_fabrication() -> None:
    q = "Do coral reef bleaching events correlate with lunar tidal amplitude across Pacific atolls?"
    out = asyncio.run(
        generate_ranked_hypotheses(
            parsed=_parsed(q),
            prior=_empty_prior(),
            max_hypotheses=5,
            llm=_EmptyLLM(),
            session_id="s1",
            emitter=_CollectingEmitter(),
            use_llm_ranking=False,
        )
    )
    assert out == [], f"empty LLM must not fabricate seeds, got {[h.text[:60] for h in out]}"
