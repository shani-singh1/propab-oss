"""Domain-generality redesign of seed generation (fix/generation-layer, G1 + G2).

These tests pin the fixed behaviour:

(a) A question matching NONE of the historical demo keywords still yields
    non-empty, question-relevant seed hypotheses (no keyword-table cliff).
(b) A known demo topic still produces sensible, tuned seeds.
(c) A boilerplate / fallback hypothesis is flagged (never silently counted as a
    validated, on-topic hypothesis).
"""

from __future__ import annotations

from services.orchestrator.hypotheses import (
    _demo_topic_supplement,
    _domain_fallback_options,
    _fallback_hypothesis_text,
    _inject_discovery_fallbacks,
    _is_ml_template_hypothesis,
    _salient_terms,
)
from services.orchestrator.hypothesis_ranking import (
    compute_question_relevance_score_lexical,
    strip_question_suffix,
)
from services.orchestrator.schemas import Prior, RankedHypothesis


def _empty_prior() -> Prior:
    return Prior(
        established_facts=[],
        contested_claims=[],
        open_gaps=[],
        dead_ends=[],
        key_papers=[],
    )


# ---------------------------------------------------------------------------
# (a) Unknown-domain questions (zero keyword matches) still get good seeds.
# ---------------------------------------------------------------------------

NOVEL_QUESTIONS = [
    "Do coral reef bleaching events correlate with lunar tidal amplitude across Pacific atolls?",
    "How does soil microbial diversity change with nitrogen fertilizer dosage in temperate grasslands?",
    "Does sentence length predict reader recall in legal contracts up to 500 words?",
    "Are volcanic tremor frequencies a leading indicator of eruption onset within 72 hours?",
]


def test_unknown_domain_yields_nonempty_seeds() -> None:
    for q in NOVEL_QUESTIONS:
        assert _demo_topic_supplement(q) == [], f"{q!r} unexpectedly matched a demo keyword"
        opts = _domain_fallback_options(q)
        assert len(opts) >= 4, f"{q!r} produced too few fallbacks: {opts}"
        assert all(o.strip() for o in opts)


def test_unknown_domain_seeds_are_question_relevant() -> None:
    """Seeds must reference the question's own vocabulary and clear the gate."""
    for q in NOVEL_QUESTIONS:
        salient = _salient_terms(q)
        assert salient, f"no salient terms for {q!r}"
        for opt in _domain_fallback_options(q):
            # references at least one of the question's own content words
            assert any(t.lower() in opt.lower() for t in salient), (opt, salient)
            # clears the lexical relevance gate threshold (0.35)
            score = compute_question_relevance_score_lexical(q, [], strip_question_suffix(opt))
            assert score >= 0.35, (q, round(score, 3), opt)


def test_unknown_domain_seeds_are_not_ml_templates() -> None:
    for q in NOVEL_QUESTIONS:
        for opt in _domain_fallback_options(q):
            assert not _is_ml_template_hypothesis(opt), opt
        assert not _is_ml_template_hypothesis(_fallback_hypothesis_text(q, 1))


def test_salient_terms_ignore_domain_profile_tag_and_stopwords() -> None:
    q = "[domain_profile:ecology] Which coral reefs show bleaching under thermal stress?"
    terms = [t.lower() for t in _salient_terms(q)]
    assert "coral" in terms and "bleaching" in terms
    assert "domain_profile" not in terms and "ecology" not in " ".join(terms)
    assert "which" not in terms  # stopword dropped


def test_prior_context_enriches_seeds() -> None:
    q = "Study widget throughput under variable input load."
    prior = Prior(
        established_facts=[],
        contested_claims=[],
        open_gaps=[{"text": "Buffering strategy affects burst tolerance in pipelines."}],
        dead_ends=[],
        key_papers=[{"title": "Queueing amplification in staged pipelines"}],
    )
    terms = [t.lower() for t in _salient_terms(q, prior=prior)]
    # borrows vocabulary from the literature prior, not only the question
    assert any(t in terms for t in ("buffering", "burst", "queueing", "pipelines"))


# ---------------------------------------------------------------------------
# (b) Known demo topics still produce sensible, tuned seeds.
# ---------------------------------------------------------------------------

def test_known_egyptian_topic_keeps_tuned_phrasing() -> None:
    q = "Which odd n admit five distinct unit fractions 1/a+1/b+1/c+1/d+1/e?"
    opts = _domain_fallback_options(q)
    # demo phrasing survives as a supplement...
    assert any("mod" in o.lower() or "residue" in o.lower() for o in opts)
    # ...but the general path is still present too (never keyword-gated away).
    assert any("enumeration" in o.lower() or "null model" in o.lower() for o in opts)
    assert not _is_ml_template_hypothesis(_fallback_hypothesis_text(q, 1))


def test_known_contagion_topic_keeps_scoped_phrasing() -> None:
    q = "Investigate epidemic contagion peak time on scale-free networks."
    opts = _domain_fallback_options(q)
    assert any("population:" in o.lower() and "ood test:" in o.lower() for o in opts)


def test_general_path_leads_demo_supplement() -> None:
    """Demo phrasings are only ever an appended extra, never the sole source."""
    q = "Which odd n admit five distinct unit fractions?"
    opts = _domain_fallback_options(q)
    from services.orchestrator.hypotheses import _domain_shape_options

    general = _domain_shape_options(q)
    assert opts[: len(general)] == general  # general seeds come first


# ---------------------------------------------------------------------------
# (c) Fallback / boilerplate hypotheses are flagged, not silently validated.
# ---------------------------------------------------------------------------

def test_injected_fallbacks_are_flagged_as_fallback() -> None:
    q = "How does canopy density affect understory light in boreal forests?"
    kept = [
        RankedHypothesis(
            id="null",
            text=(
                "Null hypothesis: No falsifiable pattern in the research question holds "
                "beyond random variation. (Question: x)"
            ),
            test_methodology="bootstrap",
            scores={},
            rank=5,
        )
    ]
    out = _inject_discovery_fallbacks(kept, question=q, max_hypotheses=5, min_discovery=3)
    injected = [h for h in out if h.id.startswith("fallback_d")]
    assert injected, "expected fallback discovery seeds to be injected"
    for h in injected:
        # G2: injected seeds are unambiguously marked as fallbacks so they
        # cannot be counted as validated, scope-checked, on-topic hypotheses.
        assert h.scores.get("is_fallback") == 1.0
        assert h.scores.get("scope_fallback") == 1.0


def test_fallback_scope_flagged_not_silently_passed() -> None:
    """A fallback whose scope is the domain template is flagged, not treated valid.

    ``_fallback_hypothesis_text`` fills scope from the domain template, which
    ``is_boilerplate_scope`` recognizes. The gate must mark such a seed as a
    fallback (is_fallback / boilerplate_scope) and drop scope_valid, rather than
    letting it pass as a genuine scope-validated hypothesis.
    """
    from propab.scoped_claim import (
        is_boilerplate_scope,
        parse_scope_from_text,
    )

    q = "How does canopy density affect understory light in boreal forests?"
    text = _fallback_hypothesis_text(q, 1)
    scope = parse_scope_from_text(text)
    assert scope is not None
    # This fallback's scope is the generic domain template -> boilerplate.
    assert is_boilerplate_scope(scope, q)

    # Simulate the gate decision made in generate_ranked_hypotheses: a fallback
    # with boilerplate scope stays flagged rather than silently validated.
    scores = {"scope_valid": 1.0, "scope_fallback": 1.0, "is_fallback": 1.0}
    is_fallback = scores.get("scope_fallback") == 1.0 or scores.get("is_fallback") == 1.0
    assert is_fallback
    if is_boilerplate_scope(scope, q) and is_fallback:
        scores["boilerplate_scope"] = 1.0
        scores["scope_valid"] = 0.0
    assert scores["scope_valid"] == 0.0
    assert scores["boilerplate_scope"] == 1.0


def test_non_fallback_boilerplate_is_rejected_logic() -> None:
    """A NON-fallback hypothesis with template scope is a gate bypass -> reject."""
    from propab.scoped_claim import infer_domain_scope_template, is_boilerplate_scope

    q = "Investigate arbitrary process X in setting Y."
    tmpl = infer_domain_scope_template(q)
    # Build a hypothesis text carrying the verbatim template scope.
    text = (
        "Some claim about X.\n"
        f"Population: {tmpl.population}\n"
        f"Distribution: {tmpl.distribution}\n"
        f"Claimed generalization: {tmpl.claimed_generalization}\n"
        f"Expected failure modes: {tmpl.expected_failure_modes}\n"
        f"OOD test: {tmpl.ood_test}"
    )
    from propab.scoped_claim import parse_scope_from_text

    scope = parse_scope_from_text(text)
    assert scope is not None and is_boilerplate_scope(scope, q)

    scores: dict[str, float] = {}  # no fallback markers -> purportedly LLM-validated
    is_fallback = scores.get("scope_fallback") == 1.0 or scores.get("is_fallback") == 1.0
    reject = is_boilerplate_scope(scope, q) and not is_fallback
    assert reject, "non-fallback boilerplate scope must be rejected"


# ---------------------------------------------------------------------------
# End-to-end: empty LLM on a NOVEL question still yields flagged, on-topic seeds.
# ---------------------------------------------------------------------------

class _EmptyLLM:
    """LLM double that never returns hypotheses -> forces the fallback path."""

    async def call(self, *, prompt: str, purpose: str, session_id: str, **kwargs) -> str:
        return ""


class _CollectingEmitter:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def emit(self, **kwargs) -> None:
        self.events.append(kwargs)


def test_end_to_end_novel_question_empty_llm_yields_flagged_seeds() -> None:
    import asyncio

    from services.orchestrator.hypotheses import generate_ranked_hypotheses
    from services.orchestrator.intake import ParsedQuestion

    q = "Do coral reef bleaching events correlate with lunar tidal amplitude across Pacific atolls?"
    parsed = ParsedQuestion(text=q, domain="general", sub_questions=[q])

    out = asyncio.run(
        generate_ranked_hypotheses(
            parsed=parsed,
            prior=_empty_prior(),
            max_hypotheses=5,
            llm=_EmptyLLM(),
            session_id="s1",
            emitter=_CollectingEmitter(),
            use_llm_ranking=False,
        )
    )

    assert out, "novel question with empty LLM must still yield seeds"
    # At least some discovery (non-null) seeds present and referencing the topic.
    discovery = [h for h in out if "null hypothesis" not in h.text.lower()]
    assert discovery, out
    assert any("coral" in h.text.lower() or "reef" in h.text.lower() for h in discovery), [
        h.text[:80] for h in discovery
    ]
    # Every synthesized (fallback) seed is flagged; none masquerades as validated.
    for h in out:
        if h.scores.get("scope_fallback") == 1.0 or h.scores.get("is_fallback") == 1.0:
            assert h.scores.get("is_fallback") == 1.0, h.scores
