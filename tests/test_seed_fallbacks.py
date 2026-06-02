"""Seed-generation fallbacks and relevance gate (fixes.md Phase 1)."""

from services.orchestrator.hypotheses import (
    _domain_fallback_options,
    _fallback_hypothesis_text,
    _inject_discovery_fallbacks,
    _is_ml_template_hypothesis,
)
from services.orchestrator.schemas import RankedHypothesis


def test_egyptian_question_gets_domain_fallbacks() -> None:
    q = "Which odd n admit five distinct unit fractions 1/a+1/b+1/c+1/d+1/e?"
    opts = _domain_fallback_options(q)
    assert any("mod" in o.lower() or "residue" in o.lower() or "enumeration" in o.lower() for o in opts)
    assert not _is_ml_template_hypothesis(_fallback_hypothesis_text(q, 1))


def test_concrete_scoped_claim_not_auto_rejected() -> None:
    q = "Test contagion on scale-free networks."
    text = (
        "Epidemic peak time on scale-free networks is more sensitive to degree exponent "
        f"than to average degree. (Question: {q})"
    )
    assert not _is_ml_template_hypothesis(text)


def test_ml_template_still_rejects_intervention_wording() -> None:
    hyp = "A targeted intervention measurably improves the primary metric against a baseline."
    assert _is_ml_template_hypothesis(hyp)


def test_inject_discovery_fallbacks_adds_three() -> None:
    q = "Compare cache policies on LRU-adversarial traces."
    kept = [
        RankedHypothesis(
            id="null",
            text=(
                "Null hypothesis: No falsifiable pattern in the research question holds beyond "
                "what random variation would produce. (Question: x)"
            ),
            test_methodology="bootstrap",
            scores={},
            rank=5,
        )
    ]
    out = _inject_discovery_fallbacks(kept, question=q, max_hypotheses=5, min_discovery=3)
    from propab.research_quality import infer_node_role

    discovery = [h for h in out if infer_node_role(h.text) != "CONTROL"]
    assert len(discovery) >= 3
