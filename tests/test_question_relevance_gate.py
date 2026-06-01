"""Tests for question relevance gate (fixes.md P0.3)."""

from services.orchestrator.hypothesis_ranking import compute_question_relevance_score_lexical


def test_math_hypothesis_scores_high_on_math_question() -> None:
    q = "Which odd integers n have a 5-term Egyptian fraction representation?"
    hyp = "Every odd n ≡ 1 (mod 4) below 1e6 admits a unit-fraction decomposition."
    score = compute_question_relevance_score_lexical(q, ["Egyptian fractions use 1/x terms"], hyp)
    assert score >= 0.35


def test_ml_template_scores_low_on_math_question() -> None:
    q = "Which odd integers n have a 5-term Egyptian fraction representation?"
    hyp = "A targeted intervention measurably improves the primary metric against a baseline."
    score = compute_question_relevance_score_lexical(q, [], hyp)
    assert score < 0.35


def test_ml_template_with_question_suffix_still_rejected() -> None:
    q = "Verify Sierpiński Egyptian fractions up to n=10000."
    hyp = (
        "Hypothesis 1: A targeted intervention measurably improves the primary metric against a baseline. "
        f"(Question: {q})"
    )
    from services.orchestrator.hypotheses import _is_ml_template_hypothesis
    from services.orchestrator.hypothesis_ranking import compute_question_relevance_score_lexical, strip_question_suffix

    assert _is_ml_template_hypothesis(hyp)
    score = compute_question_relevance_score_lexical(q, [], strip_question_suffix(hyp))
    assert score < 0.35
