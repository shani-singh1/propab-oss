"""Intake no longer guesses a domain from keywords.

Domain routing is delegated to the plugins (``DomainPlugin.matches`` via
``registry.resolve_domain_plugin``) and, in the campaign path, to the campaign's
domain tag/payload — never to a central hardcoded keyword taxonomy. These tests
lock that decision in: ``parse_question`` must NOT infer a domain, and structural
parsing (sub-questions) must still work.
"""
import asyncio

from services.orchestrator.intake import parse_question


def test_parse_question_does_not_infer_domain() -> None:
    # Previously this returned a keyword-guessed domain (deep_learning/ml_research/…).
    # The keyword taxonomy was removed; intake must leave domain unset.
    for q in (
        "How do PyTorch transformers scale with attention length?",
        "Compare Adam and SGD convergence on a non-convex loss.",
        "What is the capital of France?",
    ):
        p = asyncio.run(parse_question(q))
        assert p.domain == "", f"intake should not guess a domain, got {p.domain!r} for {q!r}"


def test_parse_question_still_splits_sub_questions() -> None:
    p = asyncio.run(parse_question("Does X hold? And does Y transfer to Z?"))
    assert p.text.startswith("Does X hold")
    assert len(p.sub_questions) >= 2
