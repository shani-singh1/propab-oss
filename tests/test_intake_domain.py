import asyncio

from services.orchestrator.intake import parse_question


def test_parse_question_dl_keywords() -> None:
    p = asyncio.run(parse_question("How do PyTorch transformers scale with attention length?"))
    assert p.domain == "deep_learning"


def test_parse_question_algo_keywords() -> None:
    p = asyncio.run(parse_question("Compare Adam and SGD convergence on a non-convex loss."))
    assert p.domain == "algorithm_optimization"


def test_parse_question_ml_research_keywords() -> None:
    p = asyncio.run(parse_question("Is the accuracy gain statistically significant with bootstrap CIs?"))
    assert p.domain == "ml_research"


def test_parse_question_default_general() -> None:
    p = asyncio.run(parse_question("What is the capital of France?"))
    assert p.domain == "general_computation"
