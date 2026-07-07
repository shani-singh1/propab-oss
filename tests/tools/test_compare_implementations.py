"""Honesty tests for the algorithm-optimization benchmarking tools.

These tools previously fabricated results: ``compare_implementations`` ignored the
agent's code/inputs and returned name-hash-derived timing + hardcoded
``all_correct: True``; ``benchmark_algorithm`` ignored the required ``code`` and reported
the complexity of a fixed built-in kernel. Both now REFUSE (there is no safe code-execution
sandbox available to core tools) rather than certify a false performance/correctness claim.
See the module docstrings for the rationale.
"""

from propab.tools.algorithm_optimization.benchmark_algorithm import benchmark_algorithm
from propab.tools.algorithm_optimization.compare_implementations import compare_implementations


def test_compare_implementations_refuses_instead_of_fabricating() -> None:
    # Two implementations where one is genuinely correct and fast and the other is
    # genuinely wrong/slow. The old tool would happily certify BOTH as all_correct and
    # pick a "fastest" from name-hash timing. The honest tool must not do that.
    r = compare_implementations(
        implementations=[
            {"name": "correct_fast", "code": "def fn(x): return sorted(x)"},
            {"name": "wrong_slow", "code": "def fn(x):\n    while True: pass"},
        ],
        test_inputs=[[3, 1, 2], [9, 8, 7]],
        n_runs=3,
    )
    # It must refuse, not confirm.
    assert r.success is False
    assert r.error is not None
    assert r.error.type == "cannot_execute_code"

    # And crucially it must emit NONE of the fabricated structured fields.
    out = r.output or {}
    assert "fastest" not in out
    assert "most_memory_efficient" not in out
    assert "correctness" not in out
    assert "performance" not in out


def test_compare_implementations_never_certifies_wrong_code_correct() -> None:
    r = compare_implementations(
        implementations=[{"name": "buggy", "code": "def fn(x): raise ValueError('nope')"}],
        test_inputs=[1, 2, 3],
        n_runs=5,
    )
    # No hardcoded all_correct:True may survive: either there is no correctness output at
    # all (refusal), or nothing claims the buggy impl is correct.
    out = r.output or {}
    correctness = out.get("correctness")
    if correctness is not None:  # would only exist if the tool claims to have measured
        for entry in correctness:
            assert entry.get("all_correct") is not True
    else:
        assert r.success is False


def test_compare_implementations_validates_inputs() -> None:
    r = compare_implementations(implementations=[], test_inputs=[1])
    assert r.success is False
    assert r.error is not None
    assert r.error.type == "validation_error"

    r2 = compare_implementations(implementations=[{"code": "x"}], test_inputs=[1])
    assert r2.success is False
    assert r2.error is not None
    assert r2.error.type == "validation_error"


def test_benchmark_algorithm_refuses_instead_of_fabricating() -> None:
    # A genuinely O(n^2) algorithm. The old tool ignored `code` and always benchmarked a
    # fixed O(n) dot-product kernel, so it could never report the truth here.
    r = benchmark_algorithm(
        code="def f(n):\n    return sum(i*j for i in range(n) for j in range(n))",
        input_sizes=[100, 200, 400],
        n_runs=3,
    )
    assert r.success is False
    assert r.error is not None
    assert r.error.type == "cannot_execute_code"

    out = r.output or {}
    assert "empirical_complexity" not in out
    assert "complexity_r2" not in out
    assert "results" not in out


def test_benchmark_algorithm_validates_inputs() -> None:
    r = benchmark_algorithm(code="   ", input_sizes=[10, 20])
    assert r.success is False
    assert r.error is not None
    assert r.error.type == "validation_error"

    r2 = benchmark_algorithm(code="def f(n): return n", input_sizes=[])
    assert r2.success is False
    assert r2.error is not None
    assert r2.error.type == "validation_error"
