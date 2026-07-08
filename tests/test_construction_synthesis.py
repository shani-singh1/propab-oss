"""
Tests for the construction-synthesis loop (generate -> execute -> verify -> refine).

No real LLM is used: a scripted ``FakeLLM`` returns construct() source strings so we
exercise the sandbox, the exact B_3 oracle, structural-failure feedback, and
best-so-far tracking deterministically. The real multiprocessing sandbox runner is
used throughout (small budgets keep it fast), including the hard-timeout path.
"""
from __future__ import annotations

import asyncio

import pytest

from propab.domain_modules.math_combinatorics.discovery import (
    b3_construction_spec,
    synthesize_construction,
)
from propab.domain_modules.math_combinatorics.discovery.construction_synthesis import (
    ConstructionSpec,
    VerificationResult,
    _extract_code,
    _run_in_sandbox,
    _screen_source,
)
from propab.domain_modules.math_combinatorics.discovery.known_witnesses import WITNESSES

_W16 = [tuple(v) for v in WITNESSES[7][0]]  # a genuine size-16 B_3 set in {0,1}^7


class FakeLLM:
    """Async LLM stub returning scripted responses in order (cycling on the last)."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def call(self, *, prompt: str, purpose: str, session_id: str, **kw) -> str:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


def _construct_code_for(points: list[tuple[int, ...]]) -> str:
    """Wrap an explicit point list in a construct(n) function (fenced code block)."""
    body = ", ".join(repr(tuple(int(x) for x in p)) for p in points)
    return f"```python\ndef construct(n):\n    return [{body}]\n```"


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# (a) A known-good construction is executed, verified, and its size recorded.
# --------------------------------------------------------------------------- #
def test_good_construction_is_verified_and_recorded():
    spec = b3_construction_spec(7, published_best=16)
    llm = FakeLLM([_construct_code_for(_W16)])
    result = _run(synthesize_construction(spec, llm=llm, max_iters=1, exec_budget_s=10.0))

    assert result["best"] is not None
    assert result["best"]["size"] == 16
    assert result["best"]["n"] == 7
    # The recorded witness is the real B_3 set the code emitted.
    assert {tuple(v) for v in result["best"]["set"]} == set(_W16)
    assert result["trace"][0]["ok"] is True
    assert result["trace"][0]["stage"] == "verify"


# --------------------------------------------------------------------------- #
# (b) Broken constructions are caught structurally and the loop keeps going.
# --------------------------------------------------------------------------- #
def test_broken_constructions_are_caught_without_crashing():
    spec = b3_construction_spec(7, published_best=16)
    raises = "```python\ndef construct(n):\n    return [1/0]\n```"
    # Duplicated vectors -> not B_3; a concrete structural failure is extracted.
    not_b3 = _construct_code_for([(0,) * 7, (0,) * 7, (1, 0, 0, 0, 0, 0, 0)])
    infinite = "```python\ndef construct(n):\n    x = 0\n    while True:\n        x += 1\n    return []\n```"

    llm = FakeLLM([raises, not_b3, infinite])
    result = _run(
        synthesize_construction(spec, llm=llm, max_iters=3, exec_budget_s=1.5)
    )

    assert result["best"] is None  # nothing valid was ever produced
    assert len(result["trace"]) == 3
    for step in result["trace"]:
        assert step["ok"] is False
        assert step["failure"]

    assert result["trace"][0]["stage"] == "execute"          # ZeroDivisionError
    assert "ZeroDivisionError" in result["trace"][0]["failure"]
    assert result["trace"][1]["stage"] == "verify"           # not a B_3 set
    assert "duplicate" in result["trace"][1]["failure"]
    assert result["trace"][2]["stage"] == "execute"          # timed out
    assert "timeout" in result["trace"][2]["failure"]


# --------------------------------------------------------------------------- #
# (c) Best-so-far strictly improves as the fake returns larger valid sets.
# --------------------------------------------------------------------------- #
def test_best_so_far_improves_across_iterations():
    spec = b3_construction_spec(7, published_best=16)
    # Prefixes of a B_3 set are themselves B_3, so these are valid at sizes 5, 10, 16.
    llm = FakeLLM([
        _construct_code_for(_W16[:5]),
        _construct_code_for(_W16[:10]),
        _construct_code_for(_W16[:16]),
    ])
    result = _run(synthesize_construction(spec, llm=llm, max_iters=3, exec_budget_s=10.0))

    sizes = [s["size"] for s in result["trace"]]
    assert sizes == [5, 10, 16]
    new_best_flags = [s.get("new_best", False) for s in result["trace"]]
    assert new_best_flags == [True, True, True]
    assert result["best"]["size"] == 16


# --------------------------------------------------------------------------- #
# Sandbox unit-level guarantees.
# --------------------------------------------------------------------------- #
def test_static_screen_blocks_imports_and_dunder_escapes():
    with pytest.raises(ValueError):
        _screen_source("import os\ndef construct(n):\n    return []")
    with pytest.raises(ValueError):
        _screen_source("def construct(n):\n    return ().__class__.__bases__")
    with pytest.raises(ValueError):
        _screen_source("def construct(n):\n    return __import__('os')")
    # A clean construction passes the screen.
    _screen_source("def construct(n):\n    return [(0,)*n]")


def test_sandbox_blocks_file_and_os_access_at_runtime():
    # `open` is absent from the restricted builtins -> NameError, reported as a failure.
    ok, pts, err = _run_in_sandbox(
        "def construct(n):\n    return [open('x')]", 3, 2.0
    )
    assert ok is False
    assert pts is None
    assert "NameError" in err or "not allowed" in err


def test_sandbox_hard_timeout_terminates_runaway_code():
    ok, pts, err = _run_in_sandbox(
        "def construct(n):\n    \n    x=0\n    while True:\n        x+=1", 3, 0.8
    )
    assert ok is False
    assert "timeout" in err


def test_extract_code_prefers_construct_block():
    text = "```python\nx = 1\n```\nthen\n```python\ndef construct(n):\n    return []\n```"
    assert "def construct" in _extract_code(text)
    assert _extract_code("no code here") is None


def test_generic_spec_tracks_best_for_non_b3_domain():
    # The loop is domain-general: a trivial spec (distinct ints, size = count) still
    # drives best-so-far growth, proving the loop logic is not B_3-specific.
    def verify(obj):
        vals = [int(v[0]) for v in obj]
        if len(set(vals)) != len(vals):
            return VerificationResult(False, len(vals), "duplicate value")
        return VerificationResult(True, len(vals), None)

    spec = ConstructionSpec(name="ints", n=1, verify=verify, object_description="distinct ints")
    llm = FakeLLM([
        _construct_code_for([(1,), (2,)]),
        _construct_code_for([(1,), (2,), (3,), (4,)]),
    ])
    result = _run(synthesize_construction(spec, llm=llm, max_iters=2, exec_budget_s=5.0))
    assert [s["size"] for s in result["trace"]] == [2, 4]
    assert result["best"]["size"] == 4
