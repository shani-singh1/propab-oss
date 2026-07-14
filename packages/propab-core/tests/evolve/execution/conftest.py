"""Shared fixtures for the WS-2 (runner + executor) tests.

Nothing here touches the network, and no test depends on `evolve/targets/` — the executor is
domain-general, so it is tested against a toy Problem defined right here.
"""
from __future__ import annotations

from typing import Any

import pytest

from propab.evolve.engine import Engine, EngineConfig
from propab.evolve.ledger import Ledger
from propab.evolve.problem import INVALID, Verdict
from propab.evolve.program import Program
from propab.evolve.runner import SandboxLimits, SandboxProgramRunner


class SumProblem:
    """Toy Problem: a candidate is a non-empty list of ints; the score is their sum.

    Deliberately tiny and exact, and — per the Problem contract — safe on garbage: a mutated program
    can emit anything, so `verify` must return INVALID rather than raise.
    """

    name = "sum-of-ints"

    def describe(self) -> str:
        return "Return a list of ints. Score = sum. Higher is better."

    def verify(self, candidate: Any) -> Verdict:
        try:
            if not isinstance(candidate, list) or not candidate:
                return INVALID
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in candidate):
                return INVALID
            return Verdict(valid=True, score=float(sum(candidate)), detail={"witness": candidate})
        except Exception:  # noqa: BLE001 - the contract: never raise, however garbage the input
            return INVALID

    def best_known(self) -> float:
        return 100.0

    def is_improvement(self, verdict: Verdict) -> bool:
        return verdict.valid and verdict.score > self.best_known()

    def seed_programs(self) -> list[str]:
        return ["def build():\n    return [1, 2, 3]\n"]


class ExplodingProblem(SumProblem):
    """A verifier that breaks its "never raises" promise — the executor must survive it anyway."""

    name = "exploding"

    def verify(self, candidate: Any) -> Verdict:
        raise RuntimeError("verifier exploded")


class StubMutator:
    """The LLM seam, stubbed. WS-2 never calls it; the Engine just needs one."""

    def mutate(self, parents: list[Program], problem: Any) -> Program:  # pragma: no cover
        return Program(code="def build():\n    return [1]\n")


# Fast sandbox limits for tests: skip the numpy pre-import (~1.5-2s per child) unless a test
# actually needs numpy warm. Keeps the suite's process-spawn cost down.
FAST_LIMITS = SandboxLimits(preimport=(), memory_mb=512, spawn_timeout_s=60.0)


def make_engine(runner: SandboxProgramRunner, tmp_path: Any, *, problem: Any = None,
                workers: int = 4, timeout_s: float = 5.0) -> Engine:
    return Engine(
        problem=problem or SumProblem(),
        mutator=StubMutator(),
        runner=runner,
        ledger=Ledger(root=tmp_path / "ledger"),
        config=EngineConfig(workers=workers, program_timeout_s=timeout_s),
    )


@pytest.fixture(scope="session")
def fast_limits() -> SandboxLimits:
    return FAST_LIMITS


@pytest.fixture(scope="module")
def runner() -> Any:
    """A module-scoped sandbox pool: spawning children is the expensive part, so share it.

    Sharing it is also a real test in itself — every test that follows reuses workers that earlier
    tests timed out, OOM'd and crashed, which is exactly the self-healing property we claim.
    """
    with SandboxProgramRunner(pool_size=4, limits=FAST_LIMITS) as r:
        yield r


@pytest.fixture
def exploding_problem() -> Any:
    return ExplodingProblem()


@pytest.fixture
def engine_for(tmp_path: Any) -> Any:
    """Build an Engine around a given runner (and optionally a different Problem)."""

    def _make(runner: SandboxProgramRunner, **kwargs: Any) -> Engine:
        return make_engine(runner, tmp_path, **kwargs)

    return _make
