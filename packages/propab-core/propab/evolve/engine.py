"""Evolve — the engine contract + loop skeleton (WS-0; WS-1 implements the population).

The loop (FunSearch/AlphaEvolve-class):

    parents  = island.sample()                       # favour high scorers
    program  = mutator.mutate(parents, problem)      # LLM writes/edits GENERATOR CODE
    result   = runner.run(program)                   # sandboxed execution -> candidates
    verdict  = max(problem.verify(c) for c in result.candidates)   # cheap exact verifier
    island.insert(program, verdict)
    if problem.is_improvement(verdict): ledger.record(...)

Islands (not one global population) exist to preserve diversity: a single population collapses onto
one lineage and stops exploring. Periodically migrate winners between islands and reset the weakest.

The LLM here is a COMMODITY. It is a mutation operator over code — nothing more. This is exactly why
the engine works with a cheap model (FunSearch used PaLM-2), and exactly why the credit for a result
accrues to the engine rather than to whoever's model we rent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .ledger import Ledger
from .problem import Problem, Verdict
from .program import ExecResult, Program, ProgramRunner


class Mutator(Protocol):
    """The LLM seam. WS-1 implements; WS-6 supplies the model router (cheap model by default,
    frontier model as an optional ceiling-raiser — NOT a prerequisite)."""

    def mutate(self, parents: list[Program], problem: Problem) -> Program:
        """Prompt = problem.describe() + PROGRAM_CONTRACT + the best parent programs and their
        scores. Returns a new candidate Program. Must never raise — on a bad completion, return a
        Program whose code is a no-op (it will simply score -inf and die)."""
        ...


class Island(Protocol):
    """One sub-population. WS-1 implements."""

    def sample(self, k: int = 2) -> list[Program]: ...
    def insert(self, program: Program) -> None: ...
    def best(self) -> Program | None: ...
    def __len__(self) -> int: ...


@dataclass
class EngineConfig:
    islands: int = 8
    workers: int = 64            # parallelism is just config — scale it (WS-2)
    program_timeout_s: float = 10.0
    migrate_every: int = 200     # steps between island migrations
    reset_weakest_every: int = 1000
    max_steps: int | None = None  # None = run until stopped


class Engine:
    """Wires Problem + Mutator + Runner + Islands + Ledger. WS-1 implements `step`/`run`;
    WS-2 supplies the parallel executor that drives many `step`s concurrently."""

    def __init__(
        self,
        problem: Problem,
        mutator: Mutator,
        runner: ProgramRunner,
        ledger: Ledger,
        config: EngineConfig | None = None,
    ) -> None:
        self.problem = problem
        self.mutator = mutator
        self.runner = runner
        self.ledger = ledger
        self.config = config or EngineConfig()

    def evaluate(self, program: Program) -> tuple[Verdict, ExecResult]:
        """Run a program and score the best candidate it emitted. Pure, no population side-effects —
        so WS-2 can call it from many workers in parallel."""
        result = self.runner.run(program, timeout_s=self.config.program_timeout_s)
        if not result.ok or not result.candidates:
            from .problem import INVALID

            return INVALID, result
        best: Verdict | None = None
        for cand in result.candidates:
            v = self.problem.verify(cand)          # cheap, exact, never raises
            if best is None or (v.valid and v.score > best.score):
                best = v
        from .problem import INVALID

        return (best or INVALID), result

    def step(self) -> Verdict:  # pragma: no cover — WS-1
        raise NotImplementedError

    def run(self) -> None:  # pragma: no cover — WS-1
        raise NotImplementedError
