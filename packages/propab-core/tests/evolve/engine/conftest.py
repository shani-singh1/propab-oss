"""Fakes for the evolve engine tests. No network, no sandbox, no LLM — by design.

The engine's whole claim is that the model is a commodity and the verifier carries the result, so the
tests exercise it with a deliberately stupid "LLM": a regex hill-climber that reads the parent code
out of the prompt and increments one number. If the loop can turn *that* into a rising score, the
mechanism — not the model — is doing the work.
"""
from __future__ import annotations

import random
import re

import pytest

from propab.evolve.ledger import Ledger, Record
from propab.evolve.problem import INVALID, Verdict
from propab.evolve.program import ExecResult, Program

# --------------------------------------------------------------------------- problem

SEED_CODE = "# family: constant\ndef build():\n    return [(1, 1, 1, 1)]\n"


class SumProblem:
    """Maximize the sum of a 4-tuple of digits. Verifier: exact, cheap, total.

    Deliberately trivial and deliberately *bounded*: max score 36, best_known 30, so a run that works
    will cross the record and exercise the ledger path.
    """

    name = "fake-sum"
    BEST_KNOWN = 30.0
    MAX_SCORE = 36.0  # (9, 9, 9, 9)

    def describe(self) -> str:
        return "Return a list of 4-tuples of ints in [0, 9]. Score = sum of the tuple. Maximize it."

    def verify(self, candidate: object) -> Verdict:
        # Total on garbage: a mutated program emits anything at all.
        try:
            values = [int(v) for v in candidate]  # type: ignore[union-attr]
        except (TypeError, ValueError):
            return INVALID
        if len(values) != 4 or any(v < 0 or v > 9 for v in values):
            return INVALID
        return Verdict(valid=True, score=float(sum(values)), detail={"values": values})

    def best_known(self) -> float:
        return self.BEST_KNOWN

    def is_improvement(self, verdict: Verdict) -> bool:
        return bool(verdict.valid and verdict.score > self.BEST_KNOWN)

    def seed_programs(self) -> list[str]:
        return [SEED_CODE]


# --------------------------------------------------------------------------- runner


class InProcRunner:
    """Executes a program in-process. Stands in for WS-2's real sandbox.

    NOT a sandbox: it has no timeout and no isolation, which is exactly why WS-2 exists. It reproduces
    the one behaviour the engine depends on — a crashing program returns ExecResult(ok=False), it does
    not propagate.
    """

    def __init__(self) -> None:
        self.runs = 0

    def run(self, program: Program, *, timeout_s: float = 10.0) -> ExecResult:
        self.runs += 1
        namespace: dict[str, object] = {}
        try:
            exec(compile(program.code, "<program>", "exec"), namespace)  # noqa: S102
            build = namespace["build"]
            output = build()  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 — a crashing program is a normal event
            return ExecResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        candidates = list(output) if isinstance(output, list) else [output]
        return ExecResult(ok=True, candidates=candidates)


class TimeoutRunner:
    """Every program "hangs" — i.e. the sandbox kills it. What the engine sees is ok=False."""

    def run(self, program: Program, *, timeout_s: float = 10.0) -> ExecResult:
        return ExecResult(ok=False, error=f"timeout after {timeout_s}s")


class ExplodingRunner:
    """A runner that is itself broken. The contract says runners never propagate; this one does."""

    def run(self, program: Program, *, timeout_s: float = 10.0) -> ExecResult:
        raise RuntimeError("sandbox is on fire")


# --------------------------------------------------------------------------- llm

_TUPLE_RE = re.compile(r"\((\d)\s*,\s*(\d)\s*,\s*(\d)\s*,\s*(\d)\)")


class HillClimbLLM:
    """The dumbest possible "model": find the best parent's tuple in the prompt, bump its smallest
    element by one. It cannot reason; it can only edit. That is the point."""

    family = "hill-climb"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        matches = _TUPLE_RE.findall(prompt)
        if not matches:
            values = [1, 1, 1, 1]
        else:
            # Parents are shown worst-to-best, so the last tuple is the best one.
            values = [int(v) for v in matches[-1]]
            i = values.index(min(values))
            values[i] = min(9, values[i] + 1)
        body = ", ".join(str(v) for v in values)
        return (
            "Here you go.\n\n"
            "```python\n"
            f"# family: {self.family}\n"
            "def build():\n"
            f"    return [({body})]\n"
            "```\n"
        )


class ScriptedLLM:
    """Returns canned completions in order (cycling the last one)."""

    def __init__(self, *completions: object) -> None:
        self.completions = list(completions)
        self.calls = 0

    def __call__(self, prompt: str) -> object:
        self.calls += 1
        return self.completions[min(self.calls - 1, len(self.completions) - 1)]


class CrashingLLM:
    """A client that is down. The run must survive it."""

    def __call__(self, prompt: str) -> str:
        raise ConnectionError("model endpoint unreachable")


class CrashingProgramLLM:
    """A model that only ever writes code that blows up at runtime."""

    def __call__(self, prompt: str) -> str:
        return "```python\ndef build():\n    raise RuntimeError('boom')\n```"


# --------------------------------------------------------------------------- ledger


class RecordingLedger(Ledger):
    """In-memory ledger. `accept=False` simulates a rejected duplicate/rediscovery."""

    def __init__(self, root, *, accept: bool = True) -> None:
        super().__init__(root)
        self.records: list[Record] = []
        self.accept = accept

    def record(self, rec: Record) -> bool:
        self.records.append(rec)
        return self.accept


class BrokenLedger(Ledger):
    """The ledger is down (disk full, WS-5 not landed). A verified improvement must NOT vanish."""

    def record(self, rec: Record) -> bool:
        raise OSError("no space left on device")


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def problem() -> SumProblem:
    return SumProblem()


@pytest.fixture
def runner() -> InProcRunner:
    return InProcRunner()


@pytest.fixture
def ledger(tmp_path) -> RecordingLedger:
    return RecordingLedger(tmp_path / "evolve")


@pytest.fixture
def rng() -> random.Random:
    return random.Random(1234)
