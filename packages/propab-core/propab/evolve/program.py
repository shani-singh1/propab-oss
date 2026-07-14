"""Evolve — the Program contract (WS-0).

The central bet, and the thing that distinguishes this engine from the CP-SAT/metaheuristic finder
that stalled: **we do not search the object space. We evolve PROGRAMS that generate objects.**

Object-space search (CP-SAT, simulated annealing) explores one construction at a time and does not
generalize. Program-space search compresses structure — a short generator that emits a whole family —
which is why FunSearch beat SOTA on cap sets with a *non-frontier* model. The LLM is a mutation
operator over code, nothing more.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

# Every program is Python source defining exactly this entry point:
ENTRYPOINT = "build"
PROGRAM_CONTRACT = f"""\
Write Python source defining:

    def {ENTRYPOINT}():
        # return a candidate, or a list of candidates
        ...

Rules:
- Deterministic unless the spec says otherwise (seed any RNG explicitly).
- No network, no file I/O, no imports outside the allowed list.
- Must return quickly; the verifier runs on whatever you emit.
"""


@dataclass
class Program:
    """A candidate generator, plus its evolutionary bookkeeping."""

    code: str
    score: float = float("-inf")     # best verified score of any candidate it emitted
    valid: bool = False
    generation: int = 0
    island: int = 0
    parents: list[str] = field(default_factory=list)   # parent program ids (provenance)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return hashlib.sha256(self.code.encode("utf-8")).hexdigest()[:16]

    def __hash__(self) -> int:  # dedupe identical code
        return hash(self.id)


@dataclass(frozen=True)
class ExecResult:
    """Outcome of running one program in the sandbox."""

    ok: bool
    candidates: list[Any] = field(default_factory=list)
    error: str | None = None
    stdout: str = ""
    seconds: float = 0.0


class ProgramRunner:
    """Contract for executing a Program -> candidates. WS-2 implements this over the existing
    sandbox (`services/worker/sub_agent_loop` / `math_combinatorics/discovery/sandbox_exec`).

    MUST be hard-sandboxed and hard-timeouted: mutated code is adversarial by accident — it will
    hang, allocate forever, and raise. A crashed program is a normal, expected event: return
    ExecResult(ok=False, ...), never propagate.
    """

    def run(self, program: Program, *, timeout_s: float = 10.0) -> ExecResult:  # pragma: no cover
        raise NotImplementedError
