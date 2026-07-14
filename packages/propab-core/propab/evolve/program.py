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

#: Candidates one program should aim to emit. The sandbox verifies ~430/sec while a model call costs
#: ~1s, so a program that emits ONE object wastes ~99.7% of the machine. Sweep, don't point.
TARGET_CANDIDATES_PER_PROGRAM = 300

PROGRAM_CONTRACT = f"""\
Write Python source defining a GENERATOR:

    def {ENTRYPOINT}():
        for <parameters of a construction family>:
            yield <candidate>

DO NOT emit a single object. Emit a FAMILY — sweep the parameters of a construction and yield every
member. Aim for ~{TARGET_CANDIDATES_PER_PROGRAM} candidates per call (a few thousand is fine if each
is cheap to build).

Why this is the whole game, measured on this system:
- The verifier checks ~430 candidates/second; asking the model for a new program costs ~1 second.
  A program that yields one object therefore uses ~0.3% of the machine. A program that yields a
  family uses all of it, at identical model cost.
- Searching one object at a time DOES NOT WORK on these landscapes. On a conjecture with a known
  counterexample, point-search (hill-climbing, simulated annealing, cross-entropy) failed 0/55 —
  the neighbourhood of the best-known object is a deceptive local optimum and the target sits
  across a valley. Sweeping a family's PARAMETERS crosses it immediately.

So: your job is to invent or vary a *construction*, not to guess an *object*. Parameterize it
(sizes, shifts, generator polynomials, block structure, symmetry group, seeds) and yield the sweep.

Rules:
- `yield` (a generator) is preferred and unambiguous. A bare `return <list>` is ambiguous when the
  candidate is itself a list, so avoid it.
- Deterministic unless the spec says otherwise (seed any RNG explicitly).
- No network, no file I/O, no imports outside the allowed list.
- Cheap per candidate: the sweep must finish inside the time budget, so prefer many cheap members
  over a few expensive ones.
- Never print, return, or claim a score. The verifier is the sole authority on how good a candidate
  is; anything a program says about itself is ignored.
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
