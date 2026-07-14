"""Evolve — the Problem contract (WS-0).

A `Problem` is the *only* thing the evolution engine knows about a domain. It is deliberately tiny:
the engine proposes PROGRAMS, runs them to get CANDIDATES, and asks the Problem to score them.

The load-bearing property — and the reason this whole approach works — is that `verify()` is
**cheap, exact, and decoupled from generation**. The verifier carries the result, not the model.
If a target cannot supply a fast exact verifier, it does not belong here (that is the GeneBench
lesson: no verifier => base-model-bound => no discovery).

Domain-general by construction: targets live in `evolve/targets/` and are thin adapters over the
existing domain-module verifiers (coding_theory.compute_min_distance, graph_invariants.verifier,
math_combinatorics.verifier). Never put domain specifics in this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# A candidate is whatever the target's programs emit (a generator matrix, a graph, a set of reals…).
# The engine never inspects it — only the Problem does.
Candidate = Any


@dataclass(frozen=True)
class Verdict:
    """The verifier's exact judgement on one candidate."""

    valid: bool                      # does it satisfy the problem's hard constraints?
    score: float                     # objective value; higher is better. Meaningless if not valid.
    detail: dict[str, Any] = field(default_factory=dict)   # witness, proof-of-work, diagnostics
    # A verified improvement MUST carry the evidence that lets a third party re-check it
    # (e.g. the witness codeword, the violating graph, the failing pair). No witness => not a result.

    @property
    def usable(self) -> bool:
        return self.valid


@runtime_checkable
class Problem(Protocol):
    """One discovery target. Implementations live in evolve/targets/."""

    name: str

    def describe(self) -> str:
        """The problem spec shown to the LLM when it mutates programs. This is the *prompt surface*:
        state the objective, the hard constraints, the candidate format, and what 'better' means."""
        ...

    def verify(self, candidate: Candidate) -> Verdict:
        """Cheap, exact, deterministic. MUST NOT call an LLM. MUST be safe on garbage input
        (a mutated program can emit anything) — return Verdict(valid=False, score=-inf) instead of
        raising."""
        ...

    def best_known(self) -> float:
        """The current record to beat. MUST come from a real, citable table/registry — never from a
        model's memory. If we cannot source it, the target is not ready to run."""
        ...

    def is_improvement(self, verdict: Verdict) -> bool:
        """True only for a genuine beat of `best_known()`. Implementations MUST reject trivial
        rediscovery (see coding_theory.trivial_rediscovery / is_table_lookup_evidence) — reproducing
        a known construction is not a discovery, and self-deception here poisons everything."""
        ...

    def seed_programs(self) -> list[str]:
        """Starting programs (Python source). Each defines `def build(): -> candidate | list[candidate]`.
        Good seeds are known constructions — evolution works by *recombining* them."""
        ...


# --------------------------------------------------------------------------- #
# Null objects so the engine/executor can be built and tested before targets land.
# --------------------------------------------------------------------------- #
NEG_INF = float("-inf")

INVALID = Verdict(valid=False, score=NEG_INF, detail={"reason": "invalid candidate"})
