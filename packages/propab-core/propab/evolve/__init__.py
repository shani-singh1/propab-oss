"""propab.evolve — LLM-guided program-evolution search over cheap exact verifiers.

The discovery engine. Domain-general by construction: it knows only the `Problem` contract; every
target (`evolve/targets/`) is a thin adapter over an existing domain-module verifier.

Why this exists (the diagnosis): propab's previous search was OBJECT-space (CP-SAT + metaheuristics,
see math_combinatorics/discovery/cp_sat_finder.py) and it stalled. This engine searches PROGRAM
space — it evolves generators, not objects — which is the one thing that made FunSearch work with a
non-frontier model.

Selection rule for any new target (all four are necessary):
  1. cheap, exact, fast verifier, decoupled from generation   (no verifier => no discovery)
  2. a REAL, sourced "best known" to beat                      (never a model's memory)
  3. search-shaped: the answer is an OBJECT, not an argument   (proofs need frontier IQ; we don't have it)
  4. many shots on goal                                        (turns a lottery into an engineering process)
"""
from __future__ import annotations

from .engine import Engine, EngineConfig, Island, Mutator
from .ledger import Ledger, Record
from .problem import INVALID, Candidate, Problem, Verdict
from .program import ENTRYPOINT, PROGRAM_CONTRACT, ExecResult, Program, ProgramRunner

__all__ = [
    "Problem",
    "Verdict",
    "Candidate",
    "INVALID",
    "Program",
    "ProgramRunner",
    "ExecResult",
    "ENTRYPOINT",
    "PROGRAM_CONTRACT",
    "Engine",
    "EngineConfig",
    "Island",
    "Mutator",
    "Ledger",
    "Record",
]
