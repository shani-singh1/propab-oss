"""Evolve — the Ledger contract (WS-0).

Every verified improvement is recorded here, together with **everything a third party needs to
re-check it without trusting us**: the program that produced it, the candidate, the verifier's
witness, and the exact verifier code.

This is the CDC credibility model, deliberately copied: OpenAI did not ask anyone to believe them —
they shipped a proof a kernel checks in 34 minutes. Our equivalent is: ship the construction AND the
verifier. Credibility comes from the checker, never from the claim.

Anti-self-deception is a first-class concern here. This project has twice reported a number that
dissolved under scrutiny. A Record is only written when:
  * the verifier says valid,
  * the score genuinely beats a REAL sourced best_known (not a model's memory), and
  * it is not a trivial rediscovery of a known construction.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Record:
    """One verified improvement — the publishable unit."""

    problem: str
    score: float
    best_known_at_time: float
    candidate: Any                        # must be JSON-serializable (or pre-serialized)
    witness: dict[str, Any]               # the verifier's evidence (codeword, violating pair, …)
    program_code: str                     # the generator that found it
    program_id: str
    verifier_fingerprint: str             # hash of the verifier source — pins WHAT checked it
    generation: int = 0
    created_at: float = field(default_factory=time.time)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


class Ledger:
    """Append-only store of verified improvements + the publication export.

    WS-5 implements this (and should reuse `math_combinatorics/discovery/record_registry.py`
    rather than reinventing record-keeping).
    """

    def __init__(self, root: str | Path = "artifacts/evolve") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def record(self, rec: Record) -> bool:  # pragma: no cover — WS-5
        """Persist a verified improvement. Returns False if it's a duplicate/rediscovery."""
        raise NotImplementedError

    def best(self, problem: str) -> Record | None:  # pragma: no cover — WS-5
        raise NotImplementedError

    def export_publishable(self, problem: str, dest: str | Path) -> Path:  # pragma: no cover — WS-5
        """Emit a self-contained bundle: the construction, the witness, and a RUNNABLE verifier
        script, so a third party can confirm the result in one command. If it isn't independently
        re-checkable, it isn't a result."""
        raise NotImplementedError
