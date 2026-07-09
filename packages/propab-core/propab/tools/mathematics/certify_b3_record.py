"""Standalone B_3 record certifier tool (the sole record gate, exposed for agents).

S1 (general-agent redesign): exposes the trusted, independent ``certify_b3_record``
certifier as a ``TOOL_SPEC`` tool so a general worker agent can VERIFY a candidate
set without ever hand-writing (and mis-writing) a B_3 check. This wrapper reuses the
audited certifier verbatim — it re-derives every threefold sum from scratch via the
independent ``is_B3`` and never trusts a claimed size.

Semantics:
  * ``is_b3``     — the set is genuinely B_3 (independent re-verification).
  * ``certified`` — the set is a valid B_3 set of the reported size (in-cube +
                    distinct + B_3). This certifies the SIZE, not a record.
  * ``beats_known`` — size strictly exceeds the best-known a(n) (A396704).
  * ``is_record`` — certified AND beats an improvable best-known (the record verdict).
"""
from __future__ import annotations

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "certify_b3_record",
    "domain": "mathematics",
    "audience": "worker",
    # Deterministic independent certifier: its evidence is a re-verified certified
    # witness, not a p-value. Satisfies the worker stop-gate (see
    # significance.any_verification_tool_ran).
    "verification_capable": True,
    "description": (
        "Independently certify a candidate B_3 set in {0,1}^n by re-deriving every "
        "threefold sum from scratch (the trusted, paranoid certify_b3_record). "
        "Returns is_b3, size, whether it is a certified valid B_3 set of that size, "
        "and whether it beats the best-known a(n) (OEIS A396704). This is the SOLE "
        "record gate — never claim a record without it."
    ),
    "params": {
        "set": {"type": "list[list[int]]", "required": True,
                 "description": "Candidate set: list of 0/1 vectors in {0,1}^n."},
        "n": {"type": "int", "required": True, "description": "Dimension of the binary cube."},
    },
    "output": {
        "is_b3": "bool — set is genuinely B_3 (independent re-verification)",
        "size": "int — number of (distinct) vectors",
        "certified": "bool — valid B_3 set of this size (in-cube + distinct + B_3)",
        "beats_known": "bool — size strictly exceeds best-known a(n)",
        "is_record": "bool — certified AND beats an improvable best-known",
        "n": "int",
        "best_known": "int|None",
        "record_status": "str|None",
        "certification": "dict — full output of the real certify_b3_record (audit)",
    },
    "example": {
        "params": {"set": [[0, 0], [1, 0], [0, 1]], "n": 2},
        "output": {"is_b3": True, "size": 3, "certified": True, "beats_known": False},
    },
}

_A396704 = "A396704"


def certify_b3_record(
    set: list | None = None,
    n: int | None = None,
) -> ToolResult:
    if set is None:
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message="Parameter 'set' (list of 0/1 vectors) is required."),
        )
    try:
        vectors = [list(v) for v in set]
    except TypeError:
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message="'set' must be a list of vectors."),
        )
    if n is None:
        # Infer n from the first vector if the caller omitted it.
        n = len(vectors[0]) if vectors else None
    if n is None:
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message="Parameter 'n' (cube dimension) is required."),
        )
    try:
        n_int = int(n)
    except (TypeError, ValueError):
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message=f"'n' must be an integer, got {n!r}."),
        )

    try:
        # Import the TRUSTED, independent certifier + best-known registry.
        from propab.domain_modules.math_combinatorics.discovery import (
            best_known,
            certify_b3_record as _real_certify_b3_record,
            record_status,
        )

        bk = best_known(_A396704, n_int)
        status = record_status(_A396704, n_int)
        size = len(vectors)

        # Pass best-known as published_best so strictly_beats is the honest record
        # signal; when absent, use size so strictly_beats is False (no reference).
        ref = int(bk) if bk is not None else size
        cert = _real_certify_b3_record(vectors, ref, expected_n=n_int)
        checks = cert.get("checks", {})

        is_b3 = bool(checks.get("is_b3"))
        # ``certified`` = valid B_3 set of the reported size (independent sub-checks),
        # deliberately NOT requiring a record — a rediscovery is still certified.
        size_certified = bool(
            checks.get("in_binary_cube") and checks.get("distinct_vectors") and is_b3
        )
        beats_known = bool(checks.get("strictly_beats_published")) if bk is not None else False
        is_record = bool(
            cert.get("certified") and bk is not None and status in ("provisional_lower_bound", "open")
        )

        return ToolResult(
            success=True,
            output={
                "is_b3": is_b3,
                "size": int(cert.get("size", size)),
                "certified": size_certified,
                "beats_known": beats_known,
                "is_record": is_record,
                "n": int(cert.get("n", n_int)),
                "best_known": bk,
                # best_known comes from the TRUSTED internal A396704 registry, not a
                # caller-supplied value — so this record needs no external corroboration.
                "best_known_source": f"reference:{_A396704}",
                "record_status": status,
                "certification": cert,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
