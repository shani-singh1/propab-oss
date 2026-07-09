"""Discovery-search tool: find as large an extremal set as the budget allows.

S1 (general-agent redesign): this exposes the trusted math-combinatorics discovery
primitive — ``find_max_b3`` (the ILS + DLS + branch-and-bound finder) gated by the
INDEPENDENT ``certify_b3_record`` certifier — as a ``TOOL_SPEC`` tool a general
worker agent can call. It is a thin, faithful wrapper: it reuses the audited search
and the audited certifier verbatim and NEVER self-reports a record.

Honesty invariants preserved here:
  * The size is always backed by a witness the finder re-verified with the
    independent ``is_B3`` (``find_max_b3`` asserts this internally).
  * ``certified`` / ``is_record`` are read from the REAL ``certify_b3_record`` (the
    sole record gate) — this wrapper does not re-implement any check.
  * ``is_record`` requires the certifier's full record verdict AND an improvable
    best-known status; a mere rediscovery (size == best-known) is CERTIFIED as a
    valid set but is NOT a record.
"""
from __future__ import annotations

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "extremal_set_search",
    "domain": "mathematics",
    "audience": "worker",
    "description": (
        "Search for a maximum-size extremal set of a combinatorial object and "
        "independently CERTIFY the result. object='b3_binary_cube' searches for the "
        "largest B_3 (threefold-sum-distinct) set in {0,1}^n (OEIS A396704) via the "
        "trusted finder, then re-verifies the witness with the independent "
        "certify_b3_record. Returns the set, its certified size, whether it beats "
        "the best-known, and the best-known/target for comparison. Never self-reports "
        "a record — certification is the sole gate."
    ),
    "params": {
        "object": {"type": "str", "required": False, "default": "b3_binary_cube",
                    "description": "Which extremal object to search (only 'b3_binary_cube' in v1)."},
        "n": {"type": "int", "required": True, "description": "Dimension of the binary cube {0,1}^n."},
        "time_budget_sec": {"type": "int", "required": False, "default": 30,
                             "description": "Wall-clock search budget in seconds."},
    },
    "output": {
        "object": "str",
        "n": "int",
        "size": "int — size of the best set found (witness-backed)",
        "set": "list[list[int]] — the best B_3 set found",
        "certified": "bool — set is an independently re-verified valid B_3 set of this size",
        "is_record": "bool — strictly beats an improvable best-known (real certifier)",
        "beats_best_known": "bool — size strictly exceeds best-known",
        "best_known": "int|None — best-known a(n) for A396704",
        "target_to_beat": "int|None — must strictly exceed this for a record",
        "record_status": "str|None — proven_optimal / provisional_lower_bound / open",
        "method": "str — finder method used",
        "proven_optimal": "bool — True only for the exact branch-and-bound branch",
        "certification": "dict — full output of the real certify_b3_record (audit)",
        "elapsed_sec": "float",
        "note": "str",
    },
    "example": {
        "params": {"object": "b3_binary_cube", "n": 7, "time_budget_sec": 30},
        "output": {"size": 16, "certified": True, "is_record": False, "best_known": 16},
    },
}

_SUPPORTED_OBJECTS = ("b3_binary_cube",)
_A396704 = "A396704"


def extremal_set_search(
    object: str = "b3_binary_cube",
    n: int | None = None,
    time_budget_sec: int = 30,
) -> ToolResult:
    obj = str(object or "b3_binary_cube").strip().lower()
    if obj not in _SUPPORTED_OBJECTS:
        return ToolResult(
            success=False,
            error=ToolError(
                type="validation_error",
                message=f"Unsupported object {object!r}; supported: {list(_SUPPORTED_OBJECTS)}.",
            ),
        )
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
    if n_int < 1:
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message="'n' must be a positive integer."),
        )
    try:
        budget = float(time_budget_sec)
    except (TypeError, ValueError):
        budget = 30.0
    budget = max(1.0, budget)

    try:
        # Import the TRUSTED discovery primitives (search + independent certifier).
        from propab.domain_modules.math_combinatorics.discovery import (
            best_known,
            certify_b3_record,
            find_max_b3,
            record_status,
        )

        res = find_max_b3(n_int, time_budget=budget)
        found_set = [list(v) for v in res.get("set", [])]
        size = int(res.get("size", len(found_set)))

        bk = best_known(_A396704, n_int)
        status = record_status(_A396704, n_int)

        # Re-verify the witness through the REAL certifier. Passing the best-known as
        # published_best makes ``strictly_beats_published`` the honest record signal.
        # When there is no table value, use ``size`` so strictly_beats is False (no
        # reference to beat) while the independent B_3 sub-checks still run.
        ref = int(bk) if bk is not None else size
        cert = certify_b3_record({"n": n_int, "set": found_set}, ref, expected_n=n_int)
        checks = cert.get("checks", {})

        # ``certified`` = independently re-verified VALID B_3 set of this size (size
        # trustworthiness), read entirely from the real certifier's sub-checks.
        size_certified = bool(
            checks.get("in_binary_cube") and checks.get("distinct_vectors") and checks.get("is_b3")
        )
        beats_known = bool(checks.get("strictly_beats_published")) if bk is not None else False
        # ``is_record`` = the certifier's FULL record verdict, and only for an
        # improvable term. A rediscovery (size == best-known) is never a record.
        is_record = bool(
            cert.get("certified") and bk is not None and status in ("provisional_lower_bound", "open")
        )

        if bk is None:
            note = (
                f"Found a {'certified ' if size_certified else ''}B_3 set of size {size} in "
                f"{{0,1}}^{n_int}; no best-known table value to compare."
            )
        elif size < bk:
            note = f"B_3 size {size} in {{0,1}}^{n_int} is below best-known {bk} (honest gap {bk - size})."
        elif size == bk:
            note = (
                f"Reproduced best-known a({n_int})={bk} for B_3 in {{0,1}}^{n_int} "
                f"({'proven-optimal rediscovery' if status == 'proven_optimal' else 'matched a search bound'}) "
                f"— certified valid set, NOT a record."
            )
        elif is_record:
            note = (
                f"CANDIDATE RECORD: certified B_3 set of size {size} beats best-known {bk} (A396704). "
                f"Certifies a lower-bound improvement only, NOT optimality."
            )
        else:
            note = (
                f"Search reported size {size} > best-known {bk} but certification returned "
                f"certified={cert.get('certified')} (status={status}); NOT surfaced as a record."
            )

        return ToolResult(
            success=True,
            output={
                "object": obj,
                "n": n_int,
                "size": size,
                "set": found_set,
                "certified": size_certified,
                "is_record": is_record,
                "beats_best_known": beats_known,
                "best_known": bk,
                "target_to_beat": bk,
                "record_status": status,
                "method": res.get("method"),
                "proven_optimal": bool(res.get("proven_optimal")),
                "certification": cert,
                "elapsed_sec": round(float(res.get("elapsed", 0.0)), 3),
                "note": note,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
