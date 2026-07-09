"""General witness certifier — the trustworthy, scalable way to turn a code-computed
combinatorial construction into a confirmed finding.

A general worker agent must WRITE flexible search code to construct an object (a Sidon
set, a sum-free set, a Golomb ruler, …) because no fixed solver DSL expresses every
property. But a result the agent's OWN code reports is self-reported and cannot be
trusted. This tool provides the missing INDEPENDENT verification layer: it re-derives an
allow-listed property FROM SCRATCH on the supplied witness (never trusting the caller's
claim), so a certified witness becomes a trustworthy LOWER-BOUND finding.

Why this scales: verification is bounded + reusable while search is infinite + specific.
One ``is this set Sidon?`` check serves every Sidon question at every scale — the trusted
layer is general, the flexible search stays with the agent (the general-agent thesis).

It certifies EXISTENCE (a witness with the property and this size EXISTS), and — when a
``published_best`` reference is supplied (e.g. from an OEIS/literature lookup) — whether
it BEATS the best-known (a candidate record). It does NOT prove optimality/maximality;
that needs an exhaustive/solver UNSAT proof, not a witness check.
"""
from __future__ import annotations

from itertools import combinations, combinations_with_replacement

from propab.tools.types import ToolError, ToolResult

# Allow-listed properties. Each is a pure, independent re-derivation from scratch that
# returns (holds: bool, violation: tuple|None). Adding a property = adding one checker
# here — verification is the general, bounded, reusable layer.
_PROPERTIES = ("sidon", "b_h", "sum_free", "golomb_ruler", "sidon_mod", "progression_free")


def _as_int_list(witness):
    vals = [int(x) for x in witness]
    return vals


def _check_bh(vals, h, modulus=None):
    """All h-fold multiset sums distinct (B_h / Sidon for h=2), optionally mod m."""
    seen = {}
    for combo in combinations_with_replacement(sorted(vals), h):
        s = sum(combo)
        if modulus:
            s %= modulus
        if s in seen:
            return False, {"sum": s, "combo_a": list(seen[s]), "combo_b": list(combo)}
        seen[s] = combo
    return True, None


def _check_golomb(vals):
    """All pairwise positive differences distinct (a Golomb ruler / B_2 difference set)."""
    seen = {}
    for a, b in combinations(sorted(set(vals)), 2):
        d = b - a
        if d in seen:
            return False, {"difference": d, "pair_a": list(seen[d]), "pair_b": [a, b]}
        seen[d] = (a, b)
    return True, None


def _check_sum_free(vals):
    """No a + b = c with a, b, c all in the set (a <= b)."""
    s = set(vals)
    for a, b in combinations_with_replacement(sorted(s), 2):
        if (a + b) in s:
            return False, {"a": a, "b": b, "c": a + b}
    return True, None


def _check_progression_free(vals, k):
    """No k-term arithmetic progression a, a+d, …, a+(k-1)d (d != 0) inside the set."""
    s = set(vals)
    sv = sorted(s)
    for a in sv:
        for d in range(1, (max(sv) - a) // max(1, (k - 1)) + 1):
            if all((a + i * d) in s for i in range(k)):
                return False, {"start": a, "diff": d, "length": k}
    return True, None


TOOL_SPEC = {
    "name": "certify_witness",
    "domain": "mathematics",
    "audience": "worker",
    # Independent re-verification: its output is a re-derived certified witness, not a
    # p-value — satisfies the worker stop-gate (see significance.any_verification_tool_ran).
    "verification_capable": True,
    "description": (
        "Independently CERTIFY a claimed combinatorial witness against an allow-listed "
        "property — the trustworthy way to turn a construction you computed in code into "
        "a confirmed finding (the honesty gate will NOT trust your code's own 'verified' "
        "flag; it WILL trust this). Supply the actual object (list of integers) + the "
        "property; the tool re-derives the property FROM SCRATCH and reports whether it "
        "holds, the size, and a self-certifying violation if it fails. Properties: "
        "'sidon' (all pairwise sums distinct), 'b_h' (all h-fold sums distinct; give h), "
        "'sum_free' (no a+b=c), 'golomb_ruler' (all pairwise differences distinct), "
        "'sidon_mod' (pairwise sums distinct mod m; give modulus), 'progression_free' (no "
        "k-term AP; give k). Pass published_best (e.g. from an OEIS/literature lookup) to "
        "learn whether the size BEATS the best-known (a candidate record). Certifies "
        "EXISTENCE/a lower bound, NOT optimality."
    ),
    "params": {
        "witness": {"type": "list[int]", "required": True,
                     "description": "The actual object to certify — a list of integers."},
        "property": {"type": "str", "required": True,
                      "description": f"One of {list(_PROPERTIES)}."},
        "h": {"type": "int", "required": False, "default": 2,
               "description": "For property='b_h': the order h (all h-fold sums distinct)."},
        "modulus": {"type": "int", "required": False,
                     "description": "For property='sidon_mod': the modulus m."},
        "k": {"type": "int", "required": False, "default": 3,
               "description": "For property='progression_free': the AP length to forbid."},
        "published_best": {"type": "int", "required": False,
                            "description": "Best-known size to beat (from OEIS/literature)."},
    },
    "output": {
        "property": "str",
        "holds": "bool — the property genuinely holds (independently re-derived)",
        "size": "int — number of distinct elements",
        "certified": "bool — holds AND elements distinct (a valid witness of this size)",
        "beats_best_known": "bool — size strictly exceeds published_best (if supplied)",
        "is_record": "bool — certified AND beats an supplied best-known",
        "best_known_source": "str — 'agent_supplied' (published_best is the CALLER's value, so a "
                             "record must be corroborated by an independent oeis_lookup before it counts)",
        "violation": "dict|None — a self-certifying counterexample when holds is False",
        "note": "str",
    },
    "example": {
        # A genuinely valid Sidon set (Mian–Chowla prefix): all pairwise sums are
        # distinct. The previous example [1,2,5,11,22,33,40] was NOT Sidon
        # (11+33 = 22+22 = 44), so it certified False — a broken example that agents
        # copied verbatim. An example must be a valid instance of what it demonstrates.
        "params": {"witness": [1, 2, 4, 8, 13, 21, 31], "property": "sidon"},
        "output": {"holds": True, "size": 7, "certified": True},
    },
}


def certify_witness(
    witness=None,
    property="sidon",
    h=2,
    modulus=None,
    k=3,
    published_best=None,
):
    if witness is None:
        return ToolResult(success=False, error=ToolError(
            type="validation_error", message="Parameter 'witness' (a list of integers) is required."))
    prop = str(property or "").strip().lower()
    if prop not in _PROPERTIES:
        return ToolResult(success=False, error=ToolError(
            type="validation_error", message=f"Unknown property {property!r}; allowed: {list(_PROPERTIES)}."))
    try:
        vals = _as_int_list(witness)
    except (TypeError, ValueError):
        return ToolResult(success=False, error=ToolError(
            type="validation_error", message="'witness' must be a list of integers."))
    if not vals:
        return ToolResult(success=False, error=ToolError(
            type="validation_error", message="'witness' must be non-empty."))

    distinct_elems = len(set(vals)) == len(vals)
    size = len(set(vals))

    try:
        if prop == "sidon":
            holds, violation = _check_bh(vals, 2, None)
        elif prop == "b_h":
            hh = int(h)
            if hh < 2:
                return ToolResult(success=False, error=ToolError(
                    type="validation_error", message="'h' must be >= 2 for b_h."))
            holds, violation = _check_bh(vals, hh, None)
        elif prop == "sidon_mod":
            if modulus is None or int(modulus) <= 0:
                return ToolResult(success=False, error=ToolError(
                    type="validation_error", message="property='sidon_mod' requires a positive 'modulus'."))
            holds, violation = _check_bh(vals, 2, int(modulus))
        elif prop == "golomb_ruler":
            holds, violation = _check_golomb(vals)
        elif prop == "sum_free":
            holds, violation = _check_sum_free(vals)
        elif prop == "progression_free":
            kk = int(k)
            if kk < 3:
                return ToolResult(success=False, error=ToolError(
                    type="validation_error", message="'k' must be >= 3 for progression_free."))
            holds, violation = _check_progression_free(vals, kk)
        else:  # unreachable (guarded above)
            holds, violation = False, None
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))

    certified = bool(holds and distinct_elems)
    bk = None if published_best is None else int(published_best)
    beats = bool(bk is not None and size > bk)
    is_record = bool(certified and beats)

    if not distinct_elems:
        note = "witness has repeated elements — not a valid set."
    elif not holds:
        note = f"property '{prop}' FAILS: {violation}."
    elif is_record:
        note = (f"CERTIFIED candidate record: a {prop} set of size {size} exists, "
                f"beating best-known {bk}. (Existence/lower-bound — NOT proven optimal.)")
    elif bk is not None and size <= bk:
        note = f"certified {prop} set of size {size}; does not beat best-known {bk}."
    else:
        note = f"certified {prop} set of size {size} (existence/lower bound; optimality not claimed)."

    return ToolResult(success=True, output={
        "property": prop,
        "holds": bool(holds),
        "size": size,
        "distinct_elements": distinct_elems,
        "certified": certified,
        "beats_best_known": beats,
        "is_record": is_record,
        "best_known": bk,
        # The reference this record is judged against was SUPPLIED BY THE CALLER, not
        # derived from a trusted registry. So is_record here is only as trustworthy as
        # published_best: the verdict layer must corroborate it against an independent
        # reference (oeis_lookup) before treating it as a discovery — otherwise an agent
        # fabricates a low published_best and manufactures a false record.
        "best_known_source": "agent_supplied",
        "violation": violation,
        "note": note,
    })
