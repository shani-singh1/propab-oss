"""Exact combinatorial enumeration & counting (M3, combinatorics cluster).

A general worker composes this to do the *repetitive* combinatorial arithmetic a
mathematician does by hand for small cases: exact counts (binomial / Catalan /
Stirling / partitions / set-partitions / compositions), and predicate-filtered
counting/generation of subsets and permutations.

HONESTY BY CONSTRUCTION
-----------------------
* Predicate-based counting *enumerates the whole space* and returns an EXACT count.
  A caller-supplied predicate is a SMALL DECLARATIVE SPEC drawn from a fixed
  allow-list (e.g. ``{"type": "sum_at_most", "k": 5}``) — never eval'd code.
* The search space is capped. If the space exceeds the cap the tool returns
  ``status="unknown"`` with the reason and ``value=None`` — it NEVER returns a
  truncated/partial count dressed up as exact.
* Closed-form counts (binomial, Catalan, Stirling, p(n), Bell, 2^(n-1)) are exact
  and do not enumerate, so they carry a larger cap.
* The returned ``objects`` list may be a bounded SAMPLE of a fully-counted space;
  it is explicitly flagged with ``objects_truncated`` so the sample is never
  mistaken for the whole enumeration. The ``value`` count stays exact and complete.

Never raises: every failure is a ``ToolResult(success=False, ...)`` (bad input) or
``success=True`` with ``status="unknown"`` (valid but oversized).
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Iterable

from propab.tools.types import ToolError, ToolResult

# --- caps (search-space limits; enumeration ops honour these) ---------------
MAX_SUBSET_UNIVERSE = 20        # 2^20 ~ 1.05e6 enumerations
MAX_PERM_UNIVERSE = 9           # 9! = 362880
MAX_PARTITION_N_GEN = 40        # p(40) = 37338 objects
MAX_SETPART_N_GEN = 11          # Bell(11) = 678570 objects
MAX_COMPOSITION_N_GEN = 18      # 2^17 = 131072 objects
MAX_FORMULA_N = 20000           # closed-form counts (no enumeration)
DEFAULT_MAX_ITEMS = 200         # returned-object sample size
HARD_MAX_ITEMS = 2000

_OPS = {
    "count_subsets_with_property", "generate", "partitions", "set_partitions",
    "permutations_with_property", "compositions", "binomial", "catalan", "stirling",
}

# Predicate allow-lists (NO code eval — declarative specs only).
_SUBSET_PRED_TYPES = {
    "sum_at_most", "sum_at_least", "sum_equals",
    "size_at_most", "size_at_least", "size_equals",
    "max_at_most", "min_at_least", "contains", "no_two_consecutive",
}
_NUMERIC_SUBSET_PRED = {
    "sum_at_most", "sum_at_least", "sum_equals",
    "max_at_most", "min_at_least", "no_two_consecutive",
}
_PERM_PRED_TYPES = {
    "derangement", "num_fixed_points_equals",
    "num_inversions_at_most", "num_inversions_equals", "num_descents_equals",
}
_PRED_NEEDS_K = {
    "sum_at_most", "sum_at_least", "sum_equals",
    "size_at_most", "size_at_least", "size_equals",
    "max_at_most", "min_at_least",
    "num_fixed_points_equals", "num_inversions_at_most",
    "num_inversions_equals", "num_descents_equals",
}


def _err(msg: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="validation_error", message=msg))


def _unknown(op: str, reason: str, **extra: Any) -> ToolResult:
    out = {"op": op, "status": "unknown", "value": None, "exact": False, "reason": reason}
    out.update(extra)
    return ToolResult(success=True, output=out)


TOOL_SPEC = {
    "name": "combinatorial_enumeration",
    "domain": "mathematics",
    "audience": "worker",
    "description": (
        "Exact combinatorial counting and (bounded) generation. op in "
        "{binomial, catalan, stirling, partitions, set_partitions, compositions, "
        "count_subsets_with_property, permutations_with_property, generate}. "
        "Predicate-based ops take a SMALL declarative predicate spec from a fixed "
        "allow-list (e.g. {'type':'sum_at_most','k':5}) — no code is eval'd. Returns "
        "EXACT counts by enumerating the full (capped) space; if the space exceeds the "
        "cap it returns status='unknown' with a reason instead of a truncated count."
    ),
    "params": {
        "op": {"type": "str", "required": True, "description": f"One of {sorted(_OPS)}."},
        "n": {"type": "int", "required": False,
              "description": "Integer parameter (universe {1..n}, or the integer for partitions/compositions/binomial/catalan/stirling)."},
        "k": {"type": "int", "required": False,
              "description": "Second integer (binomial C(n,k); stirling(n,k); exactly-k parts for compositions)."},
        "elements": {"type": "list", "required": False,
                     "description": "Explicit universe (distinct items). Overrides n for subset/permutation ops."},
        "predicate": {"type": "dict", "required": False,
                      "description": "Declarative predicate spec {'type':..., 'k':...}. See allow-list."},
        "kind": {"type": "str", "required": False,
                 "description": "For op='generate': one of subsets, permutations, partitions, set_partitions, compositions."},
        "mode": {"type": "str", "required": False,
                 "description": "'count' (default) or 'generate' for partitions/set_partitions/compositions."},
        "max_items": {"type": "int", "required": False,
                      "description": f"Max objects returned in the sample (default {DEFAULT_MAX_ITEMS}, hard cap {HARD_MAX_ITEMS})."},
        "stirling_kind": {"type": "int", "required": False,
                          "description": "Stirling kind: 2 (default, set partitions) or 1 (unsigned, cycles)."},
    },
    "output": {
        "op": "str — echoed op",
        "status": "str — 'ok' or 'unknown' (valid-but-oversized)",
        "value": "int|None — the exact count (matched count for predicate ops), None if unknown",
        "exact": "bool — True when value is an exact count",
        "space_size": "int — total size of the enumerated space (predicate/generate ops)",
        "objects": "list — bounded SAMPLE of objects (may be truncated)",
        "objects_truncated": "bool — True when 'objects' is a sample, not the whole set",
        "method": "str — how the count was obtained",
        "reason": "str — present when status='unknown'",
    },
    "example": {
        "params": {"op": "binomial", "n": 5, "k": 2},
        "output": {"op": "binomial", "status": "ok", "value": 10, "exact": True},
    },
}


# --------------------------------------------------------------------------- #
# predicate validation + evaluation (declarative; never eval)                 #
# --------------------------------------------------------------------------- #
def _validate_predicate(pred: Any, context: str, elements: list) -> str | None:
    """Return an error message if the predicate spec is invalid, else None."""
    if pred is None:
        return None
    if not isinstance(pred, dict):
        return "predicate must be an object like {'type': 'sum_at_most', 'k': 5}."
    t = pred.get("type")
    allowed = _SUBSET_PRED_TYPES if context == "subset" else _PERM_PRED_TYPES
    if t not in allowed:
        return f"unknown predicate type {t!r} for {context}; allowed: {sorted(allowed)}."
    if t in _PRED_NEEDS_K:
        if "k" not in pred:
            return f"predicate type {t!r} requires integer 'k'."
        try:
            int(pred["k"])
        except (TypeError, ValueError):
            return f"predicate 'k' must be an integer, got {pred.get('k')!r}."
    if t == "contains" and "element" not in pred:
        return "predicate type 'contains' requires 'element'."
    if context == "subset" and t in _NUMERIC_SUBSET_PRED:
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in elements):
            return f"predicate type {t!r} requires a numeric universe."
    return None


def _check_predicate(pred: dict | None, collection: tuple, base: list | None) -> bool:
    """Evaluate an allow-listed predicate. Assumes it has passed _validate_predicate."""
    if pred is None:
        return True
    t = pred["type"]
    k = pred.get("k")
    if t == "sum_at_most":
        return sum(collection) <= k
    if t == "sum_at_least":
        return sum(collection) >= k
    if t == "sum_equals":
        return sum(collection) == k
    if t == "size_at_most":
        return len(collection) <= k
    if t == "size_at_least":
        return len(collection) >= k
    if t == "size_equals":
        return len(collection) == k
    if t == "max_at_most":
        return (max(collection) <= k) if collection else True
    if t == "min_at_least":
        return (min(collection) >= k) if collection else True
    if t == "contains":
        return pred["element"] in collection
    if t == "no_two_consecutive":
        s = sorted(collection)
        return all(s[i + 1] - s[i] > 1 for i in range(len(s) - 1))
    # permutation predicates
    if t == "derangement":
        return all(collection[i] != base[i] for i in range(len(collection)))
    if t == "num_fixed_points_equals":
        return sum(1 for i in range(len(collection)) if collection[i] == base[i]) == k
    if t in ("num_inversions_at_most", "num_inversions_equals"):
        inv = sum(
            1
            for i in range(len(collection))
            for j in range(i + 1, len(collection))
            if collection[i] > collection[j]
        )
        return inv <= k if t == "num_inversions_at_most" else inv == k
    if t == "num_descents_equals":
        des = sum(1 for i in range(len(collection) - 1) if collection[i] > collection[i + 1])
        return des == k
    return False  # unreachable after validation


# --------------------------------------------------------------------------- #
# object generators                                                           #
# --------------------------------------------------------------------------- #
def _powerset(elements: list) -> Iterable[tuple]:
    for r in range(len(elements) + 1):
        yield from itertools.combinations(elements, r)


def _gen_partitions(n: int) -> Iterable[tuple]:
    from sympy.utilities.iterables import partitions
    for p in partitions(n):
        parts: list[int] = []
        for val, mult in sorted(p.items(), reverse=True):
            parts.extend([val] * mult)
        yield tuple(parts)


def _gen_set_partitions(elements: list) -> Iterable[tuple]:
    from sympy.utilities.iterables import multiset_partitions
    for part in multiset_partitions(list(elements)):
        yield tuple(tuple(block) for block in part)


def _gen_compositions(n: int, k: int | None) -> Iterable[tuple]:
    if n == 0:
        if k in (None, 0):
            yield tuple()
        return

    def rec(remaining: int, parts: list[int]):
        if k is not None and len(parts) == k:
            if remaining == 0:
                yield tuple(parts)
            return
        if remaining == 0:
            if k is None:
                yield tuple(parts)
            return
        for first in range(1, remaining + 1):
            yield from rec(remaining - first, parts + [first])

    yield from rec(n, [])


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _resolve_universe(n: Any, elements: Any) -> tuple[list | None, str | None]:
    """Return (universe_list, error). elements overrides n; else {1..n}."""
    if elements is not None:
        if not isinstance(elements, (list, tuple)):
            return None, "'elements' must be a list."
        uni = list(elements)
        if len(set(map(_hashable, uni))) != len(uni):
            return None, "'elements' must be distinct."
        return uni, None
    if n is None:
        return None, "provide 'n' (universe {1..n}) or 'elements'."
    try:
        ni = int(n)
    except (TypeError, ValueError):
        return None, f"'n' must be an integer, got {n!r}."
    if ni < 0:
        return None, "'n' must be non-negative."
    return list(range(1, ni + 1)), None


def _hashable(x: Any) -> Any:
    return tuple(x) if isinstance(x, list) else x


def _clamp_items(max_items: Any) -> int:
    if max_items is None:
        return DEFAULT_MAX_ITEMS
    try:
        mi = int(max_items)
    except (TypeError, ValueError):
        return DEFAULT_MAX_ITEMS
    return max(0, min(mi, HARD_MAX_ITEMS))


def _enumerate_and_count(
    op: str,
    objects_iter: Iterable[tuple],
    space_size: int,
    predicate: dict | None,
    base: list | None,
    max_items: int,
    method: str,
    nested: bool = False,
) -> ToolResult:
    """Full-space enumeration → exact matched count + bounded object sample."""
    matched = 0
    sample: list = []
    for obj in objects_iter:
        if _check_predicate(predicate, obj, base):
            matched += 1
            if len(sample) < max_items:
                sample.append([list(b) for b in obj] if nested else list(obj))
    return ToolResult(
        success=True,
        output={
            "op": op,
            "status": "ok",
            "value": matched,
            "exact": True,
            "space_size": space_size,
            "objects": sample,
            "objects_truncated": matched > len(sample),
            "predicate": predicate,
            "method": method,
        },
    )


# --------------------------------------------------------------------------- #
# main entry                                                                   #
# --------------------------------------------------------------------------- #
def combinatorial_enumeration(
    op: str | None = None,
    n: Any = None,
    k: Any = None,
    elements: Any = None,
    predicate: Any = None,
    kind: str | None = None,
    mode: str | None = None,
    max_items: Any = None,
    stirling_kind: Any = None,
) -> ToolResult:
    if op is None:
        return _err("Parameter 'op' is required.")
    if op not in _OPS:
        return _err(f"unknown op {op!r}; allowed: {sorted(_OPS)}.")
    try:
        max_items = _clamp_items(max_items)

        # ---- closed-form counts (exact, no enumeration) ------------------- #
        if op == "binomial":
            return _closed_form_binomial(n, k)
        if op == "catalan":
            return _closed_form_catalan(n)
        if op == "stirling":
            return _closed_form_stirling(n, k, stirling_kind)

        # ---- partitions / set_partitions / compositions ------------------- #
        if op == "partitions":
            return _op_partitions(n, mode, max_items)
        if op == "set_partitions":
            return _op_set_partitions(n, elements, mode, max_items)
        if op == "compositions":
            return _op_compositions(n, k, mode, max_items)

        # ---- predicate-filtered subsets / permutations -------------------- #
        if op == "count_subsets_with_property":
            return _op_subsets(n, elements, predicate, max_items, op)
        if op == "permutations_with_property":
            return _op_permutations(n, elements, predicate, max_items, op)

        # ---- generic generate --------------------------------------------- #
        if op == "generate":
            return _op_generate(n, k, elements, predicate, kind, max_items)

        return _err(f"unhandled op {op!r}.")
    except Exception as exc:  # noqa: BLE001 — never raise to the caller
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))


# --------------------------------------------------------------------------- #
# closed-form ops                                                              #
# --------------------------------------------------------------------------- #
def _as_int(x: Any, name: str) -> tuple[int | None, str | None]:
    try:
        return int(x), None
    except (TypeError, ValueError):
        return None, f"'{name}' must be an integer, got {x!r}."


def _closed_form_binomial(n: Any, k: Any) -> ToolResult:
    from sympy import binomial as sym_binomial
    ni, e = _as_int(n, "n")
    if e:
        return _err(e)
    ki, e = _as_int(k, "k")
    if e:
        return _err(e)
    if ni < 0:
        return _err("binomial requires n >= 0.")
    if abs(ni) > MAX_FORMULA_N:
        return _unknown("binomial", f"n exceeds formula cap {MAX_FORMULA_N}.")
    val = int(sym_binomial(ni, ki))
    return ToolResult(success=True, output={
        "op": "binomial", "status": "ok", "value": val, "exact": True,
        "method": "sympy.binomial (exact)", "n": ni, "k": ki})


def _closed_form_catalan(n: Any) -> ToolResult:
    from sympy import catalan as sym_catalan
    ni, e = _as_int(n, "n")
    if e:
        return _err(e)
    if ni < 0:
        return _err("catalan requires n >= 0.")
    if ni > MAX_FORMULA_N:
        return _unknown("catalan", f"n exceeds formula cap {MAX_FORMULA_N}.")
    return ToolResult(success=True, output={
        "op": "catalan", "status": "ok", "value": int(sym_catalan(ni)), "exact": True,
        "method": "sympy.catalan (exact)", "n": ni})


def _closed_form_stirling(n: Any, k: Any, stirling_kind: Any) -> ToolResult:
    from sympy.functions.combinatorial.numbers import stirling as sym_stirling
    ni, e = _as_int(n, "n")
    if e:
        return _err(e)
    ki, e = _as_int(k, "k")
    if e:
        return _err(e)
    if ni < 0 or ki < 0:
        return _err("stirling requires n >= 0 and k >= 0.")
    kind = 2 if stirling_kind is None else int(stirling_kind)
    if kind not in (1, 2):
        return _err("stirling_kind must be 1 (unsigned, cycles) or 2 (default, set partitions).")
    if ni > MAX_FORMULA_N:
        return _unknown("stirling", f"n exceeds formula cap {MAX_FORMULA_N}.")
    val = int(sym_stirling(ni, ki, kind=kind, signed=False))
    return ToolResult(success=True, output={
        "op": "stirling", "status": "ok", "value": val, "exact": True,
        "method": f"sympy.stirling kind={kind} (exact)", "n": ni, "k": ki, "kind": kind})


# --------------------------------------------------------------------------- #
# partition-family ops                                                         #
# --------------------------------------------------------------------------- #
def _resolve_mode(mode: str | None) -> tuple[str | None, str | None]:
    m = (mode or "count").lower()
    if m not in ("count", "generate"):
        return None, "mode must be 'count' or 'generate'."
    return m, None


def _op_partitions(n: Any, mode: str | None, max_items: int) -> ToolResult:
    from sympy import npartitions
    m, e = _resolve_mode(mode)
    if e:
        return _err(e)
    ni, e = _as_int(n, "n")
    if e:
        return _err(e)
    if ni < 0:
        return _err("partitions requires n >= 0.")
    if m == "count":
        if ni > MAX_FORMULA_N:
            return _unknown("partitions", f"n exceeds formula cap {MAX_FORMULA_N}.")
        return ToolResult(success=True, output={
            "op": "partitions", "status": "ok", "value": int(npartitions(ni)),
            "exact": True, "mode": "count", "method": "sympy.npartitions p(n) (exact)", "n": ni})
    # generate
    if ni > MAX_PARTITION_N_GEN:
        return _unknown("partitions", f"n={ni} exceeds generation cap {MAX_PARTITION_N_GEN}.",
                        space_size=int(npartitions(ni)))
    space = int(npartitions(ni))
    return _enumerate_and_count("partitions", _gen_partitions(ni), space, None, None,
                                max_items, "sympy.iterables.partitions (full enumeration)")


def _op_set_partitions(n: Any, elements: Any, mode: str | None, max_items: int) -> ToolResult:
    from sympy import bell
    m, e = _resolve_mode(mode)
    if e:
        return _err(e)
    uni, e = _resolve_universe(n, elements)
    if e:
        return _err(e)
    size = len(uni)
    if m == "count":
        if size > MAX_FORMULA_N:
            return _unknown("set_partitions", f"|universe| exceeds formula cap {MAX_FORMULA_N}.")
        return ToolResult(success=True, output={
            "op": "set_partitions", "status": "ok", "value": int(bell(size)),
            "exact": True, "mode": "count", "method": "sympy.bell Bell(n) (exact)", "n": size})
    if size > MAX_SETPART_N_GEN:
        return _unknown("set_partitions", f"|universe|={size} exceeds generation cap {MAX_SETPART_N_GEN}.",
                        space_size=int(bell(size)))
    return _enumerate_and_count("set_partitions", _gen_set_partitions(uni), int(bell(size)),
                                None, None, max_items,
                                "sympy.iterables.multiset_partitions (full enumeration)", nested=True)


def _op_compositions(n: Any, k: Any, mode: str | None, max_items: int) -> ToolResult:
    from sympy import binomial as sym_binomial
    m, e = _resolve_mode(mode)
    if e:
        return _err(e)
    ni, e = _as_int(n, "n")
    if e:
        return _err(e)
    if ni < 0:
        return _err("compositions requires n >= 0.")
    ki = None
    if k is not None:
        ki, e = _as_int(k, "k")
        if e:
            return _err(e)
        if ki < 0:
            return _err("compositions 'k' (number of parts) must be >= 0.")
    # exact count formula
    if ki is None:
        count = 1 if ni == 0 else 2 ** (ni - 1)
    else:
        count = int(sym_binomial(ni - 1, ki - 1)) if ni >= 1 and ki >= 1 else (1 if (ni == 0 and ki == 0) else 0)
    if m == "count":
        return ToolResult(success=True, output={
            "op": "compositions", "status": "ok", "value": int(count), "exact": True,
            "mode": "count", "method": "closed form (2^(n-1) or C(n-1,k-1))", "n": ni, "k": ki})
    if ni > MAX_COMPOSITION_N_GEN:
        return _unknown("compositions", f"n={ni} exceeds generation cap {MAX_COMPOSITION_N_GEN}.",
                        space_size=int(count))
    return _enumerate_and_count("compositions", _gen_compositions(ni, ki), int(count),
                                None, None, max_items, "recursive enumeration (full)")


# --------------------------------------------------------------------------- #
# predicate-filtered subset / permutation ops                                 #
# --------------------------------------------------------------------------- #
def _op_subsets(n: Any, elements: Any, predicate: Any, max_items: int, op: str) -> ToolResult:
    uni, e = _resolve_universe(n, elements)
    if e:
        return _err(e)
    e = _validate_predicate(predicate, "subset", uni)
    if e:
        return _err(e)
    m = len(uni)
    if m > MAX_SUBSET_UNIVERSE:
        return _unknown(op, f"|universe|={m} exceeds subset cap {MAX_SUBSET_UNIVERSE} (2^{m} space).",
                        space_size=None)
    return _enumerate_and_count(op, _powerset(uni), 2 ** m, predicate, None, max_items,
                                "exhaustive powerset enumeration (exact count)")


def _op_permutations(n: Any, elements: Any, predicate: Any, max_items: int, op: str) -> ToolResult:
    uni, e = _resolve_universe(n, elements)
    if e:
        return _err(e)
    e = _validate_predicate(predicate, "permutation", uni)
    if e:
        return _err(e)
    m = len(uni)
    if m > MAX_PERM_UNIVERSE:
        return _unknown(op, f"|universe|={m} exceeds permutation cap {MAX_PERM_UNIVERSE} ({m}! space).",
                        space_size=None)
    return _enumerate_and_count(op, itertools.permutations(uni), math.factorial(m),
                                predicate, list(uni), max_items,
                                "exhaustive permutation enumeration (exact count)")


# --------------------------------------------------------------------------- #
# generic generate                                                             #
# --------------------------------------------------------------------------- #
def _op_generate(n: Any, k: Any, elements: Any, predicate: Any, kind: str | None, max_items: int) -> ToolResult:
    if kind is None:
        return _err("op 'generate' requires 'kind' in {subsets, permutations, partitions, set_partitions, compositions}.")
    kind = kind.lower()
    if kind == "subsets":
        return _op_subsets(n, elements, predicate, max_items, "generate")
    if kind == "permutations":
        return _op_permutations(n, elements, predicate, max_items, "generate")
    if kind == "partitions":
        return _op_partitions(n, "generate", max_items)
    if kind == "set_partitions":
        return _op_set_partitions(n, elements, "generate", max_items)
    if kind == "compositions":
        return _op_compositions(n, k, "generate", max_items)
    return _err(f"unknown generate kind {kind!r}.")
