"""General counterexample-search framework (M5, domain-capabilities §1d).

A universal claim "for all x in S, P(x)" is best attacked by trying to BREAK it. This
tool lets a caller declare a parametrized FINITE search space S (integers in a range,
k-tuples over a range, or small labeled graphs) plus a SIMPLE allow-listed predicate P
(a small enum of predicate types with parameters — NOT arbitrary code), and searches
exhaustively (when the space is small enough) or randomly for a witness where P is FALSE.

Honesty by construction (domain-capabilities §0):
  * A counterexample is SELF-CERTIFYING: the tool re-evaluates the predicate on the
    witness from scratch and returns the concrete computed values, so the caller can
    re-check that P(x) is indeed false without trusting the search.
  * "None found" is EVIDENCE, never a proof of the universal claim. The result carries
    ``found: false``, ``exhausted: bool`` and ``checked: N``. ``exhausted: true`` means
    the ENTIRE declared FINITE space was enumerated with no counterexample — that proves
    the claim ONLY over that finite space, and the tool says so explicitly; it NEVER
    upgrades this to a universal theorem.
  * Predicate expression terms are parsed with a RESTRICTED, arithmetic-only sympy parser
    (whitelist namespace, ``__builtins__`` stripped, dunder/import/attribute screen) and
    evaluated EXACTLY on integer points — no eval, no float error.
  * The search is bounded by a point cap and a wall-clock budget; a budget cut yields
    ``exhausted: false`` (honest partial evidence). The tool never raises to the caller.
"""
from __future__ import annotations

import itertools
import random
import re
import time

import sympy as _sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

from propab.tools.types import ToolError, ToolResult

# --------------------------------------------------------------------------- #
# Restricted, ARITHMETIC-ONLY parsing (exact integer evaluation; no transcendentals
# so every point evaluates to an exact rational — no float error can hide a witness).
# --------------------------------------------------------------------------- #
_MAX_EXPR_LEN = 2000
_MAX_TUPLE_K = 10
_MAX_GRAPH_N = 8
_DEFAULT_MAX_CHECKS = 200_000
_MAX_MAX_CHECKS = 2_000_000
_DEFAULT_TIME_BUDGET = 10.0
_MAX_TIME_BUDGET = 60.0

_FORBIDDEN = re.compile(
    r"(__|\bimport\b|\bexec\b|\beval\b|\blambda\b|\bos\b|\bsys\b|\bsubprocess\b"
    r"|\bopen\b|\bcompile\b|\bglobals\b|\blocals\b|\bgetattr\b|\bsetattr\b"
    r"|\binput\b|`|\\)"
)
_ATTR_RE = re.compile(r"[A-Za-z_)\]]\s*\.\s*[A-Za-z_]")
_TRANSFORMS = standard_transformations + (convert_xor,)

_ALLOWED_NAME_LIST = (
    "Symbol", "Integer", "Rational", "Abs", "sign", "factorial", "binomial",
    "floor", "ceiling", "Min", "Max", "Mod", "gcd", "lcm",
)

_REL_OPS = ("<=", "<", ">=", ">", "==", "!=")
_PRED_TYPES = (
    "relation", "divides", "congruence", "is_prime", "is_composite",
    "is_square", "positive", "nonnegative", "nonzero",
)
_GRAPH_INVARIANTS = (
    "n_vertices", "n_edges", "max_degree", "min_degree", "n_triangles",
    "is_connected", "n_components", "is_planar", "girth", "diameter",
)


def _allowed_namespace() -> dict:
    ns: dict = {}
    for nm in _ALLOWED_NAME_LIST:
        obj = getattr(_sp, nm, None)
        if obj is not None:
            ns[nm] = obj
    ns["__builtins__"] = {}
    return ns


def _safe_sympify(text, symbol_names=None):
    if not isinstance(text, str):
        raise ValueError("expression must be a string")
    s = text.strip()
    if not s:
        raise ValueError("expression must be non-empty")
    if len(s) > _MAX_EXPR_LEN:
        raise ValueError(f"expression exceeds max length {_MAX_EXPR_LEN}")
    if _FORBIDDEN.search(s):
        raise ValueError("expression contains a forbidden token")
    if _ATTR_RE.search(s):
        raise ValueError("attribute access is not permitted")
    local_dict = {}
    for nm in symbol_names or []:
        if isinstance(nm, str) and nm:
            local_dict[nm] = _sp.Symbol(nm)
    return parse_expr(
        s, local_dict=local_dict, global_dict=_allowed_namespace(),
        transformations=_TRANSFORMS, evaluate=True,
    )


TOOL_SPEC = {
    "name": "counterexample_search",
    "domain": "mathematics",
    "audience": "worker",
    "verification_capable": True,
    "description": (
        "Search a declared FINITE space for a counterexample to a universal claim "
        "'for all x, P(x)'. 'space' is one of: {'type':'integers','low','high'}, "
        "{'type':'tuples','low','high','k'}, or {'type':'graphs','n'} (labeled graphs). "
        "'predicate' is an allow-listed type (NOT code): relation (lhs op rhs), divides, "
        "congruence, is_prime, is_composite, is_square, positive, nonnegative, nonzero "
        "(expression terms parsed in a RESTRICTED arithmetic-only namespace, evaluated "
        "EXACTLY), or graph_relation over invariants (n_edges, max_degree, n_triangles, "
        "is_connected, is_planar, girth, diameter, ...). Searches exhaustively when the "
        "space fits under the cap, else randomly. A counterexample is self-certifying "
        "(the predicate is re-evaluated on it and the values returned). 'none found' "
        "returns found=false with exhausted (whole finite space checked?) and checked=N "
        "— it NEVER claims the universal statement proven."
    ),
    "params": {
        "space": {"type": "dict", "required": True,
                  "description": "Finite space: {'type':'integers','low':a,'high':b,'var':'n'} "
                                 "| {'type':'tuples','low':a,'high':b,'k':k,'vars':[...]} "
                                 "| {'type':'graphs','n':n}."},
        "predicate": {"type": "dict", "required": True,
                      "description": "The claim P asserted true for all x. See description "
                                     "for allow-listed predicate types and their params."},
        "max_checks": {"type": "int", "required": False, "default": 200000,
                       "description": f"Point cap; exhaustive if |space| <= this, else random "
                                      f"sample this many (<= {_MAX_MAX_CHECKS})."},
        "seed": {"type": "int", "required": False, "default": 0,
                 "description": "RNG seed for random sampling (determinism)."},
        "time_budget_sec": {"type": "number", "required": False, "default": 10,
                            "description": "Wall-clock search budget (<= 60); a cut => not exhausted."},
    },
    "output": {
        "found": "bool — a counterexample was found",
        "counterexample": "dict|None — the witness point + re-evaluated predicate values (self-certifying)",
        "exhausted": "bool — the ENTIRE finite space was enumerated (proof over that finite space only)",
        "checked": "int — number of points evaluated",
        "space_size": "int|str — size of the declared space",
        "skipped": "int — points where the predicate could not be evaluated",
        "mode": "str — 'exhaustive' or 'random'",
        "note": "str — explicit statement of what a null result does and does NOT prove",
    },
    "example": {
        "params": {"space": {"type": "integers", "low": 0, "high": 50, "var": "n"},
                   "predicate": {"type": "is_prime", "expr": "n**2 - n + 41"}},
        "output": {"found": True, "counterexample": {"point": {"n": 41},
                   "evaluation": {"value": 1681, "is_prime": False}}, "exhausted": False},
    },
}


def _validation_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="validation_error", message=message))


def _execution_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="execution_error", message=message))


def _to_int_exact(val):
    """Return a Python int iff ``val`` is an exact integer, else None."""
    try:
        if getattr(val, "is_integer", False) and val.is_integer:
            return int(val)
    except (TypeError, ValueError):
        return None
    if isinstance(val, int):
        return int(val)
    return None


# --------------------------------------------------------------------------- #
# Predicate compilation for integer / tuple spaces (expression-based).
# --------------------------------------------------------------------------- #
def _compile_arith_predicate(predicate, syms):
    """Return (fn, err). fn(mapping) -> (holds: bool|None, detail: dict).

    ``holds is None`` means the predicate could not be evaluated at that point (skipped).
    """
    ptype = str(predicate.get("type", "")).strip().lower()
    names = [s.name for s in syms]

    def _parse(field):
        raw = predicate.get(field)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"predicate '{ptype}' requires a string field {field!r}.")
        return _safe_sympify(raw, names)

    try:
        if ptype == "relation":
            lhs = _parse("lhs")
            rhs = _parse("rhs")
            op = str(predicate.get("op", "")).strip()
            if op not in _REL_OPS:
                return None, f"relation 'op' must be one of {list(_REL_OPS)}; got {op!r}."

            def fn(mapping):
                lv = lhs.subs(mapping)
                rv = rhs.subs(mapping)
                if not (getattr(lv, "is_number", False) and getattr(rv, "is_number", False)):
                    return None, {"reason": "non-numeric evaluation"}
                d = lv - rv
                try:
                    if op == "<=":
                        holds = bool(d <= 0)
                    elif op == "<":
                        holds = bool(d < 0)
                    elif op == ">=":
                        holds = bool(d >= 0)
                    elif op == ">":
                        holds = bool(d > 0)
                    elif op == "==":
                        holds = bool(d == 0)
                    else:  # "!="
                        holds = bool(d != 0)
                except TypeError:
                    return None, {"reason": "non-real comparison"}
                return holds, {"lhs_value": str(lv), "rhs_value": str(rv), "op": op}

            return fn, None

        if ptype == "divides":
            div_raw = predicate.get("divisor")
            dividend = _parse("dividend")
            divisor_expr = _safe_sympify(str(div_raw), names) if not isinstance(div_raw, bool) and div_raw is not None else None
            if divisor_expr is None:
                return None, "predicate 'divides' requires a 'divisor' (int or expression)."

            def fn(mapping):
                dv = _to_int_exact(divisor_expr.subs(mapping))
                nv = _to_int_exact(dividend.subs(mapping))
                if nv is None:
                    return False, {"dividend_is_integer": False,
                                   "dividend_value": str(dividend.subs(mapping))}
                if dv is None:
                    return None, {"reason": "divisor not an integer"}
                if dv == 0:
                    holds = (nv == 0)
                else:
                    holds = (nv % dv == 0)
                return holds, {"divisor": dv, "dividend": nv}

            return fn, None

        if ptype == "congruence":
            expr = _parse("expr")
            mod = predicate.get("modulus")
            res = predicate.get("residue", 0)
            if isinstance(mod, bool) or not isinstance(mod, int) or mod <= 0:
                return None, "predicate 'congruence' requires an integer 'modulus' > 0."
            if isinstance(res, bool) or not isinstance(res, int):
                return None, "predicate 'congruence' requires an integer 'residue'."

            def fn(mapping):
                v = _to_int_exact(expr.subs(mapping))
                if v is None:
                    return False, {"expr_is_integer": False, "expr_value": str(expr.subs(mapping))}
                holds = ((v - res) % mod == 0)
                return holds, {"expr_value": v, "modulus": mod, "residue": res,
                               "expr_mod": v % mod}

            return fn, None

        if ptype in ("is_prime", "is_composite", "is_square", "positive",
                     "nonnegative", "nonzero"):
            expr = _parse("expr")

            def fn(mapping):
                raw = expr.subs(mapping)
                v = _to_int_exact(raw)
                if ptype == "is_prime":
                    if v is None:
                        return False, {"value": str(raw), "is_integer": False}
                    return bool(_sp.isprime(v)), {"value": v, "is_prime": bool(_sp.isprime(v))}
                if ptype == "is_composite":
                    if v is None:
                        return False, {"value": str(raw), "is_integer": False}
                    comp = (v > 1) and not _sp.isprime(v)
                    return bool(comp), {"value": v, "is_composite": bool(comp)}
                if ptype == "is_square":
                    if v is None or v < 0:
                        return False, {"value": str(raw), "is_perfect_square": False}
                    r = _sp.integer_nthroot(v, 2)
                    return bool(r[1]), {"value": v, "is_perfect_square": bool(r[1])}
                # sign predicates work on rationals too
                if not getattr(raw, "is_number", False) or not getattr(raw, "is_real", False):
                    return None, {"reason": "non-real value"}
                if ptype == "positive":
                    return bool(raw > 0), {"value": str(raw)}
                if ptype == "nonnegative":
                    return bool(raw >= 0), {"value": str(raw)}
                return bool(raw != 0), {"value": str(raw)}  # nonzero

            return fn, None

        return None, f"unknown predicate type {ptype!r}; allowed: {list(_PRED_TYPES)}."
    except (ValueError, SyntaxError, TypeError) as exc:
        return None, str(exc)


# --------------------------------------------------------------------------- #
# Graph predicate (invariant relation).
# --------------------------------------------------------------------------- #
def _graph_invariant(name, G, nx):
    name = str(name).strip().lower()
    if name in ("n_vertices", "n"):
        return G.number_of_nodes()
    if name in ("n_edges", "m"):
        return G.number_of_edges()
    if name == "max_degree":
        degs = [d for _, d in G.degree()]
        return max(degs) if degs else 0
    if name == "min_degree":
        degs = [d for _, d in G.degree()]
        return min(degs) if degs else 0
    if name == "n_triangles":
        return sum(nx.triangles(G).values()) // 3
    if name == "is_connected":
        return 1 if (G.number_of_nodes() > 0 and nx.is_connected(G)) else 0
    if name == "n_components":
        return nx.number_connected_components(G)
    if name == "is_planar":
        return 1 if nx.check_planarity(G)[0] else 0
    if name == "girth":
        # shortest cycle length, or a large sentinel when acyclic
        try:
            g = nx.girth(G)  # networkx >= 3.1
            return int(g) if g != float("inf") else 10 ** 9
        except Exception:  # noqa: BLE001
            return 10 ** 9 if nx.is_forest(G) else _girth_bfs(G)
    if name == "diameter":
        if G.number_of_nodes() > 0 and nx.is_connected(G):
            return nx.diameter(G)
        return 10 ** 9  # disconnected -> infinite diameter sentinel
    raise ValueError(f"unknown graph invariant {name!r}; allowed: {list(_GRAPH_INVARIANTS)}")


def _girth_bfs(G):
    import networkx as nx  # local

    best = 10 ** 9
    for src in G.nodes():
        dist = {src: 0}
        parent = {src: None}
        queue = [src]
        while queue:
            u = queue.pop(0)
            for w in G.neighbors(u):
                if w not in dist:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    queue.append(w)
                elif parent[u] != w:
                    best = min(best, dist[u] + dist[w] + 1)
    return best


def _compile_graph_predicate(predicate):
    ptype = str(predicate.get("type", "")).strip().lower()
    if ptype != "graph_relation":
        return None, (f"space 'graphs' requires predicate type 'graph_relation'; got {ptype!r}.")
    left = predicate.get("left")
    right = predicate.get("right")
    op = str(predicate.get("op", "")).strip()
    if op not in _REL_OPS:
        return None, f"graph_relation 'op' must be one of {list(_REL_OPS)}; got {op!r}."
    if not isinstance(left, str) or left.strip().lower() not in _GRAPH_INVARIANTS:
        return None, f"'left' must be a graph invariant in {list(_GRAPH_INVARIANTS)}."
    right_is_int = isinstance(right, int) and not isinstance(right, bool)
    if not right_is_int and (not isinstance(right, str) or right.strip().lower() not in _GRAPH_INVARIANTS):
        return None, "'right' must be a graph invariant name or an integer."

    def fn(G_nx):
        G, nx = G_nx
        lv = _graph_invariant(left, G, nx)
        rv = right if right_is_int else _graph_invariant(right, G, nx)
        d = lv - rv
        if op == "<=":
            holds = d <= 0
        elif op == "<":
            holds = d < 0
        elif op == ">=":
            holds = d >= 0
        elif op == ">":
            holds = d > 0
        elif op == "==":
            holds = d == 0
        else:
            holds = d != 0
        return bool(holds), {"left": left, "left_value": lv, "op": op,
                             "right": right, "right_value": rv}

    return fn, None


# --------------------------------------------------------------------------- #
# Space enumeration.
# --------------------------------------------------------------------------- #
def _build_space(space):
    """Return (kind, meta, size:int, syms) or raise ValueError."""
    if not isinstance(space, dict):
        raise ValueError("'space' must be a dict.")
    stype = str(space.get("type", "")).strip().lower()
    if stype == "integers":
        low, high = space.get("low"), space.get("high")
        if not _is_int(low) or not _is_int(high):
            raise ValueError("integers space needs integer 'low' and 'high'.")
        if high < low:
            raise ValueError("integers space needs high >= low.")
        var = str(space.get("var", "n"))
        syms = [_sp.Symbol(var)]
        return "integers", {"low": int(low), "high": int(high), "var": var}, (int(high) - int(low) + 1), syms
    if stype == "tuples":
        low, high, k = space.get("low"), space.get("high"), space.get("k")
        if not _is_int(low) or not _is_int(high) or not _is_int(k):
            raise ValueError("tuples space needs integer 'low', 'high', 'k'.")
        if high < low:
            raise ValueError("tuples space needs high >= low.")
        if not (1 <= int(k) <= _MAX_TUPLE_K):
            raise ValueError(f"tuples 'k' must be in [1, {_MAX_TUPLE_K}].")
        vnames = space.get("vars")
        if vnames is None:
            vnames = [f"x{i}" for i in range(int(k))]
        if len(vnames) != int(k):
            raise ValueError("tuples 'vars' length must equal 'k'.")
        syms = [_sp.Symbol(str(v)) for v in vnames]
        size = (int(high) - int(low) + 1) ** int(k)
        return "tuples", {"low": int(low), "high": int(high), "k": int(k),
                          "vars": [str(v) for v in vnames]}, size, syms
    if stype == "graphs":
        n = space.get("n")
        if not _is_int(n) or int(n) < 1:
            raise ValueError("graphs space needs integer 'n' >= 1.")
        if int(n) > _MAX_GRAPH_N:
            raise ValueError(f"graphs 'n' capped at {_MAX_GRAPH_N}.")
        n = int(n)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        size = 2 ** len(edges)
        return "graphs", {"n": n, "edges": edges}, size, []
    raise ValueError(f"unknown space type {stype!r}; use 'integers', 'tuples' or 'graphs'.")


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _iter_points(kind, meta, exhaustive, n_draws, rng):
    """Yield points (mapping for arith, or (edges_mask) for graphs)."""
    if kind == "integers":
        low, high = meta["low"], meta["high"]
        if exhaustive:
            for v in range(low, high + 1):
                yield v
        else:
            for _ in range(n_draws):
                yield rng.randint(low, high)
    elif kind == "tuples":
        low, high, k = meta["low"], meta["high"], meta["k"]
        if exhaustive:
            for tup in itertools.product(range(low, high + 1), repeat=k):
                yield tup
        else:
            for _ in range(n_draws):
                yield tuple(rng.randint(low, high) for _ in range(k))
    else:  # graphs
        E = len(meta["edges"])
        if exhaustive:
            for mask in range(2 ** E):
                yield mask
        else:
            hi = 2 ** E
            for _ in range(n_draws):
                yield rng.randrange(hi)


def counterexample_search(space=None, predicate=None, max_checks=_DEFAULT_MAX_CHECKS,
                          seed=0, time_budget_sec=_DEFAULT_TIME_BUDGET) -> ToolResult:
    # ---- input validation (never raise) ----
    if not isinstance(predicate, dict) or "type" not in predicate:
        return _validation_error("'predicate' must be a dict with a 'type'.")
    try:
        kind, meta, size, syms = _build_space(space)
    except ValueError as exc:
        return _validation_error(str(exc))
    try:
        cap = int(max_checks)
    except (TypeError, ValueError):
        cap = _DEFAULT_MAX_CHECKS
    cap = max(1, min(_MAX_MAX_CHECKS, cap))
    try:
        seed_i = int(seed)
    except (TypeError, ValueError):
        seed_i = 0
    try:
        budget = float(time_budget_sec)
    except (TypeError, ValueError):
        budget = _DEFAULT_TIME_BUDGET
    budget = max(0.1, min(_MAX_TIME_BUDGET, budget))

    # ---- compile the predicate ----
    nx = None
    if kind == "graphs":
        try:
            import networkx as _nx
            nx = _nx
        except Exception as exc:  # noqa: BLE001
            return _execution_error(f"networkx unavailable for graph search: {exc}")
        fn, err = _compile_graph_predicate(predicate)
    else:
        fn, err = _compile_arith_predicate(predicate, syms)
    if err:
        return _validation_error(err)

    # ---- decide mode ----
    exhaustive = size <= cap
    n_draws = size if exhaustive else cap
    rng = random.Random(seed_i)
    deadline = time.monotonic() + budget

    checked = 0
    skipped = 0
    deadline_hit = False

    try:
        for i, point in enumerate(_iter_points(kind, meta, exhaustive, n_draws, rng)):
            if (i & 511) == 0 and time.monotonic() > deadline:
                deadline_hit = True
                break

            if kind == "graphs":
                G = _mask_to_graph(point, meta, nx)
                holds, detail = fn((G, nx))
                point_repr = {"edges": _mask_to_edge_list(point, meta)}
            else:
                mapping = _point_mapping(kind, point, syms)
                holds, detail = fn(mapping)
                point_repr = {s.name: int(v) for s, v in mapping.items()}

            if holds is None:
                skipped += 1
                continue
            checked += 1

            if holds is False:
                # SELF-CERTIFY: re-evaluate the predicate from scratch on the witness.
                if kind == "graphs":
                    recheck_holds, recheck_detail = fn((_mask_to_graph(point, meta, nx), nx))
                else:
                    recheck_holds, recheck_detail = fn(_point_mapping(kind, point, syms))
                return ToolResult(success=True, output={
                    "found": True,
                    "counterexample": {
                        "point": point_repr,
                        "predicate_holds": False,
                        "evaluation": detail,
                        "recheck_predicate_holds": bool(recheck_holds) if recheck_holds is not None else None,
                        "recheck_evaluation": recheck_detail,
                    },
                    "exhausted": False,
                    "checked": checked,
                    "space_size": _size_repr(size),
                    "skipped": skipped,
                    "mode": "exhaustive" if exhaustive else "random",
                    "note": ("COUNTEREXAMPLE found: P is FALSE at the returned point. It is "
                             "self-certifying — the predicate was re-evaluated from scratch on "
                             "the witness (see recheck_*); re-check the listed values yourself. "
                             "One counterexample disproves the universal claim."),
                })

        # ---- no counterexample ----
        fully_exhausted = exhaustive and not deadline_hit
        if fully_exhausted:
            note = (
                f"No counterexample in the ENTIRE declared finite space ({_size_repr(size)} "
                f"points, all checked). This PROVES the claim over THIS finite space only — "
                "it is NOT a proof of the universal statement over any larger/infinite domain.")
        elif deadline_hit:
            note = (
                f"No counterexample in the {checked} points checked before the "
                f"{budget:g}s budget expired. This is PARTIAL evidence only — the space was "
                "NOT exhausted; the claim is neither proven nor disproven.")
        else:
            note = (
                f"No counterexample in {checked} RANDOMLY sampled points (space too large to "
                f"enumerate: {_size_repr(size)}). This is EVIDENCE, not a proof — absence of a "
                "counterexample in a sample never establishes a universal claim.")

        return ToolResult(success=True, output={
            "found": False,
            "counterexample": None,
            "exhausted": bool(fully_exhausted),
            "checked": checked,
            "space_size": _size_repr(size),
            "skipped": skipped,
            "mode": "exhaustive" if exhaustive else "random",
            "budget_expired": bool(deadline_hit),
            "note": note,
        })
    except Exception as exc:  # noqa: BLE001 — never raise to the caller
        return _execution_error(f"counterexample_search failed: {exc}")


def _point_mapping(kind, point, syms):
    if kind == "integers":
        return {syms[0]: _sp.Integer(point)}
    return {s: _sp.Integer(v) for s, v in zip(syms, point)}


def _mask_to_edge_list(mask, meta):
    edges = meta["edges"]
    return [[i, j] for idx, (i, j) in enumerate(edges) if (mask >> idx) & 1]


def _mask_to_graph(mask, meta, nx):
    G = nx.Graph()
    G.add_nodes_from(range(meta["n"]))
    for idx, (i, j) in enumerate(meta["edges"]):
        if (mask >> idx) & 1:
            G.add_edge(i, j)
    return G


def _size_repr(size):
    # keep small sizes as int; render astronomically large ones as a string.
    return int(size) if size <= 10 ** 15 else str(size)
