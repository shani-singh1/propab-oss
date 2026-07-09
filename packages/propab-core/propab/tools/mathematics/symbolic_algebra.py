"""Symbolic-algebra dispatch tool (M1, general-agent redesign).

A single, safe entry point a general worker agent uses for the repetitive
"do the algebra/calculus for me" workflow: simplify / expand / factor / solve /
differentiate / integrate / limit / series / substitute. Exact symbolic
computation via sympy.

Honesty / safety by construction:
  * Expressions are parsed with a RESTRICTED sympy parser — a curated whitelist
    namespace with ``__builtins__`` stripped, unknown names auto-symbolised, and
    a screen for dunders / imports / attribute access — so there is NO arbitrary
    ``eval`` of caller input.
  * Every op is size-capped and run under a best-effort timeout; on timeout the
    tool returns an ``execution_error`` (never hangs, never guesses).
  * Results are returned as exact strings + LaTeX. An agent that needs to TRUST
    an equality should route it through ``symbolic_verify_identity`` (the
    independent verifier) rather than reading a bare result here.
"""
from __future__ import annotations

import re
import threading

import sympy as _sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
)

from propab.tools.types import ToolError, ToolResult

# --------------------------------------------------------------------------- #
# Restricted parsing (shared shape across the mathematics tools, duplicated so
# each tool file stays self-contained per the cluster's one-file-per-tool rule).
# --------------------------------------------------------------------------- #
_MAX_EXPR_LEN = 4000
_MAX_FREE_SYMBOLS = 12
_MAX_ORDER = 20
_MAX_SERIES_TERMS = 40
_DEFAULT_TIMEOUT = 10.0
_MAX_TIMEOUT = 60.0

# Tokens that must never appear in a caller expression. The restricted namespace
# already neutralises name resolution (unknown names become Symbols and there are
# no builtins), but this is defence-in-depth against obvious injection attempts.
_FORBIDDEN = re.compile(
    r"(__|\bimport\b|\bexec\b|\beval\b|\blambda\b|\bos\b|\bsys\b|\bsubprocess\b"
    r"|\bopen\b|\bcompile\b|\bglobals\b|\blocals\b|\bgetattr\b|\bsetattr\b"
    r"|\binput\b|\beval\b|`|\\)"
)
# Attribute access on a name / closing-bracket (e.g. ``x.__class__`` or ``f(x).y``)
# is rejected; decimals like ``3.14`` are digit-dotted and never match.
_ATTR_RE = re.compile(r"[A-Za-z_)\]]\s*\.\s*[A-Za-z_]")
_TRANSFORMS = standard_transformations + (convert_xor,)

# Curated, safe namespace. Constructors used by parse_expr's codegen (Integer,
# Float, Rational, Symbol) MUST be present; everything else is standard math.
_ALLOWED_NAME_LIST = (
    "Symbol", "Integer", "Float", "Rational", "symbols", "Function",
    "pi", "E", "I", "oo", "zoo", "nan", "GoldenRatio", "EulerGamma", "Catalan",
    "sqrt", "cbrt", "root", "exp", "log", "Abs", "sign", "floor", "ceiling", "frac",
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "atan2", "acot",
    "sinh", "cosh", "tanh", "coth", "asinh", "acosh", "atanh",
    "factorial", "gamma", "loggamma", "digamma", "beta", "binomial", "rf", "ff",
    "Min", "Max", "re", "im", "conjugate", "arg",
    "erf", "erfc", "erfi", "Ei", "li", "Si", "Ci", "zeta", "polylog", "LambertW",
    "besselj", "bessely", "besseli", "besselk", "airyai", "airybi",
    "Piecewise", "Heaviside", "DiracDelta", "KroneckerDelta",
)


def _allowed_namespace() -> dict:
    ns: dict = {}
    for nm in _ALLOWED_NAME_LIST:
        obj = getattr(_sp, nm, None)
        if obj is not None:
            ns[nm] = obj
    ns["ln"] = _sp.log
    ns["__builtins__"] = {}  # no Python builtins reachable from parsed code
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
    global_dict = _allowed_namespace()
    local_dict: dict = {}
    for nm in symbol_names or []:
        if isinstance(nm, str) and nm:
            local_dict[nm] = _sp.Symbol(nm)
    return parse_expr(
        s,
        local_dict=local_dict,
        global_dict=global_dict,
        transformations=_TRANSFORMS,
        evaluate=True,
    )


class _Timeout(Exception):
    pass


def _run_with_timeout(fn, timeout_sec):
    """Best-effort timeout via a daemon thread (cross-platform, incl. Windows).

    The interpreter switches threads during ``join``, so a CPU-bound sympy call
    is abandoned (as a leaked daemon thread) after ``timeout_sec`` and we raise
    ``_Timeout`` instead of hanging the tool.
    """
    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_sec)
    if t.is_alive():
        raise _Timeout()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _latex(obj):
    try:
        return _sp.latex(obj)
    except Exception:  # noqa: BLE001
        return None


def _to_symbol(var):
    if isinstance(var, _sp.Symbol):
        return var
    if isinstance(var, str) and var.strip():
        return _sp.Symbol(var.strip())
    raise ValueError("variable must be a non-empty string")


def _resolve_single_var(expr, var):
    if var is not None:
        if isinstance(var, (list, tuple)):
            if len(var) != 1:
                raise ValueError("this operation needs exactly one variable")
            return _to_symbol(var[0])
        return _to_symbol(var)
    fs = sorted(expr.free_symbols, key=lambda s: s.name)
    if len(fs) == 1:
        return fs[0]
    if not fs:
        raise ValueError("expression has no variable; specify 'var'")
    raise ValueError(f"ambiguous variable; specify 'var' (candidates: {[s.name for s in fs]})")


_OPS = ("simplify", "expand", "factor", "solve", "diff", "integrate", "limit", "series", "subs")

TOOL_SPEC = {
    "name": "symbolic_algebra",
    "domain": "mathematics",
    "audience": "worker",
    "description": (
        "Exact symbolic algebra & calculus (sympy). op is one of "
        "simplify|expand|factor|solve|diff|integrate|limit|series|subs. 'expr' is "
        "parsed in a RESTRICTED namespace (no arbitrary eval). Returns the exact "
        "result as a string + LaTeX; 'solve' returns the solution set. To TRUST an "
        "equality, verify it with symbolic_verify_identity — do not rely on a bare "
        "result here."
    ),
    "params": {
        "op": {"type": "str", "required": True,
                "description": "simplify|expand|factor|solve|diff|integrate|limit|series|subs"},
        "expr": {"type": "str", "required": True,
                  "description": "The expression (e.g. 'sin(x)^2+cos(x)^2'). For solve, an "
                                 "equation 'lhs=rhs' (or an expression assumed =0); a system "
                                 "may be given as equations separated by ';'."},
        "var": {"type": "str|list[str]", "required": False,
                 "description": "Variable(s). Required when ambiguous. For solve, the "
                                "unknown(s); for diff/integrate/limit/series, the variable."},
        "point": {"type": "str|number|list", "required": False,
                   "description": "limit point / series expansion centre; for a definite "
                                  "integral a 2-list [a,b] of limits."},
        "subs": {"type": "dict", "required": False,
                  "description": "For op=subs: mapping {name: value} of substitutions."},
        "order": {"type": "int", "required": False, "default": 1,
                   "description": "Derivative order for op=diff (<= 20)."},
        "n": {"type": "int", "required": False, "default": 6,
               "description": "Number of series terms for op=series (<= 40)."},
        "direction": {"type": "str", "required": False, "default": "+",
                       "description": "Limit direction: '+', '-' or '+-' (both)."},
        "timeout": {"type": "number", "required": False, "default": 10,
                     "description": "Per-op wall-clock cap in seconds (<= 60)."},
    },
    "output": {
        "op": "str",
        "result": "str — exact symbolic result",
        "latex": "str|None — LaTeX of the result",
        "solutions": "list[str] — solutions (op=solve only)",
        "solution_count": "int — number of solutions (op=solve only)",
        "free_symbols": "list[str] — free symbols of the input",
        "note": "str",
    },
    "example": {
        "params": {"op": "factor", "expr": "x^2 - 1"},
        "output": {"op": "factor", "result": "(x - 1)*(x + 1)"},
    },
}


def _do_solve(expr_text, var, timeout):
    # Split a possible system on ';' or newlines; parse each equation.
    parts = [p for p in re.split(r"[;\n]", expr_text) if p.strip()]
    if not parts:
        raise ValueError("no equation to solve")
    equations = []
    all_syms: set = set()
    for part in parts:
        # A single '=' means an equation; '==','<=','>=' are not supported here.
        if part.count("=") == 1 and "==" not in part and "<" not in part and ">" not in part:
            lhs_s, rhs_s = part.split("=", 1)
            lhs = _safe_sympify(lhs_s)
            rhs = _safe_sympify(rhs_s)
            eq = _sp.Eq(lhs, rhs)
            all_syms |= lhs.free_symbols | rhs.free_symbols
        else:
            e = _safe_sympify(part)  # treated as == 0
            eq = e
            all_syms |= e.free_symbols
        equations.append(eq)

    if var is not None:
        if isinstance(var, (list, tuple)):
            symbols = [_to_symbol(v) for v in var]
        else:
            symbols = [_to_symbol(var)]
    else:
        symbols = sorted(all_syms, key=lambda s: s.name)
    if len(symbols) > _MAX_FREE_SYMBOLS:
        raise ValueError("too many unknowns")

    def work():
        target = equations[0] if len(equations) == 1 else equations
        if symbols:
            return _sp.solve(target, symbols, dict=len(symbols) > 1)
        return _sp.solve(target)

    sols = _run_with_timeout(work, timeout)
    if isinstance(sols, dict):
        sols_list = [sols]
    elif isinstance(sols, (list, tuple)):
        sols_list = list(sols)
    else:
        sols_list = [sols]
    solutions = [str(s) for s in sols_list]
    return sols_list, solutions


def symbolic_algebra(
    op=None,
    expr=None,
    var=None,
    point=None,
    subs=None,
    order=1,
    n=6,
    direction="+",
    timeout=_DEFAULT_TIMEOUT,
) -> ToolResult:
    # -------- input validation --------
    if not isinstance(op, str) or op.strip().lower() not in _OPS:
        return ToolResult(
            success=False,
            error=ToolError(
                type="validation_error",
                message=f"'op' must be one of {list(_OPS)}; got {op!r}.",
            ),
        )
    op = op.strip().lower()
    if not isinstance(expr, str) or not expr.strip():
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message="'expr' (a non-empty string) is required."),
        )
    try:
        to = float(timeout)
    except (TypeError, ValueError):
        to = _DEFAULT_TIMEOUT
    to = max(0.5, min(_MAX_TIMEOUT, to))

    # -------- parse (solve handles its own multi-equation parsing) --------
    try:
        if op != "solve":
            parsed = _safe_sympify(expr)
            if len(parsed.free_symbols) > _MAX_FREE_SYMBOLS:
                return ToolResult(
                    success=False,
                    error=ToolError(type="validation_error", message="too many free symbols in 'expr'."),
                )
    except (ValueError, SyntaxError, TypeError) as exc:
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"could not parse 'expr': {exc}"))
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"could not parse 'expr': {exc}"))

    # -------- dispatch --------
    try:
        note = ""
        if op == "solve":
            try:
                sols_list, solutions = _do_solve(expr, var, to)
            except (ValueError, SyntaxError, TypeError) as exc:
                return ToolResult(success=False, error=ToolError(type="validation_error", message=str(exc)))
            return ToolResult(
                success=True,
                output={
                    "op": op,
                    "result": str(sols_list),
                    "latex": _latex(sols_list),
                    "solutions": solutions,
                    "solution_count": len(solutions),
                    "free_symbols": sorted(str(s) for s in set().union(*[getattr(_safe_sympify(p), "free_symbols", set()) for p in re.split(r"[;\n]", expr) if p.strip()] or [set()])),
                    "note": "solve() returns concrete solutions; an empty list means no solution was found (not a proof of none).",
                },
            )

        free = sorted(str(s) for s in parsed.free_symbols)

        if op == "simplify":
            result = _run_with_timeout(lambda: _sp.simplify(parsed), to)
        elif op == "expand":
            result = _run_with_timeout(lambda: _sp.expand(parsed), to)
        elif op == "factor":
            result = _run_with_timeout(lambda: _sp.factor(parsed), to)
        elif op == "diff":
            v = _resolve_single_var(parsed, var)
            try:
                k = int(order)
            except (TypeError, ValueError):
                k = 1
            if k < 0 or k > _MAX_ORDER:
                return ToolResult(success=False, error=ToolError(type="validation_error", message=f"'order' must be in [0, {_MAX_ORDER}]."))
            result = _run_with_timeout(lambda: _sp.diff(parsed, v, k), to)
        elif op == "integrate":
            v = _resolve_single_var(parsed, var)
            if isinstance(point, (list, tuple)) and len(point) == 2:
                a = _safe_sympify(str(point[0])) if not isinstance(point[0], _sp.Basic) else point[0]
                b = _safe_sympify(str(point[1])) if not isinstance(point[1], _sp.Basic) else point[1]
                result = _run_with_timeout(lambda: _sp.integrate(parsed, (v, a, b)), to)
                note = "definite integral"
            else:
                result = _run_with_timeout(lambda: _sp.integrate(parsed, v), to)
                note = "indefinite integral (constant of integration omitted)"
        elif op == "limit":
            v = _resolve_single_var(parsed, var)
            pt = _sp.Integer(0) if point is None else (
                point if isinstance(point, _sp.Basic) else _safe_sympify(str(point))
            )
            d = str(direction) if direction in ("+", "-", "+-") else "+"
            result = _run_with_timeout(lambda: _sp.limit(parsed, v, pt, d), to)
        elif op == "series":
            v = _resolve_single_var(parsed, var)
            pt = _sp.Integer(0) if point is None else (
                point if isinstance(point, _sp.Basic) else _safe_sympify(str(point))
            )
            try:
                terms = int(n)
            except (TypeError, ValueError):
                terms = 6
            if terms < 1 or terms > _MAX_SERIES_TERMS:
                return ToolResult(success=False, error=ToolError(type="validation_error", message=f"'n' must be in [1, {_MAX_SERIES_TERMS}]."))
            result = _run_with_timeout(lambda: _sp.series(parsed, v, pt, terms), to)
        elif op == "subs":
            if not isinstance(subs, dict) or not subs:
                return ToolResult(success=False, error=ToolError(type="validation_error", message="op=subs requires a non-empty 'subs' mapping {name: value}."))
            mapping = {}
            for k, val in subs.items():
                key_sym = _to_symbol(k)
                value = val if isinstance(val, _sp.Basic) else _safe_sympify(str(val))
                mapping[key_sym] = value
            result = _run_with_timeout(lambda: parsed.subs(mapping), to)
        else:  # pragma: no cover - guarded above
            return ToolResult(success=False, error=ToolError(type="validation_error", message=f"unsupported op {op!r}."))

        out = {
            "op": op,
            "result": str(result),
            "latex": _latex(result),
            "free_symbols": free,
            "note": note,
        }
        # If the result is a pure number, also give a numeric value (honest, exact-first).
        try:
            if hasattr(result, "free_symbols") and not result.free_symbols and result.is_number:
                out["numeric"] = str(_sp.N(result, 30))
        except Exception:  # noqa: BLE001
            pass
        return ToolResult(success=True, output=out)

    except _Timeout:
        return ToolResult(
            success=False,
            error=ToolError(type="execution_error", message=f"op '{op}' timed out after {to}s (input too hard); returned nothing rather than guess."),
        )
    except (ValueError, SyntaxError, TypeError) as exc:
        return ToolResult(success=False, error=ToolError(type="validation_error", message=str(exc)))
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
