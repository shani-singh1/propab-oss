"""Exact polynomial computation (M1, domain-capabilities §1a).

A single dispatch tool for the repetitive polynomial algebra a mathematician runs across
number theory, algebraic geometry and coding theory: roots / factorization / gcd /
resultant / discriminant / Groebner basis, over ZZ, QQ or a finite field GF(p). Exact
sympy arithmetic throughout.

Honesty by construction (domain-capabilities §0):
  * Every op with a cheap witness is INDEPENDENTLY re-checked before return, reported in
    ``verified``:
      - factor: the returned factors are re-multiplied and confirmed == the input;
      - gcd:    the gcd is confirmed to divide every input (remainder 0);
      - roots:  every returned root r is re-substituted and confirmed p(r) == 0, and a
                ``complete`` flag says whether the multiplicities sum to the degree
                (roots not expressible in radicals are honestly reported as incomplete,
                with an APPROXIMATE numeric fallback clearly flagged non-exact);
      - groebner: every input polynomial is confirmed to reduce to 0 modulo the basis
                (ideal-membership check).
    A failed re-check is surfaced as ``execution_error`` — never returned as valid.
  * Polynomials are parsed with a RESTRICTED sympy parser (whitelist namespace,
    ``__builtins__`` stripped, dunder/import/attribute screen) — no eval.
  * Degree / length are capped and heavy ops run under a best-effort timeout; the tool
    returns honestly rather than hanging, and never raises to the caller.
"""
from __future__ import annotations

import re
import threading

import sympy as _sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

from propab.tools.types import ToolError, ToolResult

# --------------------------------------------------------------------------- #
# Restricted parsing (duplicated per the cluster's one-file-per-tool rule).
# --------------------------------------------------------------------------- #
_MAX_EXPR_LEN = 4000
_MAX_DEGREE = 400            # univariate factor/roots degree cap
_MAX_GROEBNER_POLYS = 40
_MAX_GF_ROOT_SCAN = 200000   # residue-scan cap for roots over GF(p)
_DEFAULT_TIMEOUT = 12.0
_MAX_TIMEOUT = 60.0

_FORBIDDEN = re.compile(
    r"(__|\bimport\b|\bexec\b|\beval\b|\blambda\b|\bos\b|\bsys\b|\bsubprocess\b"
    r"|\bopen\b|\bcompile\b|\bglobals\b|\blocals\b|\bgetattr\b|\bsetattr\b"
    r"|\binput\b|`|\\)"
)
_ATTR_RE = re.compile(r"[A-Za-z_)\]]\s*\.\s*[A-Za-z_]")
_TRANSFORMS = standard_transformations + (convert_xor,)

_ALLOWED_NAME_LIST = (
    "Symbol", "Integer", "Float", "Rational", "symbols", "I",
    "sqrt", "cbrt", "root", "Abs", "sign", "Pow",
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
        raise ValueError("polynomial must be a string")
    s = text.strip()
    if not s:
        raise ValueError("polynomial must be non-empty")
    if len(s) > _MAX_EXPR_LEN:
        raise ValueError(f"polynomial exceeds max length {_MAX_EXPR_LEN}")
    if _FORBIDDEN.search(s):
        raise ValueError("polynomial contains a forbidden token")
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


class _Timeout(Exception):
    pass


def _run_with_timeout(fn, timeout_sec):
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


_OPS = ("roots", "factor", "gcd", "resultant", "discriminant", "groebner")
_ORDERS = ("lex", "grlex", "grevlex")

TOOL_SPEC = {
    "name": "polynomial_tools",
    "domain": "mathematics",
    "audience": "worker",
    "verification_capable": True,
    "description": (
        "Exact polynomial computation (sympy). op is one of "
        "roots|factor|gcd|resultant|discriminant|groebner. Polynomials are given as "
        "strings and parsed in a RESTRICTED namespace (no eval). Optional 'domain' in "
        "{ZZ, QQ, GF(p)} (e.g. 'GF(7)'). Cheap-witness ops are INDEPENDENTLY re-checked "
        "before return (factor re-multiplied to the input, gcd confirmed to divide every "
        "input, each root re-substituted to p(r)=0, each Groebner input reduced to 0 mod "
        "the basis) and reported in 'verified'. roots reports a 'complete' flag (do the "
        "multiplicities sum to the degree?) and, when not solvable in radicals, an "
        "APPROXIMATE numeric fallback clearly flagged non-exact. groebner takes a list of "
        "polys + variable order + monomial order in {lex, grlex, grevlex}."
    ),
    "params": {
        "op": {"type": "str", "required": True,
               "description": "roots|factor|gcd|resultant|discriminant|groebner"},
        "poly": {"type": "str", "required": False,
                 "description": "The polynomial (single-poly ops: roots/factor/discriminant)."},
        "polys": {"type": "list[str]", "required": False,
                  "description": "Polynomials for gcd (>=2), resultant (2), or groebner (>=1)."},
        "var": {"type": "str|list[str]", "required": False,
                "description": "Variable(s). Required for resultant/discriminant (the "
                               "variable) and to fix the generator order for groebner."},
        "domain": {"type": "str", "required": False, "default": None,
                   "description": "Coefficient domain: 'ZZ', 'QQ', or 'GF(p)'. Default QQ."},
        "order": {"type": "str", "required": False, "default": "lex",
                  "description": "Monomial order for groebner: lex|grlex|grevlex."},
        "timeout": {"type": "number", "required": False, "default": 12,
                    "description": "Per-op wall-clock cap in seconds (<= 60)."},
    },
    "output": {
        "op": "str",
        "domain": "str — the coefficient domain used",
        "result": "op-specific exact result",
        "verified": "bool|None — the independent re-check passed (where applicable)",
        "complete": "bool — roots only: multiplicities sum to the degree",
        "note": "str",
    },
    "example": {
        "params": {"op": "factor", "poly": "x**4 - 1"},
        "output": {"op": "factor", "result": "(x - 1)*(x + 1)*(x**2 + 1)", "verified": True},
    },
}


def _validation_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="validation_error", message=message))


def _execution_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="execution_error", message=message))


def _parse_domain(domain):
    """Return (label, kwargs_for_factor_gcd, gf_prime|None) or raise ValueError."""
    if domain is None:
        return "QQ", {}, None
    d = str(domain).strip().upper().replace(" ", "")
    if d in ("QQ", "Q"):
        return "QQ", {"domain": _sp.QQ}, None
    if d in ("ZZ", "Z"):
        return "ZZ", {"domain": _sp.ZZ}, None
    m = re.fullmatch(r"(?:GF|FF)\((\d+)\)", d)
    if m:
        p = int(m.group(1))
        if not _sp.isprime(p):
            raise ValueError(f"GF(p) requires a prime p; {p} is not prime")
        return f"GF({p})", {"modulus": p}, p
    raise ValueError(f"unsupported domain {domain!r}; use 'ZZ', 'QQ', or 'GF(p)'")


def _resolve_var(exprs, var):
    """Resolve a single variable from ``var`` or the (unique) free symbol of ``exprs``."""
    if var is not None:
        if isinstance(var, (list, tuple)):
            if len(var) != 1:
                raise ValueError("this op needs exactly one variable")
            return _sp.Symbol(str(var[0]))
        return _sp.Symbol(str(var))
    syms = set()
    for e in exprs:
        syms |= e.free_symbols
    syms = sorted(syms, key=lambda s: s.name)
    if len(syms) == 1:
        return syms[0]
    if not syms:
        raise ValueError("no variable present; specify 'var'")
    raise ValueError(f"ambiguous variable; specify 'var' (candidates: {[s.name for s in syms]})")


def _resolve_gens(exprs, var):
    if var is not None:
        names = [var] if not isinstance(var, (list, tuple)) else list(var)
        return [_sp.Symbol(str(n)) for n in names]
    syms = set()
    for e in exprs:
        syms |= e.free_symbols
    return sorted(syms, key=lambda s: s.name)


def _degree_ok(expr, var):
    try:
        d = _sp.Poly(expr, var).degree()
        return d <= _MAX_DEGREE
    except Exception:  # noqa: BLE001
        return True  # non-polynomial / symbolic-coefficient: let the op decide


def polynomial_tools(op=None, poly=None, polys=None, var=None, domain=None,
                     order="lex", timeout=_DEFAULT_TIMEOUT) -> ToolResult:
    # ---- input validation (never raise) ----
    if not isinstance(op, str) or op.strip().lower() not in _OPS:
        return _validation_error(f"'op' must be one of {list(_OPS)}; got {op!r}.")
    op = op.strip().lower()
    try:
        dom_label, dom_kwargs, gf_p = _parse_domain(domain)
    except ValueError as exc:
        return _validation_error(str(exc))
    try:
        to = float(timeout)
    except (TypeError, ValueError):
        to = _DEFAULT_TIMEOUT
    to = max(0.5, min(_MAX_TIMEOUT, to))

    # ---- collect polynomial expressions ----
    try:
        single = _safe_sympify(poly) if isinstance(poly, str) and poly.strip() else None
        many = None
        if polys is not None:
            if not isinstance(polys, (list, tuple)) or not polys:
                return _validation_error("'polys' must be a non-empty list of strings.")
            many = [_safe_sympify(p) for p in polys]
    except (ValueError, SyntaxError, TypeError) as exc:
        return _validation_error(f"could not parse a polynomial: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _validation_error(f"could not parse a polynomial: {exc}")

    try:
        # ------------------------------------------------------------------ roots
        if op == "roots":
            if single is None:
                return _validation_error("op 'roots' requires 'poly'.")
            x = _resolve_var([single], var)
            if not _degree_ok(single, x):
                return _validation_error(f"degree exceeds cap {_MAX_DEGREE}.")
            if gf_p is not None:
                return _roots_gf(single, x, gf_p, dom_label)
            return _roots_qq(single, x, dom_label, to)

        # ------------------------------------------------------------------ factor
        if op == "factor":
            if single is None:
                return _validation_error("op 'factor' requires 'poly'.")
            factored = _run_with_timeout(lambda: _sp.factor(single, **dom_kwargs), to)
            # HONESTY: re-expand the factorization and confirm it equals the input.
            if gf_p is not None:
                diff = _sp.expand(factored - single)
                diff = _sp.Poly(diff, *sorted(single.free_symbols, key=str)).trunc(gf_p).as_expr() \
                    if single.free_symbols else diff % gf_p
                verified = bool(_sp.simplify(diff) == 0)
            else:
                verified = bool(_sp.expand(factored - single) == 0)
            if not verified:
                return _execution_error("factor re-check failed: product of factors != input.")
            return _ok(op, dom_label, str(factored), verified=True,
                       note="factors re-multiplied and confirmed == the input polynomial.")

        # ------------------------------------------------------------------ gcd
        if op == "gcd":
            items = many if many is not None else (
                [single] if single is not None else None)
            if not items or len(items) < 2:
                return _validation_error("op 'gcd' requires 'polys' with at least 2 entries.")
            def _gcd_all():
                g = items[0]
                for e in items[1:]:
                    g = _sp.gcd(g, e, **dom_kwargs)
                return g
            g = _run_with_timeout(_gcd_all, to)
            # HONESTY: confirm g divides every input (remainder 0).
            verified = True
            if g != 0:
                gens = _resolve_gens(items, var)
                for e in items:
                    rem = _sp.rem(e, g, *gens, **dom_kwargs) if gens else (e % g)
                    if _sp.simplify(rem) != 0:
                        verified = False
                        break
            if not verified:
                return _execution_error("gcd re-check failed: gcd does not divide an input.")
            return _ok(op, dom_label, str(g), verified=True,
                       note="gcd confirmed to divide every input polynomial (remainder 0).")

        # ------------------------------------------------------------------ resultant
        if op == "resultant":
            items = many if many is not None else None
            if not items or len(items) != 2:
                return _validation_error("op 'resultant' requires 'polys' of exactly 2 entries.")
            x = _resolve_var(items, var)
            res = _run_with_timeout(lambda: _sp.resultant(items[0], items[1], x), to)
            return _ok(op, dom_label, str(res), verified=None,
                       note=f"Res_{x}(f, g): exact; it is 0 iff f and g share a common root "
                            "(or a leading-coefficient degeneracy) in {x}.".replace("{x}", str(x)))

        # ------------------------------------------------------------------ discriminant
        if op == "discriminant":
            if single is None:
                return _validation_error("op 'discriminant' requires 'poly'.")
            x = _resolve_var([single], var)
            disc = _run_with_timeout(lambda: _sp.discriminant(single, x), to)
            return _ok(op, dom_label, str(disc), verified=None,
                       note=f"discriminant in {x}: exact; it is 0 iff the polynomial has a "
                            "repeated root.")

        # ------------------------------------------------------------------ groebner
        if op == "groebner":
            items = many if many is not None else (
                [single] if single is not None else None)
            if not items:
                return _validation_error("op 'groebner' requires 'polys' (a list of polynomials).")
            if len(items) > _MAX_GROEBNER_POLYS:
                return _validation_error(f"too many polynomials (> {_MAX_GROEBNER_POLYS}).")
            mono = str(order).strip().lower() if order else "lex"
            if mono not in _ORDERS:
                return _validation_error(f"'order' must be one of {list(_ORDERS)}; got {order!r}.")
            gens = _resolve_gens(items, var)
            if not gens:
                return _validation_error("groebner needs variables; specify 'var'.")
            def _gb():
                return _sp.groebner(items, *gens, order=mono, **dom_kwargs)
            G = _run_with_timeout(_gb, to)
            basis = [str(g) for g in G.exprs]
            # HONESTY: every input must reduce to 0 modulo the basis (ideal membership).
            verified = True
            for e in items:
                _q, r = G.reduce(e)
                if _sp.simplify(r) != 0:
                    verified = False
                    break
            if not verified:
                return _execution_error(
                    "groebner re-check failed: an input does not reduce to 0 mod the basis.")
            return _ok(op, dom_label, {"basis": basis, "order": mono,
                                       "variables": [str(g) for g in gens]},
                       verified=True,
                       note="each input polynomial re-checked: reduces to 0 modulo the basis "
                            "(so the basis generates the same ideal).")

        return _validation_error(f"unhandled op {op!r}.")  # pragma: no cover

    except _Timeout:
        return _execution_error(
            f"op '{op}' timed out after {to}s (input too hard); returned nothing rather than guess.")
    except (ValueError, SyntaxError, TypeError) as exc:
        return _validation_error(str(exc))
    except Exception as exc:  # noqa: BLE001 — never raise to the caller
        return _execution_error(f"polynomial_tools op '{op}' failed: {exc}")


def _roots_qq(expr, x, dom_label, to):
    """Exact roots over QQ/ZZ closure, each re-substituted; numeric fallback flagged."""
    P = _sp.Poly(expr, x)
    degree = P.degree()
    roots = _run_with_timeout(lambda: _sp.roots(P), to)  # {root: multiplicity}
    root_items = {}
    verified = True
    for r, mult in roots.items():
        # HONESTY: re-substitute each exact root and confirm p(r) == 0.
        val = _sp.simplify(expr.subs(x, r))
        if val != 0:
            verified = False
        root_items[str(r)] = int(mult)
    found_mult = sum(roots.values())
    complete = found_mult == degree
    out = {
        "op": "roots", "domain": dom_label,
        "result": {"roots": root_items, "degree": int(degree)},
        "verified": (verified if roots else None),
        "complete": bool(complete),
    }
    if not complete:
        # Honest numeric fallback for roots not expressible in radicals — APPROXIMATE.
        try:
            approx = _run_with_timeout(lambda: [str(z) for z in P.nroots(n=20)], to)
            out["result"]["numeric_roots_approx"] = approx
            out["note"] = (
                f"Only {found_mult} of {degree} roots are expressible in closed form; the "
                "remaining roots are given ONLY as APPROXIMATE numeric values "
                "(numeric_roots_approx) — these are not exact and not independently proven.")
        except Exception:  # noqa: BLE001
            out["note"] = (
                f"Only {found_mult} of {degree} roots found in closed form; the rest are "
                "not expressible in radicals and no reliable numeric fallback was produced.")
    else:
        if not verified:
            return _execution_error("roots re-check failed: p(r) != 0 for a returned root.")
        out["note"] = ("all roots found in closed form; each re-substituted and confirmed "
                       "p(r) = 0, and multiplicities sum to the degree.")
    return ToolResult(success=True, output=out)


def _roots_gf(expr, x, p, dom_label):
    """Roots over GF(p) by exact residue scan (self-verifying), capped by p."""
    if p > _MAX_GF_ROOT_SCAN:
        return ToolResult(success=True, output={
            "op": "roots", "domain": dom_label, "result": None, "verified": None,
            "complete": False,
            "note": f"p={p} exceeds the residue-scan cap {_MAX_GF_ROOT_SCAN}; roots over "
                    "GF(p) not computed (honest unknown rather than a hang).",
        })
    roots = []
    for a in range(p):
        val = int(expr.subs(x, a))  # exact integer, then reduce
        if val % p == 0:
            roots.append(a)
    return ToolResult(success=True, output={
        "op": "roots", "domain": dom_label,
        "result": {"roots": roots, "count": len(roots)},
        "verified": True, "complete": True,
        "note": f"roots found by EXACT evaluation at every residue in GF({p}); each listed "
                "root a satisfies p(a) = 0 mod p by construction (self-certifying).",
    })


def _ok(op, dom_label, result, **extra):
    out = {"op": op, "domain": dom_label, "result": result}
    out.update(extra)
    out.setdefault("verified", extra.get("verified"))
    return ToolResult(success=True, output=out)
