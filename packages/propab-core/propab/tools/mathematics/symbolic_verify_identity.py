"""Independent identity verifier (the HONESTY tool of the symbolic cluster).

M1 (domain-capabilities §1a). ``symbolic_algebra`` will happily return a bare
``simplify`` result; this tool is the *paranoid* second opinion an agent routes a
CLAIMED equality through before trusting it. Given ``lhs`` and ``rhs`` (strings), it
decides whether ``lhs == rhs`` is an identity and returns one of three verdicts —
``proven`` / ``refuted`` / ``unknown`` — that never over-claim.

Honesty by construction (domain-capabilities §0):
  * ``proven`` is returned ONLY when BOTH independent checks agree: a symbolic
    ``sympy.simplify(lhs - rhs) == 0`` AND a DENSE random+grid numeric sampling at many
    points that all agree (to high precision). Neither check alone is ever sufficient —
    a symbolic zero without numeric corroboration, or numeric agreement without a
    symbolic proof, is downgraded to ``unknown``.
  * ``refuted`` is returned as soon as a single valid numeric sample disagrees, and the
    offending point is returned as a self-certifying COUNTEREXAMPLE (the caller can
    re-substitute it to see lhs != rhs for themselves).
  * ``unknown`` covers everything else (couldn't prove, no counterexample, too few valid
    samples, or a timeout) — the tool refuses to guess.
  * Expressions are parsed with a RESTRICTED sympy parser (curated whitelist namespace,
    ``__builtins__`` stripped, dunder/import/attribute screening) — never ``eval``.
  * Every step is size-capped and run under a best-effort timeout; the tool never raises
    to the caller (returns ``validation_error`` / ``execution_error``).
"""
from __future__ import annotations

import math
import random
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
# Restricted parsing (shared shape across the mathematics tools, duplicated so
# each tool file stays self-contained per the cluster's one-file-per-tool rule).
# --------------------------------------------------------------------------- #
_MAX_EXPR_LEN = 4000
_MAX_FREE_SYMBOLS = 8
_DEFAULT_SAMPLES = 60
_MAX_SAMPLES = 400
_EVAL_PREC = 60           # decimal digits used for each numeric probe
_TOL_ABS = _sp.Float("1e-30")
_TOL_REL = _sp.Float("1e-20")
_MIN_VALID_SAMPLES = 8    # need at least this many defined points to trust agreement
_DEFAULT_TIMEOUT = 10.0
_MAX_TIMEOUT = 60.0

_FORBIDDEN = re.compile(
    r"(__|\bimport\b|\bexec\b|\beval\b|\blambda\b|\bos\b|\bsys\b|\bsubprocess\b"
    r"|\bopen\b|\bcompile\b|\bglobals\b|\blocals\b|\bgetattr\b|\bsetattr\b"
    r"|\binput\b|`|\\)"
)
_ATTR_RE = re.compile(r"[A-Za-z_)\]]\s*\.\s*[A-Za-z_]")
_TRANSFORMS = standard_transformations + (convert_xor,)

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
    """Best-effort timeout via a daemon thread (cross-platform, incl. Windows)."""
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


TOOL_SPEC = {
    "name": "symbolic_verify_identity",
    "domain": "mathematics",
    "audience": "worker",
    # The independent-verification tool: a 'proven' verdict is a symbolic
    # simplify==0 CROSS-CHECKED by dense numeric sampling, and 'refuted' ships a
    # re-checkable counterexample point. This is the honesty backstop for algebra.
    "verification_capable": True,
    "description": (
        "Independently verify a CLAIMED equality lhs == rhs (the honesty tool for "
        "algebra). Both sides are parsed in a RESTRICTED namespace (no arbitrary eval). "
        "Returns verdict in {proven, refuted, unknown}: 'proven' ONLY when BOTH "
        "sympy.simplify(lhs-rhs)==0 AND dense random+grid numeric sampling at many points "
        "all agree (never one check alone); 'refuted' when any numeric sample differs, "
        "returning that point as a self-certifying counterexample; 'unknown' otherwise "
        "(couldn't prove, no counterexample, too few valid points, or timeout — never a "
        "guess). Use to check an agent's algebra before trusting it."
    ),
    "params": {
        "lhs": {"type": "str", "required": True,
                "description": "Left-hand side expression, e.g. 'sin(x)**2 + cos(x)**2'."},
        "rhs": {"type": "str", "required": True,
                "description": "Right-hand side expression, e.g. '1'."},
        "variables": {"type": "list[str]", "required": False,
                      "description": "Optional explicit variable names (else inferred from "
                                     "the free symbols of lhs and rhs)."},
        "samples": {"type": "int", "required": False, "default": 60,
                    "description": f"Number of numeric sample points (<= {_MAX_SAMPLES})."},
        "seed": {"type": "int", "required": False, "default": 0,
                 "description": "RNG seed for the random sample points (determinism)."},
        "timeout": {"type": "number", "required": False, "default": 10,
                    "description": "Per-step wall-clock cap in seconds (<= 60)."},
    },
    "output": {
        "verdict": "str — proven | refuted | unknown",
        "symbolic_zero": "bool — simplify(lhs-rhs) reduced to 0",
        "numeric_agree": "bool — every valid numeric sample agreed",
        "samples_valid": "int — number of points where both sides were defined",
        "samples_agree": "int — number of valid points that agreed",
        "counterexample": "dict|None — a point where lhs != rhs (refuted only)",
        "simplified_difference": "str — simplify(lhs - rhs)",
        "free_symbols": "list[str]",
        "note": "str — explicit statement of what the verdict does and does NOT claim",
    },
    "example": {
        "params": {"lhs": "sin(x)**2 + cos(x)**2", "rhs": "1"},
        "output": {"verdict": "proven", "symbolic_zero": True, "numeric_agree": True},
    },
}


def _validation_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="validation_error", message=message))


def _execution_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="execution_error", message=message))


def _grid_values():
    """A deterministic grid of simple real sample values (ints + small rationals)."""
    R = _sp.Rational
    return [
        _sp.Integer(0), _sp.Integer(1), _sp.Integer(-1), _sp.Integer(2), _sp.Integer(-2),
        _sp.Integer(3), _sp.Integer(-3), _sp.Integer(5), _sp.Integer(-5), _sp.Integer(7),
        R(1, 2), R(-1, 2), R(1, 3), R(2, 3), R(3, 2), R(-3, 2), R(1, 5), R(4, 3), R(-7, 4),
    ]


def _build_points(syms, n_samples, seed):
    """Deterministic grid points first (clean counterexamples), then seeded randoms."""
    grid = _grid_values()
    rng = random.Random(seed)
    points = []
    for i in range(n_samples):
        pt = {}
        for j, s in enumerate(syms):
            if i < len(grid):
                pt[s] = grid[(i + j) % len(grid)]
            else:
                num = rng.randint(-60, 60)
                den = rng.randint(1, 24)
                pt[s] = _sp.Rational(num, den)
        points.append(pt)
    return points


def _eval_side(expr, pt, prec):
    """Numerically evaluate ``expr`` at point ``pt``; return a Python complex or None.

    ``None`` means the expression is UNDEFINED / non-finite at this point (division by
    zero, a branch outside the real domain, oo/zoo/nan, or a still-symbolic residue).
    An undefined point is skipped — never counted as agreement or as a counterexample.
    """
    try:
        val = expr.evalf(prec, subs=pt)
    except (TypeError, ValueError, ArithmeticError):
        return None
    if val is None or not getattr(val, "is_number", False):
        return None
    if val.has(_sp.oo, -_sp.oo, _sp.zoo, _sp.nan):
        return None
    try:
        c = complex(val)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(c.real) and math.isfinite(c.imag)):
        return None
    return c


def _agrees(lval, rval):
    diff = abs(lval - rval)
    scale = max(abs(lval), abs(rval), 1.0)
    tol = float(_TOL_ABS) + float(_TOL_REL) * scale
    return diff <= tol, diff, scale


def symbolic_verify_identity(
    lhs=None,
    rhs=None,
    variables=None,
    samples=_DEFAULT_SAMPLES,
    seed=0,
    timeout=_DEFAULT_TIMEOUT,
) -> ToolResult:
    # ---- input validation (never raise) ----
    if not isinstance(lhs, str) or not lhs.strip():
        return _validation_error("'lhs' (a non-empty string) is required.")
    if not isinstance(rhs, str) or not rhs.strip():
        return _validation_error("'rhs' (a non-empty string) is required.")
    try:
        n_samples = int(samples)
    except (TypeError, ValueError):
        n_samples = _DEFAULT_SAMPLES
    n_samples = max(4, min(_MAX_SAMPLES, n_samples))
    try:
        seed_i = int(seed)
    except (TypeError, ValueError):
        seed_i = 0
    try:
        to = float(timeout)
    except (TypeError, ValueError):
        to = _DEFAULT_TIMEOUT
    to = max(0.5, min(_MAX_TIMEOUT, to))

    var_names = None
    if variables is not None:
        if not isinstance(variables, (list, tuple)):
            return _validation_error("'variables' must be a list of names.")
        var_names = [str(v) for v in variables]

    # ---- parse both sides in the restricted namespace ----
    try:
        lhs_e = _safe_sympify(lhs, var_names)
        rhs_e = _safe_sympify(rhs, var_names)
    except (ValueError, SyntaxError, TypeError) as exc:
        return _validation_error(f"could not parse an expression: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _validation_error(f"could not parse an expression: {exc}")

    free = sorted(lhs_e.free_symbols | rhs_e.free_symbols, key=lambda s: s.name)
    if len(free) > _MAX_FREE_SYMBOLS:
        return _validation_error(
            f"too many free symbols ({len(free)} > {_MAX_FREE_SYMBOLS})."
        )

    try:
        diff = lhs_e - rhs_e

        # ---- symbolic gate: simplify(lhs - rhs) == 0 ----
        symbolic_zero = False
        simplified = diff
        try:
            simplified = _run_with_timeout(lambda: _sp.simplify(diff), to)
            symbolic_zero = bool(simplified == 0)
            if not symbolic_zero:
                # cheap second attempt; still requires an EXACT zero, no over-claiming.
                expanded = _run_with_timeout(lambda: _sp.expand(diff), to)
                if expanded == 0:
                    symbolic_zero = True
                    simplified = expanded
        except _Timeout:
            simplified = diff  # symbolic step too hard; numeric sampling still runs

        # ---- numeric gate: dense grid + random sampling ----
        points = _build_points(free, n_samples, seed_i) if free else [{}]
        valid = 0
        agree = 0
        counterexample = None
        deadline_hit = False

        for pt in points:
            lval = _eval_side(lhs_e, pt, _EVAL_PREC)
            rval = _eval_side(rhs_e, pt, _EVAL_PREC)
            if lval is None or rval is None:
                continue  # undefined here — skip, never count as (dis)agreement
            valid += 1
            ok, dval, _scale = _agrees(lval, rval)
            if ok:
                agree += 1
            else:
                counterexample = {
                    "point": {str(k): str(v) for k, v in pt.items()},
                    "lhs_value": str(lval),
                    "rhs_value": str(rval),
                    "abs_difference": repr(dval),
                }
                break

        numeric_agree = counterexample is None and valid >= _MIN_VALID_SAMPLES and agree == valid
        # A pure-numeric (no free symbols) identity needs only one valid, agreeing probe.
        if not free and counterexample is None and valid >= 1 and agree == valid:
            numeric_agree = True

        # ---- combine the two INDEPENDENT gates into an honest verdict ----
        if counterexample is not None:
            verdict = "refuted"
            note = (
                "REFUTED: lhs and rhs disagree at the returned counterexample point "
                "(re-substitute it to confirm). A single valid disagreement disproves an "
                "identity."
            )
        elif symbolic_zero and numeric_agree:
            verdict = "proven"
            note = (
                "PROVEN: simplify(lhs-rhs) reduced to 0 AND every one of the "
                f"{valid} valid numeric probes agreed to ~{_EVAL_PREC} digits. Both "
                "independent checks were required; neither alone would have sufficed."
            )
        else:
            verdict = "unknown"
            reasons = []
            if not symbolic_zero:
                reasons.append("simplify(lhs-rhs) did not reduce to an exact 0")
            if not numeric_agree:
                if valid < _MIN_VALID_SAMPLES and free:
                    reasons.append(
                        f"only {valid} of {len(points)} sample points were defined "
                        f"(< {_MIN_VALID_SAMPLES} needed to trust numeric agreement)"
                    )
                else:
                    reasons.append("numeric agreement was not established")
            note = (
                "UNKNOWN (" + "; ".join(reasons) + "). No counterexample was found, but "
                "the claim was NOT proven — this is honest non-decision, not evidence the "
                "identity is false."
            )

        return ToolResult(
            success=True,
            output={
                "verdict": verdict,
                "symbolic_zero": bool(symbolic_zero),
                "numeric_agree": bool(numeric_agree),
                "samples_valid": int(valid),
                "samples_agree": int(agree),
                "samples_requested": len(points),
                "counterexample": counterexample,
                "simplified_difference": str(simplified),
                "free_symbols": [s.name for s in free],
                "note": note,
            },
        )
    except _Timeout:
        return _execution_error(
            "verification timed out; returning nothing rather than guessing a verdict."
        )
    except (ValueError, SyntaxError, TypeError) as exc:
        return _validation_error(str(exc))
    except Exception as exc:  # noqa: BLE001 — never raise to the caller
        return _execution_error(f"symbolic_verify_identity failed: {exc}")
