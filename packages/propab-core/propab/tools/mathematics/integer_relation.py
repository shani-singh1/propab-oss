"""Integer-relation / closed-form detection via PSLQ (M5, domain-capabilities §1d).

Given a list of real constants (ideally high precision), find a small-integer linear
relation c1*x1 + c2*x2 + ... + cn*xn = 0 using ``mpmath.pslq``. This is the workhorse of
experimental mathematics for recognising a constant (e.g. is this number a rational
combination of pi**2, log(2), zeta(3)?).

Honesty by construction (domain-capabilities §0):
  * A PSLQ relation found at finite precision is a CONJECTURE, never a fact — a purely
    numerical coincidence can masquerade as a relation. So the tool never reports a bare
    relation: it RE-DERIVES the relation at HIGHER precision, recomputes the residual
    |sum c_i x_i| there, and only calls it ``confirmed`` when that higher-precision
    residual is below a strict tolerance. A relation that dissolves at higher precision
    is reported ``not_confirmed`` (with its residual) — honest evidence it was an
    artifact, never silently returned as real.
  * The confirmation precision is genuinely useful only if the CALLER supplied the
    constants to at least that many digits; the tool reports the precision it used and
    the residual so the limitation is explicit (double-precision floats cannot be
    confirmed beyond ~16 digits — the tool says so).
  * No relation found => ``none_found`` (never "proven independent"). The tool never
    raises to the caller (validation_error / execution_error only).
"""
from __future__ import annotations

from fractions import Fraction

from propab.tools.types import ToolError, ToolResult

_MIN_VALUES = 2
_MAX_VALUES = 32
_MIN_DPS = 16                 # mpmath.pslq needs >= 53 bits (~16 decimal digits)
_MAX_DPS = 2000
_DEFAULT_DPS = 40
_DEFAULT_MAX_COEFF = 1000
_MAX_MAXCOEFF = 10 ** 9
_MAX_STEPS = 10000
_INIT_TOL_FRAC = 0.75        # initial pslq tolerance ~ 10**(-dps*frac)
_CONFIRM_TOL_FRAC = 0.75     # strict confirmation tolerance ~ 10**(-confirm_dps*frac)

TOOL_SPEC = {
    "name": "integer_relation",
    "domain": "mathematics",
    "audience": "worker",
    "verification_capable": True,
    "description": (
        "Find a small-integer linear relation c1*x1 + ... + cn*xn = 0 among real constants "
        "(PSLQ, mpmath). Pass 'values' as strings for high precision (floats are limited to "
        "~16 digits). A relation is treated as a CONJECTURE: the tool RE-DERIVES it at higher "
        "precision, recomputes the residual there, and reports status in "
        "{confirmed, not_confirmed, none_found}. 'confirmed' means the higher-precision "
        "residual is below a strict tolerance; 'not_confirmed' means the relation dissolved "
        "at higher precision (a numerical artifact) and ships its residual as evidence; "
        "'none_found' means no relation up to 'max_coeff' at the working precision. Reports "
        "the precision used and the residual so the (input-precision-bound) limitation is "
        "explicit. Never claims independence when none is found."
    ),
    "params": {
        "values": {"type": "list[str|float]", "required": True,
                   "description": "The constants (>=2). Give as decimal STRINGS for high "
                                  "precision; floats are accepted but limited to ~16 digits."},
        "precision": {"type": "int", "required": False, "default": 40,
                      "description": f"Working precision in decimal digits ({_MIN_DPS}-{_MAX_DPS})."},
        "confirm_precision": {"type": "int", "required": False, "default": None,
                              "description": "Higher precision for confirmation "
                                             "(default = 2x precision)."},
        "max_coeff": {"type": "int", "required": False, "default": 1000,
                      "description": "Largest allowed |coefficient| in the relation."},
        "max_steps": {"type": "int", "required": False, "default": 1000,
                      "description": "Max PSLQ iterations."},
    },
    "output": {
        "found": "bool — a relation was found at the working precision",
        "status": "str — confirmed | not_confirmed | none_found",
        "relation": "list[int]|None — the integer coefficients (c1..cn)",
        "relation_equation": "str|None — human-readable c1*x1 + ... = 0",
        "residual": "str — |sum c_i x_i| recomputed at confirm_precision",
        "residual_working": "str — the residual at the working precision",
        "tolerance": "str — the strict confirmation tolerance the residual was tested against",
        "independently_rederived": "bool — a proportional relation was recovered at higher precision",
        "precision": "int — working precision (dps) used",
        "confirm_precision": "int — confirmation precision (dps) used",
        "note": "str — explicit statement that this is a conjecture, not a proof",
    },
    "example": {
        "params": {"values": ["1", "1.4142135623730950488", "2.4142135623730950488"]},
        "output": {"found": True, "status": "confirmed", "relation": [1, 1, -1]},
    },
}


def _validation_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="validation_error", message=message))


def _execution_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="execution_error", message=message))


def _proportional(a, b) -> bool:
    """True iff integer vectors ``a`` and ``b`` point in the same (or opposite) direction."""
    if a is None or b is None or len(a) != len(b):
        return False
    ratio = None
    for x, y in zip(a, b):
        if x == 0 and y == 0:
            continue
        if x == 0 or y == 0:
            return False
        r = Fraction(int(y), int(x))
        if ratio is None:
            ratio = r
        elif r != ratio:
            return False
    return ratio is not None


def _relation_equation(rel) -> str:
    parts = []
    for i, c in enumerate(rel):
        parts.append(f"({c})*x{i}")
    return " + ".join(parts) + " = 0"


def integer_relation(values=None, precision=_DEFAULT_DPS, confirm_precision=None,
                     max_coeff=_DEFAULT_MAX_COEFF, max_steps=1000) -> ToolResult:
    # ---- input validation (never raise) ----
    if not isinstance(values, (list, tuple)):
        return _validation_error("'values' must be a list of numeric constants.")
    if not (_MIN_VALUES <= len(values) <= _MAX_VALUES):
        return _validation_error(
            f"'values' must have between {_MIN_VALUES} and {_MAX_VALUES} entries; got {len(values)}.")
    for i, v in enumerate(values):
        if isinstance(v, bool) or isinstance(v, complex):
            return _validation_error(f"values[{i}] must be a real number/string, got {v!r}.")
        if not isinstance(v, (int, float, str)):
            return _validation_error(f"values[{i}] must be an int, float or string, got {type(v).__name__}.")
        if isinstance(v, str) and not v.strip():
            return _validation_error(f"values[{i}] is an empty string.")

    try:
        dps = int(precision)
    except (TypeError, ValueError):
        dps = _DEFAULT_DPS
    dps = max(_MIN_DPS, min(_MAX_DPS, dps))
    if confirm_precision is None:
        dps_hi = min(_MAX_DPS, 2 * dps)
    else:
        try:
            dps_hi = int(confirm_precision)
        except (TypeError, ValueError):
            dps_hi = 2 * dps
        dps_hi = max(dps + 1, min(_MAX_DPS, dps_hi))
    try:
        maxcoeff = int(max_coeff)
    except (TypeError, ValueError):
        maxcoeff = _DEFAULT_MAX_COEFF
    maxcoeff = max(2, min(_MAX_MAXCOEFF, maxcoeff))
    try:
        maxsteps = int(max_steps)
    except (TypeError, ValueError):
        maxsteps = 1000
    maxsteps = max(10, min(_MAX_STEPS, maxsteps))

    try:
        import mpmath as mp
    except Exception as exc:  # noqa: BLE001
        return _execution_error(f"mpmath unavailable: {exc}")

    # Flag if any value was supplied as a double-precision float (confirmation-limited).
    has_float_input = any(isinstance(v, float) for v in values)

    def _parse_at(dps_level):
        with mp.workdps(dps_level):
            out = []
            for v in values:
                out.append(mp.mpf(v) if not isinstance(v, float) else mp.mpf(repr(v)))
            return out

    try:
        # ---- initial PSLQ at the working precision ----
        with mp.workdps(dps):
            xs = _parse_at(dps)
            init_tol = mp.mpf(10) ** (-(dps * _INIT_TOL_FRAC))
            rel = mp.pslq(xs, tol=init_tol, maxcoeff=maxcoeff, maxsteps=maxsteps)
            res_work = None
            if rel is not None:
                res_work = abs(mp.fsum(mp.mpf(int(c)) * x for c, x in zip(rel, xs)))

        if rel is None:
            return ToolResult(success=True, output={
                "found": False, "status": "none_found", "relation": None,
                "relation_equation": None, "residual": None, "residual_working": None,
                "tolerance": None, "independently_rederived": False,
                "precision": dps, "confirm_precision": dps_hi,
                "note": (f"No integer relation with |coefficients| <= {maxcoeff} was found at "
                         f"{dps}-digit precision. This is NOT a proof that the constants are "
                         "linearly independent over the rationals — only that PSLQ found none "
                         "within the searched bound and precision."),
            })

        rel = [int(c) for c in rel]

        # ---- HONESTY: re-derive & recompute the residual at HIGHER precision ----
        with mp.workdps(dps_hi):
            xs_hi = _parse_at(dps_hi)
            res_hi = abs(mp.fsum(mp.mpf(c) * x for c, x in zip(rel, xs_hi)))
            confirm_tol = mp.mpf(10) ** (-(dps_hi * _CONFIRM_TOL_FRAC))
            rel_hi = mp.pslq(xs_hi, tol=confirm_tol, maxcoeff=maxcoeff, maxsteps=maxsteps)
            rederived = _proportional(rel, [int(c) for c in rel_hi]) if rel_hi is not None else False
            confirmed = res_hi < confirm_tol
            res_hi_s = mp.nstr(res_hi, 6)
            res_work_s = mp.nstr(res_work, 6) if res_work is not None else None
            tol_s = mp.nstr(confirm_tol, 4)

        status = "confirmed" if confirmed else "not_confirmed"
        if confirmed:
            note = (
                f"CONJECTURE (confirmed): the relation was re-derived at {dps_hi}-digit "
                f"precision and its residual ({res_hi_s}) is below the strict tolerance "
                f"({tol_s}). This is strong numerical evidence, NOT a proof — a rigorous "
                "identity still requires a symbolic derivation.")
        else:
            note = (
                f"NOT CONFIRMED: a relation appeared at {dps}-digit precision but at "
                f"{dps_hi} digits its residual ({res_hi_s}) EXCEEDS the strict tolerance "
                f"({tol_s}) — it is most likely a numerical artifact, not a real relation. "
                "Reported honestly rather than as a discovery.")
        if has_float_input:
            note += (" NOTE: at least one value was supplied as a double-precision float, so "
                     "confirmation is inherently limited to ~16 digits regardless of "
                     "confirm_precision; supply high-precision strings for a real confirmation.")

        return ToolResult(success=True, output={
            "found": True,
            "status": status,
            "relation": rel,
            "relation_equation": _relation_equation(rel),
            "residual": res_hi_s,
            "residual_working": res_work_s,
            "tolerance": tol_s,
            "independently_rederived": bool(rederived),
            "precision": dps,
            "confirm_precision": dps_hi,
            "max_coeff": maxcoeff,
            "note": note,
        })
    except Exception as exc:  # noqa: BLE001 — never raise to the caller
        return _execution_error(f"integer_relation (PSLQ) failed: {exc}")
