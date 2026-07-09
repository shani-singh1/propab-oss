"""Conjecturing tool: guess a recurrence / generating function / closed form for an
integer or rational sequence, and (optionally, network) look it up in the OEIS.

M5 (domain-capabilities §1d). The through-line of experimental mathematics is
*compute small cases → spot a pattern → conjecture*. This tool automates the pattern
step, but the design principle is **honesty by construction**: a guessed recurrence or
closed form is a CONJECTURE, never a fact. Every guess is fit on a PREFIX of the terms
and then required to PREDICT the held-out suffix it was not fit on. Only a guess that
reproduces the held-out terms is reported as ``validated_on_holdout: true`` — and even
then the verdict is ``conjecture`` with a bounded ``confidence`` (never 1.0, never
"established"/"proven"). Too few terms to fit *and* validate → ``unknown``.

Honesty mechanisms (why this never over-claims):
  * A recurrence found by ``sympy`` on ``n`` terms can have order up to ``n/2`` and so
    can OVERFIT. We (a) require ``fit_terms >= 2*order + 2`` and (b) roll the recurrence
    forward from the fit window and check it reproduces the held-out terms exactly.
  * A polynomial interpolated through ``k`` points always fits those ``k`` points; the
    only real evidence is that it also PREDICTS the held-out points — so that is what
    gates ``validated_on_holdout``.
  * The OEIS lookup is optional, network-side, short-timeout, and degrades to
    ``{"status": "unavailable"}``; a non-matching result is never surfaced as a match.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "sequence_oracle",
    "domain": "mathematics",
    "audience": "worker",
    "verification_capable": True,
    "description": (
        "Given the first terms of an integer/rational sequence, CONJECTURE a linear "
        "recurrence, a rational generating function, and/or a polynomial closed form "
        "(sympy). Every guess is fit on a prefix and validated by predicting the "
        "held-out suffix it was NOT fit on: only guesses that reproduce held-out terms "
        "are flagged validated_on_holdout=true, and the verdict is always 'conjecture' "
        "(never 'proven'). Too few terms => 'unknown'. Optionally (network, <=5s) look "
        "the terms up in the OEIS, degrading to 'unavailable' rather than fabricating a "
        "match. Use to spot a pattern in computed small cases before attempting a proof."
    ),
    "params": {
        "terms": {"type": "list[int|str]", "required": True,
                  "description": "First terms of the sequence (integers or rationals like '1/2')."},
        "max_order": {"type": "int", "required": False, "default": 6,
                      "description": "Largest linear-recurrence order to consider."},
        "holdout": {"type": "int", "required": False, "default": None,
                    "description": "How many trailing terms to hold out for validation (auto if None)."},
        "check_oeis": {"type": "bool", "required": False, "default": False,
                       "description": "If true, attempt an optional OEIS lookup (network, <=5s)."},
        "oeis_timeout": {"type": "float", "required": False, "default": 5.0,
                         "description": "OEIS request timeout in seconds (<=5)."},
    },
    "output": {
        "verdict": "str — 'conjecture' if a guess validated on held-out terms, else 'unknown'",
        "confidence": "float in [0,0.9] — never 1.0; scales with held-out terms predicted",
        "validated_on_holdout": "bool — any conjecture reproduced the held-out suffix",
        "n_terms": "int", "fit_terms": "int", "holdout_terms": "int",
        "recurrence": "dict — found/order/coefficients/relation/reproduces_fit/validated_on_holdout/generating_function",
        "closed_form": "dict — found/polynomial/degree/validated_on_holdout",
        "oeis": "dict — {'status': 'not_checked'|'unavailable'|'ok', 'matches': [...]} (never fabricated)",
        "note": "str — explicit statement that any formula is a conjecture, not established",
    },
    "example": {
        "params": {"terms": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144], "check_oeis": False},
        "output": {"verdict": "conjecture", "validated_on_holdout": True,
                    "recurrence": {"order": 2, "coefficients": [1, 1], "validated_on_holdout": True}},
    },
}


def _to_rational(x):
    """Parse a term to an exact sympy Rational; raise ValueError on junk."""
    from sympy import Integer, Rational, nsimplify

    if isinstance(x, bool):  # bools are ints in Python; reject to avoid silent nonsense
        raise ValueError("boolean is not a valid sequence term")
    if isinstance(x, int):
        return Integer(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            raise ValueError("empty term")
        return Rational(s)  # handles '3' and '3/4'
    if isinstance(x, float):
        # Exact decimal rational (no silent float error); e.g. 0.5 -> 1/2.
        return nsimplify(Rational(str(x)))
    # sympy numbers / anything sympifiable
    r = Rational(x)
    return r


def _predict_recurrence(coeffs, seed, count):
    """Roll x(k) = c0*x(k-1)+c1*x(k-2)+... forward ``count`` steps from ``seed``."""
    order = len(coeffs)
    hist = list(seed[-order:])
    out = []
    for _ in range(count):
        nxt = sum(coeffs[i] * hist[-1 - i] for i in range(order))
        out.append(nxt)
        hist.append(nxt)
    return out


def _guess_recurrence(vals_fit, max_order):
    """Return (coeffs, gf_str) for the shortest linear recurrence fitting ``vals_fit``.

    Uses sympy's ``find_linear_recurrence`` via a periodic sequence whose first period
    is exactly ``vals_fit``. Returns (None, None) when none is found.
    """
    from sympy import sequence, symbols

    x = symbols("x")
    seq = sequence(tuple(vals_fit))
    d = min(max_order, len(vals_fit) // 2)
    if d < 1:
        return None, None
    try:
        result = seq.find_linear_recurrence(len(vals_fit), d=d, gfvar=x)
    except Exception:  # noqa: BLE001 — sympy edge cases degrade to "no guess"
        return None, None
    if isinstance(result, tuple):
        coeffs, gf = result
    else:
        coeffs, gf = result, None
    if not coeffs:
        return None, None
    gf_str = str(gf) if gf is not None else None
    return list(coeffs), gf_str


def _guess_polynomial(vals_fit):
    """Interpolate a polynomial through (i, vals_fit[i]); return (poly, degree) or (None, None)."""
    from sympy import Poly, interpolate, symbols

    x = symbols("x")
    pts = list(zip(range(len(vals_fit)), vals_fit))
    try:
        p = interpolate(pts, x)
    except Exception:  # noqa: BLE001
        return None, None
    try:
        deg = Poly(p, x).degree() if p.free_symbols else 0
    except Exception:  # noqa: BLE001
        deg = None
    return p, deg


def _oeis_lookup(int_terms, timeout):
    """Optional OEIS lookup by terms. Never fabricates: returns 'unavailable' on any error.

    Only surfaces a match whose reported data actually CONTAINS our terms as a
    contiguous run (else it is flagged 'loose' and not treated as an identification).
    """
    q = ",".join(str(t) for t in int_terms)
    url = "https://oeis.org/search?" + urllib.parse.urlencode({"q": q, "fmt": "json"})
    try:
        with urllib.request.urlopen(url, timeout=min(float(timeout), 5.0)) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001 — network optional; degrade honestly
        return {"status": "unavailable", "reason": type(exc).__name__}
    results = data.get("results") or []
    needle = q
    matches = []
    for r in results[:5]:
        num = r.get("number")
        oeis_data = (r.get("data") or "")
        contiguous = needle in oeis_data
        matches.append({
            "id": f"A{int(num):06d}" if isinstance(num, int) else None,
            "name": r.get("name"),
            "contains_query_terms": bool(contiguous),
        })
    return {"status": "ok", "matches": matches,
            "note": "OEIS candidates by term match; 'contains_query_terms' flags a genuine contiguous hit."}


def sequence_oracle(
    terms: list | None = None,
    max_order: int = 6,
    holdout: int | None = None,
    check_oeis: bool = False,
    oeis_timeout: float = 5.0,
) -> ToolResult:
    # ---- input validation (never raise) ----------------------------------------
    if terms is None or not isinstance(terms, (list, tuple)):
        return ToolResult(success=False, error=ToolError(
            type="validation_error", message="Parameter 'terms' (a non-empty list of numbers) is required."))
    if len(terms) < 2:
        return ToolResult(success=False, error=ToolError(
            type="validation_error", message="Provide at least 2 terms."))
    try:
        vals = [_to_rational(t) for t in terms]
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=ToolError(
            type="validation_error", message=f"Non-numeric term in 'terms': {exc}"))
    try:
        max_order = max(1, int(max_order))
    except (TypeError, ValueError):
        max_order = 6

    try:
        from sympy import Integer, nsimplify  # noqa: F401 (import guard for env)

        L = len(vals)

        # Choose the held-out suffix used to VALIDATE every guess.
        if holdout is None:
            h = max(2, L // 4)
        else:
            try:
                h = int(holdout)
            except (TypeError, ValueError):
                h = max(2, L // 4)
        h = max(1, min(h, L - 2))  # keep at least 2 fit terms
        fit_len = L - h
        fit_vals = vals[:fit_len]
        holdout_vals = vals[fit_len:]

        # Not enough data to both fit and validate a nontrivial pattern -> honest unknown.
        if L < 5 or fit_len < 3:
            return ToolResult(success=True, output={
                "verdict": "unknown", "confidence": 0.0, "validated_on_holdout": False,
                "n_terms": L, "fit_terms": fit_len, "holdout_terms": h,
                "recurrence": {"found": False}, "closed_form": {"found": False},
                "oeis": {"status": "not_checked"},
                "note": ("Too few terms to fit a pattern AND validate it on held-out terms; "
                          "reporting 'unknown' rather than guessing (honesty by construction)."),
            })

        # ---- linear recurrence conjecture ---------------------------------------
        rec_out = {"found": False}
        coeffs, gf_str = _guess_recurrence(fit_vals, max_order)
        if coeffs is not None:
            order = len(coeffs)
            enough = fit_len >= 2 * order + 2
            # (a) does it reproduce the fit window from its first `order` seeds?
            reproduces_fit = False
            if fit_len > order:
                pred_fit = _predict_recurrence(coeffs, fit_vals[:order], fit_len - order)
                reproduces_fit = pred_fit == fit_vals[order:]
            # (b) does it PREDICT the held-out suffix? (the real conjecture test)
            pred_hold = _predict_recurrence(coeffs, fit_vals, h)
            matched = sum(1 for a, b in zip(pred_hold, holdout_vals) if a == b)
            validated = enough and reproduces_fit and matched == h
            rec_out = {
                "found": True,
                "order": order,
                "coefficients": [_num(c) for c in coeffs],
                "relation": _relation_str(coeffs),
                "enough_terms": enough,
                "reproduces_fit": reproduces_fit,
                "validated_on_holdout": validated,
                "holdout_matched": matched,
                # gf is derived from the SAME conjecture; surfaced only when validated.
                "generating_function": gf_str if validated else None,
            }

        # ---- polynomial closed-form conjecture ----------------------------------
        cf_out = {"found": False}
        poly, deg = _guess_polynomial(fit_vals)
        if poly is not None:
            from sympy import symbols
            x = symbols("x")
            pred_hold = [poly.subs(x, fit_len + i) for i in range(h)]
            matched = sum(1 for a, b in zip(pred_hold, holdout_vals) if a == b)
            validated = matched == h
            cf_out = {
                "found": True,
                "polynomial": str(poly),
                "variable": "x  (0-indexed: term k = polynomial at x=k)",
                "degree": int(deg) if deg is not None else None,
                "validated_on_holdout": validated,
                "holdout_matched": matched,
            }

        # ---- optional OEIS lookup (network) -------------------------------------
        if check_oeis:
            try:
                int_terms = [int(v) for v in vals if v == int(v)]
            except (TypeError, ValueError):
                int_terms = []
            oeis = _oeis_lookup(int_terms, oeis_timeout) if len(int_terms) == L else {
                "status": "not_checked", "reason": "OEIS lookup needs integer terms"}
        else:
            oeis = {"status": "not_checked"}

        any_valid = bool(rec_out.get("validated_on_holdout")) or bool(cf_out.get("validated_on_holdout"))
        verdict = "conjecture" if any_valid else "unknown"
        # Confidence scales with held-out evidence but is CAPPED below certainty.
        if any_valid:
            confidence = min(0.9, 0.5 + 0.08 * h)
        elif rec_out.get("found") or cf_out.get("found"):
            confidence = 0.1  # a guess exists but failed held-out validation
        else:
            confidence = 0.0

        note = (
            "Any recurrence / generating function / closed form here is a CONJECTURE, "
            "not an established fact. It was fit on the first {f} terms and "
            "{status} the {h} held-out terms it was not fit on. "
            "'validated_on_holdout' is empirical evidence over finitely many terms, "
            "NOT a proof; confidence is capped below 1.0."
        ).format(f=fit_len, h=h, status=("REPRODUCED" if any_valid else "did NOT reproduce"))

        return ToolResult(success=True, output={
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "validated_on_holdout": any_valid,
            "n_terms": L, "fit_terms": fit_len, "holdout_terms": h,
            "recurrence": rec_out,
            "closed_form": cf_out,
            "oeis": oeis,
            "note": note,
        })
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))


def _num(c):
    """Render a sympy number as int when integral, else as an exact 'p/q' string."""
    try:
        if c == int(c):
            return int(c)
    except (TypeError, ValueError):
        pass
    return str(c)


def _relation_str(coeffs):
    parts = []
    for i, c in enumerate(coeffs):
        parts.append(f"({_num(c)})*x(k-{i + 1})")
    return "x(k) = " + " + ".join(parts)
