"""Exact arithmetic and structure over a finite field GF(p^k).

M2 (domain-capabilities §1b): the reusable finite-field capability behind number
theory / cryptography / coding-theory investigations. GF(p^k) is modelled as
GF(p)[x] / (modulus) for a degree-k irreducible ``modulus`` (auto-generated
deterministically when not supplied; GF(p) is the k=1 case with modulus x). All
arithmetic is exact via ``sympy.polys.galoistools`` over the prime field GF(p).

galois is NOT installed, so this is built on sympy's dense GF(p) polynomial ops.

Ops:
  add, mul, pow, inverse   — field arithmetic on elements (poly coeff lists / ints).
  element_order            — multiplicative order of a nonzero element in GF(p^k)*.
  minimal_polynomial       — minimal polynomial of an element over GF(p).
  is_irreducible           — is a given GF(p) polynomial irreducible.
  irreducible_poly         — a deterministic irreducible polynomial of a given degree.

Honesty by construction (domain-capabilities §0):
  * ``inverse`` is re-checked (a * a^-1 == 1 in the field) before it is returned.
  * ``minimal_polynomial`` is re-checked by EVALUATING it at the element in GF(p^k)
    and confirming it is 0, and that its degree divides k.
  * ``irreducible_poly`` / a supplied ``modulus`` are re-verified irreducible via
    sympy's deterministic ``gf_irreducible_p`` (a reducible modulus is rejected — it
    would not yield a field).
  * ``element_order`` is re-checked (a^order == 1 and order | p^k - 1).
  * ``p`` is required prime; magnitude / degree caps return an honest
    ``status="unknown"`` instead of hanging. The tool never raises to the caller.
"""
from __future__ import annotations

from propab.tools.types import ToolError, ToolResult

_ALLOWED_OPS = frozenset(
    {
        "add",
        "mul",
        "pow",
        "inverse",
        "minimal_polynomial",
        "is_irreducible",
        "irreducible_poly",
        "element_order",
    }
)

# Caps — beyond these we return an honest ``unknown`` rather than risk a hang.
_P_MAX_DIGITS = 50          # magnitude of the prime p
_K_MAX = 100               # extension degree k
_ORDER_FIELD_MAX_DIGITS = 40  # p^k for ops that must factor p^k - 1 (element_order)
_IRRED_SEARCH_MAX = 5_000_000  # deterministic irreducible search space p^d cap

TOOL_SPEC = {
    "name": "finite_field_compute",
    "domain": "mathematics",
    "audience": "worker",
    "description": (
        "Exact arithmetic and structure over the finite field GF(p^k), modelled as "
        "GF(p)[x]/(modulus). op is one of: add, mul, pow, inverse, minimal_polynomial, "
        "is_irreducible, irreducible_poly, element_order. Elements and polynomials are "
        "given as coefficient lists (high-to-low) or an int for a constant. p must be "
        "prime; k>=1 (default 1 = GF(p)). If no irreducible 'modulus' is supplied for "
        "k>1 one is generated deterministically. Results are exact; inverses, minimal "
        "polynomials, orders and irreducibility are independently re-verified."
    ),
    "params": {
        "op": {"type": "str", "required": True,
                "description": "Operation (see the op list in the description)."},
        "p": {"type": "int", "required": True, "description": "Prime characteristic."},
        "k": {"type": "int", "required": False,
               "description": "Extension degree (default 1 = prime field GF(p))."},
        "modulus": {"type": "list[int]", "required": False,
                     "description": "Degree-k irreducible poly over GF(p) (coeffs high-to-low). "
                                    "Auto-generated deterministically if omitted for k>1."},
        "a": {"type": "list[int]|int", "required": False,
               "description": "First element/operand (coeff list high-to-low, or int constant)."},
        "b": {"type": "list[int]|int", "required": False,
               "description": "Second element/operand (add/mul)."},
        "e": {"type": "int", "required": False,
               "description": "Exponent for op 'pow' (may be negative if 'a' is invertible)."},
        "poly": {"type": "list[int]", "required": False,
                  "description": "Polynomial (coeffs high-to-low) for op 'is_irreducible'."},
        "degree": {"type": "int", "required": False,
                    "description": "Degree for op 'irreducible_poly' (default k)."},
    },
    "output": {
        "op": "str",
        "status": "str — 'ok' or 'unknown'",
        "result": "op-specific result (element coeff list, bool, int, or poly coeff list)",
        "coeffs": "list[int] — element/poly coefficients high-to-low (reduced), where applicable",
        "int": "int — base-p integer encoding of an element result, where applicable",
        "modulus_used": "list[int] — the degree-k irreducible modulus actually used",
        "field": "str — 'GF(p^k)' description",
        "verified": "bool — independent re-check passed (inverse/order/min-poly/irreducible)",
        "exists": "bool — inverse only: whether the element is invertible",
        "reason": "str — present when status='unknown' or a result does not exist",
    },
    "example": {
        "params": {"op": "mul", "p": 2, "k": 2, "a": [1, 0], "b": [1, 1]},
        "output": {"op": "mul", "status": "ok", "coeffs": [1], "modulus_used": [1, 1, 1]},
    },
}


# --------------------------------------------------------------------------- helpers
def _validation_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="validation_error", message=message))


def _execution_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="execution_error", message=message))


def _ok(op: str, **fields) -> ToolResult:
    return ToolResult(success=True, output={"op": op, "status": "ok", **fields})


def _unknown(op: str, reason: str) -> ToolResult:
    return ToolResult(success=True, output={"op": op, "status": "unknown", "reason": reason})


def _to_int(value, name: str):
    if isinstance(value, bool):
        return None, _validation_error(f"'{name}' must be an integer, got a bool {value!r}.")
    if isinstance(value, int):
        return value, None
    if isinstance(value, str):
        try:
            return int(value.strip()), None
        except ValueError:
            return None, _validation_error(f"'{name}' must be an integer, got {value!r}.")
    return None, _validation_error(
        f"'{name}' must be an integer, got {type(value).__name__} {value!r}."
    )


def _digits(n: int) -> int:
    return len(str(abs(int(n))))


def _parse_poly(value, name: str, p: int):
    """Parse a polynomial/element to a reduced-mod-p coeff list (high-to-low, stripped)."""
    from sympy.polys.galoistools import gf_strip

    if isinstance(value, bool):
        return None, _validation_error(f"'{name}' must be int or list[int], got a bool.")
    if isinstance(value, int):
        coeffs = [value % p]
    elif isinstance(value, (list, tuple)):
        coeffs = []
        for i, c in enumerate(value):
            ci, err = _to_int(c, f"{name}[{i}]")
            if err:
                return None, err
            coeffs.append(ci % p)
    else:
        return None, _validation_error(
            f"'{name}' must be an int or list of ints, got {type(value).__name__}."
        )
    return gf_strip(coeffs), None


def _elem_to_int(coeffs, p: int) -> int:
    """Base-p integer encoding of an element (constant term = least significant)."""
    val = 0
    for c in coeffs:  # high-to-low
        val = val * p + (c % p)
    return int(val)


def _search_irreducible(p: int, d: int):
    """Deterministic smallest monic irreducible of degree ``d`` over GF(p), or None.

    Enumerates monic degree-d polynomials by their lower coefficients as a base-p
    integer (constant term least significant) and returns the first irreducible.
    Caller must ensure p**d <= _IRRED_SEARCH_MAX.
    """
    from sympy.polys.domains import ZZ
    from sympy.polys.galoistools import gf_irreducible_p

    space = p ** d
    for idx in range(space):
        # Decode the d lower coefficients (c_{d-1} .. c_0) from idx, msb = c_{d-1}.
        lower = []
        n = idx
        for _ in range(d):
            lower.append(n % p)
            n //= p
        lower.reverse()  # now [c_{d-1}, ..., c_0]
        poly = [1] + lower  # monic, degree d, high-to-low
        if gf_irreducible_p(poly, p, ZZ):
            return poly
    return None


def _resolve_modulus(op: str, p: int, k: int, modulus):
    """Return (modulus_list, error_result, unknown_result). Exactly one is non-None.

    For k==1 the modulus is x = [1, 0] (GF(p)). For k>1 a supplied modulus is validated
    (monic, degree k, irreducible); otherwise a deterministic irreducible is generated.
    """
    from sympy.polys.domains import ZZ
    from sympy.polys.galoistools import gf_degree, gf_irreducible_p

    if k == 1:
        return [1, 0], None, None

    if modulus is not None:
        mod, err = _parse_poly(modulus, "modulus", p)
        if err:
            return None, err, None
        if gf_degree(mod) != k:
            return None, _validation_error(
                f"'modulus' must have degree k={k}, got degree {gf_degree(mod)}."
            ), None
        # Make monic (scale by inverse of leading coeff) — required for a clean field.
        if mod[0] != 1:
            from sympy.core.numbers import mod_inverse

            inv_lead = mod_inverse(mod[0], p)
            mod = [(c * inv_lead) % p for c in mod]
        if not gf_irreducible_p(mod, p, ZZ):
            return None, _validation_error(
                "'modulus' is reducible over GF(p); it does not define a field. "
                "Supply an irreducible polynomial or omit 'modulus' to auto-generate one."
            ), None
        return mod, None, None

    # Auto-generate deterministically.
    if p ** k > _IRRED_SEARCH_MAX:
        return None, None, _unknown(
            op,
            f"no modulus supplied and the search space p^k={p}^{k} exceeds the "
            f"{_IRRED_SEARCH_MAX} deterministic-search cap; supply an explicit 'modulus'.",
        )
    mod = _search_irreducible(p, k)
    if mod is None:  # unreachable: an irreducible of every degree exists over GF(p).
        return None, _execution_error(
            f"failed to find an irreducible polynomial of degree {k} over GF({p})."
        ), None
    return mod, None, None


# --------------------------------------------------------------------------- dispatch
def finite_field_compute(
    op: str | None = None,
    p=None,
    k=None,
    modulus=None,
    a=None,
    b=None,
    e=None,
    poly=None,
    degree=None,
) -> ToolResult:
    if not op or not isinstance(op, str):
        return _validation_error("Parameter 'op' (string) is required.")
    op = op.strip().lower()
    if op not in _ALLOWED_OPS:
        return _validation_error(f"Unknown op {op!r}; supported: {sorted(_ALLOWED_OPS)}.")

    # ---- prime p ----
    if p is None:
        return _validation_error("Parameter 'p' (prime) is required.")
    p_i, err = _to_int(p, "p")
    if err:
        return err
    if _digits(p_i) > _P_MAX_DIGITS:
        return _unknown(op, f"prime p has {_digits(p_i)} digits, exceeding the "
                            f"{_P_MAX_DIGITS}-digit cap.")
    if p_i < 2:
        return _validation_error("'p' must be a prime >= 2.")
    try:
        from sympy import isprime

        if not isprime(p_i):
            return _validation_error(f"'p' must be prime, got {p_i}.")

        # ---- degree k ----
        if k is None:
            k_i = 1
        else:
            k_i, err = _to_int(k, "k")
            if err:
                return err
        if k_i < 1:
            return _validation_error("'k' must be >= 1.")
        if k_i > _K_MAX:
            return _unknown(op, f"extension degree k={k_i} exceeds the cap {_K_MAX}.")

        # -------------------------------------------------------- is_irreducible
        if op == "is_irreducible":
            if poly is None:
                return _validation_error("op 'is_irreducible' requires 'poly' (coeff list).")
            f, perr = _parse_poly(poly, "poly", p_i)
            if perr:
                return perr
            from sympy.polys.domains import ZZ
            from sympy.polys.galoistools import gf_degree, gf_irreducible_p

            if gf_degree(f) < 1:
                return _validation_error(
                    "op 'is_irreducible' requires a polynomial of degree >= 1."
                )
            if _digits(p_i ** gf_degree(f)) > _ORDER_FIELD_MAX_DIGITS:
                return _unknown(op, f"p^deg has too many digits to test irreducibility.")
            res = bool(gf_irreducible_p(f, p_i, ZZ))
            return _ok(op, result=res, coeffs=f, field=f"GF({p_i})")

        # -------------------------------------------------------- irreducible_poly
        if op == "irreducible_poly":
            if degree is None:
                d_i = k_i
            else:
                d_i, err = _to_int(degree, "degree")
                if err:
                    return err
            if d_i < 1:
                return _validation_error("'degree' must be >= 1.")
            if p_i ** d_i > _IRRED_SEARCH_MAX:
                return _unknown(op, f"search space p^degree={p_i}^{d_i} exceeds the "
                                    f"{_IRRED_SEARCH_MAX} deterministic-search cap.")
            f = _search_irreducible(p_i, d_i)
            if f is None:  # unreachable
                return _execution_error(
                    f"no irreducible of degree {d_i} found over GF({p_i})."
                )
            from sympy.polys.domains import ZZ
            from sympy.polys.galoistools import gf_irreducible_p

            verified = bool(gf_irreducible_p(f, p_i, ZZ))
            return _ok(op, result=f, coeffs=f, verified=verified,
                       field=f"GF({p_i})", degree=d_i)

        # ---- resolve the field modulus for the arithmetic/structure ops ----
        mod, merr, munknown = _resolve_modulus(op, p_i, k_i, modulus)
        if merr:
            return merr
        if munknown:
            return munknown

        from sympy.polys.domains import ZZ
        from sympy.polys.galoistools import (
            gf_add,
            gf_gcdex,
            gf_mul,
            gf_pow_mod,
            gf_rem,
            gf_strip,
            gf_sub,
        )

        field_str = f"GF({p_i}^{k_i})" if k_i > 1 else f"GF({p_i})"

        def reduce_elem(elem):
            return gf_strip(gf_rem(elem, mod, p_i, ZZ))

        def fmul(x, y):
            return reduce_elem(gf_mul(x, y, p_i, ZZ))

        def to_result(coeffs):
            c = list(coeffs)
            return {"coeffs": c, "int": _elem_to_int(c, p_i)}

        # -------------------------------------------------------------- add / mul
        if op in ("add", "mul"):
            if a is None or b is None:
                return _validation_error(f"op '{op}' requires elements 'a' and 'b'.")
            ea, ea_err = _parse_poly(a, "a", p_i)
            if ea_err:
                return ea_err
            eb, eb_err = _parse_poly(b, "b", p_i)
            if eb_err:
                return eb_err
            ea, eb = reduce_elem(ea), reduce_elem(eb)
            if op == "add":
                res = reduce_elem(gf_add(ea, eb, p_i, ZZ))
            else:
                res = fmul(ea, eb)
            return _ok(op, result=res, modulus_used=mod, field=field_str, **to_result(res))

        # -------------------------------------------------------------------- pow
        if op == "pow":
            if a is None or e is None:
                return _validation_error("op 'pow' requires element 'a' and integer exponent 'e'.")
            ea, ea_err = _parse_poly(a, "a", p_i)
            if ea_err:
                return ea_err
            e_i, err = _to_int(e, "e")
            if err:
                return err
            ea = reduce_elem(ea)
            if e_i == 0:
                res = [1]
                return _ok(op, result=res, modulus_used=mod, field=field_str, **to_result(res))
            base = ea
            exp = e_i
            if e_i < 0:
                if not ea:
                    return _ok(op, result=None, exists=False, modulus_used=mod, field=field_str,
                               reason="zero element has no inverse; negative power undefined.")
                s, t, h = gf_gcdex(ea, mod, p_i, ZZ)
                if h != [1]:
                    return _ok(op, result=None, exists=False, modulus_used=mod, field=field_str,
                               reason="element is not invertible in this field.")
                base = reduce_elem(s)
                exp = -e_i
            res = gf_strip(gf_pow_mod(base, exp, mod, p_i, ZZ))
            return _ok(op, result=res, modulus_used=mod, field=field_str, **to_result(res))

        # ---------------------------------------------------------------- inverse
        if op == "inverse":
            if a is None:
                return _validation_error("op 'inverse' requires element 'a'.")
            ea, ea_err = _parse_poly(a, "a", p_i)
            if ea_err:
                return ea_err
            ea = reduce_elem(ea)
            if not ea:  # zero element
                return _ok(op, result=None, exists=False, modulus_used=mod, field=field_str,
                           reason="the zero element has no multiplicative inverse.")
            s, t, h = gf_gcdex(ea, mod, p_i, ZZ)
            if h != [1]:
                return _ok(op, result=None, exists=False, modulus_used=mod, field=field_str,
                           reason="element is not invertible (non-trivial gcd with modulus).")
            inv = reduce_elem(s)
            # HONESTY: re-check a * a^-1 == 1 in the field before returning.
            verified = fmul(ea, inv) == [1]
            if not verified:
                return _execution_error("inverse re-check failed: a * a^-1 != 1 in the field.")
            return _ok(op, result=inv, exists=True, verified=True,
                       modulus_used=mod, field=field_str, **to_result(inv))

        # ----------------------------------------------------------- element_order
        if op == "element_order":
            if a is None:
                return _validation_error("op 'element_order' requires element 'a'.")
            if _digits(p_i ** k_i) > _ORDER_FIELD_MAX_DIGITS:
                return _unknown(op, f"field size p^k={p_i}^{k_i} is too large to factor "
                                    f"p^k-1 within the {_ORDER_FIELD_MAX_DIGITS}-digit cap.")
            ea, ea_err = _parse_poly(a, "a", p_i)
            if ea_err:
                return ea_err
            ea = reduce_elem(ea)
            if not ea:
                return _validation_error(
                    "the zero element has no multiplicative order (0 is not a unit)."
                )
            from sympy import factorint

            group_order = p_i ** k_i - 1
            order = group_order
            for q in factorint(group_order):
                while order % q == 0 and gf_strip(gf_pow_mod(ea, order // q, mod, p_i, ZZ)) == [1]:
                    order //= q
            # HONESTY: re-check a^order == 1 and order divides the group order.
            verified = (
                gf_strip(gf_pow_mod(ea, order, mod, p_i, ZZ)) == [1]
                and group_order % order == 0
            )
            if not verified:
                return _execution_error("element_order re-check failed (a^order != 1).")
            return _ok(op, result=int(order), verified=True, modulus_used=mod,
                       field=field_str, group_order=int(group_order))

        # ----------------------------------------------------- minimal_polynomial
        if op == "minimal_polynomial":
            if a is None:
                return _validation_error("op 'minimal_polynomial' requires element 'a'.")
            ea, ea_err = _parse_poly(a, "a", p_i)
            if ea_err:
                return ea_err
            ea = reduce_elem(ea)
            # Conjugates under Frobenius: a, a^p, a^(p^2), ... until it cycles back to a.
            conjugates = []
            cur = ea
            for _ in range(k_i):
                conjugates.append(cur)
                cur = gf_strip(gf_pow_mod(cur, p_i, mod, p_i, ZZ))
                if cur == ea:
                    break
            # minpoly = prod (x - c) over the distinct conjugates, as a polynomial whose
            # coefficients are field elements (they collapse to GF(p) constants).
            # Represent as list of field elements, index = power of x (low-to-high).
            prod = [[1]]  # constant polynomial 1
            for c in conjugates:
                # multiply prod by (x - c): new[j] = prod[j-1] - c*prod[j]
                new = [[] for _ in range(len(prod) + 1)]
                for j in range(len(prod)):
                    # + prod[j] * x  -> contributes to new[j+1]
                    new[j + 1] = reduce_elem(gf_add(new[j + 1], prod[j], p_i, ZZ))
                    # - c * prod[j]  -> contributes to new[j]
                    term = fmul(c, prod[j])
                    new[j] = reduce_elem(gf_sub(new[j], term, p_i, ZZ))
                prod = new
            # Extract GF(p) coefficients: each prod[j] must be a constant field element.
            coeffs_low_to_high = []
            for fe in prod:
                if len(fe) > 1:
                    return _execution_error(
                        "minimal_polynomial produced a non-constant coefficient "
                        "(element not algebraic of the expected form)."
                    )
                coeffs_low_to_high.append(fe[0] % p_i if fe else 0)
            minpoly = list(reversed(coeffs_low_to_high))  # high-to-low, monic
            deg = len(minpoly) - 1
            # HONESTY: evaluate the minimal polynomial at the element in GF(p^k); must be 0.
            acc = []  # zero
            for coeff in minpoly:  # high-to-low Horner: acc = acc*a + coeff
                acc = reduce_elem(gf_add(fmul(acc, ea), [coeff] if coeff else [], p_i, ZZ))
            eval_zero = acc == []
            divides = (k_i % deg == 0) if deg > 0 else False
            if not (eval_zero and divides):
                return _execution_error(
                    f"minimal_polynomial re-check failed (eval_zero={eval_zero}, "
                    f"deg={deg} divides k={k_i}: {divides})."
                )
            return _ok(op, result=minpoly, coeffs=minpoly, degree=deg, verified=True,
                       modulus_used=mod, field=field_str)

        return _validation_error(f"Unhandled op {op!r}.")

    except Exception as exc:  # noqa: BLE001 — never raise to the caller.
        return _execution_error(f"finite_field_compute op '{op}' failed: {exc}")
