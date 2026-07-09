"""Number-theory dispatch tool (exact, honest, self-verifying).

M2 (domain-capabilities §1b): a single ``number_theory`` tool dispatching over the
repetitive number-theoretic primitives a mathematician runs across thousands of
investigations — factorization, primality, gcd/lcm, modular arithmetic, CRT, totient,
divisors, orders, primitive roots, Jacobi symbols and continued fractions — all via
``sympy.ntheory`` with exact integer arithmetic.

Honesty by construction (domain-capabilities §0):
  * ``factorint`` returns the factorization AND the re-multiplied product so the caller
    can independently confirm ``prod(p**e) == input``. The tool asserts this itself and
    FAILS LOUDLY (execution_error) if it ever mismatches — a factorization is never
    self-reported without the trivially re-checkable witness.
  * primality / order / primitive-root results come from sympy's deterministic checks.
  * Every op caps input magnitude; a too-large input returns an honest
    ``status="unknown"`` instead of hanging (never a guessed answer).
  * Inputs are validated (int or decimal string only); non-integers return a
    ``validation_error``. The tool never raises to the caller.
"""
from __future__ import annotations

from functools import reduce
from math import gcd as _math_gcd
from math import prod as _math_prod

from propab.tools.types import ToolError, ToolResult

# Ops whose evaluation needs an integer factorization (of the input or of n-1 / totient).
# These inherit the stricter factoring cap because factoring is the hard operation.
_FACTORING_OPS = frozenset(
    {
        "factorint",
        "totient",
        "divisors",
        "divisor_count",
        "multiplicative_order",
        "primitive_root",
        "is_primitive_root",
    }
)

_ALLOWED_OPS = frozenset(
    {
        "factorint",
        "isprime",
        "nextprime",
        "prevprime",
        "gcd",
        "lcm",
        "mod_inverse",
        "mod_pow",
        "crt",
        "continued_fraction",
        "jacobi_symbol",
    }
    | _FACTORING_OPS
)

# Magnitude caps (decimal-digit counts). Factoring is the hard operation, so ops that
# require a factorization are capped far lower than the cheap ones. Beyond the cap we
# return an honest ``unknown`` rather than risk hanging (see the 200-digit example).
_FACTOR_MAX_DIGITS = 40
_GENERAL_MAX_DIGITS = 2000

TOOL_SPEC = {
    "name": "number_theory",
    "domain": "mathematics",
    "audience": "worker",
    "description": (
        "Exact number-theory dispatch over sympy.ntheory. op is one of: factorint, "
        "isprime, nextprime, prevprime, gcd, lcm, mod_inverse, mod_pow, crt, totient, "
        "divisors, divisor_count, multiplicative_order, primitive_root, is_primitive_root, "
        "jacobi_symbol, continued_fraction. Integer inputs (int or decimal string; "
        "non-integers are rejected). factorint additionally returns the re-multiplied "
        "product so the caller can confirm it equals the input. Too-large inputs return "
        "status='unknown' rather than hanging."
    ),
    "params": {
        "op": {"type": "str", "required": True,
                "description": "Operation to perform (see the op list in the description)."},
        "a": {"type": "int|str", "required": False,
               "description": "Primary integer operand (int or decimal string)."},
        "b": {"type": "int|str", "required": False,
               "description": "Second integer operand (gcd/lcm second arg, mod_pow exponent, "
                              "continued_fraction denominator)."},
        "m": {"type": "int|str", "required": False,
               "description": "Modulus (mod_inverse / mod_pow / multiplicative_order / "
                              "is_primitive_root / jacobi_symbol)."},
        "values": {"type": "list[int]", "required": False,
                    "description": "List of integers for gcd/lcm over more than two operands."},
        "residues": {"type": "list[int]", "required": False,
                      "description": "CRT residues (same length as moduli)."},
        "moduli": {"type": "list[int]", "required": False,
                    "description": "CRT moduli (pairwise-coprime not required; None if no solution)."},
    },
    "output": {
        "op": "str — the operation performed",
        "status": "str — 'ok' or 'unknown' (too large to compute honestly)",
        "result": "the primary result (op-specific)",
        "factors": "dict[int,int] — factorint only: prime -> exponent",
        "product": "int — factorint only: re-multiplied prod(p**e) (must equal input)",
        "product_equals_input": "bool — factorint only: independent re-multiplication check",
        "verified": "bool — op-specific independent re-check passed (where applicable)",
        "exists": "bool — mod_inverse / crt: whether a solution exists",
        "reason": "str — present when status='unknown'",
    },
    "example": {
        "params": {"op": "factorint", "a": 360},
        "output": {
            "op": "factorint",
            "status": "ok",
            "factors": {2: 3, 3: 2, 5: 1},
            "product": 360,
            "product_equals_input": True,
        },
    },
}


def _validation_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="validation_error", message=message))


def _execution_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="execution_error", message=message))


def _ok(op: str, **fields) -> ToolResult:
    return ToolResult(success=True, output={"op": op, "status": "ok", **fields})


def _unknown(op: str, reason: str) -> ToolResult:
    return ToolResult(success=True, output={"op": op, "status": "unknown", "reason": reason})


def _to_int(value, name: str):
    """Coerce ``value`` to a Python int, accepting int or decimal string only.

    Returns (int, None) on success or (None, ToolResult validation_error). Rejects
    floats (even integral ones), bools, and non-numeric strings — the tool is exact and
    must not silently accept a lossy input.
    """
    if isinstance(value, bool):
        return None, _validation_error(f"'{name}' must be an integer, got a bool {value!r}.")
    if isinstance(value, int):
        return value, None
    if isinstance(value, str):
        s = value.strip()
        # int(s) already rejects "3.5", "1e3", "abc"; allow a leading sign.
        try:
            return int(s), None
        except ValueError:
            return None, _validation_error(
                f"'{name}' must be an integer or decimal string, got {value!r}."
            )
    return None, _validation_error(
        f"'{name}' must be an integer or decimal string, got {type(value).__name__} {value!r}."
    )


def _digits(n: int) -> int:
    return len(str(abs(int(n))))


def _too_big(op: str, ints, cap: int):
    """Return an ``unknown`` ToolResult if any operand exceeds ``cap`` digits, else None."""
    for v in ints:
        if v is not None and _digits(v) > cap:
            return _unknown(
                op,
                f"input has {_digits(v)} digits, exceeding the {cap}-digit cap for op "
                f"'{op}' — refusing to risk a hang; result is honestly unknown.",
            )
    return None


def number_theory(
    op: str | None = None,
    a=None,
    b=None,
    m=None,
    values=None,
    residues=None,
    moduli=None,
) -> ToolResult:
    if not op or not isinstance(op, str):
        return _validation_error("Parameter 'op' (string) is required.")
    op = op.strip().lower()
    if op not in _ALLOWED_OPS:
        return _validation_error(
            f"Unknown op {op!r}; supported: {sorted(_ALLOWED_OPS)}."
        )

    # ---- parse integer operands that are present ----
    a_i = b_i = m_i = None
    if a is not None:
        a_i, err = _to_int(a, "a")
        if err:
            return err
    if b is not None:
        b_i, err = _to_int(b, "b")
        if err:
            return err
    if m is not None:
        m_i, err = _to_int(m, "m")
        if err:
            return err

    cap = _FACTOR_MAX_DIGITS if op in _FACTORING_OPS else _GENERAL_MAX_DIGITS

    try:
        # ------------------------------------------------------------------ factorint
        if op == "factorint":
            if a_i is None:
                return _validation_error("op 'factorint' requires integer 'a'.")
            big = _too_big(op, [a_i], cap)
            if big:
                return big
            from sympy import factorint

            factors = {int(p): int(e) for p, e in factorint(a_i).items()}
            # HONESTY: re-multiply the returned factorization and confirm it equals the
            # input. This is an independent re-check the caller can repeat; if it ever
            # fails the tool fails loudly rather than emitting a bad factorization.
            product = _math_prod([p ** e for p, e in factors.items()]) if factors else 1
            if int(product) != int(a_i):
                return _execution_error(
                    "factorint re-multiplication mismatch: "
                    f"prod(p**e)={product} != input {a_i} (factors={factors})."
                )
            return _ok(
                op,
                result=factors,
                factors=factors,
                product=int(product),
                product_equals_input=True,
            )

        # ------------------------------------------------------------------ isprime
        if op == "isprime":
            if a_i is None:
                return _validation_error("op 'isprime' requires integer 'a'.")
            big = _too_big(op, [a_i], cap)
            if big:
                return big
            from sympy import isprime

            return _ok(op, result=bool(isprime(a_i)))

        # ------------------------------------------------------------ nextprime / prevprime
        if op in ("nextprime", "prevprime"):
            if a_i is None:
                return _validation_error(f"op '{op}' requires integer 'a'.")
            big = _too_big(op, [a_i], cap)
            if big:
                return big
            from sympy import isprime, nextprime, prevprime

            if op == "prevprime" and a_i <= 2:
                return _validation_error("prevprime is undefined for a <= 2 (no prime below 2).")
            p = int(nextprime(a_i) if op == "nextprime" else prevprime(a_i))
            return _ok(op, result=p, verified=bool(isprime(p)))

        # ------------------------------------------------------------------ gcd / lcm
        if op in ("gcd", "lcm"):
            nums = None
            if values is not None:
                parsed = []
                for i, v in enumerate(values):
                    vi, err = _to_int(v, f"values[{i}]")
                    if err:
                        return err
                    parsed.append(vi)
                if not parsed:
                    return _validation_error(f"op '{op}': 'values' must be non-empty.")
                nums = parsed
            elif a_i is not None and b_i is not None:
                nums = [a_i, b_i]
            else:
                return _validation_error(
                    f"op '{op}' requires either 'values' (list) or both 'a' and 'b'."
                )
            big = _too_big(op, nums, cap)
            if big:
                return big
            if op == "gcd":
                res = reduce(_math_gcd, (abs(x) for x in nums))
            else:
                def _lcm2(x: int, y: int) -> int:
                    if x == 0 or y == 0:
                        return 0
                    return abs(x // _math_gcd(x, y) * y)

                res = reduce(_lcm2, nums)
            return _ok(op, result=int(res))

        # ------------------------------------------------------------------ mod_inverse
        if op == "mod_inverse":
            if a_i is None or m_i is None:
                return _validation_error("op 'mod_inverse' requires integers 'a' and 'm'.")
            if m_i <= 0:
                return _validation_error("op 'mod_inverse' requires modulus 'm' > 0.")
            big = _too_big(op, [a_i, m_i], cap)
            if big:
                return big
            from sympy.core.numbers import mod_inverse

            try:
                inv = int(mod_inverse(a_i, m_i))
            except ValueError:
                # No inverse: this is a valid mathematical outcome, not a tool error.
                return _ok(op, result=None, exists=False,
                           reason=f"gcd({a_i % m_i}, {m_i}) != 1 — no inverse exists.")
            verified = ((a_i * inv) % m_i) == (1 % m_i)
            return _ok(op, result=inv, exists=True, verified=bool(verified))

        # ------------------------------------------------------------------ mod_pow
        if op == "mod_pow":
            if a_i is None or b_i is None or m_i is None:
                return _validation_error(
                    "op 'mod_pow' requires integers 'a' (base), 'b' (exponent), 'm' (modulus)."
                )
            if m_i <= 0:
                return _validation_error("op 'mod_pow' requires modulus 'm' > 0.")
            big = _too_big(op, [a_i, b_i, m_i], cap)
            if big:
                return big
            try:
                res = int(pow(a_i, b_i, m_i))
            except ValueError:
                # Negative exponent with a non-invertible base.
                return _ok(op, result=None, exists=False,
                           reason=f"base {a_i} has no inverse mod {m_i}; negative-exponent power undefined.")
            return _ok(op, result=res, exists=True)

        # ------------------------------------------------------------------ crt
        if op == "crt":
            if residues is None or moduli is None:
                return _validation_error("op 'crt' requires 'residues' and 'moduli' lists.")
            if len(residues) != len(moduli) or len(moduli) == 0:
                return _validation_error(
                    "op 'crt' requires non-empty 'residues' and 'moduli' of equal length."
                )
            res_p, mod_p = [], []
            for i, (r, mod) in enumerate(zip(residues, moduli)):
                ri, err = _to_int(r, f"residues[{i}]")
                if err:
                    return err
                mi, err = _to_int(mod, f"moduli[{i}]")
                if err:
                    return err
                if mi <= 0:
                    return _validation_error(f"op 'crt': moduli[{i}] must be > 0, got {mi}.")
                res_p.append(ri)
                mod_p.append(mi)
            big = _too_big(op, res_p + mod_p, cap)
            if big:
                return big
            from sympy.ntheory.modular import crt

            sol = crt(mod_p, res_p)
            if sol is None:
                return _ok(op, result=None, exists=False,
                           reason="no simultaneous solution (inconsistent congruences).")
            x, M = int(sol[0]), int(sol[1])
            # HONESTY: re-check the solution satisfies every congruence.
            verified = all((x - r) % mod == 0 for r, mod in zip(res_p, mod_p))
            return _ok(op, result={"solution": x, "modulus": M},
                       exists=True, verified=bool(verified))

        # ------------------------------------------------------------------ totient
        if op == "totient":
            if a_i is None:
                return _validation_error("op 'totient' requires integer 'a'.")
            if a_i < 1:
                return _validation_error("op 'totient' requires 'a' >= 1.")
            big = _too_big(op, [a_i], cap)
            if big:
                return big
            from sympy import totient

            return _ok(op, result=int(totient(a_i)))

        # ------------------------------------------------------ divisors / divisor_count
        if op in ("divisors", "divisor_count"):
            if a_i is None:
                return _validation_error(f"op '{op}' requires integer 'a'.")
            if a_i < 1:
                return _validation_error(f"op '{op}' requires 'a' >= 1.")
            big = _too_big(op, [a_i], cap)
            if big:
                return big
            from sympy import divisor_count, divisors

            if op == "divisors":
                return _ok(op, result=[int(d) for d in divisors(a_i)])
            return _ok(op, result=int(divisor_count(a_i)))

        # ------------------------------------------------------- multiplicative_order
        if op == "multiplicative_order":
            if a_i is None or m_i is None:
                return _validation_error(
                    "op 'multiplicative_order' requires integers 'a' and 'm'."
                )
            if m_i <= 1:
                return _validation_error("op 'multiplicative_order' requires modulus 'm' > 1.")
            big = _too_big(op, [a_i, m_i], cap)
            if big:
                return big
            if _math_gcd(a_i, m_i) != 1:
                return _validation_error(
                    f"op 'multiplicative_order' requires gcd(a, m) = 1 (got gcd={_math_gcd(a_i, m_i)})."
                )
            from sympy.ntheory import n_order

            order = int(n_order(a_i, m_i))
            verified = pow(a_i, order, m_i) == 1 % m_i
            return _ok(op, result=order, verified=bool(verified))

        # ------------------------------------------------------------- primitive_root
        if op == "primitive_root":
            if a_i is None:
                return _validation_error("op 'primitive_root' requires integer 'a' (the modulus n).")
            if a_i < 1:
                return _validation_error("op 'primitive_root' requires 'a' >= 1.")
            big = _too_big(op, [a_i], cap)
            if big:
                return big
            from sympy import is_primitive_root, primitive_root

            g = primitive_root(a_i)
            if g is None:
                return _ok(op, result=None, exists=False,
                           reason=f"no primitive root exists modulo {a_i}.")
            verified = bool(is_primitive_root(g, a_i))
            return _ok(op, result=int(g), exists=True, verified=verified)

        # ---------------------------------------------------------- is_primitive_root
        if op == "is_primitive_root":
            if a_i is None or m_i is None:
                return _validation_error(
                    "op 'is_primitive_root' requires integers 'a' and 'm' (the modulus)."
                )
            if m_i <= 1:
                return _validation_error("op 'is_primitive_root' requires modulus 'm' > 1.")
            big = _too_big(op, [a_i, m_i], cap)
            if big:
                return big
            if _math_gcd(a_i, m_i) != 1:
                # Not a unit -> cannot be a primitive root.
                return _ok(op, result=False,
                           reason=f"gcd(a, m) = {_math_gcd(a_i, m_i)} != 1, so a is not a unit mod m.")
            from sympy import is_primitive_root

            return _ok(op, result=bool(is_primitive_root(a_i, m_i)))

        # ----------------------------------------------------------- jacobi_symbol
        if op == "jacobi_symbol":
            if a_i is None or m_i is None:
                return _validation_error("op 'jacobi_symbol' requires integers 'a' and 'm'.")
            if m_i <= 0 or m_i % 2 == 0:
                return _validation_error(
                    "op 'jacobi_symbol' requires an odd positive modulus 'm'."
                )
            big = _too_big(op, [a_i, m_i], cap)
            if big:
                return big
            from sympy import jacobi_symbol

            return _ok(op, result=int(jacobi_symbol(a_i, m_i)))

        # -------------------------------------------------------- continued_fraction
        if op == "continued_fraction":
            if a_i is None:
                return _validation_error(
                    "op 'continued_fraction' requires integer 'a' "
                    "(with 'b' for a rational a/b, or alone for sqrt(a))."
                )
            big = _too_big(op, [a_i, b_i], cap)
            if big:
                return big
            if b_i is not None:
                if b_i == 0:
                    return _validation_error("op 'continued_fraction': denominator 'b' must be non-zero.")
                from sympy import Rational, continued_fraction

                terms = [int(t) for t in continued_fraction(Rational(a_i, b_i))]
                return _ok(op, result=terms, kind="rational", value=f"{a_i}/{b_i}",
                           periodic=False)
            # sqrt(a): periodic continued fraction (empty period => perfect square).
            if a_i < 0:
                return _validation_error(
                    "op 'continued_fraction' of sqrt(a) requires a >= 0 (pass 'b' for a rational)."
                )
            from sympy.ntheory.continued_fraction import continued_fraction_periodic

            expansion = continued_fraction_periodic(0, 1, a_i)
            has_period = len(expansion) > 1 and isinstance(expansion[-1], list)
            return _ok(op, result=expansion, kind="sqrt", value=f"sqrt({a_i})",
                       periodic=bool(has_period))

        # Unreachable: op was validated against _ALLOWED_OPS above.
        return _validation_error(f"Unhandled op {op!r}.")

    except Exception as exc:  # noqa: BLE001 — never raise to the caller.
        return _execution_error(f"number_theory op '{op}' failed: {exc}")
