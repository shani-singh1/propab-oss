"""Exact linear algebra over ℚ or a finite field 𝔽_p (M1, domain-capabilities §1a).

A single dispatch tool for the repetitive "do the matrix algebra exactly" workflow:
rank / determinant / inverse / nullspace / eigen / characteristic polynomial / RREF /
linear solve. All arithmetic is EXACT — sympy rationals (no float error) or, when a
prime ``modulus`` is given, exact arithmetic in GF(p).

Honesty by construction (domain-capabilities §0):
  * Every result that has a trivially re-checkable witness is INDEPENDENTLY re-verified
    before it is returned, and the ``verified`` flag reports that re-check:
      - inverse:   M · M⁻¹ == I (re-multiplied);
      - solve:     M · x == b (re-substituted);
      - nullspace: M · v == 0 for each returned basis vector;
      - eigen:     Σ λ·mult == trace(M) and Π λ^mult == det(M);
      - charpoly:  Cayley–Hamilton p(M) == 0 (for small matrices).
    A failed re-check is surfaced as an ``execution_error`` — never returned as valid.
  * Matrix entries given as strings are parsed with a RESTRICTED sympy parser (whitelist
    namespace, ``__builtins__`` stripped, dunder/import/attribute screen) — no eval.
  * Squareness / dimension / prime-modulus preconditions are validated; oversized inputs
    are rejected and heavy ops run under a best-effort timeout, returning honestly rather
    than hanging. The tool never raises to the caller.
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
_MAX_EXPR_LEN = 2000
_MAX_DIM = 64
_MAX_EIGEN_DIM = 24          # eigen/charpoly are much heavier — cap lower
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
    "Symbol", "Integer", "Float", "Rational", "symbols", "pi", "E", "I",
    "sqrt", "cbrt", "root", "exp", "log", "Abs", "sign",
    "sin", "cos", "tan", "asin", "acos", "atan", "gamma", "factorial", "binomial",
)


def _allowed_namespace() -> dict:
    ns: dict = {}
    for nm in _ALLOWED_NAME_LIST:
        obj = getattr(_sp, nm, None)
        if obj is not None:
            ns[nm] = obj
    ns["ln"] = _sp.log
    ns["__builtins__"] = {}
    return ns


def _safe_sympify(text):
    if not isinstance(text, str):
        raise ValueError("expression must be a string")
    s = text.strip()
    if not s:
        raise ValueError("expression must be non-empty")
    if len(s) > _MAX_EXPR_LEN:
        raise ValueError(f"entry exceeds max length {_MAX_EXPR_LEN}")
    if _FORBIDDEN.search(s):
        raise ValueError("entry contains a forbidden token")
    if _ATTR_RE.search(s):
        raise ValueError("attribute access is not permitted")
    return parse_expr(
        s, local_dict={}, global_dict=_allowed_namespace(),
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


_OPS = ("rank", "det", "inverse", "nullspace", "eigen", "charpoly", "rref", "solve")

TOOL_SPEC = {
    "name": "exact_linear_algebra",
    "domain": "mathematics",
    "audience": "worker",
    "verification_capable": True,
    "description": (
        "Exact linear algebra over QQ (default) or GF(p) (pass a prime 'modulus'). op is "
        "one of rank|det|inverse|nullspace|eigen|charpoly|rref|solve. 'matrix' is a "
        "list-of-lists of numbers or strings (strings parsed in a RESTRICTED namespace, "
        "no eval). Arithmetic is exact rational / finite-field (never float). Results with "
        "a cheap witness are INDEPENDENTLY re-checked before return (inverse re-multiplied "
        "to I, solve re-substituted into M*x=b, nullspace vectors mapped to 0, eigen "
        "cross-checked against trace & det, charpoly against Cayley-Hamilton) and reported "
        "in 'verified'. eigen is unavailable over GF(p) (needs a field extension). "
        "Oversized / non-square (where required) inputs are rejected honestly."
    ),
    "params": {
        "op": {"type": "str", "required": True,
               "description": "rank|det|inverse|nullspace|eigen|charpoly|rref|solve"},
        "matrix": {"type": "list[list]", "required": True,
                   "description": "Rows of the matrix; each entry a number or a string "
                                  "(e.g. '1/2', 'sqrt(2)', or a symbol like 'a')."},
        "b": {"type": "list", "required": False,
              "description": "For op=solve: the right-hand side vector (length = #rows)."},
        "modulus": {"type": "int", "required": False, "default": None,
                    "description": "If given (a prime p), compute over GF(p). Entries must "
                                   "be integers. eigen is not supported over GF(p)."},
        "timeout": {"type": "number", "required": False, "default": 12,
                    "description": "Per-op wall-clock cap in seconds (<= 60)."},
    },
    "output": {
        "op": "str",
        "field": "str — 'QQ' or 'GF(p)'",
        "result": "op-specific exact result (string / list / dict)",
        "verified": "bool|None — the independent re-check passed (where applicable)",
        "shape": "list[int] — [rows, cols]",
        "note": "str",
    },
    "example": {
        "params": {"op": "det", "matrix": [[1, 2], [3, 4]]},
        "output": {"op": "det", "field": "QQ", "result": "-2", "shape": [2, 2]},
    },
}


def _validation_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="validation_error", message=message))


def _execution_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=ToolError(type="execution_error", message=message))


def _parse_entry(x):
    """Parse a single matrix entry to an exact sympy object (never a lossy float)."""
    if isinstance(x, bool):
        raise ValueError(f"boolean {x!r} is not a valid matrix entry")
    if isinstance(x, int):
        return _sp.Integer(x)
    if isinstance(x, float):
        # exact decimal rational (0.5 -> 1/2), never a binary-float artifact
        return _sp.Rational(str(x))
    if isinstance(x, str):
        return _safe_sympify(x)
    if isinstance(x, _sp.Basic):
        return x
    raise ValueError(f"unsupported entry type {type(x).__name__}: {x!r}")


def _build_matrix(matrix):
    if not isinstance(matrix, (list, tuple)) or not matrix:
        raise ValueError("'matrix' must be a non-empty list of rows")
    rows = []
    ncols = None
    for i, row in enumerate(matrix):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"row {i} must be a list")
        if ncols is None:
            ncols = len(row)
        elif len(row) != ncols:
            raise ValueError(f"ragged matrix: row {i} has {len(row)} entries, expected {ncols}")
        rows.append([_parse_entry(e) for e in row])
    if ncols == 0:
        raise ValueError("matrix rows must be non-empty")
    if len(rows) > _MAX_DIM or ncols > _MAX_DIM:
        raise ValueError(f"matrix dimension exceeds cap {_MAX_DIM}")
    return _sp.Matrix(rows)


def _require_integer_entries(M, p):
    """Return an integer sympy Matrix reduced mod p, or raise if any entry is not integral."""
    ints = []
    for i in range(M.rows):
        row = []
        for j in range(M.cols):
            e = M[i, j]
            if not e.is_number or not e.is_integer:
                raise ValueError(
                    f"GF({p}) requires integer entries; entry [{i},{j}]={e} is not an integer"
                )
            row.append(int(e) % p)
        ints.append(row)
    return _sp.Matrix(ints)


def _mat_to_list(M, mod=None):
    out = []
    for i in range(M.rows):
        row = []
        for j in range(M.cols):
            v = M[i, j]
            if mod is not None:
                row.append(int(v) % mod)
            else:
                row.append(str(v))
        out.append(row)
    return out


# ---------------------------------------------------------------------------- #
# GF(p) backends via DomainMatrix.
# ---------------------------------------------------------------------------- #
def _gf_domainmatrix(Mint, p):
    from sympy import GF
    from sympy.polys.matrices import DomainMatrix

    return DomainMatrix.from_Matrix(Mint).convert_to(GF(p))


def _gf_solve(Mint, bvec, p):
    """Solve M*x = b over GF(p) via augmented RREF. Returns (solution list | None)."""
    aug = Mint.row_join(bvec)
    dm = _gf_domainmatrix(aug, p)
    R, pivots = dm.rref()
    Rm = R.to_Matrix()
    ncols = Mint.cols
    # Inconsistent iff a pivot sits in the appended RHS column.
    if ncols in pivots:
        return None
    x = [0] * ncols
    piv_rows = list(pivots)
    for r, c in enumerate(piv_rows):
        if c < ncols:
            x[c] = int(Rm[r, ncols]) % p
    return x


def exact_linear_algebra(op=None, matrix=None, b=None, modulus=None, timeout=_DEFAULT_TIMEOUT) -> ToolResult:
    # ---- input validation (never raise) ----
    if not isinstance(op, str) or op.strip().lower() not in _OPS:
        return _validation_error(f"'op' must be one of {list(_OPS)}; got {op!r}.")
    op = op.strip().lower()
    try:
        M = _build_matrix(matrix)
    except (ValueError, SyntaxError, TypeError) as exc:
        return _validation_error(f"could not build 'matrix': {exc}")
    except Exception as exc:  # noqa: BLE001
        return _validation_error(f"could not build 'matrix': {exc}")

    try:
        to = float(timeout)
    except (TypeError, ValueError):
        to = _DEFAULT_TIMEOUT
    to = max(0.5, min(_MAX_TIMEOUT, to))

    # ---- finite-field setup ----
    p = None
    if modulus is not None:
        if isinstance(modulus, bool) or not isinstance(modulus, int):
            return _validation_error("'modulus' must be an integer prime.")
        if not _sp.isprime(modulus):
            return _validation_error(f"'modulus' must be prime for GF(p); {modulus} is not prime.")
        if op == "eigen":
            return _validation_error(
                "op 'eigen' is not supported over GF(p) (eigenvalues may live in an "
                "extension field). Use charpoly over GF(p) instead."
            )
        p = int(modulus)
        try:
            M = _require_integer_entries(M, p)
        except ValueError as exc:
            return _validation_error(str(exc))

    field = f"GF({p})" if p is not None else "QQ"
    square = M.rows == M.cols
    shape = [int(M.rows), int(M.cols)]

    # Ops needing squareness.
    if op in ("det", "inverse", "eigen", "charpoly") and not square:
        return _validation_error(f"op '{op}' requires a square matrix; got shape {shape}.")
    if op in ("eigen", "charpoly") and M.rows > _MAX_EIGEN_DIM:
        return _validation_error(
            f"op '{op}' capped at {_MAX_EIGEN_DIM}x{_MAX_EIGEN_DIM}; got {shape} "
            "(refusing to risk a hang)."
        )

    try:
        # ============================================================ GF(p) path
        if p is not None:
            return _gf_dispatch(op, M, b, p, field, shape, to)

        # ============================================================ QQ / exact path
        note = "exact rational arithmetic (no float error)."

        if op == "rank":
            r = _run_with_timeout(lambda: int(M.rank()), to)
            return _ok(op, field, r, shape, note=note)

        if op == "det":
            d = _run_with_timeout(lambda: M.det(), to)
            return _ok(op, field, str(d), shape, note=note)

        if op == "inverse":
            d = _run_with_timeout(lambda: M.det(), to)
            if d == 0:
                return _ok(op, field, None, shape, verified=None, exists=False,
                           note="matrix is singular (det = 0); no inverse exists.")
            inv = _run_with_timeout(lambda: M.inv(), to)
            # HONESTY: re-multiply and confirm M*inv == I.
            prod = _run_with_timeout(lambda: _sp.simplify(M * inv), to)
            verified = bool(prod == _sp.eye(M.rows))
            if not verified:
                return _execution_error("inverse re-check failed: M*M^-1 != I.")
            return _ok(op, field, _mat_to_list(inv), shape, verified=True, exists=True,
                       note="M*M^-1 re-multiplied and confirmed == I.")

        if op == "nullspace":
            basis = _run_with_timeout(lambda: M.nullspace(), to)
            # HONESTY: confirm every returned basis vector maps to 0.
            verified = True
            for v in basis:
                if _sp.simplify(M * v) != _sp.zeros(M.rows, 1):
                    verified = False
                    break
            if basis and not verified:
                return _execution_error("nullspace re-check failed: M*v != 0 for some basis vector.")
            return _ok(op, field, [_mat_to_list(v) for v in basis], shape,
                       verified=(True if basis else None), dimension=len(basis),
                       note=("each basis vector v was re-checked to satisfy M*v = 0."
                             if basis else "trivial nullspace (only the zero vector)."))

        if op == "eigen":
            evals = _run_with_timeout(lambda: M.eigenvals(), to)
            eig = {str(val): int(mult) for val, mult in evals.items()}
            # HONESTY cross-check: sum(lambda*mult) == trace and prod(lambda^mult) == det.
            trace = _run_with_timeout(lambda: M.trace(), to)
            det = _run_with_timeout(lambda: M.det(), to)
            sum_ev = _sp.simplify(sum(val * mult for val, mult in evals.items()))
            prod_ev = _sp.simplify(_sp.prod([val ** mult for val, mult in evals.items()]))
            verified = bool(_sp.simplify(sum_ev - trace) == 0 and _sp.simplify(prod_ev - det) == 0)
            complete = sum(evals.values()) == M.rows
            return _ok(op, field, eig, shape, verified=verified, complete=bool(complete),
                       note=("eigenvalues cross-checked: sum(lambda*mult) == trace and prod(lambda^mult) == det. "
                             + ("all eigenvalues found (mult sums to n)."
                                if complete else
                                "NOTE: multiplicities do not sum to n — some eigenvalues are "
                                "not expressible in closed form (honestly incomplete).")))

        if op == "charpoly":
            lam = _sp.Symbol("lambda")
            cp = _run_with_timeout(lambda: M.charpoly(lam), to)
            poly_expr = cp.as_expr()
            coeffs = [str(c) for c in cp.all_coeffs()]
            verified = None
            if M.rows <= 12:  # Cayley-Hamilton re-check (heavy) only for small matrices
                ch_at_M = _run_with_timeout(lambda: _apply_poly(cp, M), to)
                verified = bool(ch_at_M == _sp.zeros(M.rows, M.rows))
                if not verified:
                    return _execution_error("charpoly re-check failed: p(M) != 0 (Cayley-Hamilton).")
            return _ok(op, field, {"polynomial": str(poly_expr), "coefficients": coeffs},
                       shape, verified=verified,
                       note=("characteristic polynomial det(lambda*I - M); "
                             + ("re-checked via Cayley-Hamilton p(M)=0."
                                if verified else "Cayley-Hamilton check skipped (matrix too large).")))

        if op == "rref":
            R, pivots = _run_with_timeout(lambda: M.rref(), to)
            return _ok(op, field, {"rref": _mat_to_list(R), "pivots": [int(c) for c in pivots],
                                   "rank": len(pivots)}, shape, note=note)

        if op == "solve":
            if b is None:
                return _validation_error("op 'solve' requires a right-hand side 'b'.")
            try:
                bvec = _build_vector(b, M.rows)
            except (ValueError, SyntaxError, TypeError) as exc:
                return _validation_error(f"could not build 'b': {exc}")
            try:
                sol, params = _run_with_timeout(lambda: M.gauss_jordan_solve(bvec), to)
            except ValueError:
                return _ok(op, field, None, shape, exists=False, verified=None,
                           note="the system M*x = b is INCONSISTENT — no solution exists.")
            # HONESTY: re-substitute the particular solution (free params -> 0) into M*x = b.
            particular = sol
            if params.free_symbols:
                particular = sol.subs({s: 0 for s in params.free_symbols})
            verified = bool(_sp.simplify(M * particular - bvec) == _sp.zeros(M.rows, 1))
            if not verified:
                return _execution_error("solve re-check failed: M*x != b for the returned solution.")
            has_free = bool(params.free_symbols)
            return _ok(op, field, {"solution": _mat_to_list(sol),
                                   "particular": _mat_to_list(particular),
                                   "has_free_parameters": has_free,
                                   "free_parameters": sorted(str(s) for s in params.free_symbols)},
                       shape, verified=True, exists=True,
                       note=("M*x = b re-substituted and confirmed for the particular solution."
                             + (" Infinitely many solutions (free parameters present)." if has_free else "")))

        return _validation_error(f"unhandled op {op!r}.")  # pragma: no cover

    except _Timeout:
        return _execution_error(
            f"op '{op}' timed out after {to}s (matrix too hard); returned nothing rather than guess."
        )
    except (ValueError, SyntaxError, TypeError) as exc:
        return _validation_error(str(exc))
    except Exception as exc:  # noqa: BLE001 — never raise to the caller
        return _execution_error(f"exact_linear_algebra op '{op}' failed: {exc}")


def _apply_poly(cp, M):
    """Evaluate the characteristic polynomial ``cp`` at the matrix ``M`` (for C–H check)."""
    coeffs = cp.all_coeffs()  # highest degree first
    n = M.rows
    result = _sp.zeros(n, n)
    power = _sp.eye(n)
    for c in reversed(coeffs):  # ascending: c0 + c1 M + c2 M^2 + ...
        result = result + c * power
        power = power * M
    return _sp.simplify(result)


def _build_vector(b, n):
    if not isinstance(b, (list, tuple)) or not b:
        raise ValueError("'b' must be a non-empty list")
    # accept a flat list or a list of single-element lists
    flat = []
    for e in b:
        if isinstance(e, (list, tuple)):
            if len(e) != 1:
                raise ValueError("'b' rows must be scalars or single-element lists")
            flat.append(_parse_entry(e[0]))
        else:
            flat.append(_parse_entry(e))
    if len(flat) != n:
        raise ValueError(f"'b' has length {len(flat)}, expected {n} (one per row).")
    return _sp.Matrix(flat)


def _ok(op, field, result, shape, **extra):
    out = {"op": op, "field": field, "result": result, "shape": shape}
    out.update(extra)
    out.setdefault("verified", extra.get("verified"))
    return ToolResult(success=True, output=out)


def _gf_dispatch(op, Mint, b, p, field, shape, to):
    """GF(p) operations via DomainMatrix + integer-result-mod-p (det/charpoly)."""
    note = f"exact arithmetic in GF({p})."
    square = Mint.rows == Mint.cols

    if op == "rank":
        dm = _gf_domainmatrix(Mint, p)
        return _ok(op, field, int(dm.rank()), shape, note=note)

    if op == "det":
        # det is a polynomial in the entries, so (integer det) mod p == det over GF(p).
        d = int(_run_with_timeout(lambda: Mint.det(), to)) % p
        return _ok(op, field, d, shape, note=note)

    if op == "inverse":
        try:
            inv = Mint.inv_mod(p)
        except (ValueError, _sp.matrices.common.NonInvertibleMatrixError):
            return _ok(op, field, None, shape, exists=False, verified=None,
                       note=f"matrix is singular over GF({p}); no inverse exists.")
        prod = (Mint * inv).applyfunc(lambda e: int(e) % p)
        verified = bool(prod == _sp.eye(Mint.rows))
        if not verified:
            return _execution_error(f"GF({p}) inverse re-check failed: M*M^-1 != I.")
        return _ok(op, field, _mat_to_list(inv, mod=p), shape, verified=True, exists=True,
                   note=f"M*M^-1 re-multiplied mod {p} and confirmed == I.")

    if op == "nullspace":
        dm = _gf_domainmatrix(Mint, p)
        ns = dm.nullspace().to_Matrix()  # rows are basis vectors
        basis = [[int(ns[i, j]) % p for j in range(ns.cols)] for i in range(ns.rows)]
        # HONESTY: each basis vector v satisfies M*v == 0 mod p.
        verified = True
        for vec in basis:
            prod = [sum(int(Mint[i, j]) * vec[j] for j in range(Mint.cols)) % p for i in range(Mint.rows)]
            if any(prod):
                verified = False
                break
        if basis and not verified:
            return _execution_error(f"GF({p}) nullspace re-check failed: M*v != 0.")
        return _ok(op, field, basis, shape, verified=(True if basis else None),
                   dimension=len(basis),
                   note=(f"each basis vector re-checked: M*v = 0 mod {p}." if basis
                         else "trivial nullspace over GF(p)."))

    if op == "charpoly":
        # charpoly coefficients are integer polynomials in the entries -> reduce mod p.
        lam = _sp.Symbol("lambda")
        cp = _run_with_timeout(lambda: Mint.charpoly(lam), to)
        coeffs = [int(c) % p for c in cp.all_coeffs()]
        return _ok(op, field, {"coefficients": coeffs}, shape, verified=None,
                   note=f"characteristic-polynomial coefficients reduced mod {p} "
                        "(highest degree first).")

    if op == "rref":
        dm = _gf_domainmatrix(Mint, p)
        R, pivots = dm.rref()
        Rm = R.to_Matrix()
        return _ok(op, field, {"rref": [[int(Rm[i, j]) % p for j in range(Rm.cols)]
                                        for i in range(Rm.rows)],
                               "pivots": [int(c) for c in pivots], "rank": len(pivots)},
                   shape, note=note)

    if op == "solve":
        if b is None:
            return _validation_error("op 'solve' requires a right-hand side 'b'.")
        try:
            bvec = _require_integer_entries(_build_vector(b, Mint.rows), p)
        except (ValueError, SyntaxError, TypeError) as exc:
            return _validation_error(f"could not build 'b': {exc}")
        x = _gf_solve(Mint, bvec, p)
        if x is None:
            return _ok(op, field, None, shape, exists=False, verified=None,
                       note=f"the system M*x = b is INCONSISTENT over GF({p}) — no solution.")
        # HONESTY: re-substitute x into M*x == b mod p.
        prod = [sum(int(Mint[i, j]) * x[j] for j in range(Mint.cols)) % p for i in range(Mint.rows)]
        target = [int(bvec[i]) % p for i in range(Mint.rows)]
        verified = prod == target
        if not verified:
            return _execution_error(f"GF({p}) solve re-check failed: M*x != b.")
        return _ok(op, field, {"solution": x}, shape, verified=True, exists=True,
                   note=f"M*x = b re-substituted and confirmed mod {p}.")

    return _validation_error(f"op '{op}' not supported over GF(p).")
