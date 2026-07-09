"""Linear / mixed-integer programming tool (M4).

``linear_optimization`` solves ``minimize/maximize c·x`` subject to
``A_ub x <= b_ub``, ``A_eq x = b_eq`` and per-variable bounds, with an optional
integrality mask. Pure LPs go to ``scipy.optimize.linprog`` (HiGHS); as soon as any
variable is declared integer the problem is a MILP and is routed to OR-tools
``pywraplp`` (SCIP, falling back to CBC).

Honesty invariants (mirrors ``constraint_solve`` / ``cp_sat_finder``):
  * A returned ``x`` is INDEPENDENTLY re-checked against every declared constraint
    (inequalities, equalities, bounds, integrality) and the reported objective is
    recomputed as ``c·x`` — all within tolerance — BEFORE it is returned. A violation
    means a solver/encoding bug and is raised (``execution_error``), never surfaced as
    a valid optimum.
  * ``infeasible`` / ``unbounded`` are reported only when the solver actually proves
    them; a budget timeout is ``unknown`` — never a fabricated status.

Outcomes (``output["status"]``):
  * ``"optimal"``    — a proven optimal solution.
  * ``"feasible"``   — a valid solution found but optimality not proven (MILP timeout).
  * ``"infeasible"`` — the solver proved no feasible ``x`` exists.
  * ``"unbounded"``  — the objective is unbounded on the feasible region.
  * ``"unknown"``    — the budget/solver could not decide (honest non-answer).
"""
from __future__ import annotations

import math
import time
from typing import Any

from propab.tools.types import ToolError, ToolResult

_MAX_VARS = 5000
_MAX_CONSTRAINTS = 20000
_TOL = 1e-6  # absolute tolerance for the independent feasibility re-check
_INT_TOL = 1e-6  # how close an integer variable's value must be to a whole number

TOOL_SPEC = {
    "name": "linear_optimization",
    "domain": "mathematics",
    "audience": "worker",
    # Deterministic optimizer whose returned x is independently re-checked against all
    # declared constraints (evidence is a re-verified feasible point, not a p-value).
    "verification_capable": True,
    "description": (
        "Solve a linear program (LP) or mixed-integer linear program (MILP): "
        "minimize or maximize c·x subject to A_ub x <= b_ub, A_eq x = b_eq, and "
        "per-variable bounds, with an optional integrality mask. Pure LPs use "
        "scipy.optimize.linprog (HiGHS); any integer variable routes the problem to "
        "OR-tools MILP (SCIP/CBC). Returns status in "
        "{optimal, feasible, infeasible, unbounded, unknown}, the optimal value, and x. "
        "The returned x is independently re-checked against every constraint (within "
        "tolerance) and the objective recomputed as c·x before returning; infeasible / "
        "unbounded are reported only when proven, and a timeout is unknown."
    ),
    "params": {
        "c": {
            "type": "list[float]", "required": True,
            "description": "Objective coefficient vector; its length n is the number of variables.",
        },
        "A_ub": {
            "type": "list[list[float]]", "required": False, "default": None,
            "description": "Inequality matrix (each row length n) for A_ub x <= b_ub.",
        },
        "b_ub": {
            "type": "list[float]", "required": False, "default": None,
            "description": "Inequality right-hand side; length = number of rows of A_ub.",
        },
        "A_eq": {
            "type": "list[list[float]]", "required": False, "default": None,
            "description": "Equality matrix (each row length n) for A_eq x = b_eq.",
        },
        "b_eq": {
            "type": "list[float]", "required": False, "default": None,
            "description": "Equality right-hand side; length = number of rows of A_eq.",
        },
        "bounds": {
            "type": "list[list[float]]", "required": False, "default": None,
            "description": (
                "Per-variable [low, high] bounds (use null for -inf/+inf). Length n. "
                "Default (matching scipy) is [0, null] for every variable."
            ),
        },
        "integrality": {
            "type": "list[int]", "required": False, "default": None,
            "description": (
                "Length-n mask: 1 = variable must be integer, 0 = continuous. Any 1 "
                "makes this a MILP solved with OR-tools. Default: all continuous (LP)."
            ),
        },
        "sense": {
            "type": "str", "required": False, "default": "minimize",
            "description": "'minimize' or 'maximize'.",
        },
        "time_budget_sec": {
            "type": "float", "required": False, "default": 10.0,
            "description": "Wall-clock solve budget in seconds (MILP).",
        },
    },
    "output": {
        "status": "str — optimal | feasible | infeasible | unbounded | unknown",
        "optimal_value": "float|None — value of c·x at the returned solution",
        "x": "list[float]|None — the solution vector",
        "proven": "bool — True for optimal/infeasible/unbounded (solver closed the question)",
        "verified": "bool|None — x was independently re-checked against all constraints",
        "is_milp": "bool — whether an integer variable forced the MILP path",
        "solver": "str — backend used (scipy_highs / ortools_SCIP / ortools_CBC)",
        "num_variables": "int",
        "num_constraints": "int",
        "elapsed_sec": "float",
        "note": "str",
    },
    "example": {
        "params": {
            "c": [1, 1],
            "A_ub": [[1, 1]],
            "b_ub": [4],
            "bounds": [[0, None], [0, None]],
            "sense": "maximize",
        },
        "output": {"status": "optimal", "optimal_value": 4.0},
    },
}


def _err(message: str, type: str = "validation_error") -> ToolResult:
    return ToolResult(success=False, error=ToolError(type=type, message=message))


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    return None


def _matrix(raw: Any, n: int, name: str) -> tuple[list[list[float]] | None, str | None]:
    """Validate an m x n matrix of finite numbers."""
    if not isinstance(raw, (list, tuple)):
        return None, f"'{name}' must be a list of rows."
    rows: list[list[float]] = []
    for r, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) != n:
            return None, f"'{name}' row {r} must have length {n} (number of variables)."
        conv: list[float] = []
        for val in row:
            f = _num(val)
            if f is None:
                return None, f"'{name}' row {r} has a non-finite/non-numeric entry {val!r}."
            conv.append(f)
        rows.append(conv)
    return rows, None


def _vector(raw: Any, length: int, name: str) -> tuple[list[float] | None, str | None]:
    if not isinstance(raw, (list, tuple)) or len(raw) != length:
        return None, f"'{name}' must be a list of length {length}."
    out: list[float] = []
    for val in raw:
        f = _num(val)
        if f is None:
            return None, f"'{name}' has a non-finite/non-numeric entry {val!r}."
        out.append(f)
    return out, None


def linear_optimization(
    c: list | None = None,
    A_ub: list | None = None,
    b_ub: list | None = None,
    A_eq: list | None = None,
    b_eq: list | None = None,
    bounds: list | None = None,
    integrality: list | None = None,
    sense: str = "minimize",
    time_budget_sec: float = 10.0,
) -> ToolResult:
    # ---- Input validation -----------------------------------------------------
    if not isinstance(c, (list, tuple)) or not c:
        return _err("Parameter 'c' must be a non-empty objective coefficient vector.")
    c_vec = []
    for val in c:
        f = _num(val)
        if f is None:
            return _err(f"'c' has a non-finite/non-numeric entry {val!r}.")
        c_vec.append(f)
    n = len(c_vec)
    if n > _MAX_VARS:
        return _err(f"Too many variables ({n} > {_MAX_VARS}).")

    sense_l = str(sense).strip().lower()
    if sense_l in ("min", "minimize"):
        sense_l = "minimize"
    elif sense_l in ("max", "maximize"):
        sense_l = "maximize"
    else:
        return _err("'sense' must be 'minimize' or 'maximize'.")

    # Inequality block.
    A_ub_m: list[list[float]] = []
    b_ub_v: list[float] = []
    if A_ub is not None or b_ub is not None:
        if A_ub is None or b_ub is None:
            return _err("Provide both 'A_ub' and 'b_ub', or neither.")
        A_ub_m, msg = _matrix(A_ub, n, "A_ub")
        if A_ub_m is None:
            return _err(msg)  # type: ignore[arg-type]
        b_ub_v, msg = _vector(b_ub, len(A_ub_m), "b_ub")
        if b_ub_v is None:
            return _err(msg)  # type: ignore[arg-type]

    # Equality block.
    A_eq_m: list[list[float]] = []
    b_eq_v: list[float] = []
    if A_eq is not None or b_eq is not None:
        if A_eq is None or b_eq is None:
            return _err("Provide both 'A_eq' and 'b_eq', or neither.")
        A_eq_m, msg = _matrix(A_eq, n, "A_eq")
        if A_eq_m is None:
            return _err(msg)  # type: ignore[arg-type]
        b_eq_v, msg = _vector(b_eq, len(A_eq_m), "b_eq")
        if b_eq_v is None:
            return _err(msg)  # type: ignore[arg-type]

    if len(A_ub_m) + len(A_eq_m) > _MAX_CONSTRAINTS:
        return _err(f"Too many constraints ({len(A_ub_m) + len(A_eq_m)} > {_MAX_CONSTRAINTS}).")

    # Bounds (default (0, None) per variable, matching scipy).
    bnds: list[tuple[float | None, float | None]] = [(0.0, None)] * n
    if bounds is not None:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != n:
            return _err(f"'bounds' must be a list of {n} [low, high] pairs.")
        parsed_bnds: list[tuple[float | None, float | None]] = []
        for i, pair in enumerate(bounds):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return _err(f"bounds[{i}] must be a [low, high] pair (null allowed).")
            lo = None if pair[0] is None else _num(pair[0])
            hi = None if pair[1] is None else _num(pair[1])
            if pair[0] is not None and lo is None:
                return _err(f"bounds[{i}] low is non-numeric {pair[0]!r}.")
            if pair[1] is not None and hi is None:
                return _err(f"bounds[{i}] high is non-numeric {pair[1]!r}.")
            if lo is not None and hi is not None and lo > hi:
                return _err(f"bounds[{i}] low {lo} > high {hi}.")
            parsed_bnds.append((lo, hi))
        bnds = parsed_bnds

    # Integrality mask.
    int_mask = [0] * n
    if integrality is not None:
        if not isinstance(integrality, (list, tuple)) or len(integrality) != n:
            return _err(f"'integrality' must be a length-{n} mask of 0/1.")
        for i, m in enumerate(integrality):
            mi = _num(m)
            if mi is None or mi not in (0.0, 1.0):
                return _err(f"integrality[{i}] must be 0 or 1, got {m!r}.")
            int_mask[i] = int(mi)
    is_milp = any(int_mask)

    try:
        budget = max(1e-3, float(time_budget_sec))
    except (TypeError, ValueError):
        budget = 10.0

    problem = {
        "c": c_vec, "A_ub": A_ub_m, "b_ub": b_ub_v, "A_eq": A_eq_m, "b_eq": b_eq_v,
        "bounds": bnds, "int_mask": int_mask, "sense": sense_l,
    }

    try:
        if is_milp:
            result = _solve_milp(problem, budget)
        else:
            result = _solve_lp(problem)
    except AssertionError as exc:
        # Honesty backstop: a returned x failed the independent feasibility re-check.
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))

    return ToolResult(success=True, output=result)


# ---------------------------------------------------------------------------
# LP backend: scipy.optimize.linprog (HiGHS).
# ---------------------------------------------------------------------------
def _solve_lp(p: dict) -> dict:
    from scipy.optimize import linprog

    c = p["c"]
    # scipy minimizes; negate the objective for a maximize request.
    c_solver = c if p["sense"] == "minimize" else [-ci for ci in c]
    A_ub = p["A_ub"] or None
    b_ub = p["b_ub"] or None
    A_eq = p["A_eq"] or None
    b_eq = p["b_eq"] or None
    start = time.time()
    res = linprog(
        c_solver, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=p["bounds"], method="highs",
    )
    elapsed = time.time() - start

    # scipy status: 0 optimal, 1 iteration limit, 2 infeasible, 3 unbounded, 4 numerical.
    status_map = {0: "optimal", 2: "infeasible", 3: "unbounded", 1: "unknown", 4: "unknown"}
    status = status_map.get(res.status, "unknown")

    x = None
    optimal_value = None
    verified: bool | None = None
    if status in ("optimal", "feasible") and res.x is not None:
        x = [float(v) for v in res.x]
        _verify_feasible(p, x)  # raises on violation (honesty backstop)
        optimal_value = _dot(c, x)
        verified = True

    return _package(status, optimal_value, x, verified, is_milp=False,
                    solver="scipy_highs", p=p, elapsed=elapsed,
                    proven=status in ("optimal", "infeasible", "unbounded"))


# ---------------------------------------------------------------------------
# MILP backend: OR-tools pywraplp (SCIP, fallback CBC).
# ---------------------------------------------------------------------------
def _solve_milp(p: dict, budget: float) -> dict:
    from ortools.linear_solver import pywraplp

    backend = "SCIP"
    solver = pywraplp.Solver.CreateSolver(backend)
    if solver is None:
        backend = "CBC"
        solver = pywraplp.Solver.CreateSolver(backend)
    if solver is None:
        raise RuntimeError("No OR-tools MILP backend available (SCIP/CBC).")

    inf = solver.infinity()
    n = len(p["c"])
    xs = []
    for i in range(n):
        lo, hi = p["bounds"][i]
        lo = -inf if lo is None else lo
        hi = inf if hi is None else hi
        if p["int_mask"][i]:
            xs.append(solver.IntVar(lo, hi, f"x{i}"))
        else:
            xs.append(solver.NumVar(lo, hi, f"x{i}"))

    for row, rhs in zip(p["A_ub"], p["b_ub"]):
        solver.Add(sum(row[i] * xs[i] for i in range(n)) <= rhs)
    for row, rhs in zip(p["A_eq"], p["b_eq"]):
        solver.Add(sum(row[i] * xs[i] for i in range(n)) == rhs)

    obj = sum(p["c"][i] * xs[i] for i in range(n))
    if p["sense"] == "minimize":
        solver.Minimize(obj)
    else:
        solver.Maximize(obj)

    solver.SetTimeLimit(int(budget * 1000))
    start = time.time()
    rc = solver.Solve()
    elapsed = time.time() - start

    status_map = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
        pywraplp.Solver.ABNORMAL: "unknown",
        pywraplp.Solver.NOT_SOLVED: "unknown",
    }
    status = status_map.get(rc, "unknown")

    x = None
    optimal_value = None
    verified: bool | None = None
    if status in ("optimal", "feasible"):
        x = [float(v.solution_value()) for v in xs]
        # Snap near-integers so the re-check's integrality test is not tripped by a
        # solver returning 2.9999999999; the tolerance guards genuine violations.
        _verify_feasible(p, x)  # raises on violation (honesty backstop)
        optimal_value = _dot(p["c"], x)
        verified = True

    return _package(status, optimal_value, x, verified, is_milp=True,
                    solver=f"ortools_{backend}", p=p, elapsed=elapsed,
                    proven=status in ("optimal", "infeasible", "unbounded"))


# ---------------------------------------------------------------------------
# Independent feasibility re-verification (honesty backstop).
# ---------------------------------------------------------------------------
def _verify_feasible(p: dict, x: list[float]) -> None:
    """Re-check ``x`` against every declared constraint from scratch.

    Raises ``AssertionError`` on any violation beyond tolerance — a solver that
    reports an infeasible point as optimal is a bug that must never surface as a
    valid answer. Purely independent: it re-evaluates the caller's own A/b/bounds/
    integrality, not the solver's internal model.
    """
    violations: list[str] = []
    n = len(p["c"])
    # Bounds.
    for i in range(n):
        lo, hi = p["bounds"][i]
        if lo is not None and x[i] < lo - _TOL:
            violations.append(f"x[{i}]={x[i]:.9g} < low {lo}")
        if hi is not None and x[i] > hi + _TOL:
            violations.append(f"x[{i}]={x[i]:.9g} > high {hi}")
    # Inequalities A_ub x <= b_ub.
    for r, (row, rhs) in enumerate(zip(p["A_ub"], p["b_ub"])):
        lhs = sum(row[i] * x[i] for i in range(n))
        if lhs > rhs + _TOL + _TOL * abs(rhs):
            violations.append(f"A_ub row {r}: {lhs:.9g} > {rhs}")
    # Equalities A_eq x == b_eq.
    for r, (row, rhs) in enumerate(zip(p["A_eq"], p["b_eq"])):
        lhs = sum(row[i] * x[i] for i in range(n))
        if abs(lhs - rhs) > _TOL + _TOL * abs(rhs):
            violations.append(f"A_eq row {r}: {lhs:.9g} != {rhs}")
    # Integrality.
    for i in range(n):
        if p["int_mask"][i] and abs(x[i] - round(x[i])) > _INT_TOL:
            violations.append(f"x[{i}]={x[i]:.9g} not integer")
    if violations:
        raise AssertionError(
            "Returned solution violates the declared constraints "
            f"(solver/encoding bug): {violations[:5]}"
        )


def _dot(c: list[float], x: list[float]) -> float:
    return float(sum(ci * xi for ci, xi in zip(c, x)))


def _package(status, optimal_value, x, verified, *, is_milp, solver, p, elapsed, proven) -> dict:
    if status == "optimal":
        note = f"Proven optimal objective = {optimal_value:.9g} ({solver})."
    elif status == "feasible":
        note = (
            f"Feasible solution found (objective = {optimal_value:.9g}); optimality NOT "
            f"proven within the budget ({solver})."
        )
    elif status == "infeasible":
        note = f"Proven infeasible: no x satisfies the declared constraints ({solver})."
    elif status == "unbounded":
        note = f"Objective is unbounded on the feasible region ({solver})."
    else:
        note = f"Solver could not decide within the budget ({solver}); honest 'unknown'."
    return {
        "status": status,
        "optimal_value": optimal_value,
        "x": x,
        "proven": proven,
        "verified": verified,
        "is_milp": is_milp,
        "solver": solver,
        "num_variables": len(p["c"]),
        "num_constraints": len(p["A_ub"]) + len(p["A_eq"]),
        "elapsed_sec": round(elapsed, 4),
        "note": note,
    }
