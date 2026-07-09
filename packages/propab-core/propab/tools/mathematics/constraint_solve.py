"""General CP/SAT decision + optimization tool (M4).

``constraint_solve`` exposes Google OR-tools CP-SAT as a DATA-DRIVEN solver: the caller
declares a model (integer/boolean variables with domains, linear constraints,
AllDifferent groups, and an optional linear objective) as JSON-like *data* — never code
— and the tool builds the ``CpModel``, solves within a time + worker budget, and reports
an honest outcome. It generalises the audited ``decide_b3_cpsat`` idiom in
``domain_modules/math_combinatorics/discovery/cp_sat_finder.py`` to any combinatorial
feasibility / optimization problem.

Honesty invariants (mirrors ``cp_sat_finder`` / ``extremal_set_search``):
  * Any returned assignment is INDEPENDENTLY re-checked against the DECLARED model —
    every variable domain, every linear constraint, and every AllDifferent group is
    re-evaluated on the solution — before it is returned. A violation means an
    encoding/solver bug and is raised (surfaced as ``execution_error``), NEVER returned
    as a valid solution.
  * ``unsat`` is reported ONLY when CP-SAT proves INFEASIBLE. A budget timeout is
    ``unknown`` — never ``unsat``.
  * Model size is capped; oversized / malformed models are rejected as
    ``validation_error`` rather than silently truncated.

Outcomes (``output["outcome"]``):
  * ``"sat"``     — a feasible assignment was found (decision problem, or an objective
                    problem whose optimality the budget could not prove).
  * ``"optimal"`` — an objective problem solved to proven optimality.
  * ``"unsat"``   — CP-SAT PROVED the model infeasible.
  * ``"unknown"`` — the budget expired before CP-SAT decided (honest timeout).
"""
from __future__ import annotations

import time
from typing import Any

from propab.tools.types import ToolError, ToolResult

# ---------------------------------------------------------------------------
# Model-size caps (reject rather than silently truncate an oversized model).
# ---------------------------------------------------------------------------
_MAX_VARS = 5000
_MAX_CONSTRAINTS = 20000
_MAX_DOMAIN_WIDTH = 10_000_000  # |high - low| per integer variable
_MAX_ALLDIFF_GROUPS = 2000
_VALID_OPS = ("<=", ">=", "==")

TOOL_SPEC = {
    "name": "constraint_solve",
    "domain": "mathematics",
    "audience": "worker",
    # Deterministic solver: a SAT/optimal assignment is independently re-checked
    # against the declared model, and UNSAT is a real infeasibility proof — evidence
    # is a re-verified witness, not a p-value. Satisfies the worker stop-gate.
    "verification_capable": True,
    "description": (
        "General constraint-programming / SAT solver over a DECLARED model (data, not "
        "code). Declare integer/boolean variables with domains, linear constraints "
        "(coeffs, op in {<=,>=,==}, rhs), optional AllDifferent groups, and an optional "
        "linear objective (minimize/maximize); the tool builds an OR-tools CP-SAT model, "
        "solves within a time+worker budget, and returns outcome in "
        "{sat, optimal, unsat, unknown} with the variable assignment for sat/optimal. "
        "Every returned assignment is independently re-checked against the declared "
        "constraints before it is returned; unsat is reported only when CP-SAT proves "
        "infeasibility, and a timeout is unknown (never unsat). Coefficients, bounds and "
        "rhs must be integers (CP-SAT is integer-only)."
    ),
    "params": {
        "variables": {
            "type": "list[dict]", "required": True,
            "description": (
                "Variable declarations. Each: {'name': str, 'type': 'int'|'bool', "
                "'low': int, 'high': int}. For type 'bool' the domain is [0,1] and "
                "low/high may be omitted. Names must be unique."
            ),
        },
        "constraints": {
            "type": "list[dict]", "required": False, "default": [],
            "description": (
                "Linear constraints. Each: {'coeffs': {var_name: int, ...} (or list of "
                "[name, coeff] pairs), 'op': '<='|'>='|'==', 'rhs': int}."
            ),
        },
        "all_different": {
            "type": "list[list[str]]", "required": False, "default": [],
            "description": "Groups of variable names that must take pairwise-distinct values.",
        },
        "objective": {
            "type": "dict", "required": False, "default": None,
            "description": (
                "Optional linear objective: {'sense': 'minimize'|'maximize', "
                "'coeffs': {var_name: int, ...}}. Omit for a pure feasibility decision."
            ),
        },
        "time_budget_sec": {
            "type": "float", "required": False, "default": 10.0,
            "description": "Wall-clock solve budget in seconds.",
        },
        "workers": {
            "type": "int", "required": False, "default": 8,
            "description": "Number of CP-SAT search workers.",
        },
    },
    "output": {
        "outcome": "str — sat | optimal | unsat | unknown",
        "assignment": "dict[str,int]|None — variable -> value for sat/optimal",
        "objective_value": "int|None — value of the declared objective at the assignment",
        "bound": "float|None — best objective bound (optimality gap witness) when an objective is set",
        "proven": "bool — True for optimal and unsat (CP-SAT closed the question)",
        "verified": "bool|None — the assignment was independently re-checked against the declared model",
        "status": "str — raw CP-SAT status name",
        "num_variables": "int",
        "num_constraints": "int",
        "elapsed_sec": "float",
        "note": "str",
    },
    "example": {
        "params": {
            "variables": [
                {"name": "x", "type": "int", "low": 0, "high": 5},
                {"name": "y", "type": "int", "low": 0, "high": 5},
            ],
            "constraints": [{"coeffs": {"x": 1, "y": 1}, "op": "<=", "rhs": 4}],
            "objective": {"sense": "maximize", "coeffs": {"x": 1, "y": 1}},
            "time_budget_sec": 5,
        },
        "output": {"outcome": "optimal", "objective_value": 4, "verified": True},
    },
}


def _err(message: str, type: str = "validation_error") -> ToolResult:
    return ToolResult(success=False, error=ToolError(type=type, message=message))


def _as_int(value: Any) -> int | None:
    """Return ``value`` as an int iff it is integer-valued, else ``None``.

    Accepts a Python int or a float exactly equal to an integer (CP-SAT is
    integer-only; a genuinely fractional value must be rejected, not rounded).
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _normalize_coeffs(raw: Any, known: dict[str, Any], where: str) -> tuple[dict[str, int] | None, str | None]:
    """Coerce a coeffs spec (dict or list of [name, coeff] pairs) to {name: int}."""
    pairs: list[tuple[Any, Any]]
    if isinstance(raw, dict):
        pairs = list(raw.items())
    elif isinstance(raw, (list, tuple)):
        pairs = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                return None, f"{where}: each coeff pair must be [name, coeff], got {item!r}."
            pairs.append((item[0], item[1]))
    else:
        return None, f"{where}: 'coeffs' must be a dict or a list of [name, coeff] pairs."

    out: dict[str, int] = {}
    for name, coeff in pairs:
        name = str(name)
        if name not in known:
            return None, f"{where}: unknown variable {name!r} in coeffs."
        ci = _as_int(coeff)
        if ci is None:
            return None, f"{where}: coefficient for {name!r} must be an integer, got {coeff!r}."
        out[name] = out.get(name, 0) + ci
    if not out:
        return None, f"{where}: 'coeffs' is empty."
    return out, None


def constraint_solve(
    variables: list | None = None,
    constraints: list | None = None,
    all_different: list | None = None,
    objective: dict | None = None,
    time_budget_sec: float = 10.0,
    workers: int = 8,
) -> ToolResult:
    # ---- Input validation (never raise; return validation_error) --------------
    if not isinstance(variables, list) or not variables:
        return _err("Parameter 'variables' must be a non-empty list of variable declarations.")
    if len(variables) > _MAX_VARS:
        return _err(f"Too many variables ({len(variables)} > {_MAX_VARS}).")

    constraints = constraints or []
    all_different = all_different or []
    if not isinstance(constraints, list):
        return _err("'constraints' must be a list.")
    if not isinstance(all_different, list):
        return _err("'all_different' must be a list of name groups.")
    if len(constraints) > _MAX_CONSTRAINTS:
        return _err(f"Too many constraints ({len(constraints)} > {_MAX_CONSTRAINTS}).")
    if len(all_different) > _MAX_ALLDIFF_GROUPS:
        return _err(f"Too many all_different groups ({len(all_different)} > {_MAX_ALLDIFF_GROUPS}).")

    # Parse variable declarations into an independent spec (used for re-verification).
    var_domains: dict[str, tuple[int, int]] = {}
    for i, spec in enumerate(variables):
        if not isinstance(spec, dict) or "name" not in spec:
            return _err(f"variables[{i}] must be a dict with a 'name'.")
        name = str(spec["name"])
        if name in var_domains:
            return _err(f"Duplicate variable name {name!r}.")
        vtype = str(spec.get("type", "int")).strip().lower()
        if vtype in ("bool", "boolean", "binary"):
            low, high = 0, 1
        elif vtype in ("int", "integer"):
            low = _as_int(spec.get("low"))
            high = _as_int(spec.get("high"))
            if low is None or high is None:
                return _err(f"variables[{i}] ({name!r}): integer variable needs integer 'low' and 'high'.")
            if low > high:
                return _err(f"variables[{i}] ({name!r}): low {low} > high {high}.")
            if high - low > _MAX_DOMAIN_WIDTH:
                return _err(f"variables[{i}] ({name!r}): domain width {high - low} exceeds cap {_MAX_DOMAIN_WIDTH}.")
        else:
            return _err(f"variables[{i}] ({name!r}): type must be 'int' or 'bool', got {vtype!r}.")
        var_domains[name] = (low, high)

    # Parse constraints into an independent spec: (coeffs, op, rhs).
    parsed_cons: list[tuple[dict[str, int], str, int]] = []
    for i, con in enumerate(constraints):
        if not isinstance(con, dict):
            return _err(f"constraints[{i}] must be a dict.")
        op = str(con.get("op", "")).strip()
        if op not in _VALID_OPS:
            return _err(f"constraints[{i}]: 'op' must be one of {list(_VALID_OPS)}, got {op!r}.")
        rhs = _as_int(con.get("rhs"))
        if rhs is None:
            return _err(f"constraints[{i}]: 'rhs' must be an integer, got {con.get('rhs')!r}.")
        coeffs, msg = _normalize_coeffs(con.get("coeffs"), var_domains, f"constraints[{i}]")
        if coeffs is None:
            return _err(msg)  # type: ignore[arg-type]
        parsed_cons.append((coeffs, op, rhs))

    # Parse AllDifferent groups.
    parsed_alldiff: list[list[str]] = []
    for i, group in enumerate(all_different):
        if not isinstance(group, (list, tuple)) or len(group) < 2:
            return _err(f"all_different[{i}] must be a list of at least two variable names.")
        names = [str(g) for g in group]
        for nm in names:
            if nm not in var_domains:
                return _err(f"all_different[{i}]: unknown variable {nm!r}.")
        parsed_alldiff.append(names)

    # Parse objective.
    obj_sense: str | None = None
    obj_coeffs: dict[str, int] | None = None
    if objective is not None:
        if not isinstance(objective, dict):
            return _err("'objective' must be a dict {'sense':..., 'coeffs':...} or omitted.")
        sense = str(objective.get("sense", "")).strip().lower()
        if sense not in ("minimize", "maximize", "min", "max"):
            return _err("objective['sense'] must be 'minimize' or 'maximize'.")
        obj_sense = "minimize" if sense in ("minimize", "min") else "maximize"
        obj_coeffs, msg = _normalize_coeffs(objective.get("coeffs"), var_domains, "objective")
        if obj_coeffs is None:
            return _err(msg)  # type: ignore[arg-type]

    try:
        budget = max(1e-3, float(time_budget_sec))
    except (TypeError, ValueError):
        budget = 10.0
    try:
        n_workers = max(1, int(workers))
    except (TypeError, ValueError):
        n_workers = 8

    # ---- Build + solve the CP-SAT model ---------------------------------------
    try:
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        var_objs: dict[str, Any] = {}
        for name, (low, high) in var_domains.items():
            var_objs[name] = model.NewIntVar(low, high, name)

        for coeffs, op, rhs in parsed_cons:
            expr = sum(c * var_objs[nm] for nm, c in coeffs.items())
            if op == "<=":
                model.Add(expr <= rhs)
            elif op == ">=":
                model.Add(expr >= rhs)
            else:  # "=="
                model.Add(expr == rhs)

        for names in parsed_alldiff:
            model.AddAllDifferent([var_objs[nm] for nm in names])

        if obj_coeffs is not None:
            oexpr = sum(c * var_objs[nm] for nm, c in obj_coeffs.items())
            if obj_sense == "minimize":
                model.Minimize(oexpr)
            else:
                model.Maximize(oexpr)

        # Catch a malformed model before solving (report as validation_error).
        validation_msg = model.Validate()
        if validation_msg:
            return _err(f"CP-SAT rejected the model: {validation_msg}")

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = budget
        solver.parameters.num_search_workers = n_workers
        solver.parameters.random_seed = 0  # determinism
        start = time.time()
        status = solver.Solve(model)
        elapsed = time.time() - start
        status_name = solver.StatusName(status)

        has_solution = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assignment: dict[str, int] | None = None
        objective_value: int | None = None
        bound: float | None = None
        verified: bool | None = None

        if has_solution:
            assignment = {name: int(solver.Value(v)) for name, v in var_objs.items()}
            # ---- INDEPENDENT re-verification against the DECLARED model --------
            # Re-evaluate every declared constraint on the solution from scratch. A
            # violation is an encoding/solver bug — raise it, never return a bogus
            # "solution" as valid (the core honesty invariant).
            violations = _recheck(assignment, var_domains, parsed_cons, parsed_alldiff)
            if violations:
                raise AssertionError(
                    "CP-SAT returned an assignment that violates the declared model "
                    f"(encoding/solver bug): {violations[:5]}"
                )
            verified = True
            if obj_coeffs is not None:
                objective_value = sum(c * assignment[nm] for nm, c in obj_coeffs.items())
                bound = float(solver.BestObjectiveBound())
                # Cross-check the objective we recomputed matches CP-SAT's own value.
                solver_obj = solver.ObjectiveValue()
                if abs(solver_obj - objective_value) > 1e-6:
                    raise AssertionError(
                        f"Objective mismatch: recomputed {objective_value} != CP-SAT {solver_obj}."
                    )

        # ---- Map status to an honest outcome ------------------------------------
        if status == cp_model.OPTIMAL:
            outcome = "optimal" if obj_coeffs is not None else "sat"
            proven = True
        elif status == cp_model.FEASIBLE:
            # A solution exists but optimality was not proven within the budget.
            outcome = "sat"
            proven = False
        elif status == cp_model.INFEASIBLE:
            outcome = "unsat"  # PROVEN infeasible — the only path to unsat.
            proven = True
        else:  # UNKNOWN / MODEL_INVALID -> honest non-answer, never unsat.
            outcome = "unknown"
            proven = False

        note = _make_note(outcome, obj_coeffs is not None, objective_value, status_name)

        return ToolResult(
            success=True,
            output={
                "outcome": outcome,
                "assignment": assignment,
                "objective_value": objective_value,
                "bound": bound,
                "proven": proven,
                "verified": verified,
                "status": status_name,
                "num_variables": len(var_domains),
                "num_constraints": len(parsed_cons),
                "elapsed_sec": round(elapsed, 4),
                "note": note,
            },
        )
    except AssertionError as exc:
        # Honesty backstop tripped: a returned solution failed independent re-check.
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))


def _recheck(
    assignment: dict[str, int],
    var_domains: dict[str, tuple[int, int]],
    constraints: list[tuple[dict[str, int], str, int]],
    all_different: list[list[str]],
) -> list[str]:
    """Independently re-evaluate the DECLARED model on ``assignment``.

    Returns a list of human-readable violation strings (empty iff the assignment
    genuinely satisfies every declared domain, linear constraint and AllDifferent
    group). This is deliberately a from-scratch re-derivation that does NOT consult
    the CP-SAT model — it is the check that makes a returned solution trustworthy.
    """
    violations: list[str] = []
    # Domains.
    for name, (low, high) in var_domains.items():
        val = assignment.get(name)
        if val is None or not (low <= val <= high):
            violations.append(f"{name}={val} outside domain [{low},{high}]")
    # Linear constraints.
    for coeffs, op, rhs in constraints:
        lhs = sum(c * assignment.get(nm, 0) for nm, c in coeffs.items())
        ok = (lhs <= rhs) if op == "<=" else (lhs >= rhs) if op == ">=" else (lhs == rhs)
        if not ok:
            violations.append(f"constraint {coeffs} {op} {rhs} violated (lhs={lhs})")
    # AllDifferent.
    for names in all_different:
        vals = [assignment.get(nm) for nm in names]
        if len(set(vals)) != len(vals):
            violations.append(f"all_different{names} has repeated values {vals}")
    return violations


def _make_note(outcome: str, has_obj: bool, obj_value: int | None, status_name: str) -> str:
    if outcome == "optimal":
        return f"Proven optimal objective = {obj_value} (CP-SAT status {status_name})."
    if outcome == "sat":
        if has_obj:
            return (
                f"Feasible assignment found (objective = {obj_value}); optimality NOT "
                f"proven within the budget (status {status_name})."
            )
        return f"Feasible assignment found and independently re-verified (status {status_name})."
    if outcome == "unsat":
        return "CP-SAT PROVED the declared model infeasible (no assignment exists)."
    return (
        f"Budget expired before CP-SAT decided (status {status_name}); honest 'unknown' "
        "— not reported as unsat."
    )
