"""
Self-contained, dependency-free sandbox executor for model-written construct(n).

This module deliberately imports NOTHING from ``propab`` (only stdlib). It can
therefore be launched as a bare script -- ``python sandbox_exec.py`` -- which starts
a fresh interpreter in ~100ms instead of re-importing the whole discovery package
(ortools et al.). That keeps the wall-clock budget in ``_run_in_sandbox`` a measure
of the *construction's* runtime, not process/import warmup.

Protocol (when run as ``__main__``): read ``{"code": str, "n": int}`` as JSON on
stdin; write ``{"status": "ok", "points": [[...], ...]}`` or
``{"status": "err", "error": "..."}`` as JSON on stdout.

The safety layers (static AST screen + restricted builtins) live here so the child
enforces them itself; the parent (``construction_synthesis``) adds process isolation
and the hard timeout. See ``construction_synthesis`` for the full safety model and
its documented limits.
"""
from __future__ import annotations

import ast
import json
import sys
from typing import Any

# Hard cap on emitted points (bounds memory + serialization cost).
_MAX_POINTS = 200_000


def _build_safe_builtins() -> dict[str, Any]:
    import builtins as _b

    allow = [
        "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod", "enumerate",
        "filter", "float", "frozenset", "hex", "int", "len", "list", "map", "max",
        "min", "ord", "pow", "print", "range", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "zip",
    ]
    safe = {name: getattr(_b, name) for name in allow if hasattr(_b, name)}
    safe.update({"True": True, "False": False, "None": None})
    return safe


_SAFE_BUILTINS = _build_safe_builtins()

_FORBIDDEN_NAMES = frozenset({
    "__import__", "eval", "exec", "compile", "open", "input", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "hasattr", "memoryview", "breakpoint",
    "help", "exit", "quit", "object", "type", "super", "classmethod", "staticmethod",
})


def _screen_source(code: str) -> None:
    """Static AST screen. Raises ``ValueError`` with a structural reason if unsafe.

    Blocks imports, dunder attribute access (the classic ``().__class__.__bases__``
    escape), dunder names, and forbidden builtins -- all before execution.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("import statements are not allowed in the sandbox")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"dunder attribute access '{node.attr}' is not allowed")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id.endswith("__"):
                raise ValueError(f"dunder name '{node.id}' is not allowed")
            if node.id in _FORBIDDEN_NAMES:
                raise ValueError(f"use of '{node.id}' is not allowed in the sandbox")


def _coerce_points(obj: Any) -> list[tuple[int, ...]]:
    """Coerce a construct() return value into a list of int tuples (capped)."""
    if obj is None:
        raise ValueError("construct(n) returned None")
    out: list[tuple[int, ...]] = []
    for v in obj:  # non-iterable -> TypeError, reported as a failure by the caller
        out.append(tuple(int(x) for x in v))
        if len(out) > _MAX_POINTS:
            raise ValueError(f"construction exceeded {_MAX_POINTS} points")
    return out


def _execute_construct(code: str, n: int) -> list[tuple[int, ...]]:
    """Screen, exec, and run ``construct(n)`` under restricted globals."""
    _screen_source(code)
    import math as _math
    import itertools as _itertools
    import random as _random

    sandbox_globals: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "math": _math,
        "itertools": _itertools,
        "random": _random,
    }
    exec(compile(code, "<construction>", "exec"), sandbox_globals)  # noqa: S102 - sandboxed
    fn = sandbox_globals.get("construct")
    if not callable(fn):
        raise ValueError("code did not define a callable 'construct'")
    return _coerce_points(fn(n))


def main() -> None:
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
        code = str(data["code"])
        n = int(data["n"])
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write(json.dumps({"status": "err", "error": f"bad request: {exc}"}))
        return
    try:
        pts = _execute_construct(code, n)
        sys.stdout.write(json.dumps({"status": "ok", "points": [list(p) for p in pts]}))
    except BaseException as exc:  # noqa: BLE001 - report ANY failure structurally
        sys.stdout.write(json.dumps({"status": "err", "error": f"{type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
