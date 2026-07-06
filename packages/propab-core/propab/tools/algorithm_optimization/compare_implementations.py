from __future__ import annotations

from typing import Any

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "compare_implementations",
    "domain": "algorithm_optimization",
    "description": (
        "Compare competing implementations for speed / memory / correctness. This tool "
        "REFUSES rather than fabricating: it has no safe path to execute the agent-provided "
        "code from within propab-core (the only sandbox is the Docker-backed "
        "services/worker sandbox, which core tools do not and must not import), so it "
        "cannot measure real timing/memory or verify correctness. It returns success=False "
        "instead of emitting fabricated fastest/most_memory_efficient/correctness results."
    ),
    "params": {
        "implementations": {"type": "list[dict]", "required": True},
        "test_inputs": {"type": "list", "required": True},
        "check_outputs": {"type": "bool", "required": False, "default": True},
        "n_runs": {"type": "int", "required": False, "default": 10},
    },
    "output": {
        "error": "ToolError (tool refuses; cannot execute arbitrary code to verify the claim)",
    },
    "example": {
        "params": {
            "implementations": [
                {"name": "a", "code": "def fn(x): return x"},
                {"name": "b", "code": "def fn(x): return x"},
            ],
            "test_inputs": [1, 2, 3],
            "n_runs": 5,
        },
        "output": {},
    },
}


def compare_implementations(
    implementations: list[dict[str, Any]],
    test_inputs: list[Any],
    check_outputs: bool = True,
    n_runs: int = 10,
) -> ToolResult:
    """Refuse to compare implementations rather than fabricate results.

    A prior version of this tool ignored the ``implementations`` code and ``test_inputs``
    entirely: it timed a fixed matmul seeded by each impl's NAME, derived
    ``peak_memory_mb`` from ``hash(name)``, and hardcoded ``all_correct: True`` /
    ``failing_inputs: []`` for every implementation. That let a hypothesis "confirm" a
    false performance or correctness claim against numbers that never touched the agent's
    real code.

    There is no safe way to execute arbitrary agent-provided code from inside
    ``propab-core``: the only execution sandbox is the Docker-backed
    ``services/worker/sandbox.py``, which the core tools deliberately do not import (a
    ``services`` dependency from ``propab-core`` would be a layering inversion, and Docker
    is not guaranteed available in the tool-call context). The wave-1 fix to
    ``deep_learning/evaluate_model.py`` set the precedent: when honest measurement is
    impossible, fail closed rather than fabricate.

    So this tool measures nothing and asserts nothing. It validates its inputs and then
    returns ``success=False`` with a message that the claim cannot be verified here.
    """
    # Validate inputs so callers still get actionable errors for malformed requests,
    # but never emit a fabricated fastest / correctness / performance result.
    if not implementations:
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message="implementations required."),
        )
    for impl in implementations:
        if not isinstance(impl, dict) or "name" not in impl:
            return ToolResult(
                success=False,
                error=ToolError(
                    type="validation_error",
                    message="Each implementation must be a dict with 'name'.",
                ),
            )

    _ = (test_inputs, check_outputs, n_runs)  # accepted, but intentionally not "measured"
    return ToolResult(
        success=False,
        error=ToolError(
            type="cannot_execute_code",
            message=(
                "compare_implementations cannot execute the provided implementations or "
                "test_inputs: there is no safe code-execution sandbox available to core "
                "tools, so real timing, peak memory, and output correctness cannot be "
                "measured. Refusing to return fabricated fastest / most_memory_efficient / "
                "correctness results. To benchmark real implementations, run them through "
                "the worker sandbox (services/worker) or a dedicated harness and report the "
                "measured numbers; a speed or correctness claim cannot be verified by this "
                "tool."
            ),
        ),
    )
