from __future__ import annotations

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "benchmark_algorithm",
    "domain": "algorithm_optimization",
    "description": (
        "Benchmark an algorithm's empirical time/space complexity from its source code. "
        "This tool REFUSES rather than fabricating: it has no safe path to execute the "
        "agent-provided code from within propab-core (the only sandbox is the "
        "Docker-backed services/worker sandbox, which core tools do not and must not "
        "import), so it cannot measure real timing or fit a real complexity curve. It "
        "returns success=False instead of emitting a fabricated empirical_complexity / r2."
    ),
    "params": {
        "code": {"type": "str", "required": True},
        "input_sizes": {"type": "list[int]", "required": True},
        "n_runs": {"type": "int", "required": False, "default": 3},
        "measure": {"type": "list[str]", "required": False, "default": ["time"]},
    },
    "output": {
        "error": "ToolError (tool refuses; cannot execute the provided code to measure complexity)",
    },
    "example": {"params": {"code": "def f(n): ...", "input_sizes": [100, 500, 1000]}, "output": {}},
}


def benchmark_algorithm(code: str, input_sizes: list, n_runs: int = 3, measure: list | None = None) -> ToolResult:
    """Refuse to benchmark rather than fabricate an empirical complexity.

    A prior version ignored the required ``code`` param entirely and always benchmarked a
    fixed dot-product kernel, then reported an ``empirical_complexity`` from the slope of
    *that* kernel's timing (with two branches that both returned ``"O(n)"``, making
    ``O(log n)`` / ``O(n log n)`` unreachable) and hardcoded ``complexity_r2 = 0.85``
    whenever only two input sizes were given. None of those numbers described the agent's
    actual algorithm, yet a hypothesis could "confirm" a complexity claim against them.

    As with ``compare_implementations`` and the wave-1 ``evaluate_model`` fix, there is no
    safe way to execute arbitrary agent-provided code from inside ``propab-core`` (the only
    sandbox is the Docker-backed ``services/worker/sandbox.py``, which core tools must not
    import, and Docker is not guaranteed available). Measuring a fixed built-in kernel and
    labelling the result as the agent's complexity is fabrication. So this tool validates
    its inputs and then fails closed.
    """
    if not isinstance(code, str) or not code.strip():
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message="code (algorithm source) required."),
        )
    if not input_sizes:
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message="input_sizes required."),
        )

    _ = (n_runs, measure)  # accepted, but intentionally not used to fabricate a measurement
    return ToolResult(
        success=False,
        error=ToolError(
            type="cannot_execute_code",
            message=(
                "benchmark_algorithm cannot execute the provided code, so it cannot measure "
                "real timing or fit an empirical complexity curve for it. Refusing to return "
                "a fabricated empirical_complexity / complexity_r2 derived from a fixed "
                "built-in kernel rather than your algorithm. To measure real complexity, run "
                "the code through the worker sandbox (services/worker) or a dedicated harness "
                "over the input_sizes and fit the timing there; a complexity claim cannot be "
                "verified by this tool."
            ),
        ),
    )
