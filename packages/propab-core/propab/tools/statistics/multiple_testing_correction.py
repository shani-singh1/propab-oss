"""S1 — multiple-testing correction (BH-FDR / Bonferroni / Holm).

The missing guard behind most false discoveries. Raising many hypotheses inflates
the family-wise / false-discovery rate, so a raw ``p < alpha`` *per test* is NOT a
finding. This tool applies a correct correction over a **vector** of p-values and
returns the adjusted p-values (q-values), the reject decision, the number rejected,
and the data threshold (the largest raw p-value still rejected).

Honesty by construction:
  * **Benjamini-Hochberg** is implemented exactly in numpy — sort ascending,
    ``q_i = p_i * m / rank``, then enforce monotonicity from the largest rank down,
    capped at 1. This is the guard behind most false biology findings, so it is
    cross-checked in the tests against ``scipy.stats.false_discovery_control`` AND a
    hand-worked textbook example.
  * ``statsmodels`` is NOT a dependency here — BH/Holm/Bonferroni are pure numpy on
    purpose (a sort + a monotone comparison), so nothing silently imports it.
  * Refuses to silently pass. An empty vector, an all-NaN vector, or ANY value that
    is not a finite number in ``[0, 1]`` (NaN / inf / out-of-range) -> a
    ``validation_error``, never a fabricated q-value.
  * Ties get identical q-values (verified in the tests).
"""
from __future__ import annotations

import numpy as np

from propab.tools.types import ToolError, ToolResult

# Canonical method names + common aliases a worker agent might emit.
_METHOD_ALIASES = {
    "benjamini_hochberg": "benjamini_hochberg",
    "benjamini-hochberg": "benjamini_hochberg",
    "benjamini_hochberg_fdr": "benjamini_hochberg",
    "bh": "benjamini_hochberg",
    "bh_fdr": "benjamini_hochberg",
    "fdr": "benjamini_hochberg",
    "fdr_bh": "benjamini_hochberg",
    "bonferroni": "bonferroni",
    "bonf": "bonferroni",
    "holm": "holm",
    "holm_bonferroni": "holm",
    "holm-bonferroni": "holm",
}

TOOL_SPEC = {
    "name": "multiple_testing_correction",
    "domain": "statistics",
    "audience": "worker",
    # Emits significance decisions from p-values: surfaced to every significance
    # workflow AND never auto-filled from the spec example (that would inject a
    # placeholder p-vector and manufacture fake q-values).
    "significance_capable": True,
    "description": (
        "Correct a VECTOR of p-values for multiple testing and return adjusted "
        "p-values (q-values), the boolean reject vector, the number rejected, and the "
        "p-value threshold. method in {benjamini_hochberg (default, FDR), bonferroni, "
        "holm}. Use this whenever more than one hypothesis is tested — a raw p<alpha "
        "per test is not a finding. BH is implemented exactly (numpy, no statsmodels). "
        "Rejects empty / all-NaN / out-of-[0,1] inputs instead of silently passing."
    ),
    "params": {
        "p_values": {
            "type": "list[float]",
            "required": True,
            "description": "Vector of raw p-values, each in [0, 1]. NaN/inf/out-of-range -> validation_error.",
        },
        "method": {
            "type": "str",
            "required": False,
            "default": "benjamini_hochberg",
            "description": "benjamini_hochberg (default) | bonferroni | holm.",
        },
        "alpha": {
            "type": "float",
            "required": False,
            "default": 0.05,
            "description": "Target level in (0, 1). BH controls FDR at alpha; Bonferroni/Holm control FWER.",
        },
    },
    "output": {
        "method": "str — normalized correction method used",
        "alpha": "float — level applied",
        "n_tests": "int — number of hypotheses (m)",
        "adjusted_p_values": "list[float] — q-values, same order as input p_values",
        "reject": "list[bool] — True where the hypothesis is rejected (adjusted p <= alpha)",
        "n_rejected": "int — number of hypotheses rejected",
        "p_value_threshold": "float — largest RAW p-value that is rejected (0.0 if none)",
    },
    "example": {
        "params": {
            "p_values": [0.001, 0.008, 0.039, 0.041, 0.042, 0.06],
            "method": "benjamini_hochberg",
            "alpha": 0.05,
        },
        "output": {
            "adjusted_p_values": [0.006, 0.024, 0.0504, 0.0504, 0.0504, 0.06],
            "n_rejected": 2,
            "p_value_threshold": 0.008,
        },
    },
}


def _bh_qvalues(p_sorted: np.ndarray, m: int) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values for an ascending-sorted p-vector.

    q_i = p_i * m / rank_i, then enforce monotone non-decreasing from the LARGEST
    rank down (running minimum backwards), capped at 1. Ties inherit equal q's.
    """
    ranks = np.arange(1, m + 1, dtype=float)
    q = p_sorted * m / ranks
    # Monotonicity: q_(i) = min(q_(i), q_(i+1), ..., q_(m)).
    q = np.minimum.accumulate(q[::-1])[::-1]
    return np.minimum(q, 1.0)


def _holm_qvalues(p_sorted: np.ndarray, m: int) -> np.ndarray:
    """Holm (step-down) adjusted p-values for an ascending-sorted p-vector.

    q_(i) = min(1, max_{j<=i} (m - j + 1) * p_(j)); enforce monotone non-decreasing
    from the SMALLEST rank up (running maximum forward), capped at 1.
    """
    factors = np.arange(m, 0, -1, dtype=float)  # m, m-1, ..., 1
    q = p_sorted * factors
    q = np.maximum.accumulate(q)
    return np.minimum(q, 1.0)


def multiple_testing_correction(
    p_values: list | None = None,
    method: str = "benjamini_hochberg",
    alpha: float = 0.05,
) -> ToolResult:
    if p_values is None:
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message="Parameter 'p_values' is required."),
        )

    # Normalize the method up front so a bad method fails clearly.
    method_key = str(method).strip().lower()
    method_norm = _METHOD_ALIASES.get(method_key)
    if method_norm is None:
        return ToolResult(
            success=False,
            error=ToolError(
                type="validation_error",
                message=(
                    f"Unknown method {method!r}. Use one of: benjamini_hochberg (default), "
                    "bonferroni, holm."
                ),
            ),
        )

    try:
        a = float(alpha)
    except (TypeError, ValueError):
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message=f"alpha must be a number in (0, 1); got {alpha!r}."),
        )
    if not (0.0 < a < 1.0) or not np.isfinite(a):
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message=f"alpha must be in (0, 1); got {a}."),
        )

    try:
        p = np.asarray(p_values, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message=f"p_values must be a numeric vector: {exc}"),
        )

    m = int(p.shape[0])
    if m == 0:
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message="p_values is empty — nothing to correct."),
        )
    if not np.all(np.isfinite(p)):
        # Covers all-NaN AND any-NaN/inf: never silently drop or pass NaN entries.
        return ToolResult(
            success=False,
            error=ToolError(
                type="validation_error",
                message="p_values contains NaN or inf; every p-value must be a finite number in [0, 1].",
            ),
        )
    if np.any(p < 0.0) or np.any(p > 1.0):
        lo, hi = float(np.min(p)), float(np.max(p))
        return ToolResult(
            success=False,
            error=ToolError(
                type="validation_error",
                message=f"p_values out of range: every value must be in [0, 1]; observed [{lo}, {hi}].",
            ),
        )

    try:
        # Stable sort so tie ordering (and thus the inverse mapping) is deterministic.
        order = np.argsort(p, kind="mergesort")
        p_sorted = p[order]

        if method_norm == "bonferroni":
            q_sorted = np.minimum(p_sorted * m, 1.0)
        elif method_norm == "holm":
            q_sorted = _holm_qvalues(p_sorted, m)
        else:  # benjamini_hochberg
            q_sorted = _bh_qvalues(p_sorted, m)

        # Map adjusted p-values back to the ORIGINAL input order.
        q = np.empty(m, dtype=float)
        q[order] = q_sorted

        # Reject decision on the unrounded q (adjusted p <= alpha).
        reject = q <= a
        n_rejected = int(np.count_nonzero(reject))
        # Threshold = the largest RAW p-value that is rejected (0.0 if none).
        threshold = float(np.max(p[reject])) if n_rejected > 0 else 0.0

        return ToolResult(
            success=True,
            output={
                "method": method_norm,
                "alpha": a,
                "n_tests": m,
                "adjusted_p_values": [round(float(v), 10) for v in q],
                "reject": [bool(v) for v in reject],
                "n_rejected": n_rejected,
                "p_value_threshold": round(threshold, 10),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
