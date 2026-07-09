"""Honesty-critical label-shuffle null for leave-one-group-out (LOFO) R².

S1 (general-agent redesign): exposes the CORRECT within-group target-permutation
LOFO null as a ``TOOL_SPEC`` tool so a general worker agent never hand-writes a
(buggy) null. It reuses the audited ``_family_label_shuffle_null`` from the mandrake
adapter verbatim — that primitive permutes the TARGET ``y`` WITHIN each group,
preserving each group's marginal outcome distribution and the LOFO partition while
destroying the X→y relationship.

Why within-group target shuffle (not split shuffle): shuffling the group/split
variable leaves X→y intact, so the null R² tracks the observed value and the test
has NO power (the genomics/enzyme/mandrake bug class). Permuting y within group is
the correct null — a real signal beats it; noise does not.

Includes the degenerate-target guard: a (near-)constant ``y`` carries no learnable
signal, so it reports no signal outright instead of a spurious result.
"""
from __future__ import annotations

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "label_shuffle_null",
    "domain": "statistics",
    "audience": "worker",
    "description": (
        "Correct leave-one-group-out (LOFO) permutation null: permutes the TARGET y "
        "WITHIN each group (never the split) to break X→y while preserving the group "
        "structure. Returns observed_lofo_r2, null_p (P[null R² >= observed]) and "
        "null_p95. Use this instead of writing your own null — a split-shuffle null "
        "has no power. Guards degenerate (constant) targets."
    ),
    "params": {
        "X": {"type": "list[list[float]]", "required": True, "description": "Feature matrix (n_samples x n_features)."},
        "y": {"type": "list[float]", "required": True, "description": "Target vector (length n_samples)."},
        "groups": {"type": "list", "required": True,
                    "description": "Group / family label per sample (defines the LOFO split)."},
        "n_perm": {"type": "int", "required": False, "default": 200, "description": "Number of within-group permutations."},
        "model": {"type": "str", "required": False, "default": "ridge",
                   "description": "Baseline model for LOFO R²: 'ridge' (default) or 'linear'."},
    },
    "output": {
        "observed_lofo_r2": "float — leave-one-group-out R² of X->y",
        "null_p": "float — P[null R² >= observed] under the within-group target shuffle",
        "null_p95": "float — 95th percentile of the null R² distribution",
        "significant": "bool — null_p < 0.05 and observed > 0",
        "degenerate_target": "bool — True when var(y) ~ 0 (no learnable signal)",
        "n_perm": "int",
        "n_groups": "int",
    },
    "example": {
        "params": {
            "X": [[0.1], [0.2], [0.9], [1.1], [2.0], [2.1]],
            "y": [0.1, 0.2, 0.9, 1.1, 2.0, 2.1],
            "groups": ["a", "a", "b", "b", "c", "c"],
            "n_perm": 100,
        },
        "output": {"observed_lofo_r2": 0.0, "null_p": 1.0, "null_p95": 0.0},
    },
}


def label_shuffle_null(
    X: list | None = None,
    y: list | None = None,
    groups: list | None = None,
    n_perm: int = 200,
    model: str = "ridge",
) -> ToolResult:
    if X is None or y is None or groups is None:
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message="Parameters 'X', 'y' and 'groups' are all required."),
        )
    try:
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float).ravel()
        ga = np.asarray(groups)
    except (TypeError, ValueError) as exc:
        return ToolResult(success=False, error=ToolError(type="validation_error", message=f"Malformed inputs: {exc}"))

    if Xa.ndim != 2:
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message=f"X must be 2D (n_samples x n_features); got shape {Xa.shape}."),
        )
    if Xa.shape[0] != ya.shape[0] or ga.shape[0] != ya.shape[0]:
        return ToolResult(
            success=False,
            error=ToolError(
                type="validation_error",
                message=f"Length mismatch: X={Xa.shape[0]}, y={ya.shape[0]}, groups={ga.shape[0]} must match.",
            ),
        )
    if ya.shape[0] < 4:
        return ToolResult(
            success=False,
            error=ToolError(type="validation_error", message="Need at least 4 samples for a LOFO null."),
        )

    try:
        n_perm_int = max(1, int(n_perm))
    except (TypeError, ValueError):
        n_perm_int = 200

    n_groups = int(len(np.unique(ga)))

    # Degenerate-target guard: a (near-)constant y carries no learnable signal.
    # Permuting a constant leaves it unchanged, so any "R²" is spurious — report
    # no signal outright (mirrors the genomics verifier's degenerate branch).
    if not np.isfinite(np.nanvar(ya)) or float(np.nanvar(ya)) < 1e-12:
        return ToolResult(
            success=True,
            output={
                "observed_lofo_r2": 0.0,
                "null_p": 1.0,
                "null_p95": 1.0,
                "significant": False,
                "degenerate_target": True,
                "n_perm": n_perm_int,
                "n_groups": n_groups,
            },
        )

    if n_groups < 2:
        return ToolResult(
            success=True,
            output={
                "observed_lofo_r2": 0.0,
                "null_p": 1.0,
                "null_p95": 0.0,
                "significant": False,
                "degenerate_target": False,
                "n_perm": n_perm_int,
                "n_groups": n_groups,
                "note": "Fewer than 2 groups — leave-one-group-out is undefined; no signal reported.",
            },
        )

    try:
        # Reuse the AUDITED within-group target-shuffle null verbatim.
        from propab.domain_adapters.mandrake_adapter import (
            _family_label_shuffle_null,
            _make_model,
        )

        mdl = _make_model(str(model or "ridge"))
        observed, null_p, null_samples = _family_label_shuffle_null(
            Xa, ya, ga, mdl, n_perm=n_perm_int
        )
        null_p95 = float(np.percentile(null_samples, 95)) if null_samples else 0.0
        significant = bool(null_p < 0.05 and observed > 0.0)
        return ToolResult(
            success=True,
            output={
                "observed_lofo_r2": round(float(observed), 6),
                "null_p": round(float(null_p), 6),
                "null_p95": round(float(null_p95), 6),
                "significant": significant,
                "degenerate_target": False,
                "n_perm": n_perm_int,
                "n_groups": n_groups,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
