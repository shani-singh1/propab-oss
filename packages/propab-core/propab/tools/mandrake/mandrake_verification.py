"""MandrakeVerificationTool — real LOFO/LOGO on mandrake-data (fixes.md P4)."""
from __future__ import annotations

from pathlib import Path

from propab.domain_adapters.mandrake_adapter import MandrakeAdapter, MandrakeExperimentSpec
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "mandrake_verification",
    "domain": "mandrake",
    "significance_capable": True,
    "description": "Run LOFO regression on Mandrake tabular data; returns mean_r2, bootstrap_ci, family_breakdown.",
    "params": {
        "feature_subset": {"type": "list[str]", "required": True},
        "methodology": {"type": "str", "required": False, "default": "LOFO"},
        "target_column": {"type": "str", "required": False, "default": "pe_efficiency_pct"},
        "family_column": {"type": "str", "required": False, "default": "rt_family"},
        "metric": {"type": "str", "required": False, "default": "lofo_r2"},
        "baseline_model": {"type": "str", "required": False, "default": "ridge"},
        "compare_features": {"type": "list[str]", "required": False},
        "mechanism_id": {"type": "str", "required": False},
        "artifacts_dir": {"type": "str", "required": False},
    },
    "output": {
        "mean_r2": "float", "lofo_r2": "float", "lofo_gap": "float",
        "bootstrap_ci": "list[float]", "permutation_p": "float",
        "family_breakdown": "dict", "feature_subset": "list[str]",
        "p_value": "float", "metric_value": "float", "confidence_interval": "list[float]",
    },
    "example": {"params": {"feature_subset": ["t70_raw", "t75_raw"]}, "output": {"mean_r2": -0.12}},
}


def mandrake_verification(
    feature_subset: list[str],
    methodology: str = "LOFO",
    target_column: str = "pe_efficiency_pct",
    family_column: str = "rt_family",
    metric: str = "lofo_r2",
    baseline_model: str = "ridge",
    compare_features: list[str] | None = None,
    mechanism_id: str | None = None,
    artifacts_dir: str | None = None,
) -> ToolResult:
    if not feature_subset:
        return ToolResult(success=False, output={}, error=ToolError(type="invalid_input", message="feature_subset required"))
    spec = MandrakeExperimentSpec(
        feature_subset=list(feature_subset), methodology=methodology,
        target_column=target_column, family_column=family_column, metric=metric,
        baseline_model=baseline_model, compare_features=list(compare_features or []),
        mechanism_id=mechanism_id,
    )
    try:
        adapter = MandrakeAdapter()
        result = adapter.run_experiment(spec)
        if artifacts_dir:
            adapter.write_artifacts(Path(artifacts_dir), spec, result)
        return ToolResult(success=True, output=result)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, output={}, error=ToolError(type="experiment_failed", message=str(exc)))
