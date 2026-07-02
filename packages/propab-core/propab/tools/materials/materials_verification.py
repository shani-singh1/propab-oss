"""MaterialsVerificationTool — crystal-system LOFO on matbench dielectric."""
from __future__ import annotations

from pathlib import Path

from propab.domain_adapters.materials_adapter import MaterialsAdapter, MaterialsExperimentSpec
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "materials_verification",
    "domain": "materials",
    "significance_capable": True,
    "description": (
        "Run cross-crystal-system LOFO on matbench dielectric data; "
        "returns lofo_r2, label_shuffle_null_p95, lofo_gap, family_leakage_confirmed."
    ),
    "params": {
        "feature_subset": {"type": "list[str]", "required": True},
        "methodology": {"type": "str", "required": False, "default": "LOFO"},
        "target_column": {"type": "str", "required": False, "default": "dielectric"},
        "family_column": {"type": "str", "required": False, "default": "crystal_system"},
        "metric": {"type": "str", "required": False, "default": "lofo_r2"},
        "baseline_model": {"type": "str", "required": False, "default": "ridge"},
        "compare_features": {"type": "list[str]", "required": False},
        "artifacts_dir": {"type": "str", "required": False},
    },
    "output": {
        "mean_r2": "float",
        "lofo_r2": "float",
        "lofo_gap": "float",
        "label_shuffle_null_p95": "float",
        "label_shuffle_permutation_p": "float",
        "family_leakage_confirmed": "bool",
        "bootstrap_ci": "list[float]",
        "permutation_p": "float",
        "family_breakdown": "dict",
        "feature_subset": "list[str]",
        "p_value": "float",
        "metric_value": "float",
        "confidence_interval": "list[float]",
    },
    "example": {
        "params": {"feature_subset": ["n_sites", "n_elements", "mean_Z"]},
        "output": {"lofo_r2": 0.12, "lofo_gap": 0.08},
    },
}


def materials_verification(
    feature_subset: list[str],
    methodology: str = "LOFO",
    target_column: str = "dielectric",
    family_column: str = "crystal_system",
    metric: str = "lofo_r2",
    baseline_model: str = "ridge",
    compare_features: list[str] | None = None,
    artifacts_dir: str | None = None,
) -> ToolResult:
    if not feature_subset:
        return ToolResult(
            success=False,
            output={},
            error=ToolError(type="invalid_input", message="feature_subset required"),
        )
    spec = MaterialsExperimentSpec(
        feature_subset=list(feature_subset),
        methodology=methodology,
        target_column=target_column,
        family_column=family_column,
        metric=metric,
        baseline_model=baseline_model,
        compare_features=list(compare_features or []),
    )
    try:
        adapter = MaterialsAdapter()
        result = adapter.run_experiment(spec)
        if artifacts_dir:
            adapter.write_artifacts(Path(artifacts_dir), spec, result)
        return ToolResult(success=True, output=result)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            success=False,
            output={},
            error=ToolError(type="experiment_failed", message=str(exc)),
        )
