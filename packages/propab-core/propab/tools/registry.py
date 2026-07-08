from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from dataclasses import dataclass
from typing import Any

from propab.tools.types import ToolResult

logger = logging.getLogger(__name__)

# Audience scoping: a tool/skill may be usable by the orchestrator only, workers
# only, or both. Absent/unknown values default to "both" so nothing is ever hidden
# by accident (additive, non-breaking). Domain-general — no domain knowledge here.
AUDIENCES = ("orchestrator", "worker", "both")
_DEFAULT_AUDIENCE = "both"


def normalize_audience(value: Any) -> str:
    """Coerce a spec's ``audience`` value to a valid audience, defaulting to "both".

    An absent, empty, or unrecognised value degrades to "both" (never hidden).
    """
    if value is None:
        return _DEFAULT_AUDIENCE
    v = str(value).strip().lower()
    return v if v in AUDIENCES else _DEFAULT_AUDIENCE


def audience_matches(entry_audience: str, requested: str) -> bool:
    """True when a tool with ``entry_audience`` should be exposed to ``requested``.

    A tool is visible to a requested audience when it is scoped to that audience
    OR to "both".
    """
    return entry_audience == requested or entry_audience == _DEFAULT_AUDIENCE

# Common LLM parameter name synonyms: (canonical_name → [aliases])
_PARAM_SYNONYMS: dict[str, list[str]] = {
    "n_steps": ["epochs", "num_steps", "num_epochs", "steps", "n_epochs", "training_steps"],
    "learning_rate": ["lr", "alpha", "step_size"],
    "batch_size": ["bs", "mini_batch_size", "batch"],
    "hidden_dims": ["hidden_layers", "layers", "layer_sizes", "hidden_sizes", "hidden_units"],
    "output_dim": ["num_classes", "n_classes", "output_size", "num_outputs", "n_outputs"],
    "input_dim": ["input_size", "n_features", "num_features", "n_inputs"],
    "results_a": ["group_a", "condition_a", "treatment", "sample_a", "data_a"],
    "results_b": ["group_b", "condition_b", "control", "sample_b", "data_b"],
    "values": ["data", "samples", "measurements", "observations", "metrics"],
    "n_repeats": ["n_runs", "repeats", "num_repeats", "num_runs"],
    "weight_decay": ["l2_reg", "l2", "wd", "regularization"],
    "our_results": ["results", "our_values", "experiment_results"],
    "baseline_results": ["baseline_values", "baseline", "reference_results"],
    "baseline_value": ["literature_mean", "baseline_mean", "population_mean", "reference_value"],
    "ci": ["confidence", "confidence_level", "alpha_level", "conf_level", "confidence_percent"],
    "n_bootstrap": ["n_boot", "bootstrap_samples", "num_bootstrap"],
    "experiment_code": ["code", "experiment", "script"],
    "n_runs": ["runs", "num_runs", "repeats"],
}


@dataclass(slots=True)
class ToolEntry:
    spec: dict[str, Any]
    fn: Any
    domain: str
    audience: str = _DEFAULT_AUDIENCE


class ToolRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, ToolEntry] = {}
        self._scan()

    def _scan(self) -> None:
        package_name = "propab.tools"
        package = importlib.import_module(package_name)
        for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            if module_info.ispkg:
                continue
            if module_info.name.endswith(".registry") or module_info.name.endswith(".types"):
                continue
            module = importlib.import_module(module_info.name)
            spec = getattr(module, "TOOL_SPEC", None)
            if not isinstance(spec, dict):
                continue
            fn = getattr(module, spec.get("name", ""), None)
            if fn is None:
                continue
            self._registry[spec["name"]] = ToolEntry(
                spec=spec,
                fn=fn,
                domain=spec["domain"],
                audience=normalize_audience(spec.get("audience")),
            )

    def get_all_specs(self) -> list[dict[str, Any]]:
        return [entry.spec for entry in self._registry.values()]

    def get_cluster(self, domain: str) -> list[dict[str, Any]]:
        return [entry.spec for entry in self._registry.values() if entry.domain == domain]

    def get_for(self, audience: str) -> list[dict[str, Any]]:
        """Specs visible to ``audience`` — those scoped to it OR to "both".

        A tool without an explicit ``audience`` defaults to "both" and is therefore
        returned for every requested audience (additive / non-breaking).
        """
        requested = normalize_audience(audience)
        return [
            entry.spec
            for entry in self._registry.values()
            if audience_matches(entry.audience, requested)
        ]

    def get_cluster_for(self, domain: str, audience: str) -> list[dict[str, Any]]:
        """``get_cluster`` narrowed to tools visible to ``audience`` (scope OR "both")."""
        requested = normalize_audience(audience)
        return [
            entry.spec
            for entry in self._registry.values()
            if entry.domain == domain and audience_matches(entry.audience, requested)
        ]

    def get_significance_tools(self) -> list[dict[str, Any]]:
        """All tools that can produce p-values, effect sizes, or confidence intervals."""
        return [
            entry.spec
            for entry in self._registry.values()
            if entry.spec.get("significance_capable", False)
        ]

    def get_cluster_with_significance(self, domain: str) -> list[dict[str, Any]]:
        """Domain cluster always augmented with significance-capable tools."""
        cluster = {s["name"]: s for s in self.get_cluster(domain)}
        for sig_spec in self.get_significance_tools():
            cluster.setdefault(sig_spec["name"], sig_spec)
        return list(cluster.values())

    def call(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        entry = self._registry[tool_name]
        filtered = self._filter_params(entry.fn, params, tool_name, spec=entry.spec)
        return entry.fn(**filtered)

    def _filter_params(
        self, fn: Any, params: dict[str, Any], tool_name: str, spec: dict | None = None
    ) -> dict[str, Any]:
        """
        Filter params to only those accepted by the function, applying synonym
        remapping first. Logs unknown params at DEBUG level instead of crashing.
        """
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            return params

        # Accept **kwargs functions as-is
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return params

        accepted = set(sig.parameters.keys())
        result: dict[str, Any] = {}
        remaining = dict(params)

        # First pass: direct matches
        for k, v in list(remaining.items()):
            if k in accepted:
                result[k] = v
                del remaining[k]

        # Second pass: synonym remapping (canonical → aliases)
        for canonical, aliases in _PARAM_SYNONYMS.items():
            if canonical in accepted and canonical not in result:
                for alias in aliases:
                    if alias in remaining:
                        result[canonical] = remaining.pop(alias)
                        logger.debug(
                            "Tool %s: remapped param '%s' -> '%s'",
                            tool_name, alias, canonical,
                        )
                        break

        # Reverse synonym pass: if LLM used canonical name for an aliased tool
        for k, v in list(remaining.items()):
            for canonical, aliases in _PARAM_SYNONYMS.items():
                if k in aliases and canonical in accepted and canonical not in result:
                    result[canonical] = v
                    remaining.pop(k, None)
                    logger.debug(
                        "Tool %s: remapped alias '%s' -> '%s'",
                        tool_name, k, canonical,
                    )
                    break

        # Log unknown params that were stripped
        if remaining:
            logger.debug(
                "Tool %s: stripped unknown params: %s",
                tool_name,
                list(remaining.keys()),
            )

        # Fill missing required params from spec examples only for non–significance tools.
        # Filling results_a/results_b from TOOL_SPEC leaks identical p-values across experiments.
        if spec and not spec.get("significance_capable"):
            example_params = (spec.get("example") or {}).get("params") or {}
            for name, p in sig.parameters.items():
                if (
                    name not in result
                    and p.default is inspect.Parameter.empty
                    and p.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                    and name in example_params
                ):
                    result[name] = example_params[name]
                    logger.debug(
                        "Tool %s: filled missing required param %r from spec example",
                        tool_name,
                        name,
                    )

        return result
