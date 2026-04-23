from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any

from propab.tools.types import ToolResult


@dataclass(slots=True)
class ToolEntry:
    spec: dict[str, Any]
    fn: Any
    domain: str


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
            self._registry[spec["name"]] = ToolEntry(spec=spec, fn=fn, domain=spec["domain"])

    def get_all_specs(self) -> list[dict[str, Any]]:
        return [entry.spec for entry in self._registry.values()]

    def get_cluster(self, domain: str) -> list[dict[str, Any]]:
        return [entry.spec for entry in self._registry.values() if entry.domain == domain]

    def call(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        entry = self._registry[tool_name]
        return entry.fn(**params)
