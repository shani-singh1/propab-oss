from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ToolError:
    type: str
    message: str
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolResult:
    success: bool
    output: dict[str, Any] | None = None
    error: ToolError | None = None
