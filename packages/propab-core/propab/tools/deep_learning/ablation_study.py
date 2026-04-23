from __future__ import annotations

from propab.tools.types import ToolResult

TOOL_SPEC = {
    "name": "ablation_study",
    "domain": "deep_learning",
    "description": "Compare scalar scores for named ablation configs (v1 synthetic scoring).",
    "params": {
        "configs": {"type": "list[dict]", "required": True, "description": "List of {name: str, score_hint: float}"},
        "metric": {"type": "str", "required": False, "default": "score"},
    },
    "output": {"results": "list", "best": "str", "summary": "str"},
    "example": {"params": {"configs": [{"name": "full", "score_hint": 0.9}, {"name": "no_dropout", "score_hint": 0.85}]}, "output": {}},
}


def ablation_study(configs: list, metric: str = "score") -> ToolResult:
    rows = []
    for c in configs:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "cfg"))
        hint = float(c.get("score_hint", 0.5))
        rows.append({"config": name, "metric": round(hint, 4)})
    if not rows:
        return ToolResult(success=True, output={"results": [], "best": "", "summary": "No configs."})
    best = max(rows, key=lambda r: r["metric"])["config"]
    return ToolResult(
        success=True,
        output={"results": rows, "best": best, "summary": f"Best ablation by synthetic hint: {best}."},
    )
