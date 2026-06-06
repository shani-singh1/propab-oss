"""Phase F — lineage metrics for population-style search (not strict tree depth)."""
from __future__ import annotations

from typing import Any


def lineage_stats(nodes: dict[str, Any]) -> dict[str, Any]:
    lengths: list[int] = []
    generations: list[int] = []
    for n in nodes.values():
        if isinstance(n, dict):
            lengths.append(int(n.get("lineage_length") or n.get("depth", 0) or 1))
            generations.append(int(n.get("generation") or 0))
        else:
            lengths.append(int(getattr(n, "lineage_length", None) or getattr(n, "depth", 0) or 1))
            generations.append(int(getattr(n, "generation", 0) or 0))
    return {
        "max_lineage": max(lengths) if lengths else 0,
        "avg_lineage": round(sum(lengths) / len(lengths), 3) if lengths else 0.0,
        "max_generation": max(generations) if generations else 0,
        "population_size": len(nodes),
    }
