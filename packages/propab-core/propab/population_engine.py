"""Phase F — population view over flat-generation hypothesis waves."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def population_snapshot(nodes: dict[str, Any]) -> dict[str, Any]:
    """Treat nodes as a population grouped by generation, not tree depth."""
    by_gen: dict[int, list[dict[str, Any]]] = defaultdict(list)
    species: Counter[str] = Counter()

    for nid, n in nodes.items():
        if isinstance(n, dict):
            g = int(n.get("generation") or 0)
            theme = n.get("primary_theme") or n.get("theme_id") or "general"
            verdict = n.get("verdict") or "pending"
            role = n.get("node_role") or "DISCOVERY"
        else:
            g = int(getattr(n, "generation", 0) or 0)
            theme = getattr(n, "primary_theme", None) or getattr(n, "theme_id", None) or "general"
            verdict = getattr(n, "verdict", "pending")
            role = getattr(n, "node_role", "DISCOVERY")
        by_gen[g].append({"id": nid, "verdict": verdict, "theme": theme, "role": role})
        species[theme] += 1

    waves = {
        str(g): {
            "count": len(items),
            "verdicts": dict(Counter(i["verdict"] for i in items)),
            "themes": dict(Counter(i["theme"] for i in items)),
        }
        for g, items in sorted(by_gen.items())
    }
    return {
        "waves": waves,
        "species_histogram": dict(species),
        "n_generations": len(by_gen),
    }
