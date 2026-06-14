"""Neighbor attribution for predictions (fixes.md P3)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from propab.layer05.state_embedding_index import StateIndexEntry


@dataclass
class NeighborAttribution:
    campaign_id: str
    similarity: float
    weight: float
    growth_rate_early: float
    plateau_position: float
    entropy_start: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_neighbor_attribution(
    neighbors: list[tuple[StateIndexEntry, float]],
) -> list[NeighborAttribution]:
    if not neighbors:
        return []
    weights = [1.0 / (d + 0.01) for _, d in neighbors]
    wsum = sum(weights) or 1.0
    out: list[NeighborAttribution] = []
    for (entry, dist), w in zip(neighbors, weights):
        feats = entry.features_v2 or entry.features
        growth = feats[10] if len(feats) > 10 else 0.0
        plateau = feats[11] if len(feats) > 11 else 0.5
        ent_start = float(entry.snapshots[0].get("theme_entropy") or 0) if entry.snapshots else 0.0
        out.append(NeighborAttribution(
            campaign_id=entry.campaign_id,
            similarity=round(1.0 / (1.0 + dist), 4),
            weight=round(w / wsum, 4),
            growth_rate_early=round(growth, 4),
            plateau_position=round(plateau, 4),
            entropy_start=round(ent_start, 4),
        ))
    return out


def attribution_summary(attributions: list[NeighborAttribution]) -> dict[str, Any]:
    if not attributions:
        return {"sparse": True, "n_neighbors": 0, "regime_mismatch": False}
    weights = [a.weight for a in attributions]
    growths = [a.growth_rate_early for a in attributions]
    spread = max(growths) - min(growths) if growths else 0.0
    return {
        "sparse": len(attributions) < 2,
        "n_neighbors": len(attributions),
        "top_neighbor": attributions[0].campaign_id,
        "top_weight": attributions[0].weight,
        "growth_spread": round(spread, 4),
        "regime_mismatch": spread > 0.25,
    }
