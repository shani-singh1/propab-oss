"""Reciprocal rank fusion (ARCHITECTURE.md §5.3) — dependency-free helper."""

from __future__ import annotations

from collections import defaultdict


def rrf_merge(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in rankings:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
