"""
Phase C — cluster confirmed claims into compact theories.
"""
from __future__ import annotations

from collections import defaultdict

from propab.knowledge_graph import Claim, KnowledgeGraph, Theory, new_id


def form_theories_from_claims(claims: list[Claim], *, min_support: int = 2) -> list[Theory]:
    """Group confirmed claims by primary theme into theory objects."""
    by_theme: dict[str, list[Claim]] = defaultdict(list)
    for c in claims:
        if c.verdict == "confirmed":
            by_theme[c.theme].append(c)

    theories: list[Theory] = []
    for theme, group in by_theme.items():
        if len(group) < min_support:
            continue
        ids = [c.id for c in group]
        texts = [c.text[:200] for c in group[:5]]
        theories.append(Theory(
            id=new_id("theory"),
            name=f"{theme}_contagion_theory",
            assumptions=[f"Network theme: {theme}", "Competing diffusion models apply"],
            mechanism_summary="; ".join(texts[:3]),
            predictions=[c.text[:300] for c in group[:4]],
            failure_regions=[],
            supporting_claim_ids=ids,
            themes=[theme],
        ))
    return theories


def merge_theories_into_graph(graph: KnowledgeGraph, theories: list[Theory]) -> int:
    added = 0
    existing = {t.name for t in graph.theories.values()}
    for th in theories:
        if th.name in existing:
            continue
        graph.add_theory(th)
        for cid in th.supporting_claim_ids:
            graph.link(th.id, cid, "supports")
        added += 1
    return added
