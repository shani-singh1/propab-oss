"""
Phase C — cluster confirmed claims into compact theories.
"""
from __future__ import annotations

from collections import defaultdict

from propab.knowledge_graph import Claim, KnowledgeGraph, Theory, new_id

# Substrings that mark a domain (or theme) as genuinely about network
# diffusion / contagion. Only then is contagion/diffusion framing accurate;
# for any other domain it is cross-domain contamination (LL5).
_NETWORK_DIFFUSION_MARKERS = (
    "network_diffusion",
    "contagion",
    "diffusion",
    "epidemic",
    "graph",
)


def _is_network_diffusion(domain: str | None, theme: str) -> bool:
    """True only for genuine network-diffusion campaigns/themes.

    We check the campaign's ``domain`` (plugin id or policy bucket, e.g.
    ``network_diffusion`` / ``graphs``) and fall back to the ``theme`` text so
    a diffusion theme still gets network framing even when the domain hint is
    missing. Anything else (materials, number theory, ...) stays domain-neutral.
    """
    hay = f"{domain or ''} {theme}".lower()
    return any(marker in hay for marker in _NETWORK_DIFFUSION_MARKERS)


def form_theories_from_claims(
    claims: list[Claim],
    *,
    min_support: int = 2,
    domain: str | None = None,
) -> list[Theory]:
    """Group confirmed claims by primary theme into theory objects.

    ``domain`` is the campaign's domain hint (plugin ``domain_id`` or policy
    bucket). Theory naming/assumptions are derived from it: contagion/diffusion
    wording is used ONLY for genuine network-diffusion campaigns; every other
    domain gets a domain-neutral ``{theme}_theory`` with generic assumptions,
    so a math/materials campaign no longer inherits nonsense network framing.
    """
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
        if _is_network_diffusion(domain, theme):
            name = f"{theme}_contagion_theory"
            assumptions = [f"Network theme: {theme}", "Competing diffusion models apply"]
        else:
            name = f"{theme}_theory"
            assumptions = [
                f"Theme: {theme}",
                "Supported by confirmed claims from prior campaigns",
                "Competing explanations remain possible",
            ]
        theories.append(Theory(
            id=new_id("theory"),
            name=name,
            assumptions=assumptions,
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
