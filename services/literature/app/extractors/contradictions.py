"""
Contradiction detector — two documents claiming different things about the
same quantity.

Unlike the other extractors, this one is inherently cross-document: it needs
the whole indexed claim set, not one ``FullTextDocument`` at a time. It still
subclasses ``BaseExtractor`` for interface consistency, but the real entry
point is ``find_contradictions``, called once per retrieval after all claims
from all sources have been extracted and embedded.
"""
from __future__ import annotations

from itertools import combinations

from services.literature.app.extractors._bounds import intervals_disjoint, parse_bounds
from services.literature.app.extractors.base import BaseExtractor
from services.literature.app.indexer.embeddings import cosine_similarity
from services.literature.app.models import Contradiction, ExtractedClaim, FullTextDocument

_CLUSTER_SIMILARITY = 0.80


class ContradictionsExtractor(BaseExtractor):
    name = "contradictions"

    async def extract(self, doc: FullTextDocument) -> list[Contradiction]:
        # Single-document contradiction detection is a degenerate case of the
        # real cross-document analysis; not meaningful here.
        return []

    async def find_contradictions(self, claims: list[ExtractedClaim]) -> list[Contradiction]:
        contradictions: list[Contradiction] = []
        has_embeddings = all(c.embedding for c in claims) and len(claims) > 1
        clusters = self._cluster_by_similarity(claims) if has_embeddings else [claims]

        for cluster in clusters:
            for a, b in combinations(cluster, 2):
                if a.source_doi and a.source_doi == b.source_doi and a.location == b.location:
                    continue  # same claim indexed twice, not a contradiction
                bounds_a = parse_bounds(a.verbatim)
                bounds_b = parse_bounds(b.verbatim)
                found = False
                for subj_a, lo_a, hi_a in bounds_a:
                    for subj_b, lo_b, hi_b in bounds_b:
                        if subj_a != subj_b:
                            continue
                        if intervals_disjoint((lo_a, hi_a), (lo_b, hi_b)):
                            ctype = "direct"
                            if a.source_year and b.source_year and a.source_year != b.source_year:
                                ctype = "superseded"
                            contradictions.append(
                                Contradiction(
                                    claim_a=a, claim_b=b,
                                    contradiction_type=ctype,
                                    requires_investigation=True,
                                )
                            )
                            found = True
                            break
                    if found:
                        break
        return contradictions

    def _cluster_by_similarity(self, claims: list[ExtractedClaim]) -> list[list[ExtractedClaim]]:
        clusters: list[list[ExtractedClaim]] = []
        assigned = [False] * len(claims)
        for i, claim in enumerate(claims):
            if assigned[i]:
                continue
            cluster = [claim]
            assigned[i] = True
            for j in range(i + 1, len(claims)):
                if assigned[j]:
                    continue
                if cosine_similarity(claim.embedding, claims[j].embedding) >= _CLUSTER_SIMILARITY:
                    cluster.append(claims[j])
                    assigned[j] = True
            clusters.append(cluster)
        return clusters
