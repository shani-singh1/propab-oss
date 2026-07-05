"""
Gap mapper's raw material — the space between what is proven and what is
conjectured or open. This is what Propab should be targeting.

Also cross-document like ``contradictions.py``: a gap is identified by
pairing a proven bound on a quantity with a conjectured/open stronger bound
on the *same* quantity — the interval between them is the frontier. Where no
matching conjecture exists, an ``OpenProblem`` with no adjacent proven bound
becomes a gap with an empty ``best_known_bound`` (still worth surfacing —
gap_mapper.py ranks these lower since they are less "tight").
"""
from __future__ import annotations

from services.literature.app.extractors._bounds import parse_bounds
from services.literature.app.extractors.base import BaseExtractor
from services.literature.app.models import ExtractedClaim, FullTextDocument, KnowledgeGap, OpenProblem


class GapsExtractor(BaseExtractor):
    name = "gaps"

    async def extract(self, doc: FullTextDocument) -> list[KnowledgeGap]:
        return []

    async def find_gaps(
        self, claims: list[ExtractedClaim], open_problems: list[OpenProblem]
    ) -> list[KnowledgeGap]:
        gaps: list[KnowledgeGap] = []
        proven = [c for c in claims if c.status == "proven"]
        conjectured_or_open = [c for c in claims if c.status in ("conjectured", "open")]

        proven_by_subject: dict[str, list[ExtractedClaim]] = {}
        for c in proven:
            for subj, lo, hi in parse_bounds(c.verbatim):
                proven_by_subject.setdefault(subj, []).append(c)

        matched_subjects: set[str] = set()
        for c in conjectured_or_open:
            for subj, lo, hi in parse_bounds(c.verbatim):
                candidates = proven_by_subject.get(subj)
                if not candidates:
                    continue
                matched_subjects.add(subj)
                best = candidates[0]
                gaps.append(
                    KnowledgeGap(
                        description=f"Gap between proven and {c.status} bound on '{subj}'",
                        what_is_known=best.verbatim,
                        what_is_open=c.verbatim,
                        best_known_bound=best.verbatim,
                        last_progress=max(best.source_year, c.source_year),
                        computationally_approachable=_looks_approachable(c.text),
                        approachable_angle=(
                            "Numerically tighten the bound toward the conjectured value"
                            if _looks_approachable(c.text) else ""
                        ),
                    )
                )

        # Open problems with no matched proven bound are still gaps — just
        # untethered ones (gap_mapper.py ranks by tightness, so these sink
        # to the bottom unless computationally approachable).
        for op in open_problems:
            gaps.append(
                KnowledgeGap(
                    description=op.statement,
                    what_is_open=op.statement,
                    last_progress=op.year,
                    computationally_approachable=op.computationally_approachable,
                    approachable_angle=op.approachable_angle,
                )
            )

        return gaps


def _looks_approachable(text: str) -> bool:
    lower = text.lower()
    return any(
        w in lower
        for w in ("comput", "numerical", "algorithm", "enumerat", "search", "bound", "estimate")
    )
