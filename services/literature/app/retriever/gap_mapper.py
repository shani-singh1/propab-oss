"""
Gap mapper — given a domain profile, produce a structured map of the open
frontier: what the literature says is unresolved and worth attacking, ranked
so Propab frames campaigns around real gaps instead of "what sounds
interesting."

Ranking (agent3.md):
1. computationally approachable (true > false)
2. how long open (longer = more pressing)
3. alignment with the domain's own search terms / research focus
4. tightness of the gap (not always knowable for a bare OpenProblem — treated
   as a tiebreaker only when a matching KnowledgeGap narrows it down)
"""
from __future__ import annotations

import re
import time
from typing import Any

import httpx

from services.literature.app.context import PipelineContext
from services.literature.app.extractors.open_problems import PROBLEM_MARKER_RE, OpenProblemsExtractor
from services.literature.app.models import FullTextDocument, GapsResponse, OpenProblem

_CURRENT_YEAR = time.gmtime().tm_year


def _alignment_score(statement: str, search_terms: list[str]) -> float:
    lower = statement.lower()
    hits = sum(1 for t in search_terms if t.lower() in lower)
    return min(1.0, hits / max(1, len(search_terms))) if search_terms else 0.0


def _rank_key(op: OpenProblem, search_terms: list[str]) -> tuple:
    age = (_CURRENT_YEAR - op.year) if op.year else 0
    alignment = _alignment_score(op.statement, search_terms)
    return (
        1 if op.computationally_approachable else 0,
        age,
        alignment,
    )


async def _scrape_open_problem_source(url: str) -> list[OpenProblem]:
    """Best-effort generic extraction from a domain-listed open-problem page
    (e.g. an Erdős-problems-style list). Uses the same marker regex as
    ``OpenProblemsExtractor`` — if the page doesn't use those markers, this
    returns nothing rather than guessing."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            return []
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text)
    except Exception:
        return []
    out = []
    for m in PROBLEM_MARKER_RE.finditer(text):
        statement = re.sub(r"\s+", " ", m.group(2)).strip()[:400]
        if len(statement) > 15:
            out.append(OpenProblem(statement=statement, context=f"scraped from {url}"))
    return out


async def map_gaps(ctx: PipelineContext, *, domain_id: str, profile: dict[str, Any]) -> GapsResponse:
    search_terms = list(profile.get("search_terms", []) or [])
    all_problems: list[OpenProblem] = list(await ctx.structured_store.get_open_problems(domain_id))
    seen_statements = {p.statement for p in all_problems}

    # 1. Best-effort scrape of domain-declared open-problem lists.
    for entry in profile.get("open_problem_sources", []) or []:
        url = entry.get("url", "")
        if not url:
            continue
        for op in await _scrape_open_problem_source(url):
            if op.statement not in seen_statements:
                seen_statements.add(op.statement)
                all_problems.append(op)

    # 2. Fresh search across relevant sources for "<search term> open problem"
    # style queries, extracting via OpenProblemsExtractor.
    extractor = ctx.open_problems_extractor
    relevant_sources = {n: s for n, s in ctx.sources.items() if s.is_relevant(profile)}
    for term in search_terms[:5]:
        query = f"{term} open problem"
        for name, source in relevant_sources.items():
            try:
                docs = await source.search(query, profile)
            except Exception:
                continue
            for raw in docs[:5]:
                try:
                    full_doc: FullTextDocument = await source.fetch_full_text(raw)
                except Exception:
                    continue
                for op in await extractor.extract(full_doc):
                    if op.statement not in seen_statements:
                        seen_statements.add(op.statement)
                        all_problems.append(op)

    try:
        await ctx.structured_store.save_open_problems(domain_id, all_problems)
    except Exception:
        pass

    ranked = sorted(all_problems, key=lambda op: _rank_key(op, search_terms), reverse=True)
    return GapsResponse(frontier_questions=ranked)
