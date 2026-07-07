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


async def _scrape_open_problem_source(url: str, *, timeout: float = 15.0) -> list[OpenProblem]:
    """Best-effort generic extraction from a domain-listed open-problem page
    (e.g. an Erdős-problems-style list). Uses the same marker regex as
    ``OpenProblemsExtractor`` — if the page doesn't use those markers, this
    returns nothing rather than guessing."""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
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


async def scrape_declared_open_problem_sources(
    profile: dict[str, Any], *, seen_statements: set[str] | None = None, timeout: float = 15.0
) -> list[OpenProblem]:
    """Scrape every ``open_problem_sources[].url`` declared in the domain
    profile, de-duplicated against ``seen_statements``. This is the *cheap*
    half of gap discovery (one HTTP GET per declared URL, no per-search-term
    source fan-out) and is safe to call from the latency-sensitive ``/prior``
    path — it is what makes ``PriorResponse.open_gaps`` populate from a
    domain's curated frontier list even when fetched abstracts carry no
    ``Problem:``/conjecture markers of their own.

    Domain-general: the URLs come entirely from the profile; this function
    knows nothing about any specific domain."""
    seen = seen_statements if seen_statements is not None else set()
    out: list[OpenProblem] = []
    for entry in profile.get("open_problem_sources", []) or []:
        url = entry.get("url", "") if isinstance(entry, dict) else ""
        if not url:
            continue
        for op in await _scrape_open_problem_source(url, timeout=timeout):
            if op.statement not in seen:
                seen.add(op.statement)
                out.append(op)
    return out


async def search_open_problems(
    ctx: PipelineContext,
    profile: dict[str, Any],
    *,
    seen_statements: set[str] | None = None,
    max_terms: int = 5,
    max_docs_per_source: int = 5,
    source_names: list[str] | None = None,
) -> list[OpenProblem]:
    """Fresh search across relevant sources for "<search term> open problem"
    queries, extracting explicit open problems via ``OpenProblemsExtractor``.

    Parameterized so the same discovery logic serves two callers with very
    different latency budgets: ``/gaps`` runs it unbounded, while ``/prior``
    runs a trimmed variant (few terms, one primary source) that fits the prior
    deadline. Domain-general: terms and source selection come from the profile.

    ``source_names`` restricts which relevant sources are queried (used by the
    prior path to hit only the single fastest/primary source); ``None`` = all
    relevant sources."""
    seen = seen_statements if seen_statements is not None else set()
    search_terms = list(profile.get("search_terms", []) or [])
    extractor = ctx.open_problems_extractor
    relevant_sources = {n: s for n, s in ctx.sources.items() if s.is_relevant(profile)}
    if source_names is not None:
        relevant_sources = {n: s for n, s in relevant_sources.items() if n in source_names}

    out: list[OpenProblem] = []
    for term in search_terms[:max_terms]:
        query = f"{term} open problem"
        for _name, source in relevant_sources.items():
            try:
                docs = await source.search(query, profile)
            except Exception:
                continue
            for raw in docs[:max_docs_per_source]:
                try:
                    full_doc: FullTextDocument = await source.fetch_full_text(raw)
                except Exception:
                    continue
                for op in await extractor.extract(full_doc):
                    if op.statement not in seen:
                        seen.add(op.statement)
                        out.append(op)
    return out


async def map_gaps(ctx: PipelineContext, *, domain_id: str, profile: dict[str, Any]) -> GapsResponse:
    search_terms = list(profile.get("search_terms", []) or [])
    all_problems: list[OpenProblem] = list(await ctx.structured_store.get_open_problems(domain_id))
    seen_statements = {p.statement for p in all_problems}

    # 1. Best-effort scrape of domain-declared open-problem lists.
    all_problems.extend(
        await scrape_declared_open_problem_sources(profile, seen_statements=seen_statements)
    )

    # 2. Fresh search across relevant sources for "<search term> open problem"
    # style queries, extracting via OpenProblemsExtractor.
    all_problems.extend(
        await search_open_problems(ctx, profile, seen_statements=seen_statements)
    )

    try:
        await ctx.structured_store.save_open_problems(domain_id, all_problems)
    except Exception:
        pass

    ranked = sorted(all_problems, key=lambda op: _rank_key(op, search_terms), reverse=True)
    return GapsResponse(frontier_questions=ranked)
