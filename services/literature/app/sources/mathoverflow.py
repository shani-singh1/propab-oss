"""
MathOverflow / Stack Exchange source — expert discussion of open problems.

The site to query is domain-driven, not hardcoded: math domains use
``mathoverflow``, biology uses ``biology``, etc. The domain profile does not
mandate a site explicitly (agent3.md leaves the site implicit per domain), so
we infer it from ``source_priorities``/``classification_codes`` with
``mathoverflow`` as the sane default for any domain that lists this source at
all — a math-flavored microservice default, overridable by a profile that
sets ``stackexchange_site`` explicitly.
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from services.literature.app.models import FullTextDocument, RawDocument, TabMatch
from services.literature.app.sources.base import BaseSource, RateLimiter

_BASE = "https://api.stackexchange.com/2.3"

_KNOWN_ANSWER_RE = re.compile(r"(this (?:is|was) (?:solved|resolved|proven|answered) (?:in|by)|solved in|proved in)", re.I)
_OPEN_ANSWER_RE = re.compile(r"(this is (?:still |currently )?open|no(?:t| known)? (?:proof|counterexample) is known|remains open|unknown whether)", re.I)


class MathOverflowSource(BaseSource):
    name = "mathoverflow"
    supported_domains: tuple[str, ...] = ()

    def __init__(self, *, min_interval_sec: float = 1.0, http_timeout: float = 30.0,
                 user_agent: str = "propab-literature/0.1") -> None:
        super().__init__(http_timeout=http_timeout, user_agent=user_agent)
        self._rate_limiter = RateLimiter(min_interval_sec)

    def _site(self, profile: dict[str, Any]) -> str:
        return profile.get("stackexchange_site") or (
            "biology" if "pubmed" in (profile.get("source_priorities") or []) and
            "arxiv" not in (profile.get("source_priorities") or []) else "mathoverflow"
        )

    async def search(self, query: str, profile: dict[str, Any]) -> list[RawDocument]:
        site = self._site(profile)
        terms = [query] + list(profile.get("search_terms", []) or [])
        client = await self._get_client()
        out: list[RawDocument] = []
        seen: set[int] = set()
        for term in terms[:6]:  # bound fan-out; search terms can be long lists
            await self._rate_limiter.wait()
            resp = await client.get(
                f"{_BASE}/search/advanced",
                params={
                    "order": "desc",
                    "sort": "relevance",
                    "q": term,
                    "site": site,
                    "filter": "withbody",
                    "pagesize": 30,
                },
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            for item in data.get("items", []):
                qid = item.get("question_id")
                if qid in seen:
                    continue
                seen.add(qid)
                out.append(
                    RawDocument(
                        source="mathoverflow",
                        external_id=str(qid),
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        year=_year_from_epoch(item.get("creation_date")),
                        abstract=_strip_html(item.get("body", ""))[:500],
                        extra={
                            "is_answered": item.get("is_answered", False),
                            "accepted_answer_id": item.get("accepted_answer_id"),
                            "site": site,
                        },
                    )
                )
        return out

    async def fetch_full_text(self, doc: RawDocument) -> FullTextDocument:
        site = doc.extra.get("site", "mathoverflow")
        client = await self._get_client()
        await self._rate_limiter.wait()
        resp = await client.get(
            f"{_BASE}/questions/{doc.external_id}/answers",
            params={"order": "desc", "sort": "votes", "site": site, "filter": "withbody"},
        )
        answers = resp.json().get("items", []) if resp.status_code == 200 else []
        answer_texts = []
        classification = "unclassified"
        for a in answers:
            body = _strip_html(a.get("body", ""))
            tag = " [ACCEPTED]" if a.get("is_accepted") else ""
            answer_texts.append(f"Answer{tag} (score {a.get('score', 0)}):\n{body}")
            if a.get("is_accepted"):
                if _KNOWN_ANSWER_RE.search(body):
                    classification = "settled_with_reference"
                elif _OPEN_ANSWER_RE.search(body):
                    classification = "confirmed_open"
        # classification ("settled_with_reference" | "confirmed_open" | "unclassified")
        # is folded into body_text so the claims extractor can pick it up as a
        # high-confidence novelty signal without a dedicated model field.
        body_text = f"[novelty_signal: {classification}]\n\n" + doc.abstract + "\n\n" + "\n\n".join(answer_texts)
        return FullTextDocument(
            source="mathoverflow",
            external_id=doc.external_id,
            title=doc.title,
            url=doc.url,
            year=doc.year,
            body_text=body_text,
            extraction_method="stackexchange_api",
            extraction_quality=1.0 if answers else 0.3,
        )

    async def check_tabulated(self, values: dict[str, Any]) -> list[TabMatch]:
        return []

    async def health(self) -> bool:
        try:
            client = await self._get_client()
            resp = await client.get(f"{_BASE}/info", params={"site": "mathoverflow"}, timeout=10.0)
            return resp.status_code in (200, 429)
        except httpx.HTTPError:
            return False


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").replace("&quot;", '"').replace("&amp;", "&").strip()


def _year_from_epoch(epoch: Any) -> int:
    if not epoch:
        return 0
    try:
        import datetime

        return datetime.datetime.utcfromtimestamp(int(epoch)).year
    except Exception:
        return 0
