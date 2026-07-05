"""
OEIS source — sequence search and tabulated-value matching.

Critical for combinatorics/number theory novelty checks: the key question the
pipeline asks this source is "does the value F(n) = X at index n already
appear in a known sequence?" This is answered two ways: (1) a cached lookup
against the domain profile's declared ``tabulation_sources`` OEIS ids
(fetched once at startup and kept warm), and (2) an on-demand OEIS full-text
search when the cached set doesn't cover it.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from services.literature.app.models import FullTextDocument, RawDocument, TabMatch
from services.literature.app.sources.base import BaseSource, RateLimiter


def _oeis_results(payload: Any) -> list[dict[str, Any]]:
    """The OEIS search API returns a bare JSON list of hits on success, or the
    literal JSON ``null`` when there are no matches — never a ``{"results": [...]}``
    wrapper. Normalize both shapes to a list."""
    if isinstance(payload, list):
        return payload
    return []


def _parse_oeis_values(data_field: str) -> list[int]:
    if not data_field:
        return []
    out = []
    for tok in data_field.split(","):
        tok = tok.strip()
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out


class OeisSource(BaseSource):
    name = "oeis"
    supported_domains: tuple[str, ...] = ("math_combinatorics", "graph_invariants", "genomics")

    def __init__(self, *, min_interval_sec: float = 1.0, http_timeout: float = 30.0,
                 user_agent: str = "propab-literature/0.1") -> None:
        super().__init__(http_timeout=http_timeout, user_agent=user_agent)
        self._rate_limiter = RateLimiter(min_interval_sec)
        self._sequence_cache: dict[str, dict[str, Any]] = {}

    async def warm_cache(self, sequence_ids: list[str]) -> None:
        """Fetch and cache the domain profile's declared OEIS sequences so
        novelty checks can do an O(1) cached-range check before any network
        call (agent3.md: 'checks the candidate value against the cached
        tabulation range first before doing a full search')."""
        for seq_id in sequence_ids:
            if seq_id in self._sequence_cache:
                continue
            seq = await self._fetch_sequence(seq_id)
            if seq is not None:
                self._sequence_cache[seq_id] = seq

    async def search(self, query: str, profile: dict[str, Any]) -> list[RawDocument]:
        await self._rate_limiter.wait()
        client = await self._get_client()
        resp = await client.get("https://oeis.org/search", params={"q": query, "fmt": "json"})
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except Exception:
            return []
        results = _oeis_results(data)
        out = []
        for r in results:
            seq_id = f"A{r.get('number', 0):06d}"
            out.append(
                RawDocument(
                    source="oeis",
                    external_id=seq_id,
                    title=r.get("name", ""),
                    url=f"https://oeis.org/{seq_id}",
                    abstract=r.get("comment", [""])[0] if r.get("comment") else "",
                    extra={"data": r.get("data", ""), "formula": r.get("formula", [])},
                )
            )
        return out

    async def search_by_values(self, values: list[int]) -> list[RawDocument]:
        """Search OEIS for sequences containing a specific run of computed
        values — the exact operation used for novelty checking."""
        query = ",".join(str(v) for v in values)
        return await self.search(query, profile={})

    async def fetch_full_text(self, doc: RawDocument) -> FullTextDocument:
        seq_id = doc.external_id
        seq = self._sequence_cache.get(seq_id) or await self._fetch_sequence(seq_id)
        if seq is None:
            return FullTextDocument(source="oeis", external_id=seq_id, title=doc.title, url=doc.url)
        comments = "\n".join(seq.get("comment", []) or [])
        links = seq.get("link", []) or []
        formulas = "\n".join(seq.get("formula", []) or [])
        body = f"{seq.get('name', '')}\n\n{comments}\n\nFormula:\n{formulas}\n\nLinks:\n" + "\n".join(links)
        return FullTextDocument(
            source="oeis",
            external_id=seq_id,
            title=seq.get("name", doc.title),
            url=f"https://oeis.org/{seq_id}",
            body_text=body,
            extraction_method="oeis_api",
            extraction_quality=1.0,
        )

    async def check_tabulated(self, values: dict[str, Any]) -> list[TabMatch]:
        """``values`` is expected to look like {"sequence_hint": [...], "index": n,
        "value": X} or {"index": n, "value": X} for a direct point lookup."""
        matches: list[TabMatch] = []
        index = values.get("index")
        value = values.get("value")
        candidate_ids = values.get("candidate_sequence_ids") or list(self._sequence_cache.keys())

        for seq_id in candidate_ids:
            seq = self._sequence_cache.get(seq_id)
            if seq is None:
                continue
            data_values = _parse_oeis_values(seq.get("data", ""))
            offset_raw = seq.get("offset", "0,1")
            try:
                offset = int(str(offset_raw).split(",")[0])
            except ValueError:
                offset = 0
            if index is not None:
                pos = int(index) - offset
                if 0 <= pos < len(data_values):
                    matched_value = data_values[pos]
                    is_match = value is None or _numeric_close(matched_value, value)
                    matches.append(
                        TabMatch(
                            source="oeis",
                            identifier=seq_id,
                            matched=is_match,
                            matched_index=index,
                            matched_value=matched_value,
                            url=f"https://oeis.org/{seq_id}",
                        )
                    )
                    continue
            if value is not None:
                try:
                    target = int(value)
                    if target in data_values:
                        pos = data_values.index(target)
                        matches.append(
                            TabMatch(
                                source="oeis",
                                identifier=seq_id,
                                matched=True,
                                matched_index=pos + offset,
                                matched_value=target,
                                url=f"https://oeis.org/{seq_id}",
                            )
                        )
                except (TypeError, ValueError):
                    continue
        # If nothing in the warm cache matched and a raw sequence of values was
        # given, fall back to a live OEIS search (Step: novelty checking).
        if not matches and values.get("sequence"):
            hits = await self.search_by_values(list(values["sequence"]))
            for h in hits:
                matches.append(
                    TabMatch(source="oeis", identifier=h.external_id, matched=True, url=h.url)
                )
        return matches

    async def health(self) -> bool:
        try:
            client = await self._get_client()
            resp = await client.get("https://oeis.org/search", params={"q": "1,2,3", "fmt": "json"}, timeout=10.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def _fetch_sequence(self, seq_id: str) -> dict[str, Any] | None:
        await self._rate_limiter.wait()
        client = await self._get_client()
        num = seq_id.lstrip("Aa")
        try:
            resp = await client.get("https://oeis.org/search", params={"q": f"id:A{num}", "fmt": "json"})
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except json.JSONDecodeError:
            return None
        results = _oeis_results(data)
        return results[0] if results else None


def _numeric_close(a: Any, b: Any, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b
