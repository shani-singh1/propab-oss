"""
BaseSource — the contract every external source fulfills. Sources never know
about each other, and never know anything domain-specific: the ``profile``
dict passed to every method is the domain plugin's ``literature_profile()``
output, and a source only reads the keys it understands (e.g. an OEIS source
reads ``tabulation_sources``, a source that has never heard of OEIS ignores it).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from services.literature.app.models import FullTextDocument, RawDocument, TabMatch


class RateLimiter:
    """Simple per-source polite rate limiter — one request in flight at a time,
    with a minimum interval between requests, matching each API's published
    etiquette guidance rather than hammering it in parallel."""

    def __init__(self, min_interval_sec: float) -> None:
        self._min_interval = min_interval_sec
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        async with self._lock:
            elapsed = time.monotonic() - self._last_call
            remaining = self._min_interval - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_call = time.monotonic()


class BaseSource:
    name: str = "base"
    # Empty tuple/list = all domains. A source declares which domain
    # source_priorities it responds to; the pipeline filters on this.
    supported_domains: tuple[str, ...] = ()

    def __init__(self, *, http_timeout: float = 30.0, user_agent: str = "propab-literature/0.1") -> None:
        self._timeout = http_timeout
        self._user_agent = user_agent
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": self._user_agent},
                follow_redirects=True,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def is_relevant(self, profile: dict[str, Any]) -> bool:
        """Whether this source should run at all for the given domain profile."""
        priorities = [p.lower() for p in profile.get("source_priorities", []) or []]
        if not priorities:
            # No profile / default profile → keyword-search fallback sources only.
            return self.name in {"arxiv", "semantic_scholar", "crossref"}
        return self.name in priorities

    async def search(self, query: str, profile: dict[str, Any]) -> list[RawDocument]:
        raise NotImplementedError

    async def fetch_full_text(self, doc: RawDocument) -> FullTextDocument:
        raise NotImplementedError

    async def check_tabulated(self, values: dict[str, Any]) -> list[TabMatch]:
        raise NotImplementedError

    async def health(self) -> bool:
        """Cheap reachability probe used by GET /health. Default: assume healthy
        (a source without a defined probe should not fail health checks it
        cannot meaningfully answer)."""
        return True
