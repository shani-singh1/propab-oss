"""
Per-domain sandbox CPU timeouts (ARCHITECTURE.md §13.2, table after line 1133).

``SANDBOX_TIMEOUT_SEC`` is the default for domains not listed in the table.
Per-domain defaults apply for listed domains. Override any domain with
``SANDBOX_TIMEOUT_<DOMAIN>`` in uppercase with underscores, e.g.
``SANDBOX_TIMEOUT_DEEP_LEARNING=600``.
"""

from __future__ import annotations

import os
import re


_DOMAIN_DEFAULT_SEC: dict[str, int] = {
    "general_computation": 30,
    "mathematics": 60,
    "statistics": 60,
    "data_analysis": 60,
    "ml_research": 120,
    "algorithm_optimization": 180,
    "deep_learning": 300,
}


def _normalize_domain(domain: str) -> str:
    d = (domain or "general_computation").strip().lower().replace("-", "_")
    return re.sub(r"[^a-z0-9_]", "", d) or "general_computation"


def effective_sandbox_timeout_sec(domain: str, global_default: int) -> int:
    """
    Effective CPU timeout for sandboxed code for this routing domain.

    ``global_default`` comes from ``Settings.sandbox_timeout_sec`` and applies
    to domains not listed in the architecture table (e.g. ``chemistry``).
    """
    g = max(5, min(int(global_default), 7200))
    key = _normalize_domain(domain)
    base = _DOMAIN_DEFAULT_SEC.get(key, g)
    env_name = f"SANDBOX_TIMEOUT_{key.upper()}"
    raw = os.environ.get(env_name)
    if raw is not None and raw.strip() != "":
        try:
            return max(5, min(int(raw.strip()), 7200))
        except ValueError:
            pass
    return max(5, min(base, 7200))
