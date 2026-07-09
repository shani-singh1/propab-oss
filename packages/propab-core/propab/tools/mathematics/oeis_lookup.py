"""OEIS lookup — the scalable best-known REFERENCE for combinatorial discovery.

To recognize a computed object as NOVEL you need a best-known value to beat. A hardcoded
per-sequence registry does not scale; OEIS is the maintained database of best-known values
for millions of sequences, so a lookup scales to any known object. Given the first terms
of a sequence (e.g. the maximum object size for n=1,2,3,…) or an A-number, this returns the
matching OEIS sequence(s): id, name, the KNOWN terms, offset, and keywords (``more``/``hard``
mark where the sequence runs out — i.e. where NEW terms would be novel).

Honesty: it reports exactly what OEIS returns (provenance-tagged). A network failure/timeout
degrades to ``status: 'unavailable'`` (NEVER a fabricated match); no match → ``'not_found'``.
Use the returned known terms as the reference to pass as ``published_best`` to certify_witness.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from propab.tools.types import ToolError, ToolResult

_OEIS_URL = "https://oeis.org/search"
_DEFAULT_TIMEOUT = 6.0


def _fetch_oeis(query: str, timeout: float) -> dict:
    """GET the OEIS JSON API. Isolated so tests can monkeypatch it (no real network)."""
    url = f"{_OEIS_URL}?{urllib.parse.urlencode({'q': query, 'fmt': 'json'})}"
    req = urllib.request.Request(url, headers={"User-Agent": "propab-oeis-lookup/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed host)
        return json.loads(resp.read().decode("utf-8"))


TOOL_SPEC = {
    "name": "oeis_lookup",
    "domain": "mathematics",
    "audience": "worker",
    "description": (
        "Look up a sequence in OEIS (the On-Line Encyclopedia of Integer Sequences) — the "
        "scalable best-known REFERENCE for deciding whether a computed value is novel. Pass "
        "the first `terms` of your computed sequence (e.g. the maximum object size for "
        "n=1,2,3,…) to identify it, OR an `anum` (e.g. 'A003022'). Returns the matching "
        "sequence(s): id, name, the KNOWN terms, offset, and keywords (keyword 'more'/'hard' "
        "flag where the sequence is only known so far — a term beyond that is potentially "
        "novel). Use the known terms as the reference (published_best) for certify_witness. "
        "Reports exactly what OEIS returns; a network problem degrades to 'unavailable', "
        "never a fabricated answer."
    ),
    "params": {
        "terms": {"type": "list[int]", "required": False,
                   "description": "First terms of the sequence to identify (>= 4 recommended)."},
        "anum": {"type": "str", "required": False,
                  "description": "An OEIS A-number, e.g. 'A003022'."},
        "max_results": {"type": "int", "required": False, "default": 3,
                         "description": "How many matching sequences to return."},
        "timeout_sec": {"type": "float", "required": False, "default": 6.0},
    },
    "output": {
        "status": "str — 'ok' | 'not_found' | 'unavailable'",
        "query": "str",
        "results": "list[dict] — each {anum, name, terms:list[int], offset, keyword}",
        "note": "str",
    },
    "example": {"params": {"terms": [1, 2, 2, 3, 3, 4]}, "output": {"status": "ok"}},
}


def _parse_data_terms(data: str) -> list[int]:
    out = []
    for tok in str(data or "").split(","):
        tok = tok.strip()
        if tok.lstrip("-").isdigit():
            out.append(int(tok))
    return out


def oeis_lookup(terms=None, anum=None, max_results=3, timeout_sec=6.0):
    if not terms and not anum:
        return ToolResult(success=False, error=ToolError(
            type="validation_error", message="Provide either 'terms' (list of ints) or 'anum'."))
    if anum:
        query = str(anum).strip()
    else:
        try:
            query = ",".join(str(int(t)) for t in terms)
        except (TypeError, ValueError):
            return ToolResult(success=False, error=ToolError(
                type="validation_error", message="'terms' must be a list of integers."))
    try:
        timeout = max(1.0, float(timeout_sec))
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIMEOUT

    try:
        data = _fetch_oeis(query, timeout)
    except Exception as exc:  # noqa: BLE001 — network/parse failure degrades gracefully
        return ToolResult(success=True, output={
            "status": "unavailable", "query": query, "results": [],
            "note": f"OEIS lookup unavailable ({type(exc).__name__}); proceed without a reference or retry.",
        })

    # The OEIS fmt=json API returns a JSON LIST of sequence dicts directly; older/other
    # shapes wrap them in {"results": [...]}. Handle both, defensively.
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = data.get("results") or []
    else:
        raw = []
    if not raw:
        return ToolResult(success=True, output={
            "status": "not_found", "query": query, "results": [],
            "note": "No OEIS match — the sequence may be novel, or the terms/offset differ.",
        })

    try:
        n = max(1, int(max_results))
    except (TypeError, ValueError):
        n = 3
    results = []
    for seq in raw[:n]:
        num = seq.get("number")
        results.append({
            "anum": f"A{int(num):06d}" if isinstance(num, int) else str(num),
            "name": seq.get("name"),
            "terms": _parse_data_terms(seq.get("data")),
            "offset": seq.get("offset"),
            "keyword": seq.get("keyword"),
        })
    return ToolResult(success=True, output={
        "status": "ok", "query": query, "results": results,
        "note": ("Reference from OEIS. Use a matched sequence's known terms as published_best; "
                 "keyword 'more'/'hard' marks where new terms would be novel."),
    })
