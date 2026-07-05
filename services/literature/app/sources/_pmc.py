"""
Shared PubMed Central full-text fetch/parse — used by both ``pubmed.py``
(which discovers a PMCID via NCBI's id-converter from a bare PMID) and
``europepmc.py`` (whose search results already carry the PMCID directly, so
it skips the id-conversion round-trip entirely and calls straight in here).

Not every PMC-indexed article allows this: many journals participate in PMC
without granting bulk-download/text-mining rights, and ``efetch`` returns
those with the article's <front> metadata but an explicit comment ("does not
allow downloading of the full text") instead of a <body>. Detected and
treated as "no full text available" — never guessed at.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx

_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_NO_FULLTEXT_MARKER = "does not allow downloading"


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_table(table_wrap: ET.Element) -> str | None:
    """Renders a JATS <table> as a compact pipe-delimited text block — good
    enough for an LLM to read the values out of, without trying to preserve
    full column-span/HTML layout fidelity."""
    table = table_wrap.find(".//table")
    if table is None:
        return None
    rows = []
    for tr in table.iter("tr"):
        cells = [_clean_whitespace("".join(cell.itertext())) for cell in list(tr)]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    if not rows:
        return None
    caption_el = table_wrap.find(".//caption")
    caption = _clean_whitespace("".join(caption_el.itertext())) if caption_el is not None else ""
    header = f"Table ({caption}):" if caption else "Table:"
    return header + "\n" + "\n".join(rows)


async def fetch_pmc_fulltext(
    client: httpx.AsyncClient, pmcid: str, *, api_key: str = ""
) -> dict[str, Any] | None:
    """Returns ``{"title", "body_text", "captions", "tables"}`` if this
    article's full text is available in PMC's OA subset, else ``None``
    (never raises — a PMC lookup failure must fall back to an abstract,
    not break retrieval)."""
    params = {"db": "pmc", "id": pmcid, "rettype": "full", "retmode": "xml"}
    if api_key:
        params["api_key"] = api_key
    try:
        resp = await client.get(_EFETCH_URL, params=params)
        if resp.status_code != 200 or not resp.text:
            return None
        if _NO_FULLTEXT_MARKER in resp.text:
            return None
        root = ET.fromstring(resp.text)
    except Exception:
        return None

    body_el = root.find(".//body")
    if body_el is None:
        return None

    # One entry per <p>, joined with a paragraph break — never a single
    # itertext() over the whole <body>, which would concatenate every
    # paragraph into one undifferentiated blob and destroy the paragraph
    # boundaries downstream chunking/sentence-splitting depend on.
    paragraphs = [_clean_whitespace("".join(p.itertext())) for p in body_el.iter("p")]
    body_text = "\n\n".join(p for p in paragraphs if p)
    if len(body_text) < 200:
        return None

    tables = [t for t in (_extract_table(tw) for tw in body_el.iter("table-wrap")) if t]
    if tables:
        body_text = body_text + "\n\n" + "\n\n".join(tables)

    title_el = root.find(".//article-title")
    title = _clean_whitespace("".join(title_el.itertext())) if title_el is not None else ""
    captions = [_clean_whitespace("".join(cap.itertext())) for cap in root.iter("caption")]
    captions = [c for c in captions if c]

    return {"title": title, "body_text": body_text, "captions": captions, "tables": tables}
