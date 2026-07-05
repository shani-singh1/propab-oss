"""
arXiv source — full-text retrieval, not abstract retrieval.

Dual extraction path: LaTeX e-print (ground truth, exact notation) is tried
first; PDF text extraction is the fallback when no LaTeX source exists or the
LaTeX parse looks too thin (Step 8 quality check in agent3.md). We use
``pypdf`` for the PDF fallback rather than nougat: nougat requires a multi-GB
vision-transformer checkpoint and a GPU-friendly torch install, which is not
a reasonable dependency for a service whose job is text/citation extraction,
not layout recovery. ``fetch_full_text`` records which path was used in
``extraction_method`` so a future nougat backend can be swapped in behind the
same interface without changing any caller.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx

from services.literature.app.models import FullTextDocument, RawDocument, TabMatch
from services.literature.app.sources.base import BaseSource, RateLimiter
from services.literature.app.sources._latex import flatten_inputs, parse_latex_document

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?")

# Signal phrases used both for extraction-quality scoring (Step 8) and
# available for extractors/claims.py to reuse if desired.
CLAIM_SIGNAL_PHRASES = (
    "we show", "we prove", "we establish", "it is known that", "it is unknown",
    "it remains open", "we conjecture", "open problem", "open question",
    "it follows that", "this implies", "as a consequence", "we note that",
    "one can show", "the following is known", "is not known whether",
    "it would be interesting to determine",
)


def normalize_arxiv_id(raw: str) -> str:
    m = _ARXIV_ID_RE.search(raw)
    if not m:
        return raw.strip().strip("/")
    return m.group(1)


class ArxivSource(BaseSource):
    name = "arxiv"
    supported_domains: tuple[str, ...] = ()  # relevant to all domains by default

    def __init__(
        self,
        *,
        cache_dir: str = "./data/literature_cache",
        max_results: int = 200,
        min_interval_sec: float = 3.0,
        http_timeout: float = 30.0,
        user_agent: str = "propab-literature/0.1",
    ) -> None:
        super().__init__(http_timeout=http_timeout, user_agent=user_agent)
        self._cache_dir = Path(cache_dir) / "arxiv"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_results = max_results
        self._rate_limiter = RateLimiter(min_interval_sec)

    # -- search --------------------------------------------------------------

    async def search(self, query: str, profile: dict[str, Any]) -> list[RawDocument]:
        terms = [query] + list(profile.get("search_terms", []) or [])
        search_query = " OR ".join(f"all:{_quote(t)}" for t in terms if t.strip())
        classes = (profile.get("classification_codes", {}) or {}).get("arxiv", [])
        if classes:
            cat_query = " OR ".join(f"cat:{c}" for c in classes)
            search_query = f"({search_query}) AND ({cat_query})"

        client = await self._get_client()
        results: list[RawDocument] = []
        start = 0
        page_size = min(100, self._max_results)
        while start < self._max_results:
            await self._rate_limiter.wait()
            params = {
                "search_query": search_query,
                "start": start,
                "max_results": min(page_size, self._max_results - start),
                "sortBy": "relevance",
            }
            resp = await client.get("http://export.arxiv.org/api/query", params=params)
            resp.raise_for_status()
            batch = _parse_atom_feed(resp.text)
            if not batch:
                break
            results.extend(batch)
            start += len(batch)
            if len(batch) < page_size:
                break
        return results

    # -- full text -------------------------------------------------------------

    async def fetch_full_text(self, doc: RawDocument) -> FullTextDocument:
        arxiv_id = normalize_arxiv_id(doc.external_id)
        cached = self._read_cache(arxiv_id)
        if cached is not None:
            return cached

        latex_source = await self._fetch_eprint(arxiv_id)
        if latex_source is not None:
            full_doc = self._extract_from_latex(doc, arxiv_id, latex_source)
            quality = _extraction_quality(full_doc)
            full_doc.extraction_quality = quality
            if quality >= 0.02:
                self._write_cache(arxiv_id, full_doc)
                return full_doc
            # Step 8: thin LaTeX parse — fall back to PDF, but keep whatever
            # LaTeX gave us if PDF extraction fails entirely.
            fallback = await self._fetch_pdf_text(doc, arxiv_id)
            best = fallback if fallback is not None and fallback.extraction_quality > quality else full_doc
            self._write_cache(arxiv_id, best)
            return best

        pdf_doc = await self._fetch_pdf_text(doc, arxiv_id)
        if pdf_doc is not None:
            self._write_cache(arxiv_id, pdf_doc)
            return pdf_doc

        abstract_only = FullTextDocument(
            source="arxiv",
            external_id=arxiv_id,
            title=doc.title,
            authors=doc.authors,
            year=doc.year,
            doi=doc.doi,
            url=doc.url,
            body_text=doc.abstract,
            extraction_method="abstract_only",
            extraction_quality=0.0,
        )
        self._write_cache(arxiv_id, abstract_only)
        return abstract_only

    async def check_tabulated(self, values: dict[str, Any]) -> list[TabMatch]:
        # arXiv is not a tabulation authority itself (OEIS/domain sources are);
        # a source with no tabulation role returns no matches rather than
        # raising, so pipeline fan-out over all sources stays simple.
        return []

    async def health(self) -> bool:
        try:
            client = await self._get_client()
            resp = await client.get(
                "http://export.arxiv.org/api/query",
                params={"search_query": "all:test", "max_results": 1},
                timeout=10.0,
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    # -- internals -------------------------------------------------------------

    async def _fetch_eprint(self, arxiv_id: str) -> dict[str, str] | None:
        await self._rate_limiter.wait()
        client = await self._get_client()
        try:
            resp = await client.get(f"https://arxiv.org/e-print/{arxiv_id}")
        except httpx.HTTPError:
            return None
        if resp.status_code != 200 or not resp.content:
            return None
        content = resp.content
        # Single-file sources are sometimes raw .tex (not gzipped) or gzip-only.
        try:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as tar:
                files: dict[str, str] = {}
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith((".tex", ".bbl")):
                        f = tar.extractfile(member)
                        if f is not None:
                            try:
                                files[Path(member.name).name] = f.read().decode("utf-8", errors="ignore")
                            except Exception:
                                continue
                return files or None
        except tarfile.ReadError:
            try:
                text = content.decode("utf-8", errors="ignore")
                if "\\documentclass" in text or "\\begin{document}" in text:
                    return {f"{arxiv_id}.tex": text}
            except Exception:
                return None
            return None

    def _extract_from_latex(self, doc: RawDocument, arxiv_id: str, files: dict[str, str]) -> FullTextDocument:
        main_candidates = [f for f in files if "\\documentclass" in files[f]] or list(files)
        main_name = main_candidates[0]
        flattened = flatten_inputs(files[main_name], files)
        parsed = parse_latex_document(flattened)

        # Supplementary .tex files (Step 4): parse independently and merge —
        # tables/environments there are as citable as the main body.
        for fname, content in files.items():
            if fname == main_name:
                continue
            supp = parse_latex_document(content)
            for t in supp.tables_raw:
                t["location"] = t["location"] + " (supplementary)"
                t["is_supplementary"] = True
            parsed.tables_raw.extend(supp.tables_raw)
            parsed.latex_environments.extend(supp.latex_environments)
            parsed.footnotes.extend(supp.footnotes)
            parsed.captions.extend(supp.captions)

        return FullTextDocument(
            source="arxiv",
            external_id=arxiv_id,
            title=doc.title,
            authors=doc.authors,
            year=doc.year,
            doi=doc.doi,
            url=doc.url or f"https://arxiv.org/abs/{arxiv_id}",
            body_text=parsed.body_text,
            latex_environments=parsed.latex_environments,
            tables_raw=parsed.tables_raw,
            footnotes=parsed.footnotes,
            captions=parsed.captions,
            bibliography=parsed.bibliography,
            cite_sentences=parsed.cite_sentences,
            extraction_method="latex",
            is_appendix_included=True,
        )

    async def _fetch_pdf_text(self, doc: RawDocument, arxiv_id: str) -> FullTextDocument | None:
        await self._rate_limiter.wait()
        client = await self._get_client()
        try:
            resp = await client.get(f"https://arxiv.org/pdf/{arxiv_id}")
        except httpx.HTTPError:
            return None
        if resp.status_code != 200 or not resp.content:
            return None
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(resp.content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return None
        if not text.strip():
            return None
        return FullTextDocument(
            source="arxiv",
            external_id=arxiv_id,
            title=doc.title,
            authors=doc.authors,
            year=doc.year,
            doi=doc.doi,
            url=doc.url or f"https://arxiv.org/abs/{arxiv_id}",
            body_text=text,
            extraction_method="pdf_text",
            extraction_quality=_text_signal_density(text),
            is_appendix_included=True,
        )

    def _cache_path(self, arxiv_id: str) -> Path:
        safe = hashlib.sha1(arxiv_id.encode()).hexdigest()[:16]
        return self._cache_dir / f"{safe}.json"

    def _read_cache(self, arxiv_id: str) -> FullTextDocument | None:
        path = self._cache_path(arxiv_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return FullTextDocument(**data)
        except Exception:
            return None

    def _write_cache(self, arxiv_id: str, doc: FullTextDocument) -> None:
        path = self._cache_path(arxiv_id)
        path.write_text(doc.model_dump_json(), encoding="utf-8")


def _quote(term: str) -> str:
    term = term.strip()
    if " " in term:
        return f'"{term}"'
    return term


def _parse_atom_feed(xml_text: str) -> list[RawDocument]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out: list[RawDocument] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        id_el = entry.find(f"{_ATOM_NS}id")
        if id_el is None or not id_el.text:
            continue
        arxiv_id = normalize_arxiv_id(id_el.text)
        title = (entry.findtext(f"{_ATOM_NS}title") or "").strip().replace("\n", " ")
        summary = (entry.findtext(f"{_ATOM_NS}summary") or "").strip()
        authors = ", ".join(
            (a.findtext(f"{_ATOM_NS}name") or "").strip()
            for a in entry.findall(f"{_ATOM_NS}author")
        )
        published = entry.findtext(f"{_ATOM_NS}published") or ""
        year = int(published[:4]) if published[:4].isdigit() else 0
        doi = entry.findtext(f"{_ARXIV_NS}doi") or ""
        out.append(
            RawDocument(
                source="arxiv",
                external_id=arxiv_id,
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                abstract=summary,
            )
        )
    return out


def _extraction_quality(doc: FullTextDocument) -> float:
    """(# math environments found) / (estimated paragraphs) — Step 8 check."""
    n_envs = len(doc.latex_environments)
    n_paragraphs = max(1, doc.body_text.count("\n\n"))
    return round(n_envs / n_paragraphs, 4)


def _text_signal_density(text: str) -> float:
    lower = text.lower()
    hits = sum(lower.count(p) for p in CLAIM_SIGNAL_PHRASES)
    paragraphs = max(1, text.count("\n\n"))
    return round(hits / paragraphs, 4)
