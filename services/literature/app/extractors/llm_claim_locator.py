"""
LLM-assisted claim location — with verbatim extraction kept 100% code-side.

The problem this solves: ``claims.py``'s linguistic scan requires an exact
signal phrase ("we show", "it is known that", ...) to find a claim. That
works for arXiv's hedge-heavy prose but has near-zero recall on PubMed/
bioRxiv abstracts (measured live: running the full extraction pipeline
against real biology papers found real documents but extracted zero
``established_facts`` from them — see CHANGELOG.md). An LLM reading the same
abstract can identify "this sentence states the answer" without needing a
magic phrase.

The problem that creates: an LLM asked to "extract the claim" will paraphrase
it. Paraphrase breaks the citation contract outright — the whole point of
``verbatim`` is that it is checkable against the source (see
``evaluator/metrics.citation_verification_rate``), and a paraphrase is not
findable in the source it claims to quote.

The fix, and the only thing that makes this module safe to use: the LLM is
never asked for claim *text*. It is asked for a *sentence index* into a
list this module builds and numbers itself. Verbatim extraction is a pure
lookup — ``sentences[index]`` — performed entirely by this module, using the
LLM's response only for *which* index to look up. Even if a model ignores
instructions and echoes text back in its response, that text is discarded;
only the index is read. This is enforced in code (see ``locate_claims``
below), not by prompt wording alone, and is covered by
``tests/test_llm_claim_locator.py::test_ignores_llm_supplied_text_field``
specifically to keep it that way.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from services.literature.app.llm_client import gemini_generate
from services.literature.app.models import ClaimStatus, ClaimType, ExtractedClaim, FullTextDocument

_VALID_CLAIM_TYPES: frozenset[str] = frozenset(ClaimType.__args__)  # type: ignore[attr-defined]
_VALID_STATUSES: frozenset[str] = frozenset(ClaimStatus.__args__)  # type: ignore[attr-defined]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d])")
_INDEX_RE = re.compile(r'"index"\s*:\s*(\d+)')


def split_sentences(text: str) -> list[str]:
    """Deterministic, code-only sentence split — paragraph-scoped (never
    crosses a ``\\n\\n`` boundary) for the same reason ``claims.py``'s
    linguistic scan is: a sentence-terminator lookbehind can miss (e.g. an
    abbreviation, or ".)" not followed by whitespace), and without paragraph
    scoping a "sentence" can run on and splice unrelated text across a
    structural seam. This is the ONE place that defines what "sentence N"
    means — the LLM only ever sees indices into this exact list."""
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text or ""):
        for sent in _SENTENCE_SPLIT_RE.split(paragraph):
            cleaned = sent.strip()
            if len(cleaned) >= 15:
                sentences.append(cleaned)
    return sentences


def build_location_prompt(sentences: list[str]) -> str:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return (
        "Below is a numbered list of sentences from a scientific document. "
        "Identify which sentences state a specific factual, quantitative, or "
        "scientific claim (something proven, observed, conjectured, or explicitly "
        "left open) — not filler, background, or purely procedural sentences.\n\n"
        "For each claim sentence, report ONLY its index and a status/type judgment. "
        "Do NOT restate, quote, or paraphrase the sentence itself — report the index only; "
        "the exact sentence will be looked up separately.\n\n"
        "Respond with exactly one JSON object on a single line, in this exact shape:\n"
        '{"claims": [{"index": 3, "claim_type": "observation", "status": "proven"}, ...]}\n\n'
        f"claim_type must be one of: {sorted(_VALID_CLAIM_TYPES)}\n"
        f"status must be one of: {sorted(_VALID_STATUSES)}\n\n"
        f"Sentences:\n{numbered}"
    )


def _parse_locations(raw: str) -> list[dict[str, Any]]:
    """Parses the LLM's response into a list of {index, claim_type, status}
    dicts. Any "text"/"verbatim"/"sentence" key the model includes anyway is
    dropped here — deliberately never read past this function, so it cannot
    leak into an ExtractedClaim.verbatim downstream."""
    m = re.search(r'"claims"\s*:\s*(\[.*\])', raw, re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if not isinstance(item, dict) or "index" not in item:
            continue
        try:
            idx = int(item["index"])
        except (TypeError, ValueError):
            continue
        out.append({"index": idx, "claim_type": item.get("claim_type"), "status": item.get("status")})
    return out


def _claim_id(text: str, location: str) -> str:
    return hashlib.sha1(f"{location}:{text}".encode("utf-8", "ignore")).hexdigest()[:16]


async def locate_claims(
    doc: FullTextDocument, *, api_key: str, model: str, max_sentences: int = 200
) -> list[ExtractedClaim]:
    """Find claims in ``doc.body_text`` via LLM sentence-location + code-only
    verbatim extraction. Returns ``[]`` (never raises into the caller) if no
    API key is configured, the document is empty, or the LLM call/response
    is unusable — a disabled or flaky locator must degrade to "no extra
    claims found," not break the pipeline.
    """
    if not api_key:
        return []
    sentences = split_sentences(doc.body_text)[:max_sentences]
    if not sentences:
        return []

    prompt = build_location_prompt(sentences)
    try:
        raw = await gemini_generate(prompt, api_key=api_key, model=model)
    except Exception:
        return []

    locations = _parse_locations(raw)
    claims: list[ExtractedClaim] = []
    seen_indices: set[int] = set()
    for loc in locations:
        idx = loc["index"]
        if idx in seen_indices or not (0 <= idx < len(sentences)):
            continue
        seen_indices.add(idx)

        # The verbatim field is ALWAYS this lookup — never anything derived
        # from the LLM's own response text, by construction.
        verbatim = sentences[idx]

        claim_type = loc.get("claim_type")
        if claim_type not in _VALID_CLAIM_TYPES:
            claim_type = "observation"
        status = loc.get("status")
        if status not in _VALID_STATUSES:
            status = "unknown"

        location_label = f"sentence {idx} (LLM-located)"
        claims.append(
            ExtractedClaim(
                text=verbatim,
                claim_type=claim_type,  # type: ignore[arg-type]
                status=status,  # type: ignore[arg-type]
                verbatim=verbatim,
                source=doc.source,
                source_doi=doc.doi,
                source_title=doc.title,
                source_authors=doc.authors,
                source_year=doc.year,
                source_url=doc.url,
                location=location_label,
                claim_id=_claim_id(verbatim, location_label),
            )
        )
    return claims
