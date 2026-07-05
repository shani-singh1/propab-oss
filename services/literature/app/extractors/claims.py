"""
Claims extractor — every statement anywhere in a document that asserts
something is true, proven, conjectured, or open. No part of a paper is out
of scope: main text, appendices (already merged into ``latex_environments``/
``tables_raw`` by the arXiv source with an "appendix ..." location), footnotes,
figure/table captions, remarks after theorems, bibliography annotations, and
quantitative statements inside proof bodies.

Domain-agnostic: this module has no idea what a Sidon set or a GTEx tissue
is. It only knows how to recognize the *shape* of a scientific claim.
"""
from __future__ import annotations

import hashlib
import re

from services.literature.app.extractors.base import BaseExtractor
from services.literature.app.models import ClaimStatus, ExtractedClaim, FullTextDocument

SIGNAL_PHRASES = (
    "we show", "we prove", "we establish", "it is known that", "it is unknown",
    "it remains open", "we conjecture", "open problem", "open question",
    "it follows that", "this implies", "as a consequence", "we note that",
    "one can show", "the following is known", "is not known whether",
    "it would be interesting to determine", "it is not known whether",
    "as a corollary", "the following is immediate",
)

_REMARK_START_RE = re.compile(
    r"^\s*(note|remark|we observe|in particular|it is worth noting)\b", re.I
)
_OPEN_RE = re.compile(
    r"\b(open problem|open question|remains open|is not known whether|it is unknown|"
    r"it would be interesting to determine)\b", re.I
)
_DISPROVEN_RE = re.compile(r"\b(disprove[sd]?|counterexample (?:shows|demonstrates)|is false|refute[sd]?)\b", re.I)
_CONJECTURE_RE = re.compile(r"\bconjectur", re.I)
_PROOF_QUANT_RE = re.compile(
    r"\b(this gives us that|we obtain a bound of|this yields|we obtain that|"
    r"which gives|we deduce that)\b", re.I
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\$])")

_ENV_TO_STATUS: dict[str, ClaimStatus] = {
    "theorem": "proven", "lemma": "proven", "proposition": "proven",
    "corollary": "proven", "conjecture": "conjectured", "claim": "proven",
    "observation": "proven", "remark": "unknown", "definition": "unknown",
    "example": "unknown",
}


def _claim_id(text: str, location: str) -> str:
    return hashlib.sha1(f"{location}:{text}".encode("utf-8", "ignore")).hexdigest()[:16]


def _infer_status(text: str, default: ClaimStatus) -> ClaimStatus:
    if _OPEN_RE.search(text):
        return "open"
    if _DISPROVEN_RE.search(text):
        return "disproven"
    if _CONJECTURE_RE.search(text):
        return "conjectured"
    return default


class ClaimsExtractor(BaseExtractor):
    name = "claims"

    async def extract(self, doc: FullTextDocument) -> list[ExtractedClaim]:
        claims: list[ExtractedClaim] = []
        seen_verbatim: set[str] = set()

        def _add(text: str, claim_type: str, status: ClaimStatus, location: str) -> None:
            text = text.strip()
            if len(text) < 8 or text in seen_verbatim:
                return
            seen_verbatim.add(text)
            claims.append(
                ExtractedClaim(
                    text=text,
                    claim_type=claim_type,  # type: ignore[arg-type]
                    status=status,
                    verbatim=text,
                    source=doc.source,
                    source_doi=doc.doi,
                    source_title=doc.title,
                    source_authors=doc.authors,
                    source_year=doc.year,
                    source_url=doc.url,
                    location=location,
                    claim_id=_claim_id(text, location),
                )
            )

        # 1. Structural: LaTeX math environments (Step 3). Proof bodies are
        # context, not standalone claims — but their quantitative sentences are
        # separately mined below.
        for env in doc.latex_environments:
            name = env.get("env", "")
            content = env.get("content", "")
            location = env.get("location", "body")
            if name == "proof":
                for sent in _SENTENCE_SPLIT_RE.split(content):
                    if _PROOF_QUANT_RE.search(sent):
                        _add(sent, "proof_intermediate", "proven", f"{location}, proof body")
                continue
            status = _infer_status(content, _ENV_TO_STATUS.get(name, "unknown"))
            claim_type = name if name in (
                "theorem", "lemma", "proposition", "corollary", "conjecture",
                "claim", "observation", "remark", "definition", "example",
            ) else "observation"
            _add(content, claim_type, status, location)

        # 2. Footnotes with a mathematical/scientific assertion.
        for i, fn in enumerate(doc.footnotes, start=1):
            if _looks_like_assertion(fn):
                _add(fn, "footnote_claim", _infer_status(fn, "unknown"), f"footnote {i}")

        # 3. Figure/table captions that assert something.
        for i, cap in enumerate(doc.captions, start=1):
            if _looks_like_assertion(cap):
                _add(cap, "caption_claim", _infer_status(cap, "unknown"), f"caption {i}")

        # 4. Remarks immediately following a theorem-like environment, and any
        # free-text paragraph opening with a remark marker — caught by the
        # linguistic scan below via _REMARK_START_RE.

        # 5. Linguistic scan over body text for signal phrases not already
        # captured structurally (covers papers with no LaTeX environments —
        # i.e. PDF-extracted or abstract-only documents too). Sentence-split
        # within each paragraph separately, never across a paragraph break:
        # a document's structural seams ("\n\n") — section breaks, injected
        # "Answer (score N):" markers between Q&A turns, etc. — are not
        # sentence boundaries the LaTeX/prose splitter recognizes, so without
        # this a "claim" can silently splice unrelated text across them.
        body = doc.body_text or ""
        for paragraph in re.split(r"\n\s*\n", body):
            for sent in _SENTENCE_SPLIT_RE.split(paragraph):
                sent_stripped = sent.strip()
                if len(sent_stripped) < 15 or len(sent_stripped) > 600:
                    continue
                lower = sent_stripped.lower()
                has_signal = any(p in lower for p in SIGNAL_PHRASES)
                is_remark = bool(_REMARK_START_RE.match(sent_stripped))
                if not (has_signal or is_remark):
                    continue
                claim_type = "remark" if is_remark else "observation"
                status = _infer_status(sent_stripped, "unknown" if is_remark else "proven")
                _add(sent_stripped, claim_type, status, "body")

        # Bibliography annotations (sentences citing another work) are handled
        # by extract_bibliography_annotations() below — the arXiv source
        # parses \cite{} sentences separately from FullTextDocument.body_text,
        # so the pipeline calls that entry point directly when available.
        return claims


def _looks_like_assertion(text: str) -> bool:
    lower = text.lower()
    if any(p in lower for p in SIGNAL_PHRASES):
        return True
    if re.search(r"[<>=≈≤≥]|\bn\s*=\s*\d|\d+\s*(?:>|<|≥|≤)|monoton|density|converges|bound", lower):
        return True
    return False


async def extract_bibliography_annotations(doc: FullTextDocument) -> list[ExtractedClaim]:
    """Sentences citing another work (``doc.cite_sentences``, populated by
    the arXiv source's LaTeX parser) treated as claims about what the cited
    work established — often more precise than the cited paper's own
    abstract (agent3.md Step 5)."""
    out = []
    seen: set[str] = set()
    for cs in doc.cite_sentences:
        text = cs.get("text", "").strip()
        if len(text) < 15 or text in seen:
            continue
        seen.add(text)
        out.append(
            ExtractedClaim(
                text=text,
                claim_type="bibliography_annotation",
                status=_infer_status(text, "proven"),
                verbatim=text,
                source=doc.source,
                source_doi=doc.doi,
                source_title=doc.title,
                source_authors=doc.authors,
                source_year=doc.year,
                source_url=doc.url,
                location="bibliography annotation",
                claim_id=_claim_id(text, "bibliography annotation"),
            )
        )
    return out
