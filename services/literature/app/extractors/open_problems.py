"""
Open-problems extractor — explicitly stated unresolved questions.

This is the raw material for ``gap_mapper.py``: Propab should target what the
literature says is open and approachable by computation, not "what sounds
interesting." Survey papers with explicit problem lists are the highest-value
source (agent3.md: "these are gold") — this extractor treats every list item
following a "Problem:"/"Open problem:"/"Question:" marker as its own entry,
in addition to whole paragraphs matching the same markers.
"""
from __future__ import annotations

import re

from services.literature.app.extractors.base import BaseExtractor
from services.literature.app.models import FullTextDocument, OpenProblem

PROBLEM_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?(Open\s+[Pp]roblem|Problem|Question|Conjecture)\s*\d*\s*(?:\*\*)?[:.]\s*(.+?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)
_OPEN_SIGNAL_RE = re.compile(
    r"\b(it is unknown|it remains open|it would be interesting to (?:know|determine)|"
    r"is not known whether|open (?:problem|question))\b", re.I
)
_APPROACHABLE_RE = re.compile(
    r"\b(comput(?:e|ation|ationally)|numerical(?:ly)?|algorithm|enumerat|search|"
    r"exhaustive|brute[- ]force|simulat|bound|estimate)\b", re.I
)
_PREFIX_STRIP_RE = re.compile(
    r"^\s*(?:\*\*)?(?:Open\s+[Pp]roblem|Problem|Question|Conjecture)\s*\d*\s*(?:\*\*)?[:.]\s*", re.IGNORECASE
)


def _approachable_angle(text: str) -> str:
    m = _APPROACHABLE_RE.search(text)
    if not m:
        return ""
    word = m.group(1).lower()
    return f"Candidate approach signalled by '{word}' in the statement — worth a computational search/bound-tightening pass."


class OpenProblemsExtractor(BaseExtractor):
    name = "open_problems"

    async def extract(self, doc: FullTextDocument) -> list[OpenProblem]:
        out: list[OpenProblem] = []
        seen: set[str] = set()

        def _add(statement: str, context: str = "") -> None:
            statement = re.sub(r"\s+", " ", statement).strip()
            if len(statement) < 10 or statement in seen:
                return
            seen.add(statement)
            approachable = bool(_APPROACHABLE_RE.search(statement))
            out.append(
                OpenProblem(
                    statement=statement,
                    source_doi=doc.doi,
                    source_title=doc.title,
                    stated_by=doc.authors,
                    year=doc.year,
                    context=context,
                    computationally_approachable=approachable,
                    approachable_angle=_approachable_angle(statement) if approachable else "",
                )
            )

        # 1. Explicit "Problem:"/"Open problem:"/"Question:" markers.
        body = doc.body_text or ""
        for m in PROBLEM_MARKER_RE.finditer(body):
            _add(m.group(2))

        # 2. LaTeX conjecture/claim environments with open status, and any
        # sentence carrying an open-status signal phrase.
        for env in doc.latex_environments:
            content = env.get("content", "")
            if env.get("env") == "conjecture" or _OPEN_SIGNAL_RE.search(content):
                _add(content, context=f"from {env.get('location', 'body')} ({env.get('env')})")

        for sent in re.split(r"(?<=[.!?])\s+", body):
            if _OPEN_SIGNAL_RE.search(sent) and 10 < len(sent) < 500:
                _add(_PREFIX_STRIP_RE.sub("", sent))

        return out
