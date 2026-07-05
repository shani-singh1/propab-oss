"""
Minimal but real LaTeX structural parser used by the arXiv source.

We deliberately do not depend on a full LaTeX-to-AST library (pylatexenc is
built for single-macro expansion, not environment extraction across a whole
paper made of many .tex files with \\input chains). Instead this module does
brace-balanced environment extraction, which is what actually matters for
"pull out every theorem/lemma/footnote/caption/tabular verbatim" — the exact
mathematical notation must survive untouched, so we never render or expand
macros, we only locate balanced spans.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field

_ENV_NAMES = (
    "theorem", "lemma", "proposition", "corollary", "conjecture", "claim",
    "observation", "remark", "definition", "example", "proof",
)

_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
_SECTION_RE = re.compile(r"\\(section|subsection|subsubsection)\*?\{")
_APPENDIX_RE = re.compile(r"\\appendix\b")
_BEGIN_RE = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
_CITE_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CITE_RE = re.compile(r"\\cite[a-zA-Z]*(?:\[[^\]]*\])?\{([^}]+)\}")
_BIBITEM_RE = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}")


def strip_comments(tex: str) -> str:
    return "\n".join(_COMMENT_RE.sub("", line) for line in tex.split("\n"))


def _find_matching_brace(text: str, open_idx: int) -> int:
    """``open_idx`` points at an opening ``{``. Return index just past the
    matching ``}``, or -1 if unbalanced (truncated source)."""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _find_env_span(text: str, begin_match: re.Match) -> tuple[int, int, str] | None:
    """Given a \\begin{env} match, find the balanced \\end{env}, respecting
    nested environments of the same name."""
    env = begin_match.group(1)
    depth = 1
    pos = begin_match.end()
    begin_re = re.compile(r"\\begin\{" + re.escape(env) + r"\}")
    end_re = re.compile(r"\\end\{" + re.escape(env) + r"\}")
    while pos < len(text):
        nb = begin_re.search(text, pos)
        ne = end_re.search(text, pos)
        if ne is None:
            return None
        if nb is not None and nb.start() < ne.start():
            depth += 1
            pos = nb.end()
            continue
        depth -= 1
        pos = ne.end()
        if depth == 0:
            return begin_match.start(), pos, text[begin_match.end():ne.start()]
    return None


@dataclass
class ParsedDocument:
    body_text: str
    latex_environments: list[dict] = field(default_factory=list)
    tables_raw: list[dict] = field(default_factory=list)
    footnotes: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    bibliography: list[dict] = field(default_factory=list)
    cite_sentences: list[dict] = field(default_factory=list)


def _section_index(tex: str) -> list[tuple[int, str]]:
    """Positions of section-like headers -> a human location label, plus the
    \\appendix boundary so later spans can be labeled 'appendix B' etc."""
    marks: list[tuple[int, str]] = []
    appendix_start = None
    am = _APPENDIX_RE.search(tex)
    if am:
        appendix_start = am.start()
    appendix_letter_counter = 0
    for m in _SECTION_RE.finditer(tex):
        end = _find_matching_brace(tex, tex.index("{", m.end() - 1))
        title = tex[m.end():end - 1].strip() if end > 0 else ""
        title = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", title)[:60]
        kind = m.group(1)
        if appendix_start is not None and m.start() >= appendix_start:
            if kind == "section":
                appendix_letter_counter += 1
                label = f"appendix {chr(64 + appendix_letter_counter)}"
                if title:
                    label += f" ({title})"
            else:
                label = f"appendix ({title})" if title else "appendix"
        else:
            label = f"{kind} \"{title}\"" if title else kind
        marks.append((m.start(), label))
    return marks


def _location_for(pos: int, section_marks: list[tuple[int, str]]) -> str:
    if not section_marks:
        return "body"
    positions = [p for p, _ in section_marks]
    idx = bisect_right(positions, pos) - 1
    if idx < 0:
        return "preamble/abstract"
    return section_marks[idx][1]


def parse_latex_document(tex: str) -> ParsedDocument:
    """Parse a single (already \\input-flattened) LaTeX source string into
    the structural pieces the pipeline needs. Never modifies the verbatim
    text of a matched span — callers extract ``text[start:end]`` unchanged.
    """
    tex = strip_comments(tex)
    section_marks = _section_index(tex)
    doc = ParsedDocument(body_text=tex)

    # 1. Environments (theorem/lemma/.../proof), brace/env-balanced.
    seen_spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        m = _BEGIN_RE.search(tex, pos)
        if not m:
            break
        env = m.group(1).rstrip("*")
        if env not in _ENV_NAMES and env not in ("table", "figure", "tabular"):
            pos = m.end()
            continue
        span = _find_env_span(tex, m)
        if span is None:
            pos = m.end()
            continue
        start, end, content = span
        if env in _ENV_NAMES:
            doc.latex_environments.append(
                {
                    "env": env,
                    "content": content.strip(),
                    "location": _location_for(start, section_marks),
                }
            )
        elif env in ("table", "figure"):
            # captions are pulled separately (may be nested); tables get their
            # tabular body recorded here if present.
            tab_m = re.search(r"\\begin\{tabular\}", content)
            if tab_m:
                inner = _BEGIN_RE.search(content, tab_m.start())
                inner_span = _find_env_span(content, inner) if inner else None
                if inner_span:
                    doc.tables_raw.append(
                        {
                            "raw": inner_span[2].strip(),
                            "location": _location_for(start, section_marks) + f" ({env})",
                            "is_appendix": "appendix" in _location_for(start, section_marks),
                        }
                    )
        pos = end
    # Bare \begin{tabular} not wrapped in a table float.
    pos = 0
    while True:
        m = re.search(r"\\begin\{tabular\}", tex[pos:])
        if not m:
            break
        abs_start = pos + m.start()
        bm = _BEGIN_RE.match(tex, abs_start)
        span = _find_env_span(tex, bm) if bm else None
        if span:
            already = any(span[2].strip() == t["raw"] for t in doc.tables_raw)
            if not already:
                doc.tables_raw.append(
                    {
                        "raw": span[2].strip(),
                        "location": _location_for(abs_start, section_marks) + " (tabular)",
                        "is_appendix": "appendix" in _location_for(abs_start, section_marks),
                    }
                )
            pos = span[1]
        else:
            pos = abs_start + len("\\begin{tabular}")

    # 2. Footnotes — \footnote{...}, numbered in document order.
    pos = 0
    fn_i = 0
    while True:
        m = re.search(r"\\footnote\{", tex[pos:])
        if not m:
            break
        abs_start = pos + m.start()
        brace_idx = pos + m.end() - 1
        end = _find_matching_brace(tex, brace_idx)
        if end == -1:
            break
        fn_i += 1
        doc.footnotes.append(tex[brace_idx + 1:end - 1].strip())
        pos = end

    # 3. Captions — \caption{...}, including subfigures.
    pos = 0
    while True:
        m = re.search(r"\\caption\{", tex[pos:])
        if not m:
            break
        brace_idx = pos + m.end() - 1
        end = _find_matching_brace(tex, brace_idx)
        if end == -1:
            break
        doc.captions.append(tex[brace_idx + 1:end - 1].strip())
        pos = end

    # 4. Bibliography — \bibitem entries, each followed by its raw citation text.
    bibitems = list(_BIBITEM_RE.finditer(tex))
    for i, m in enumerate(bibitems):
        end = bibitems[i + 1].start() if i + 1 < len(bibitems) else min(len(tex), m.end() + 800)
        raw = tex[m.end():end].strip()
        raw = re.sub(r"\s+", " ", raw)[:500]
        doc.bibliography.append({"key": m.group(1), "raw": raw})

    # 5. Sentences containing \cite{...} — the citing paper's own annotation
    # of what the cited work established. The verbatim requirement applies
    # here too: "text" must be the exact source text (raw \cite{...} markup
    # and all), never a paraphrase substituting a "[cite]" placeholder — a
    # rewritten claim can't be verified against the source it claims to quote.
    # Split into paragraphs first: a sentence-terminator lookbehind can miss
    # (e.g. ".)" — the period isn't immediately followed by whitespace), which
    # would otherwise let a "sentence" run on through a paragraph break and
    # splice unrelated text from the next paragraph into one claim.
    plain = re.sub(r"\\(?:begin|end)\{[^}]*\}", " ", tex)
    for paragraph in re.split(r"\n\s*\n", plain):
        for sent in _CITE_SENTENCE_SPLIT_RE.split(paragraph):
            if "\\cite" in sent:
                cite_m = _CITE_RE.search(sent)
                keys = cite_m.group(1).split(",") if cite_m else []
                verbatim = sent.strip()
                if len(verbatim) > 15:
                    doc.cite_sentences.append({"text": verbatim, "keys": [k.strip() for k in keys]})

    # body_text for linguistic scanning: strip only \label/\ref (pure noise,
    # non-semantic cross-reference plumbing). \cite{...} is deliberately left
    # intact — cite_sentences above quotes raw text including \cite{...}
    # markup, and that verbatim must actually be findable in body_text for
    # citation-verification re-fetch checks to mean anything.
    doc.body_text = re.sub(r"\\(?:label|ref)\{[^}]*\}", "", tex)
    return doc


def flatten_inputs(main_tex: str, file_lookup: dict[str, str], _depth: int = 0) -> str:
    """Recursively substitute \\input{X}/\\include{X} with the referenced
    file's content so multi-file arXiv sources parse as one document."""
    if _depth > 6:
        return main_tex

    def _sub(m: re.Match) -> str:
        name = m.group(1)
        for candidate in (name, name + ".tex"):
            if candidate in file_lookup:
                return flatten_inputs(file_lookup[candidate], file_lookup, _depth + 1)
        return ""

    return _INPUT_RE.sub(_sub, main_tex)
