from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from propab.llm import LLMClient
from propab.paper_compiler import _latex_escape


def _template_env() -> Path:
    return Path(__file__).resolve().parent / "templates"


def render_paper_tex(
    *,
    title: str,
    abstract: str,
    introduction: str,
    methods_tex: str,
    results_tex: str,
    figures_tex: str,
    discussion: str,
    conclusion: str,
    references_tex: str,
) -> str:
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(_template_env())), autoescape=False)
    tpl = env.get_template("paper_arxiv.tex.j2")
    return tpl.render(
        title=_latex_escape(title.strip()[:200] or "Research report"),
        abstract=abstract,
        introduction=introduction,
        methods_tex=methods_tex,
        results_tex=results_tex,
        figures_tex=figures_tex,
        discussion=discussion,
        conclusion=conclusion,
        references_tex=references_tex,
    )


def _confirmed_hypothesis_count(synthesis: dict[str, Any] | None) -> int:
    ledger = (synthesis or {}).get("ledger")
    if not isinstance(ledger, dict):
        return 0
    confirmed = ledger.get("confirmed")
    return len(confirmed) if isinstance(confirmed, list) else 0


def _fallback_from_context(question: str, prior: dict[str, Any], synthesis: dict[str, Any] | None) -> dict[str, str]:
    syn = synthesis or {}
    if syn.get("short_circuit"):
        ans = str(syn.get("short_answer") or "")
        body = _latex_escape(ans[:2000])
        return {
            "abstract": body or _latex_escape(question[:1200]),
            "introduction": _latex_escape(
                "This report documents a literature-aligned answer produced without full hypothesis testing."
            ),
            "discussion": _latex_escape(
                "The short-circuit path skipped multi-hypothesis experiments; see abstract and prior literature."
            ),
            "conclusion": _latex_escape(
                "Established literature supports the stated answer; reproduce with full experiments if stakes are high."
            ),
        }
    kpapers = prior.get("key_papers") or []
    cite = ""
    if isinstance(kpapers, list) and kpapers:
        first = kpapers[0] if isinstance(kpapers[0], dict) else {}
        cite = str(first.get("title", ""))[:200]
    intro_plain = (
        f"We address: {question[:900]}. "
        f"Prior retrieval surfaced {len(kpapers)} seed papers"
        + (f", including ``{cite}''." if cite else ".")
    )
    ledger = syn.get("ledger") or {}
    if not isinstance(ledger, dict):
        ledger = {}
    n_c = len(ledger.get("confirmed") or [])
    n_r = len(ledger.get("refuted") or [])
    n_i = len(ledger.get("inconclusive") or [])
    abs_plain = (
        f"This session investigated: {question[:700]}. "
        f"Automated sub-agent runs produced stored traces summarized in Results "
        f"(confirmed={n_c}, refuted={n_r}, inconclusive={n_i}). "
    )
    if n_c == 0:
        abs_plain += (
            "No hypothesis met the platform confirmation bar against the recorded tool outputs; "
            "do not treat narrative optimizer rankings as established facts without independent replication."
        )
    else:
        abs_plain += "Claims tied to confirmed hypotheses are summarized in Results; other rows remain tentative."

    disc_plain = (
        f"Automated experiments concluded with ledger "
        f"(confirmed={n_c}, refuted={n_r}, inconclusive={n_i}). "
        "Limitations include sandboxed execution and tool proxies where noted in outputs."
    )
    concl_plain = (
        "Primary takeaways depend on hypotheses marked confirmed in the results section; "
        "inconclusive rows indicate insufficient signal or tooling limits."
    )
    return {
        "abstract": _latex_escape(abs_plain[:2000]),
        "introduction": _latex_escape(intro_plain),
        "discussion": _latex_escape(disc_plain),
        "conclusion": _latex_escape(concl_plain),
    }


async def generate_prose_sections(
    *,
    llm: LLMClient | None,
    session_id: str,
    question: str,
    prior: dict[str, Any] | None,
    synthesis: dict[str, Any] | None,
) -> dict[str, str]:
    prior = prior or {}
    use_llm = llm is not None and (
        str(getattr(llm, "provider", "") or "").lower() in ("ollama", "gemini")
        or bool(str(getattr(llm, "api_key", "") or "").strip())
    )
    # LLM prose routinely asserts definitive rankings when every hypothesis is inconclusive;
    # the ledger from synthesis is authoritative for what may be claimed at high level.
    if use_llm and _confirmed_hypothesis_count(synthesis) == 0:
        use_llm = False
    if not use_llm:
        return _fallback_from_context(question, prior, synthesis)

    prior_blob = json.dumps(prior, ensure_ascii=False)[:6000]
    syn_blob = json.dumps(synthesis or {}, ensure_ascii=False)[:4000]

    async def one(purpose: str, instruction: str) -> str:
        prompt = (
            f"You write one concise LaTeX-free paragraph for an ML research report.\n"
            f"Research question: {question}\n"
            f"Context JSON (truncated): {prior_blob}\n"
            f"Synthesis JSON (truncated): {syn_blob}\n"
            f"Task: {instruction}\n"
            "Output plain text only, no markdown, no citations markup, <= 180 words."
        )
        raw = (await llm.call(prompt=prompt, purpose=purpose, session_id=session_id)).strip()
        if raw.startswith("[") or raw.startswith("{"):
            return ""
        return _latex_escape(raw[:2500])

    abstract = await one("paper.abstract", "Write the abstract summarizing problem, approach, and high-level outcome.")
    introduction = await one("paper.introduction", "Write the introduction: motivation, gap, and what was executed.")
    discussion = await one("paper.discussion", "Write discussion: interpret experiment ledger, limitations, rival explanations.")
    conclusion = await one("paper.conclusion", "Write a tight conclusion: what was learned, what remains open, one recommendation.")

    fb = _fallback_from_context(question, prior, synthesis)
    return {
        "abstract": abstract or fb["abstract"],
        "introduction": introduction or fb["introduction"],
        "discussion": discussion or fb["discussion"],
        "conclusion": conclusion or fb["conclusion"],
    }
